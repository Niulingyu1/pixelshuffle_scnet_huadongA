#!/bin/bash
# 归档（2026-09-18）：正式长跑不走 Slurm。
# 现行入口：bash /public/home/acd7koea4a/work/scripts/launch_platform_train.sh
# 本文件仅作历史参考，不要当现行命令。
#SBATCH --job-name=downscale_train_ddp_multi
#SBATCH --partition=qdagnormal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=dcu:8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm_ddp_multi_%j.out
#SBATCH --error=logs/slurm_ddp_multi_%j.err

# 多节点多卡（DCU/GPU）分布式训练：torch.distributed DDP + torchrun + RDMA(IB/RoCE)。
# 每个节点起 1 个 Slurm task，task 内部再用 torchrun 拉起 --gres 指定数量的进程
# （一进程对应一张 DCU），跨节点梯度 all-reduce 走 NCCL/RCCL + IB，对应平台用户手册
# 《RDMA：使用高性能网络进行分布式训练》
# https://www.scnet.cn/help/docs/mainsite/ai/model-training/rdma/
#
# 正式配置（与 slurm/train_ddp_single_node.slurm 一致，跑通单节点后再扩到多节点）：
#   --no_cbam --hr_aux_mode stage1（不开 CBAM、HR 辅助仅 Stage1 注入）
#   --norm_type group --ema_decay 0.999（方案 B：GroupNorm + EMA）
#   --warmup_ratio 0.03 --val_interval 2 --batch_size 2（方案 A：按真实步数调 warmup、
#   降低验证频率、多卡后提高单卡 batch）
#
# 使用前必读：
#   1. `sinfo` / `sacctmgr show assoc user=$USER` 确认本账号可用的 AI/DCU 分区名，
#      替换下面的 --partition（示例中 qdagnormal 为占位）。
#   2. --nodes 改为实际申请节点数；--gres=dcu:N 改为每节点实际可用加速卡数。
#   3. 首次运行建议先跑通 slurm/train_ddp_single_node.slurm（单节点多卡），
#      确认 DDP/SyncBN(或 GroupNorm)/checkpoint/EMA 逻辑正常后再扩到多节点，减少排障成本。
#
# 提交: cd /public/home/acd7koea4a/work && sbatch slurm/train_ddp_multi_node.slurm

set -euo pipefail
WORK_ROOT="/public/home/acd7koea4a/work"
cd "${SLURM_SUBMIT_DIR:-${WORK_ROOT}}"

source env/activate.sh

# DCU 计算/通信库（Hygon DTK + RCCL，RCCL 实现与 NCCL 相同的 API/环境变量名）。
# 若集群实际是 NVIDIA GPU，可换成 `module load nvidia/cuda/12.1`。
module load compiler/dtk/25.04 2>/dev/null || true
module load app/rccl/dtk-25.04/26.01.1 2>/dev/null || true
# 跨节点 RDMA 走 IB/RoCE 时的 SHCA 插件（如平台提供，找不到则忽略，退回 TCP/Socket）
module load app/rccl/shca_rdma_plugins/v8 2>/dev/null || true

mkdir -p logs runs

# --- 平台推荐的 RDMA/NCCL 默认变量（参考 SCNet 用户手册，可按需覆盖） ---
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0           # 启用 InfiniBand/RoCE
export NCCL_IB_HCA=mlx5            # 实际网卡名请用 `ibv_devices` 核对
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=136
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_SOCKET_IFNAME=eth      # Pod/容器内一般是 eth0；裸金属节点请核对实际接口名

# --- rendezvous：MASTER_ADDR 取本次作业分配到的第一个节点 ---
NODELIST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}")
export MASTER_ADDR=$(echo "${NODELIST}" | head -n1)
export MASTER_PORT="${MASTER_PORT:-29500}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"   # = --gres=dcu:N 中的 N
NNODES="${SLURM_NNODES}"

echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}  NNODES=${NNODES}  NPROC_PER_NODE=${NPROC_PER_NODE}"

# 每个节点各起一个 srun task，task 内 torchrun 再拉起 NPROC_PER_NODE 个进程；
# --node_rank 用 Slurm 的节点序号（SLURM_NODEID），保证各节点 rank 区间不重叠
srun --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" bash -c '
    torchrun \
        --nnodes='"${NNODES}"' \
        --nproc_per_node='"${NPROC_PER_NODE}"' \
        --node_rank="${SLURM_NODEID}" \
        --master_addr='"${MASTER_ADDR}"' \
        --master_port='"${MASTER_PORT}"' \
        train.py \
        --hdf5_root "$(python -c "from paths import HDF5_ROOT; print(HDF5_ROOT)")" \
        --epochs 100 \
        --batch_size 2 \
        --accum_steps 2 \
        --val_fraction 0.2 \
        --val_interval 2 \
        --num_workers 4 \
        --no_cbam \
        --hr_aux_mode stage1 \
        --norm_type group \
        --ema_decay 0.999 \
        --warmup_ratio 0.03 \
        --early_stop_patience 20 \
        --run_dir runs/exp_prod_ddp_multi_node
'
