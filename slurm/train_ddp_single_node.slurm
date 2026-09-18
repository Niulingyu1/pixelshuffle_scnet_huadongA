#!/bin/bash
# 归档（2026-09-18）：正式长跑不走 Slurm。
# 现行入口：bash /public/home/acd7koea4a/work/scripts/launch_platform_train.sh
# 本文件仅作历史参考，不要当现行命令。
#SBATCH --job-name=downscale_train_ddp1n
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=dcu:8
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm_ddp1n_%j.out
#SBATCH --error=logs/slurm_ddp1n_%j.err

# 正式训练（单节点多卡 DDP）：确认的正式配置——
#   --no_cbam：不开 CBAM 注意力
#   --hr_aux_mode stage1：HR 辅助仅在 Stage1（360×720）注入，Stage2-3/Head 不再拼接
#   分布式：torch.distributed DDP + torchrun（本脚本）
# 叠加方案 A（调参）+ 方案 B（架构/训练机制改进）：
#   方案 A：--warmup_ratio 0.03（按真实 total_steps 的 3% 自动换算 warmup_steps，
#           不必再手动跑一次看日志回填）；多卡后 --batch_size 提到 2（DDP 下每卡独立
#           显存，--no_cbam + hr_aux_mode=stage1 比全量配置更省显存，--val_interval 2
#           降低全图验证频率的开销
#   方案 B：--norm_type group（GroupNorm 替代 BatchNorm2d，不依赖 batch 维统计量，
#           对每卡 batch 仍然较小的场景更稳健）；--ema_decay 0.999（验证/交付用 EMA
#           权重，checkpoint 额外保存 model_ema）
#   --compile 默认未开启：torch.compile 与本平台 DCU+RCCL 后端的兼容性未经验证，
#           建议先在小规模冒烟（slurm/train_smoke_test_ddp.slurm）里追加 --compile
#           验证无误后再加到这里（见脚本末尾注释）
#
# 数据/输出路径见 paths.py（HDF5_ROOT/HDF5_HALF 等），--gres 按实际单节点可用加速卡数调整
#（本平台每节点 8 张 Hygon DCU，见 `scontrol show node <nodename>` 的 Gres=dcu:Hygon:8）。
#
# 提交前请用 `sinfo` / `sacctmgr show assoc user=$USER` 确认本账号可用的 AI/DCU 分区名，
# 替换下面的 --partition（示例中 qdagnormal 为占位，不同集群/项目分区名不同）。
# 若需限定 shard 的 batch_tag 白名单（排除坏批次），追加 --manifests <tag...>。
#
# 提交: cd /public/home/acd7koea4a/work && sbatch slurm/train_ddp_single_node.slurm

set -euo pipefail
WORK_ROOT="/public/home/acd7koea4a/work"
cd "${SLURM_SUBMIT_DIR:-${WORK_ROOT}}"

source env/activate.sh

# DCU 计算/通信库（Hygon DTK + RCCL，RCCL 兼容 NCCL 的 API 与环境变量名）。
# 若本节点实际是 NVIDIA GPU，可换成 `module load nvidia/cuda/12.1`。
module load compiler/dtk/25.04 2>/dev/null || true
module load app/rccl/dtk-25.04/26.01.1 2>/dev/null || true

mkdir -p logs runs

# 单节点内多卡通信走 P2P/共享内存，无需 IB；这里仍设置平台推荐的默认变量
# （参考 SCNet 用户手册《RDMA：使用高性能网络进行分布式训练》），便于日后扩展到多节点时一致。
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=136
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_SOCKET_IFNAME=eth

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"   # = --gres=dcu:N 中的 N

torchrun \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT:-29500}" \
    train.py \
    --hdf5_root "$(python -c 'from paths import HDF5_ROOT; print(HDF5_ROOT)')" \
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
    --run_dir runs/exp_prod_ddp_1node

# 验证过 --compile 与本平台组合无误后，可在上面追加一行：
#     --compile \
# 并先在 slurm/train_smoke_test_ddp.slurm 上单独验证过（见该脚本注释）。
