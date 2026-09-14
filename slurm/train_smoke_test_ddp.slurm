#!/bin/bash
#SBATCH --job-name=downscale_smoke_ddp
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=dcu:2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_smoke_ddp_%j.out
#SBATCH --error=logs/slurm_smoke_ddp_%j.err

# 分布式冒烟测试：在正式扩展到 8 卡（slurm/train_ddp_single_node.slurm）之前，
# 先用 2 张卡 + 单个 shard（16 个样本）验证 DDP 路径本身是否正常：
#   进程组初始化 / DistributedSampler 切分 / 梯度 all-reduce / 验证指标跨 rank 聚合
#   （Loss/val、MAE_val_*、MAE_val_resource/mean 均应与单卡结果在合理误差范围内一致）
#   / rank0-only checkpoint 落盘（best.pt、best_resource.pt，含 model_ema）
# 同时验证正式配置：--no_cbam --hr_aux_mode stage1 --norm_type group --ema_decay。
#
# 若这里通过，再提交 slurm/train_ddp_single_node.slurm（8 卡、正式数据、100 epoch）。
# 若要顺带验证 --compile 与本平台 DCU+RCCL 的兼容性，可在下面追加 --compile 后单独跑一次
# 对比（先不加，确认基础 DDP 路径通了再加，避免多个新变量同时排查）。
#
# 提交: cd /public/home/acd7koea4a/work && sbatch slurm/train_smoke_test_ddp.slurm
# 注意: --partition 需替换为本账号实际可用分区；--gres=dcu:2 若平台是 NVIDIA GPU，
#       改为 --gres=gpu:2 并把下方 module load 换成 nvidia/cuda/12.1。

set -euo pipefail
WORK_ROOT="/public/home/acd7koea4a/work"
cd "${SLURM_SUBMIT_DIR:-${WORK_ROOT}}"

source env/activate.sh
module load compiler/dtk/25.04 2>/dev/null || true
module load app/rccl/dtk-25.04/26.01.1 2>/dev/null || true

mkdir -p logs runs

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=136
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_SOCKET_IFNAME=eth

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"   # = --gres=dcu:N 中的 N

torchrun \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT:-29500}" \
    train.py \
    --hdf5_root "${WORK_ROOT}/smoke_test_data" \
    --seasons MAM \
    --manifests cra1p5_full \
    --val_fraction 0.25 \
    --epochs 3 \
    --val_interval 1 \
    --batch_size 1 \
    --accum_steps 1 \
    --num_workers 2 \
    --no_cbam \
    --hr_aux_mode stage1 \
    --norm_type group \
    --ema_decay 0.999 \
    --warmup_ratio 0.2 \
    --early_stop_patience 0 \
    --save_top_k 1 \
    --run_dir runs/smoke_ddp_cra1p5_full_0038

# 通过标准：
#   - 日志出现 "[dist] rank=.../... backend=nccl 初始化完成"（每个 rank 各一行）
#   - 打印的 "Dataset: 16 samples (train=..., val=...)" 与单卡冒烟一致（数据集本身不受
#     world_size 影响，只是 DistributedSampler 按 rank 再切分 train/val）
#   - 各 epoch 正常完成，无 NCCL/RCCL 报错或卡死
#   - checkpoints/ 下正常生成 best.pt / best_resource.pt（仅 rank0 落盘）
