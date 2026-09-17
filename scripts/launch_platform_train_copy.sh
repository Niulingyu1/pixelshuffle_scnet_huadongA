#!/bin/bash
# 正式配置的 1 epoch 全量路径探测（容器/vcjob）。
#
# 架构与损失与 scripts/launch_platform_train.sh 相同，只把 --epochs 改成 1，
# 用来确认全量 HDF5 可读、DDP 能走完 validate()/落盘。MAE 没有业务意义。
# 架构冒烟（16 样本）请用 launch_platform_smoke_single.sh / smoke_ddp.sh。
#
# 启动命令：
#   bash /public/home/acd7koea4a/work/scripts/launch_platform_train_copy.sh
#
# NPROC_PER_NODE 须与控制台「每实例加速卡数量」一致（默认 2）。

set -euo pipefail

WORK_ROOT="/public/home/acd7koea4a/work"
cd "${WORK_ROOT}"
mkdir -p logs runs

python -c "import torch, h5py, netCDF4; print('torch', torch.__version__, '| cuda/dtk available:', torch.cuda.is_available())"

HDF5_ROOT="$(python -c 'from paths import HDF5_ROOT; print(HDF5_ROOT)')"
echo "HDF5_ROOT=${HDF5_ROOT}"
ls "${HDF5_ROOT}" >/dev/null || { echo "错误: HDF5_ROOT 不可访问，请检查挂载/路径配置" >&2; exit 1; }

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

echo "WORLD_SIZE=${WORLD_SIZE:-1} RANK=${RANK:-0} MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} " \
     "MASTER_PORT=${MASTER_PORT:-23456} NPROC_PER_NODE=${NPROC_PER_NODE}"

torchrun \
    --nnodes="${WORLD_SIZE:-1}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-23456}" \
    train.py \
    --hdf5_root "${HDF5_ROOT}" \
    --seasons MAM JJA SON DJF \
    --manifests cra1p5_full \
    --epochs 1 \
    --batch_size 1 \
    --accum_steps 2 \
    --val_fraction 0.2 \
    --val_interval 1 \
    --num_workers 4 \
    --no_cbam \
    --hr_aux_mode stage1 \
    --norm_type group \
    --ema_decay 0.999 \
    --warmup_ratio 0.03 \
    --early_stop_patience 0 \
    --save_top_k 1 \
    --run_dir runs/probe_platform_1epoch
