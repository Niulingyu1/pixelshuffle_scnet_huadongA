#!/bin/bash
# 本仓库唯一正式长跑入口：SCNet「模型训练」控制台（容器/vcjob）。
# 不用 slurm/*.slurm（那些脚本已归档，仅作历史参考）。
#
# 现行配置（DOWNSCALE_README.md 第 1、4 节应与本脚本一致，不要反过来对齐 slurm）：
#   架构：--no_cbam --hr_aux_mode stage1 --norm_type group --ema_decay 0.999
#   损失：train.py v2 默认（面积加权 TailMAE + PatchExtreme 0.1 + WPS 0.05 + Phys 0.02）
#   调参：--warmup_ratio 0.03 --val_interval 2
#   8 卡保持 --batch_size 1；不要提到 2
#   数据：paths.HDF5_ROOT（hdf5_norm_fp16）+ --manifests cra1p5_full
#
# 控制台：https://www.scnet.cn/help/docs/mainsite/ai/model-training/
#   启动命令：
#     bash /public/home/acd7koea4a/work/scripts/launch_platform_train.sh
#
# 平台注入 WORLD_SIZE/RANK/MASTER_*（语义是实例数，不是卡数）。
# NPROC_PER_NODE 必须与「每实例加速卡数量」一致（默认 8；2 卡时 export NPROC_PER_NODE=2）。
#
# 预标准化数据冒烟：scripts/launch_platform_smoke_norm_single.sh / smoke_norm_ddp.sh
# 全量 1 epoch 探测：scripts/launch_platform_train_copy.sh
# 8 卡短跑：scripts/launch_platform_short_8gpu.sh
# 历史纯 MAE 对齐（exp01_no_all_100）：launch_platform_train_bw_a800_aligned_benchmark.sh

set -euo pipefail

WORK_ROOT="/public/home/acd7koea4a/work"
cd "${WORK_ROOT}"
mkdir -p logs runs

# --- 容器内 Python 环境 ---
# 镜像自带 python/torch 即可，不要 source env/activate.sh（那是 Slurm 登录节点的 conda+module）。
python -c "import torch, h5py, netCDF4; print('torch', torch.__version__, '| cuda/dtk available:', torch.cuda.is_available())"

# --- 数据路径 ---
HDF5_ROOT="$(python -c 'from paths import HDF5_ROOT; print(HDF5_ROOT)')"
echo "HDF5_ROOT=${HDF5_ROOT}"
ls "${HDF5_ROOT}" >/dev/null || { echo "错误: HDF5_ROOT 不可访问，请检查挂载/路径配置" >&2; exit 1; }

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

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
    --epochs 100 \
    --batch_size 1 \
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
    --run_dir runs/exp_prod_ddp_platform
