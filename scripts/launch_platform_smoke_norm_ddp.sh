#!/bin/bash
# 预标准化 fp16 数据切换后的 2 卡 DDP 冒烟。
# bash 命令两边相同；控制台选卡和镜像不同。
# 仅在 launch_platform_smoke_norm_single.sh 通过后再提交。
#
# 控制台（两边都是每实例 2 卡 / 1 实例）：
#   BW1000：加速卡=BW1000，镜像=与单卡相同的 DTK PyTorch
#   A800  ：加速卡=A800， 镜像=与单卡相同的 CUDA PyTorch
#   启动命令（相同）：
#     bash /public/home/acd7koea4a/work/scripts/launch_platform_smoke_norm_ddp.sh
#
# 多卡：保留平台注入的 WORLD_SIZE/RANK/MASTER_*，不要 unset。

set -euo pipefail

WORK_ROOT="/public/home/acd7koea4a/work"
cd "${WORK_ROOT}"
mkdir -p logs runs

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

python - <<PY
import os, sys, torch
nproc = int(os.environ.get("NPROC_PER_NODE", "${NPROC_PER_NODE}"))
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
print("platform WORLD_SIZE(实例数)=", os.environ.get("WORLD_SIZE"),
      "RANK(实例)=", os.environ.get("RANK"))
print("MASTER_ADDR=", os.environ.get("MASTER_ADDR"),
      "MASTER_PORT=", os.environ.get("MASTER_PORT"))
if not torch.cuda.is_available():
    raise SystemExit("未检测到加速卡")
if torch.cuda.device_count() < nproc:
    raise SystemExit(
        f"可见卡数 {torch.cuda.device_count()} < NPROC_PER_NODE={nproc}，"
        "请把控制台「每实例加速卡数量」改成 2"
    )
for i in range(torch.cuda.device_count()):
    print(f"  [{i}]", torch.cuda.get_device_name(i))
PY

python - <<'PY'
import importlib, subprocess, sys
missing = []
for m in ("h5py", "netCDF4", "tensorboard"):
    try:
        importlib.import_module(m)
    except ImportError:
        missing.append(m)
if missing:
    print("缺少依赖，安装:", " ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "h5py", "netCDF4", "tensorboard"])
print("deps ok")
PY

SMOKE_ROOT="${WORK_ROOT}/smoke_norm_fp16_data"
NORM_ROOT="$(python -c 'from paths import HDF5_ROOT; print(HDF5_ROOT)')"
SHARD="${NORM_ROOT}/MAM/shard_cra1p5_full_0038.h5"
if [ ! -f "${SHARD}" ]; then
    echo "错误: 找不到 ${SHARD}" >&2
    exit 1
fi
rm -rf "${SMOKE_ROOT}"
mkdir -p "${SMOKE_ROOT}/MAM"
ln -s "${SHARD}" "${SMOKE_ROOT}/MAM/shard_cra1p5_full_0038.h5"
echo "HDF5_ROOT(norm)=${NORM_ROOT}"
echo "smoke overlay=${SMOKE_ROOT} -> ${SHARD}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
LOG="logs/smoke_norm_ddp_$(date +%Y%m%d_%H%M%S).log"

torchrun \
    --nnodes="${WORLD_SIZE:-1}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-23456}" \
    train.py \
    --hdf5_root "${SMOKE_ROOT}" \
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
    --run_dir runs/smoke_norm_fp16_ddp \
    2>&1 | tee "${LOG}"

echo "日志: ${LOG}"
echo "产物: runs/smoke_norm_fp16_ddp"
echo "通过标准："
echo "  [dataset] ... pre_normalized=True  n_samples=16"
echo "  Using device: cuda:0  (distributed=True, world_size=2)"
echo "  3 个 epoch 完成，无 NCCL 报错 / stats_sha256 报错"
echo "  仅 rank0 落盘 checkpoints/best.pt、best_resource.pt"
