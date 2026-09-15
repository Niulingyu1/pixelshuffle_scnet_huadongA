#!/bin/bash
# SCNet「模型训练」2 卡 DDP 冒烟（A800 / NVIDIA CUDA 镜像）。
# 仅在单卡冒烟 scripts/launch_platform_smoke_single.sh 通过后再提交。
#
# 控制台：
#   加速卡型号     = A800
#   每实例加速卡数量 = 2
#   实例数         = 1
#   训练镜像       = 与单卡冒烟相同的 CUDA PyTorch 镜像
#   启动命令：
#     bash /public/home/acd7koea4a/work/scripts/launch_platform_smoke_ddp.sh
#
# 多卡：保留平台注入的 WORLD_SIZE/RANK/MASTER_*（语义是实例数），
# 用 torchrun --nproc_per_node 把每实例卡数展开成进程。不要 unset。

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

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
LOG="logs/smoke_ddp_$(date +%Y%m%d_%H%M%S).log"

torchrun \
    --nnodes="${WORLD_SIZE:-1}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-23456}" \
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
    --run_dir runs/smoke_a800_ddp \
    2>&1 | tee "${LOG}"

echo "日志: ${LOG}"
echo "产物: runs/smoke_a800_ddp"
echo "通过标准："
echo "  Using device: cuda:0  (distributed=True, world_size=2)"
echo "  仅 rank0 落盘 checkpoints/best.pt、best_resource.pt"
