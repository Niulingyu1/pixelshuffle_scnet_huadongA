#!/bin/bash
# SCNet「模型训练」8 卡全量路径短跑（A800 / NVIDIA CUDA 镜像）。
#
# 目的：用正式数据 + 正式架构，验证 8 卡 DDP 读盘 / 显存 / validate() / 落盘。
# 只跑 2 个 epoch，MAE 没有业务意义。通过后再把 --epochs 改成 100。
#
# 控制台：https://www.scnet.cn/help/docs/mainsite/ai/model-training/
#   加速卡型号     = A800
#   每实例加速卡数量 = 8
#   实例数         = 1
#   训练镜像       = 与 Notebook 单卡冒烟相同的 CUDA PyTorch 镜像
#   启动命令：
#     bash /public/home/acd7koea4a/work/scripts/launch_platform_short_8gpu.sh
#
# 多卡：保留平台注入的 WORLD_SIZE/RANK/MASTER_*，不要 unset。
# 仍用 batch_size=1；单卡冒烟余量约 23 GiB，但 8 卡还有 NCCL 缓冲，不要这次就提 bs=2。

set -euo pipefail

WORK_ROOT="/public/home/acd7koea4a/work"
cd "${WORK_ROOT}"
mkdir -p logs runs

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

python - <<PY
import os, sys, torch
nproc = int(os.environ.get("NPROC_PER_NODE", "8"))
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
print("platform WORLD_SIZE(实例数)=", os.environ.get("WORLD_SIZE"),
      "RANK(实例)=", os.environ.get("RANK"))
print("MASTER_ADDR=", os.environ.get("MASTER_ADDR"),
      "MASTER_PORT=", os.environ.get("MASTER_PORT"))
if not torch.cuda.is_available():
    raise SystemExit("未检测到加速卡：请确认镜像是 CUDA PyTorch，加速卡选了 A800")
if torch.cuda.device_count() < nproc:
    raise SystemExit(
        f"可见卡数 {torch.cuda.device_count()} < NPROC_PER_NODE={nproc}，"
        "请把控制台「每实例加速卡数量」改成 8"
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

LOG="logs/short_8gpu_$(date +%Y%m%d_%H%M%S).log"

torchrun \
    --nnodes="${WORLD_SIZE:-1}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-23456}" \
    train.py \
    --hdf5_root /public/share/acd7koea4a/hdf5 \
    --seasons MAM JJA SON DJF \
    --manifests cra1p5_full \
    --val_fraction 0.2 \
    --val_interval 1 \
    --epochs 2 \
    --batch_size 1 \
    --accum_steps 2 \
    --num_workers 4 \
    --no_cbam \
    --hr_aux_mode stage1 \
    --norm_type group \
    --ema_decay 0.999 \
    --warmup_ratio 0.03 \
    --log_interval 20 \
    --early_stop_patience 0 \
    --save_top_k 1 \
    --run_dir runs/short_a800_8gpu \
    2>&1 | tee "${LOG}"

echo "日志: ${LOG}"
echo "产物: runs/short_a800_8gpu"
echo "通过标准："
echo "  Using device: cuda:0  (distributed=True, world_size=8)"
echo "  [mem] rank0 首个 optimizer step 后：max_reserved 距离 device_total 有健康余量"
echo "  两个 epoch 都跑完 validate()，无 OOM"
echo "  仅 rank0 落盘 checkpoints/best.pt、best_resource.pt"
