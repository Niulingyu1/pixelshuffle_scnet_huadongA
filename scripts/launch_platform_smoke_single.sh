#!/bin/bash
# SCNet「模型训练」单卡冒烟（A800 / NVIDIA CUDA 镜像）。
#
# 控制台：https://www.scnet.cn/help/docs/mainsite/ai/model-training/
#   加速卡型号     = A800（不要选 BW1000 / DTK）
#   每实例加速卡数量 = 1
#   实例数         = 1
#   训练镜像       = CUDA 版 PyTorch（不要选 dtk / 海光 DCU 镜像）
#   启动命令：
#     bash /public/home/acd7koea4a/work/scripts/launch_platform_smoke_single.sh
#
# 目的：验证正式架构在 A800 上前向 / 反向 / 验证 / 存盘都通。
# 数据只有 16 个样本，MAE 没有业务意义。通过后再做 2 卡 DDP 冒烟。
#
# 关键：平台会注入 RANK/WORLD_SIZE（语义是「实例数」，不是卡数）。
# train.py 只要看到这两个变量就会 init DDP。单卡必须 unset，否则变成
# 「1 卡假 DDP」，多占通信缓冲，DCU 上曾因此 OOM。

set -euo pipefail

WORK_ROOT="/public/home/acd7koea4a/work"
cd "${WORK_ROOT}"
mkdir -p logs runs

# 单卡：清掉平台注入的分布式变量，走 python -u 而不是 torchrun
unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT

python - <<'PY'
import os, sys, torch
print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
print("RANK in env:", "RANK" in os.environ, "WORLD_SIZE in env:", "WORLD_SIZE" in os.environ)
if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
    raise SystemExit("分布式环境变量仍在，单卡必须先 unset 再跑")
if not torch.cuda.is_available():
    raise SystemExit("未检测到加速卡：请确认镜像是 CUDA PyTorch，加速卡选了 A800")
print("device[0]:", torch.cuda.get_device_name(0))
PY

# 镜像通常有 torch，不一定有 h5py / netCDF4 / tensorboard
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

LOG="logs/smoke_single_$(date +%Y%m%d_%H%M%S).log"

python -u train.py \
    --hdf5_root "${WORK_ROOT}/smoke_test_data" \
    --seasons MAM \
    --manifests cra1p5_full \
    --val_fraction 0.1 \
    --epochs 3 \
    --val_interval 1 \
    --batch_size 1 \
    --accum_steps 2 \
    --num_workers 2 \
    --no_cbam \
    --hr_aux_mode stage1 \
    --norm_type group \
    --ema_decay 0.999 \
    --warmup_ratio 0.2 \
    --early_stop_patience 0 \
    --save_top_k 1 \
    --run_dir runs/smoke_a800_single \
    2>&1 | tee "${LOG}"

echo "日志: ${LOG}"
echo "产物: runs/smoke_a800_single"
echo "通过标准："
echo "  Using device: cuda:0  (distributed=False, world_size=1)"
echo "  Model: ... use_cbam=False  hr_aux_mode=stage1  norm_type=group"
echo "  [EMA] 已启用，decay=0.999"
echo "  checkpoints/ 下有 best.pt、best_resource.pt"
