#!/bin/bash
# 预标准化 fp16 数据切换后的单卡冒烟。
# bash 命令两边相同；控制台选卡和镜像不同，见脚本末尾说明。
#
# 与 launch_platform_smoke_single.sh 的差别：读 hdf5_norm_fp16 的 1 个 shard
# （不是 work/smoke_test_data 的 fp32 小集），验证：
#   Dataset 识别 pre_normalized=True、stats_sha256 与当前 STATS_FILE 一致、
#   DataLoader num_workers>0、x/y=bf16、hr_aux=fp32、前向/反向/验证/落盘。
#
# 控制台（两边都是 1 卡 / 1 实例）：
#   BW1000：加速卡=BW1000，镜像=DTK PyTorch（不要 cuda12.x，不要 source env/activate.sh）
#   A800  ：加速卡=A800， 镜像=CUDA PyTorch（不要 DTK）
#   启动命令（相同）：
#     bash /public/home/acd7koea4a/work/scripts/launch_platform_smoke_norm_single.sh
#
# 通过后再跑 launch_platform_smoke_norm_ddp.sh。全量 1 epoch 用
# launch_platform_train_copy.sh，不要用已归档的 slurm/train_smoke_test*.slurm。

set -euo pipefail

WORK_ROOT="/public/home/acd7koea4a/work"
cd "${WORK_ROOT}"
mkdir -p logs runs

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
    raise SystemExit(
        "未检测到加速卡。BW1000 必须用 DTK 镜像且不要 source env/activate.sh；"
        "A800 必须用 CUDA PyTorch 镜像。"
    )
print("device[0]:", torch.cuda.get_device_name(0))
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

LOG="logs/smoke_norm_single_$(date +%Y%m%d_%H%M%S).log"

python -u train.py \
    --hdf5_root "${SMOKE_ROOT}" \
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
    --run_dir runs/smoke_norm_fp16_single \
    2>&1 | tee "${LOG}"

echo "日志: ${LOG}"
echo "产物: runs/smoke_norm_fp16_single"
echo "通过标准："
echo "  [dataset] ... pre_normalized=True  n_samples=16"
echo "  Using device: cuda:0  (distributed=False, world_size=1)"
echo "  3 个 epoch 都跑完 validate()，无 dtype / stats_sha256 报错"
echo "  checkpoints/ 下有 best.pt、best_resource.pt"
