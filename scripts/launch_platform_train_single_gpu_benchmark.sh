#!/bin/bash
# SCNet「模型训练」单卡速度基准（正式架构 + v2 损失）。
#
# 控制台配置：
#   - 每实例加速卡数量 = 1
#   - 实例数 = 1
#   - 启动命令：
#       bash /public/home/acd7koea4a/work/scripts/launch_platform_train_single_gpu_benchmark.sh
#
# 不使用 torchrun；并 unset 平台注入的 RANK/WORLD_SIZE，避免「1 卡假 DDP」。
# 测的是正式配置（no_cbam / stage1 / group / ema + v2 损失）的单卡吞吐，不是旧消融基线。
#
# 若要对齐历史 A800 exp01_no_all_100（hr_aux=none + 纯 MAE），改用
# scripts/launch_platform_train_bw_a800_aligned_benchmark.sh。
#
# 环境变量（可选）：
#   HDF5_ROOT   数据目录；未设置时自动选择：hdf5 -> hdf5_half -> hdf5_mini
#   EPOCHS      默认 1
#   RUN_TAG     输出目录后缀，默认时间戳
#   BENCHMARK_QUICK=1  使用 hdf5_mini 做快速连通性/吞吐冒烟

set -euo pipefail

WORK_ROOT="/public/home/acd7koea4a/work"
cd "${WORK_ROOT}"
mkdir -p logs runs

unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT 2>/dev/null || true

python - <<'PY'
import os, torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("RANK in env:", "RANK" in os.environ, "WORLD_SIZE in env:", "WORLD_SIZE" in os.environ)
if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
    raise SystemExit("分布式环境变量仍在，单卡必须先 unset 再跑")
if torch.cuda.is_available():
    print("device_name:", torch.cuda.get_device_name(0))
PY

if [[ -z "${HDF5_ROOT:-}" ]]; then
    if [[ "${BENCHMARK_QUICK:-0}" == "1" ]]; then
        HDF5_ROOT="$(python -c 'from paths import HDF5_MINI; print(HDF5_MINI)')"
    else
        HDF5_ROOT="$(python -c 'from paths import resolve_hdf5_train_root; print(resolve_hdf5_train_root())')"
    fi
fi
EPOCHS="${EPOCHS:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="runs/benchmark_official_1gpu_${RUN_TAG}"
LOG_FILE="logs/benchmark_official_1gpu_${RUN_TAG}.log"

echo "HDF5_ROOT=${HDF5_ROOT}"
echo "config: no_cbam / hr_aux=stage1 / norm=group / ema=0.999 / v2 loss defaults"
if [[ ! -d "${HDF5_ROOT}" ]]; then
    echo "错误: HDF5_ROOT 不存在或不可访问: ${HDF5_ROOT}" >&2
    echo "已检查候选: hdf5, hdf5_half, hdf5_mini（见 paths.resolve_hdf5_train_root）" >&2
    echo "可手动指定: export HDF5_ROOT=/public/share/acd7koea4a/hdf5_mini" >&2
    exit 1
fi
ls "${HDF5_ROOT}" >/dev/null || { echo "错误: HDF5_ROOT 目录不可列出: ${HDF5_ROOT}" >&2; exit 1; }

python -u train.py \
    --hdf5_root "${HDF5_ROOT}" \
    --epochs "${EPOCHS}" \
    --val_interval 1 \
    --batch_size 1 \
    --accum_steps 2 \
    --val_fraction 0.2 \
    --num_workers 4 \
    --no_cbam \
    --hr_aux_mode stage1 \
    --norm_type group \
    --ema_decay 0.999 \
    --warmup_ratio 0.03 \
    --log_interval 20 \
    --early_stop_patience 0 \
    --save_top_k 1 \
    --run_dir "${RUN_DIR}" \
    2>&1 | tee "${LOG_FILE}"

echo "完成。日志: ${LOG_FILE}  产物: ${RUN_DIR}"
