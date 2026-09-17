#!/bin/bash
# SCNet「模型训练」单卡测速 — 对齐历史 A800 exp01_no_all_100（不是当前正式训练）。
#
# 对齐基准：train_exp01_no_all_100.log（A800，改分布式代码之前）
#   - 全量 hdf5（14685 样本，val_fraction=0.2）
#   - base_ch=256，no_cbam，hr_aux_mode=none（17.38M 参数）
#   - 纯 MAE：关尾部加权、面积加权、v2 增量项、旧版 SpatialExtreme/FFT/Grad
#   - batch_size=1，accum_steps=4，num_workers=4，log_interval=50
#   - 单卡非分布式（distributed=False，无 DDP/NCCL/SyncBN 开销）
#
# 当前正式训练请用 scripts/launch_platform_train.sh（stage1 + group + ema + v2 损失）。
#
# 控制台配置：
#   - 每实例加速卡数量 = 1
#   - 实例数 = 1
#   - 启动命令：
#       bash /public/home/acd7koea4a/work/scripts/launch_platform_train_bw_a800_aligned_benchmark.sh
#
# 环境变量（可选）：
#   HDF5_ROOT   默认 HDF5_ROOT（全量 hdf5）
#   EPOCHS      默认 1（与 A800 日志对比看 Epoch 001 done in 即可）
#   RUN_TAG     输出目录后缀，默认时间戳

set -euo pipefail

WORK_ROOT="/public/home/acd7koea4a/work"
cd "${WORK_ROOT}"
mkdir -p logs runs

# 平台 vcjob 可能注入 RANK/WORLD_SIZE，会误触发 DDP 单卡路径；
# 本脚本强制纯单卡，与 A800 历史日志一致。
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT NODE_RANK 2>/dev/null || true

python - <<'PY'
import os
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("RANK in env:", os.environ.get("RANK", "<unset>"))
print("WORLD_SIZE in env:", os.environ.get("WORLD_SIZE", "<unset>"))
if torch.cuda.is_available():
    print("device_name:", torch.cuda.get_device_name(0))
PY

HDF5_ROOT="${HDF5_ROOT:-$(python -c 'from paths import HDF5_ROOT; print(HDF5_ROOT)')}"
EPOCHS="${EPOCHS:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="runs/benchmark_bw_a800_aligned_${RUN_TAG}"
LOG_FILE="logs/benchmark_bw_a800_aligned_${RUN_TAG}.log"

echo "=== 历史 A800 对齐配置（exp01_no_all_100，非正式训练）==="
echo "HDF5_ROOT=${HDF5_ROOT}"
echo "base_ch=256 | hr_aux=none | pure MAE (v2 extras off) | batch=1 accum=4 | distributed=False"
echo "对比 A800 基线: Epoch 001 done in 17997.0s (~5.00h)"
echo "日志: ${LOG_FILE}"

if [[ ! -d "${HDF5_ROOT}" ]]; then
    echo "错误: HDF5_ROOT 不存在或不可访问: ${HDF5_ROOT}" >&2
    exit 1
fi
ls "${HDF5_ROOT}" >/dev/null || { echo "错误: HDF5_ROOT 目录不可列出: ${HDF5_ROOT}" >&2; exit 1; }

python -u train.py \
    --hdf5_root "${HDF5_ROOT}" \
    --epochs "${EPOCHS}" \
    --val_interval 1 \
    --val_fraction 0.2 \
    --batch_size 1 \
    --accum_steps 4 \
    --num_workers 4 \
    --base_ch 256 \
    --no_cbam \
    --hr_aux_mode none \
    --loss_gamma 0.0 \
    --var_weights "" \
    --no_area_weight \
    --lambda_extreme 0.0 \
    --lambda_patch_extreme 0.0 \
    --lambda_wps 0.0 \
    --lambda_phys 0.0 \
    --lambda_freq 0.0 \
    --lambda_grad 0.0 \
    --extreme_vars "" \
    --log_interval 50 \
    --early_stop_patience 0 \
    --run_dir "${RUN_DIR}" \
    2>&1 | tee "${LOG_FILE}"

echo "完成。日志: ${LOG_FILE}  产物: ${RUN_DIR}"
echo "请对比日志中的 'Epoch 001 done in' 与 A800 的 17997.0s"
