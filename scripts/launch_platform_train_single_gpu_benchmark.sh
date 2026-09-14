#!/bin/bash
# SCNet「模型训练」单卡（BW）速度基准测试。
#
# 控制台配置：
#   - 每实例加速卡数量 = 1
#   - 实例数 = 1
#   - 启动命令：
#       bash /public/home/acd7koea4a/work/scripts/launch_platform_train_single_gpu_benchmark.sh
#
# 说明：不使用 torchrun，train.py 以单卡模式运行（distributed=False），
# 避免 DDP/NCCL 开销，测的是单张 BW 卡的真实训练吞吐。
#
# 环境变量（可选）：
#   HDF5_ROOT   数据目录；未设置时自动选择：hdf5 -> hdf5_half -> hdf5_mini
#   EPOCHS      默认 1（看 1 个 epoch 耗时即可估算总时长）
#   RUN_TAG     输出目录后缀，默认时间戳
#   BENCHMARK_QUICK=1  使用 hdf5_mini 做快速连通性/吞吐冒烟（约数分钟）

set -euo pipefail

WORK_ROOT="/public/home/acd7koea4a/work"
cd "${WORK_ROOT}"
mkdir -p logs runs

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
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
RUN_DIR="runs/benchmark_bw1gpu_${RUN_TAG}"
LOG_FILE="logs/benchmark_bw1gpu_${RUN_TAG}.log"

echo "HDF5_ROOT=${HDF5_ROOT}"
if [[ ! -d "${HDF5_ROOT}" ]]; then
    echo "错误: HDF5_ROOT 不存在或不可访问: ${HDF5_ROOT}" >&2
    echo "已检查候选: hdf5, hdf5_half, hdf5_mini（见 paths.resolve_hdf5_train_root）" >&2
    echo "可手动指定: export HDF5_ROOT=/public/share/acd7koea4a/hdf5_mini" >&2
    exit 1
fi
ls "${HDF5_ROOT}" >/dev/null || { echo "错误: HDF5_ROOT 目录不可列出: ${HDF5_ROOT}" >&2; exit 1; }

# 单卡：不设置 RANK/WORLD_SIZE，直接 python（train.py 自动走单卡路径）
python -u train.py \
    --hdf5_root "${HDF5_ROOT}" \
    --epochs "${EPOCHS}" \
    --val_interval 1 \
    --batch_size 1 \
    --accum_steps 4 \
    --val_fraction 0.2 \
    --num_workers 4 \
    --base_ch 128 \
    --no_cbam \
    --hr_aux_mode none \
    --loss_gamma 0.0 --lambda_freq 0.0 --lambda_grad 0.0 \
    --log_interval 20 \
    --early_stop_patience 0 \
    --run_dir "${RUN_DIR}" \
    2>&1 | tee "${LOG_FILE}"

echo "完成。日志: ${LOG_FILE}  产物: ${RUN_DIR}"
