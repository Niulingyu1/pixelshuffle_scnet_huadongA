#!/bin/bash
#SBATCH --job-name=downscale_infer
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# HDF5 测试集推理（需 checkpoint 与 hdf5_test 数据）
# 提交前修改 CKPT 路径
# 提交: cd /public/home/acd7koea4a/work && sbatch slurm/infer_gpu.slurm

set -euo pipefail
WORK_ROOT="/public/home/acd7koea4a/work"
cd "${SLURM_SUBMIT_DIR:-${WORK_ROOT}}"

source env/activate.sh
module load nvidia/cuda/12.1

CKPT="${CKPT:-runs/exp01_no_all/checkpoints/best.pt}"
OUT_DIR="${OUT_DIR:-$(python -c 'from paths import INFER_OUT; print(INFER_OUT / "test_no_all")')}"

mkdir -p logs "$(dirname "${OUT_DIR}")"

nvidia-smi

python -u infer.py \
    --input_source hdf5 \
    --hdf5_root "$(python -c 'from paths import HDF5_TEST; print(HDF5_TEST)')" \
    --ckpt "${CKPT}" \
    --out_dir "${OUT_DIR}" \
    --output_mode per_sample \
    --output_format nc \
    --output_space physical \
    --amp_bf16 \
    --auto_model_cfg
