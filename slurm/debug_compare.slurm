#!/bin/bash
#SBATCH --job-name=downscale_cmp_debug
# HDF5 测试集推理 + 多模型对比（debug：hdf5_test 小样本）
# 提交: cd /public/home/acd7koea4a/work && sbatch slurm/debug_compare.slurm
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_cmp_debug_%j.out
#SBATCH --error=logs/slurm_cmp_debug_%j.err

set -euo pipefail
WORK_ROOT="/public/home/acd7koea4a/work"
cd "${SLURM_SUBMIT_DIR:-${WORK_ROOT}}"

source env/activate.sh
module load nvidia/cuda/12.1 || true
bash eval/run_comparison.sh debug
