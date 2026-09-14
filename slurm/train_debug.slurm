#!/bin/bash
#SBATCH --job-name=downscale_debug
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# 快速调试训练（数据路径见 paths.py HDF5_MINI）
# 提交: cd /public/home/acd7koea4a/work && sbatch slurm/train_debug.slurm

set -euo pipefail
WORK_ROOT="/public/home/acd7koea4a/work"
cd "${SLURM_SUBMIT_DIR:-${WORK_ROOT}}"

source env/activate.sh
module load nvidia/cuda/12.1

mkdir -p logs runs

nvidia-smi

python train.py \
    --hdf5_root "$(python -c 'from paths import HDF5_MINI; print(HDF5_MINI)')" \
    --epochs 3 \
    --val_interval 1 \
    --num_workers 4 \
    --seasons DJF \
    --base_ch 128 \
    --no_cbam \
    --hr_aux_mode none \
    --loss_gamma 0.0 --lambda_freq 0.0 --lambda_grad 0.0 \
    --run_dir runs/debug_slurm
