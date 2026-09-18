#!/bin/bash
# 归档（2026-09-18）：正式长跑不走 Slurm。
# 现行入口：bash /public/home/acd7koea4a/work/scripts/launch_platform_train.sh
# 本文件仅作历史参考，不要当现行命令。
#SBATCH --job-name=downscale_train
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=90G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# 单卡回退脚本（无多卡资源时使用；有多卡请优先用 slurm/train_ddp_single_node.slurm）。
# 正式配置（与 DDP 脚本一致）：
#   --no_cbam --hr_aux_mode stage1（不开 CBAM、HR 辅助仅 Stage1 注入）
#   --norm_type group --ema_decay 0.999（方案 B：GroupNorm + EMA）
#   --warmup_ratio 0.03（方案 A：按真实 total_steps 自动换算 warmup_steps）
#   单卡无多卡显存冗余，--batch_size 保持 1，--accum_steps 提到 4 维持有效 batch。
#
# 数据路径见 paths.py HDF5_ROOT；若需限定 batch_tag 白名单（排除坏批次），追加 --manifests <tag...>。
# 提交: cd /public/home/acd7koea4a/work && sbatch slurm/train_gpu.slurm
# 注意: --partition 需替换为本账号实际可用分区（sinfo / sacctmgr show assoc user=$USER 确认）

set -euo pipefail
WORK_ROOT="/public/home/acd7koea4a/work"
cd "${SLURM_SUBMIT_DIR:-${WORK_ROOT}}"

source env/activate.sh
module load nvidia/cuda/12.1

mkdir -p logs runs

nvidia-smi

python -u train.py \
    --hdf5_root "$(python -c 'from paths import HDF5_ROOT; print(HDF5_ROOT)')" \
    --epochs 100 \
    --batch_size 1 \
    --accum_steps 4 \
    --val_fraction 0.2 \
    --val_interval 2 \
    --num_workers 4 \
    --no_cbam \
    --hr_aux_mode stage1 \
    --norm_type group \
    --ema_decay 0.999 \
    --warmup_ratio 0.03 \
    --early_stop_patience 20 \
    --run_dir runs/exp_prod_single_gpu
