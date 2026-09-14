#!/bin/bash
#SBATCH --job-name=downscale_smoke
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_smoke_%j.out
#SBATCH --error=logs/slurm_smoke_%j.err

# 冒烟测试（单卡）：仅用单个 shard（shard_cra1p5_full_0038.h5，16 个样本）验证
# **正式训练确定的架构/机制**是否在真实 GPU 上跑通：
#   --no_cbam --hr_aux_mode stage1（正式配置：不开 CBAM、HR 辅助仅 Stage1 注入）
#   --norm_type group --ema_decay 0.999（方案 B：GroupNorm + EMA，含 checkpoint 的
#   model_ema 落盘）
#   --warmup_ratio（方案 A：按真实 total_steps 自动换算 warmup_steps）
# 分布式路径请另用 slurm/train_smoke_test_ddp.slurm 验证（本脚本仅单卡）。
#
# 提交: cd /public/home/acd7koea4a/work && sbatch slurm/train_smoke_test.slurm
# 注意: --partition 需替换为本账号实际可用分区（sinfo / sacctmgr show assoc user=$USER 确认）

set -euo pipefail
WORK_ROOT="/public/home/acd7koea4a/work"
cd "${SLURM_SUBMIT_DIR:-${WORK_ROOT}}"

source env/activate.sh
module load nvidia/cuda/12.1 || true

mkdir -p logs runs

nvidia-smi

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
    --run_dir runs/smoke_cra1p5_full_0038

# 通过标准（除此前版本已有的几项外，新增）：
#   - 日志出现 "[EMA] 已启用，decay=0.999"
#   - 日志出现 "[warmup] --warmup_ratio=... → warmup_steps=..."
#   - checkpoints/best.pt、best_resource.pt 内含 "model_ema" 字段（可用
#     python -c "import torch; print(torch.load('runs/smoke_cra1p5_full_0038/checkpoints/best.pt', map_location='cpu').keys())"
#     核对）
#   - Model 配置行打印 norm_type=group  use_cbam=False  hr_aux_mode=stage1
