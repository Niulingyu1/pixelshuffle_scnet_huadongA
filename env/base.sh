#!/bin/bash
# 基础 conda 环境：module + pytorch_downscale + PATH 修复
# 用法: source env/base.sh
# 被 activate.sh 与 ~/.bashrc（交互式）引用，请勿在此设置项目变量或 cd

module purge
module load anaconda3/2023.09

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch_downscale
# module 的 base 路径优先级更高，需手动前置 env bin
export PATH="${CONDA_PREFIX}/bin:${PATH}"
# 避免 ~/.local 中残留包覆盖 conda 环境（曾误装 torch 到用户目录）
export PYTHONNOUSERSITE=1
