#!/bin/bash
# 可选运行时调优（路径见 paths.py，不在此设置）
# 用法: source env/project.sh

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
