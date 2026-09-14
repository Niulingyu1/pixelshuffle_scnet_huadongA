#!/bin/bash
# SCNet「模型训练」控制台任务的启动脚本（容器/vcjob 场景，非 Slurm sbatch）。
#
# 使用场景：在 https://www.scnet.cn 控制台 -> 人工智能服务 -> 模型训练 -> 创建训练任务
# 里，「启动命令」一栏直接粘贴：
#
#   bash /public/home/acd7koea4a/work/scripts/launch_platform_train.sh
#
# 平台会为容器自动注入以下环境变量（见《环境变量列表》
# https://www.scnet.cn/help/docs/mainsite/ai/model-training/environment-variable/）：
#   WORLD_SIZE  = 本次任务的实例（容器）数
#   RANK        = 当前实例序号（0 起始）
#   MASTER_ADDR = 主实例（worker-0）hostname
#   MASTER_PORT = 通信端口（默认 23456）
# 若开启 RDMA，还会自动注入 NCCL_IB_* 等变量，无需手动设置。
#
# train.py 本身通过检测 torchrun 为每个子进程设置的 RANK/WORLD_SIZE/LOCAL_RANK
# 环境变量来判断是否启用 DistributedDataParallel，因此本脚本只需正确调用
# torchrun，train.py 无需任何改动（单卡/多卡/多实例行为一致）。
#
# NPROC_PER_NODE 需与控制台「每实例加速卡数量」一致（本例为 2）；
# 若后续调整实例规格，改这一个变量即可。

set -euo pipefail

WORK_ROOT="/public/home/acd7koea4a/work"
cd "${WORK_ROOT}"

# --- 容器内 Python 环境 ---
# 镜像 jupyterlab-pytorch:2.7.1-ubuntu22.04-dtk26.04-py3.11-devel 自带 DTK 26.04 +
# PyTorch 2.7.1，直接用镜像自带 python 即可，无需（也不应该）再 source env/activate.sh
# （该脚本是为 Slurm 登录节点的 conda + module 环境写的，容器内没有对应 module 系统）。
# 若镜像里缺少本项目依赖，取消下面一行注释按需安装（首次运行建议先手动装好并固化进自定义镜像，
# 避免每次任务启动都重新装包浪费时间）：
# pip install -q numpy h5py netCDF4 tensorboard

python -c "import torch, h5py, netCDF4; print('torch', torch.__version__, '| cuda/dtk available:', torch.cuda.is_available())"

# --- 数据路径 ---
# 数据路径由 paths.py 统一定义（DATA_ROOT=/public/share/acd7koea4a）。
# 若容器内挂载点不同，可在启动前 export HDF5_ROOT=... 或改 paths.py。
HDF5_ROOT="$(python -c 'from paths import HDF5_ROOT; print(HDF5_ROOT)')"
echo "HDF5_ROOT=${HDF5_ROOT}"
ls "${HDF5_ROOT}" >/dev/null || { echo "错误: HDF5_ROOT 不可访问，请检查挂载/路径配置" >&2; exit 1; }

# --- 分布式启动 ---
# 与 SCNet《模型训练最佳实践》示例一致的写法：
#   --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT
# 均由平台自动注入，无需手动修改；--nproc_per_node 按实例内加速卡数设置。
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

echo "WORLD_SIZE=${WORLD_SIZE:-1} RANK=${RANK:-0} MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} " \
     "MASTER_PORT=${MASTER_PORT:-23456} NPROC_PER_NODE=${NPROC_PER_NODE}"

torchrun \
    --nnodes="${WORLD_SIZE:-1}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-23456}" \
    train.py \
    --hdf5_root "${HDF5_ROOT}" \
    --epochs 100 \
    --batch_size 1 \
    --accum_steps 4 \
    --val_fraction 0.2 \
    --num_workers 4 \
    --base_ch 128 \
    --no_cbam \
    --hr_aux_mode none \
    --loss_gamma 0.0 --lambda_freq 0.0 --lambda_grad 0.0 \
    --early_stop_patience 20 \
    --run_dir runs/exp01_ddp_platform
