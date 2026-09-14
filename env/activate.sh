#!/bin/bash
# 进入项目上下文：工作目录 + 按需加载 conda（Notebook 可跳过）
# 路径统一在 paths.py 定义，无需环境变量
# 用法: source env/activate.sh

_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_ENV_DIR}/project.sh"

_PROJECT_HOME="${DOWNSCALE_HOME:-${HOME}}"
# Notebook 中 HOME 常为 /root；优先用本脚本所在项目目录（env/ 的上一级）
_PROJECT_ROOT="${DOWNSCALE_WORK:-$(
    cd "${_ENV_DIR}/.." && pwd
)}"

_activate_pytorch_downscale_direct() {
    local env_name="pytorch_downscale"
    local env_root=""
    local project_home=""

    project_home="$(cd "${_PROJECT_ROOT}/.." 2>/dev/null && pwd || true)"

    for candidate in \
        "${_PROJECT_HOME}/.conda/envs/${env_name}" \
        "${project_home}/.conda/envs/${env_name}" \
        "${HOME}/.conda/envs/${env_name}"; do
        if [ -x "${candidate}/bin/python" ]; then
            env_root="${candidate}"
            break
        fi
    done
    [ -n "${env_root}" ] || return 1

    export CONDA_PREFIX="${env_root}"
    export CONDA_DEFAULT_ENV="${env_name}"
    export PATH="${env_root}/bin:${PATH}"
    export PYTHONNOUSERSITE=1
}

_use_base=0
if [ "${CONDA_DEFAULT_ENV:-}" = "pytorch_downscale" ]; then
    : # 已在目标 conda 环境中
elif [ "${DOWNSCALE_SKIP_CONDA:-}" = "1" ]; then
    : # 显式跳过（Notebook 等自带 PyTorch 的场景）
elif _activate_pytorch_downscale_direct; then
    : # 优先直接激活（Notebook / 无 module 登录节点）
elif type module &>/dev/null 2>&1; then
    _use_base=1
elif ! python -c "import torch, netCDF4, h5py" &>/dev/null; then
    echo "警告: 无法激活 pytorch_downscale，且当前 Python 缺少 torch/netCDF4/h5py" >&2
fi

if [ "$_use_base" = 1 ]; then
    # shellcheck disable=SC1091
    source "${_ENV_DIR}/base.sh"
fi

if ! cd "${_PROJECT_ROOT}" 2>/dev/null; then
    echo "警告: 无法进入项目目录 ${_PROJECT_ROOT}（Notebook 可设置 export DOWNSCALE_WORK=/public/home/acd7koea4a/work）" >&2
else
    :
fi

_env_label="${CONDA_DEFAULT_ENV:-system}"
_data_dir=""
if _py="$(command -v python3 2>/dev/null || command -v python 2>/dev/null)" && [ -n "${_py}" ]; then
    _data_dir="$("${_py}" -c 'from paths import DATA_ROOT; print(DATA_ROOT)' 2>/dev/null || true)"
fi
echo "Python: $("${_py:-python}" --version 2>&1) | 环境: ${_env_label}"
echo "工作目录: $(pwd)"
echo "数据目录: ${_data_dir:-<paths.py 未加载>}"
