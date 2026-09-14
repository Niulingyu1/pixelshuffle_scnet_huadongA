"""
SCNet / qdcs 集群路径配置。

数据根目录：/public/share/acd7koea4a（HDF5、静态场、CESM 等）
代码与训练产物：/public/home/acd7koea4a/work（runs、logs、infer_out）

所有路径在此统一定义；修改数据目录只需改本文件中的 DATA_ROOT。
"""
from __future__ import annotations

from pathlib import Path

# 家目录 / 代码区
HOME = Path("/public/home/acd7koea4a")
WORK_ROOT = HOME / "work"

# 数据根目录（团队共享存储）
DATA_ROOT = Path("/public/share/acd7koea4a")

# 静态场与归一化统计
STATIC_DIR = DATA_ROOT / "static"


def _resolve_stats_file() -> Path:
    """优先 states/global_stats_state.json，回退至 static/ 下已有统计文件。"""
    for p in (
        DATA_ROOT / "states" / "global_stats_state.json",
        STATIC_DIR / "global_stats_state.json",
        STATIC_DIR / "global_stats.json",
    ):
        if p.exists():
            return p
    return DATA_ROOT / "states" / "global_stats_state.json"


STATS_FILE = _resolve_stats_file()

# HDF5 数据集（按用途分目录，从旧机迁移后放到对应子目录）
HDF5_ROOT = DATA_ROOT / "hdf5"          # 全量训练集 1979–2019
HDF5_HALF = DATA_ROOT / "hdf5_half"     # 半量训练集 2000–2019
HDF5_MINI = DATA_ROOT / "hdf5_mini"     # 调试用小集
HDF5_TEST = DATA_ROOT / "hdf5_test"       # 测试集 2020–2021
HDF5_CESM = DATA_ROOT / "hdf5_cesm"       # CESM 转换后推理输入


def resolve_hdf5_train_root(*, prefer_half: bool = False) -> Path:
    """返回第一个存在的 HDF5 训练目录。

    默认优先全量 ``hdf5``，再回退 ``hdf5_half``、``hdf5_mini``。
    半量集未迁移时不会因此失败。
    """
    if prefer_half:
        candidates = (HDF5_HALF, HDF5_ROOT, HDF5_MINI)
    else:
        candidates = (HDF5_ROOT, HDF5_HALF, HDF5_MINI)
    for p in candidates:
        if p.is_dir():
            return p
    return HDF5_ROOT


# CESM 原始数据
CESM_ROOT = DATA_ROOT / "cesm"
# 推理输出放在 work 区：登录节点 / Notebook 上 /public 常为只读
INFER_OUT = WORK_ROOT / "infer_out"
EXAMPLE_NC = DATA_ROOT / "example" / "obs_20000101.nc"

# 训练产物
RUNS_DIR = WORK_ROOT / "runs"
LOGS_DIR = WORK_ROOT / "logs"
