"""
# PixelShuffle 下采样模型的数据集定义。
#
# 每个样本返回三元组：
#   x_lr   : (8, 180, 360)   bfloat16
#              8 个归一化 LR 气候变量（LR 静态特征与 cos(SZA)_lr 已移入 hr_aux 路径，
#              不再拼入 LR 输入，避免重复引入干扰）
#   hr_aux : ( 7, 1801, 3600) float32
#              6 个 HR 静态特征 + 1 个 cos(SZA)_hr（本次保持 fp32，不随 HDF5 转换）
#   y_hr   : ( 8, 1801, 3600) bfloat16  （归一化后的高分辨率 8 变量）
#
# 两种 HDF5 数据源（按 shard metadata.normalized 自动识别，同一 hdf5_root 内不得混用）：
#   - 预标准化 fp16（全量训练 hdf5_norm_fp16）：磁盘已是 z-score 后的 float16，
#     读出后转为 bfloat16，不再做 (x-mean)/std
#   - 未标准化 fp32（hdf5_mini / 原始 hdf5 备份 / CESM 训练格式）：读出后在 float32
#     上做 z-score，再转为 bfloat16
#
# Dataset.__getitem__ 出口的 x_lr/y_hr 一律为 bfloat16，与 train.py 的
# autocast(dtype=bfloat16) 对齐。hr_aux 保持 float32。
#
# cos(SZA)（太阳天顶角余弦）在线计算，采用模块级 LRU 缓存。
# LR 特征缓存最大为 400 条（约 100 MB），HR 特征缓存最大为 50 条（约 1.3 GB）。
# 注：_load_lr_static / _cos_sza_lr 函数仍保留，供 infer.py 单独调用。
"""

# 数据流：HDF5 读 LR/HR 变量与日期 → [未标准化则 z-score] → 转 bf16 → 拼 hr_aux → 返回。

from __future__ import annotations

import functools
import json
import math
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# 默认路径（见 paths.py）；运行时可改模块变量 STATIC_DIR / STATS_FILE
# ---------------------------------------------------------------------------
from paths import HDF5_ROOT, STATIC_DIR, STATS_FILE

VARIABLES = ["TAS", "PRE", "wind10", "Q", "2M_RH", "2M_TMAX", "2M_TMIN", "FSDS"]
SEASONS   = ["MAM", "JJA", "SON", "DJF"]

# ---------------------------------------------------------------------------
# 全局统计加载（懒加载，模块级单例）
# ---------------------------------------------------------------------------

_NORM_MEAN: np.ndarray | None = None   # (8,) float32
_NORM_STD:  np.ndarray | None = None   # (8,) float32


def _load_norm_stats() -> tuple[np.ndarray, np.ndarray]:
    global _NORM_MEAN, _NORM_STD
    if _NORM_MEAN is None:
        stats_path = Path(STATS_FILE)
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"找不到归一化统计文件: {stats_path}。"
                "Notebook 需挂载 /public/share/acd7koea4a，"
                "或使用家目录真实副本 /public/home/acd7koea4a/local_data "
                "（paths.py 会在 share 不可用时自动回退）。"
            )
        with open(stats_path) as f:
            d = json.load(f)
        means, stds = [], []
        for v in VARIABLES:
            if "var_all" in d:
                b = d["var_all"][v]
                mn = b["mean"]
                std = math.sqrt(b["M2"] / b["n"])
            elif "lr" in d:
                b = d["lr"][v]
                mn = b["mean"]
                std = float(b["std"])
            else:
                raise KeyError(
                    f"stats file {STATS_FILE} missing var_all/lr for {v}"
                )
            means.append(mn)
            stds.append(std)
        _NORM_MEAN = np.array(means, dtype=np.float32)
        _NORM_STD  = np.array(stds,  dtype=np.float32)
    return _NORM_MEAN, _NORM_STD


# ---------------------------------------------------------------------------
# cos(SZA) 网格（纬度/经度）加载器 — 首次访问时加载一次
# ---------------------------------------------------------------------------

_LAT_LR: np.ndarray | None = None   # (180,)
_LON_LR: np.ndarray | None = None   # (360,)
_LAT_HR: np.ndarray | None = None   # (1801,)
_LON_HR: np.ndarray | None = None   # (3600,)


def _ensure_lat_lon() -> None:
    global _LAT_LR, _LON_LR, _LAT_HR, _LON_HR
    if _LAT_LR is None:
        _LAT_LR = np.load(STATIC_DIR / "lat_lr.npy").astype(np.float64)
        _LON_LR = np.load(STATIC_DIR / "lon_lr.npy").astype(np.float64)
        _LAT_HR = np.load(STATIC_DIR / "lat_hr.npy").astype(np.float64)
        _LON_HR = np.load(STATIC_DIR / "lon_hr.npy").astype(np.float64)


# ---------------------------------------------------------------------------
# cos(SZA) 计算辅助函数
# ---------------------------------------------------------------------------

def _doy_from_date_str(date_str: str | bytes) -> int:
    """Convert 'YYYYMMDD' (or bytes) to day-of-year (1-based)."""
    if isinstance(date_str, (bytes, np.bytes_)):
        date_str = date_str.decode()
    year  = int(date_str[:4])
    month = int(date_str[4:6])
    day   = int(date_str[6:8])
    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        days_in_month[2] = 29
    return sum(days_in_month[:month]) + day


def _daily_mean_cos_sza(
    lat_deg: np.ndarray,   # 1-D latitude array  [degrees]
    doy: int,
) -> np.ndarray:
    """
    # 计算一维纬度数组的逐日日均 cos(SZA)。
    #
    # 使用标准的日照公式（逐日日均无经度依赖）：
    #     δ  = 23.45° × sin(2π/365 × (DOY − 80))              # 太阳赤纬角
    #     H₀ = arccos(−tan(lat) × tan(δ))                     # 日落时角（sunset hour angle）
    #     <cos(SZA)> = (H₀·sin(lat)·sin(δ) + cos(lat)·cos(δ)·sin(H₀)) / π
    #     最终裁剪至 [0, 1] 区间
    #
    # 返回: (len(lat_deg),) float32
    """
    lat = np.deg2rad(lat_deg)
    dec = np.deg2rad(23.45 * math.sin(2 * math.pi / 365 * (doy - 80)))

    tan_lat = np.tan(lat)
    cos_sza_pole = -tan_lat * math.tan(dec)
    # 高纬极区 arccos 定义域保护，避免数值越界
    cos_sza_pole = np.clip(cos_sza_pole, -1.0, 1.0)
    H0 = np.arccos(cos_sza_pole)  # sunset hour angle (0 at polar night, π at midnight sun)

    result = (
        H0 * np.sin(lat) * math.sin(dec)
        + np.cos(lat) * math.cos(dec) * np.sin(H0)
    ) / math.pi
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _build_cos_sza_2d(lats: np.ndarray, doy: int) -> np.ndarray:
    """
    # 返回 (H, W) cos(SZA) 数组，其中 H = len(lats)，每行复制到所有经度（逐日均无经度依赖）
    # Return (H, W) cos(SZA) array where H = len(lats) and each row is
    # replicated across all longitudes (daily mean has no lon dependence).
    # """
    row = _daily_mean_cos_sza(lats, doy)          # (H,)
    return np.broadcast_to(row[:, None], (len(row), 1)).copy()  # (H,1) 用于广播


# ---------------------------------------------------------------------------
# cos(SZA) 按日期 LRU 缓存：主进程 fork 的 DataLoader worker 可复用同进程内缓存
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=400)
def _cos_sza_lr(date_str: str) -> np.ndarray:
    """Return (1, 180, 360) float32, read-only, cached."""
    _ensure_lat_lon()
    doy = _doy_from_date_str(date_str)
    col = _daily_mean_cos_sza(_LAT_LR, doy)  # (180,)
    arr = np.tile(col[:, None], (1, len(_LON_LR))).astype(np.float32)   # (180, 360)
    arr = arr[None]         # (1, 180, 360)
    arr.flags.writeable = False
    return arr


@functools.lru_cache(maxsize=50)
def _cos_sza_hr(date_str: str) -> np.ndarray:
    """Return (1, 1801, 3600) float32, read-only, cached."""
    _ensure_lat_lon()
    doy = _doy_from_date_str(date_str)
    col = _daily_mean_cos_sza(_LAT_HR, doy)  # (1801,)
    arr = np.tile(col[:, None], (1, len(_LON_HR))).astype(np.float32)   # (1801, 3600)
    arr = arr[None]         # (1, 1801, 3600)
    arr.flags.writeable = False
    return arr


# ---------------------------------------------------------------------------
# 静态特征加载（模块级单例）
# ---------------------------------------------------------------------------

_LR_STATIC: np.ndarray | None = None   # (6, 180, 360)
_HR_STATIC: np.ndarray | None = None   # (6, 1801, 3600)


def _load_lr_static() -> np.ndarray:
    global _LR_STATIC
    if _LR_STATIC is None:
        dem      = np.load(STATIC_DIR / "dem_lr_norm.npy").astype(np.float32)       # (1,180,360)
        latlon   = np.load(STATIC_DIR / "latlon_sincos_lr.npy").astype(np.float32)  # (4,180,360)
        lsm      = np.load(STATIC_DIR / "land_sea_mask_lr.npy").astype(np.float32)  # (1,180,360)
        _LR_STATIC = np.concatenate([dem, latlon, lsm], axis=0)                     # (6,180,360)
    return _LR_STATIC


def _load_hr_static() -> np.ndarray:
    global _HR_STATIC
    if _HR_STATIC is None:
        dem      = np.load(STATIC_DIR / "dem_hr_norm.npy").astype(np.float32)       # (1,1801,3600)
        latlon   = np.load(STATIC_DIR / "latlon_sincos_hr.npy").astype(np.float32)  # (4,1801,3600)
        lsm      = np.load(STATIC_DIR / "land_sea_mask_hr.npy").astype(np.float32)  # (1,1801,3600)
        _HR_STATIC = np.concatenate([dem, latlon, lsm], axis=0)                     # (6,1801,3600)
    return _HR_STATIC


# ---------------------------------------------------------------------------
# 分片索引构建器
# ---------------------------------------------------------------------------

import re

# shard 文件名约定：shard_{batch_tag}_{序号}.h5，batch_tag 允许含下划线
# （如 "cra1p5_full"），因此用“最后一段纯数字”反推 batch_tag，而非用第一个 "_" 切分。
_SHARD_NAME_RE = re.compile(r"^shard_(?P<tag>.+)_(?P<idx>\d+)\.h5$")


def _shard_batch_tag(h5path: Path) -> str | None:
    """从 shard 文件名中解析出 batch_tag（如 'cra1p5_full'）；不匹配约定则返回 None。"""
    m = _SHARD_NAME_RE.match(h5path.name)
    return m.group("tag") if m else None


def is_pre_normalized_shard(h5path: Path | str) -> bool:
    """True iff the shard metadata marks pre-normalized z-score fp16 storage.

    Missing metadata group or missing ``normalized`` attr → False (legacy fp32).
    """
    with h5py.File(h5path, "r") as f:
        if "metadata" not in f:
            return False
        return bool(f["metadata"].attrs.get("normalized", False))


def _build_index(
    hdf5_root: Path,
    seasons: Sequence[str],
    manifests: Sequence[str] | None = None,
) -> tuple[list[tuple[str, int]], bool]:
    """
    Returns (index, pre_normalized).

    index: flat list of (hdf5_file_path, sample_idx_within_file).
    pre_normalized: True if every included shard is marked normalized.

    将各季节目录下 shard_*.h5 展平为全局样本索引，供 Dataset.__getitem__ 随机访问。
    同一 hdf5_root 下不得混用预标准化 / 未标准化 shard。

    Args:
        hdf5_root:  directory containing season sub-folders.
        seasons:    list of season names to include.
        manifests:  可选，允许的 batch_tag 白名单（如 ["cra1p5_full"]）。非空时，
                    只保留文件名匹配 shard_{tag}_{序号}.h5 且 tag 属于该白名单的
                    shard；文件名不符合该约定的 shard 一律跳过并打印警告（避免
                    静默漏读/误读）。None 或空序列 → 不过滤，纳入全部 shard_*.h5
                    （当前全新转换的数据只有单一 batch_tag=cra1p5_full 时，不传
                    该参数即可；后续混合多批次、需要排除某个坏批次时再启用）。
    """
    allowed = set(manifests) if manifests else None
    index: list[tuple[str, int]] = []
    flags: dict[str, bool] = {}
    for season in seasons:
        season_dir = hdf5_root / season
        if not season_dir.exists():
            continue
        h5_files = sorted(season_dir.glob("shard_*.h5"))
        for h5path in h5_files:
            if allowed is not None:
                tag = _shard_batch_tag(h5path)
                if tag is None:
                    print(f"[dataset] 警告：{h5path.name} 不符合 shard_{{tag}}_{{idx}}.h5 "
                          f"命名约定，manifests 过滤下已跳过")
                    continue
                if tag not in allowed:
                    continue
            with h5py.File(h5path, "r") as f:
                n = f["data/x"].shape[0]
                normalized = False
                if "metadata" in f:
                    normalized = bool(f["metadata"].attrs.get("normalized", False))
            flags[str(h5path)] = bool(normalized)
            index.extend((str(h5path), i) for i in range(n))
    uniq = set(flags.values())
    if len(uniq) > 1:
        true_files = [p for p, v in flags.items() if v]
        false_files = [p for p, v in flags.items() if not v]
        raise ValueError(
            f"{hdf5_root}: 同一 hdf5_root 混有预标准化与未标准化 shard，"
            f"normalized=True 例: {true_files[:3]} ; "
            f"normalized=False 例: {false_files[:3]}"
        )
    pre_normalized = bool(next(iter(uniq))) if uniq else False
    return index, pre_normalized


# ---------------------------------------------------------------------------
# 降尺度数据集
# ---------------------------------------------------------------------------

class DownscaleDataset(Dataset):
    """
    # 用于气候变量降尺度的 PyTorch Dataset。

    # 每个样本包含：
    #     x_lr   : ( 8, 180, 360)   bfloat16  —— 标准化后的低分辨率 8 个气候变量
    #     hr_aux : ( 7, 1801, 3600) float32   —— 高分辨率静态因子 + cos_sza_hr
    #     y_hr   : ( 8, 1801, 3600) bfloat16  —— 标准化后的高分辨率目标变量

    # 参数说明:
    #     hdf5_root:   包含各个季节子文件夹的根目录
    #     seasons:     需要加载哪些季节（默认全部四季）
    #     static_dir:  static/*.npy 文件的路径
    #     stats_file:  global_stats_state.json 的路径
    #     manifests:   可选，允许的 batch_tag 白名单（见 _build_index），用于排除坏批次

    """

    def __init__(
        self,
        hdf5_root:  Path | str = HDF5_ROOT,
        seasons:    Sequence[str] = SEASONS,
        static_dir: Path | str = STATIC_DIR,
        stats_file: Path | str = STATS_FILE,
        manifests:  Sequence[str] | None = None,
    ):
        super().__init__()
        global STATIC_DIR, STATS_FILE
        STATIC_DIR = Path(static_dir)
        STATS_FILE = Path(stats_file)

        self.hdf5_root = Path(hdf5_root)
        self.index, self.pre_normalized = _build_index(self.hdf5_root, seasons, manifests)

        # 全局 mean/std，与 HDF5 中 8 变量顺序一致（反标准化 / 未标准化路径仍需要）
        self.norm_mean, self.norm_std = _load_norm_stats()   # (8,) each

        # fork 子进程后写时复制，多 worker 共享只读大数组内存
        # lr_static 不再拼入 x_lr，但保留加载供子类或外部调用
        self.lr_static = _load_lr_static()   # (6, 180, 360)
        self.hr_static = _load_hr_static()   # (6, 1801, 3600)

        print(
            f"[dataset] hdf5_root={self.hdf5_root}  pre_normalized={self.pre_normalized}  "
            f"n_samples={len(self.index)}  static_dir={STATIC_DIR}  stats_file={STATS_FILE}"
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h5path, sample_i = self.index[idx]

        with h5py.File(h5path, "r") as f:
            x_raw = f["data/x"][sample_i]       # (8, 180, 360)
            y_raw = f["data/y"][sample_i]       # (8, 1801, 3600)
            date  = f["data/dates"][sample_i]   # bytes, e.g. b'19790101'

        date_str = date.decode() if isinstance(date, (bytes, np.bytes_)) else str(date)

        if self.pre_normalized:
            # 磁盘已是 z-score fp16；进程内统一转 bf16，不再做 (x-mean)/std
            x = torch.from_numpy(np.array(x_raw, copy=True)).to(torch.bfloat16)
            y = torch.from_numpy(np.array(y_raw, copy=True)).to(torch.bfloat16)
        else:
            # 与训练目标一致：LR/HR 动态变量均用同一组全局 mean/std 做 z-score
            x = x_raw.astype(np.float32)   # (8, 180, 360)
            y = y_raw.astype(np.float32)   # (8, 1801, 3600)
            mean = self.norm_mean[:, None, None]   # (8, 1, 1)
            std  = self.norm_std[:, None, None]
            x = (x - mean) / std
            y = (y - mean) / std
            x = torch.from_numpy(np.ascontiguousarray(x)).to(torch.bfloat16)
            y = torch.from_numpy(np.ascontiguousarray(y)).to(torch.bfloat16)

        # 日尺度平均 cos(SZA)，仅随纬度与日期变化；缓存返回只读数组
        sza_hr = _cos_sza_hr(date_str)   # (1, 1801, 3600)

        # 网络 LR 输入：仅 8 个归一化气候变量（去掉 LR 静态与 cos_sza_lr，避免重复干扰）
        x_lr = x   # (8, 180, 360)

        # HR 辅助（不进 y）：6 HR 静态 + 1 cos_sza_hr = 7 通道；保持 fp32
        hr_aux = np.concatenate([self.hr_static, sza_hr], axis=0)    # (7, 1801, 3600)

        # copy：避免与 h5 缓冲区共享可写内存，防止 DataLoader 多线程竞态
        return (
            x_lr,
            torch.from_numpy(hr_aux.copy()),
            y,
        )

    # ------------------------------------------------------------------
    # 方便：返回归一化统计量作为张量
    # ------------------------------------------------------------------
    def get_norm_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std) tensors of shape (8,) for denormalisation."""
        return (
            torch.from_numpy(self.norm_mean.copy()),
            torch.from_numpy(self.norm_std.copy()),
        )
