"""
将官方 CESM 日尺度数据（NetCDF/npy）转换为可与 `infer.py` 直接对接的 HDF5 分片。

脚本作用说明：
  - 训练/验证/测试用 HDF5 通常由 npy（coarse/resized）字典生成，而官方 CESM 数据主要以 daily NetCDF（或部分 npy 字典）形式分发。
  - 为保证推理链路的高效与健壮性，本脚本将 CESM 每日物理场转为与推理一致的数据排布。
  - 默认行为是保留 CESM 原生 180×360 网格顺序，保证输出 /data/x 与 coarse/*.npy 语义最大程度一致。
  - 如需与训练集静态 LR 网格完全对齐，可通过指定参数开启重网格模式。

输出结构（每个分片）：
  /data/x      float16，形状 (N, 8, 180, 360)   （预处理后，未做 z-score 标准化）
  /data/dates  vlen 字节数组，形状 (N,)，每项为 b"YYYYMMDD"
  /metadata/*  可选自描述属性/网格信息

重要约定：
  - 通道顺序（channel dim=1）：["TAS","PRE","wind10","Q","2M_RH","2M_TMAX","2M_TMIN","FSDS"]
  - PRE: 先 clip>=0，再 log1p，再 z-score（与训练 HDF5 预处理一致）
  - Q: 单位由 kg/kg ×1000 转为 g/kg，再做 z-score
  - 默认网格语义：保持原始 CESM 180×360 排列（通常 lat=-90..90, lon=0..359），以最大程度拟合 coarse/*.npy
  - 可选：强制重网格至训练静态 LR 网格（/public/share/acd7koea4a/static/lat_lr.npy, lon_lr.npy）

使用示例：


 CUDA_VISIBLE_DEVICES=2 conda run -n pytorch_downscale python prepare_hdf5_cesm.py \
    --cesm_root /public/share/acd7koea4a/cesm \
    --out_hdf5_root /public/share/acd7koea4a/hdf5_cesm_com \
    --samples_per_shard 100 \
    --date_start 20210101 --date_end 20211231

CUDA_VISIBLE_DEVICES=7 nohup python prepare_hdf5_cesm.py \
    --cesm_root /public/share/acd7koea4a/cesm_no_mbc \
    --out_hdf5_root /public/share/acd7koea4a/hdf5_cesm_no_mbc \
    --samples_per_shard 100 \
    --date_start 20000101 --date_end 20001231 \
    --overwrite > prepare_hdf5_cesm_no_mbc.log 2>&1 & echo "作业PID: $!" >> prepare_hdf5_cesm_no_mbc.log





  # 如需强制重网格至训练静态 LR
  conda run -n pytorch_downscale python prepare_hdf5_cesm.py \
    --cesm_root /public/share/acd7koea4a/cesm \
    --out_hdf5_root /public/share/acd7koea4a/hdf5_cesm_regridded \
    --grid_semantics training_static \
    --samples_per_shard 100 \
    --date_start 20000101 --date_end 20000101

后续用上述 HDF5 目录作推理输入：
  conda run -n pytorch_downscale python infer.py \
    --input_source hdf5 --hdf5_root /public/share/acd7koea4a/hdf5_cesm --seasons DJF MAM JJA SON \
    --ckpt ... --out_dir ... --output_mode per_sample --output_format nc --auto_model_cfg
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import h5py
import netCDF4
import numpy as np

from paths import CESM_ROOT, STATIC_DIR


CHANNEL_ORDER = (
    "TAS",
    "PRE",
    "wind10",
    "Q",
    "2M_RH",
    "2M_TMAX",
    "2M_TMIN",
    "FSDS",
)

SEASON_MONTHS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}


@dataclass(frozen=True)
class CesmVarMap:
    tas: str = "t2m"
    pre: str = "pr"
    wind10: str = "wind10"
    q: str = "q"
    rh: str = "rhmin"
    tmax: str = "tmax"
    tmin: str = "tmin"
    fsds: str = "fsds"


def _normalize_date_yyyymmdd(s: str) -> str:
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else s


def _date_in_range(date_yyyymmdd: str, start: str | None, end: str | None) -> bool:
    date_yyyymmdd = _normalize_date_yyyymmdd(date_yyyymmdd)
    start = _normalize_date_yyyymmdd(start) if start else None
    end = _normalize_date_yyyymmdd(end) if end else None
    if start and date_yyyymmdd < start:
        return False
    if end and date_yyyymmdd > end:
        return False
    return True


def _parse_date_from_name(path: Path) -> str:
    m = re.search(r"(\d{8})", path.name)
    if not m:
        raise ValueError(f"Could not parse YYYYMMDD from: {path.name}")
    return m.group(1)


def _season_from_date(date_yyyymmdd: str) -> str:
    m = int(date_yyyymmdd[4:6])
    for s, months in SEASON_MONTHS.items():
        if m in months:
            return s
    raise ValueError(f"Bad month in date: {date_yyyymmdd}")


def _data_compression_kwargs(gzip_level: int) -> dict:
    return {"compression": "gzip", "compression_opts": int(gzip_level)}


def _load_static_lr_grid(static_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    lat = np.load(static_dir / "lat_lr.npy").astype(np.float32)
    lon = np.load(static_dir / "lon_lr.npy").astype(np.float32)
    return lat, lon


def _load_static_hr_grid(static_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    lat = np.load(static_dir / "lat_hr.npy").astype(np.float32)
    lon = np.load(static_dir / "lon_hr.npy").astype(np.float32)
    return lat, lon


def _normalize_lon_convention(lon: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Return a strictly increasing native lon axis plus a convention tag.

    For coarse-native semantics we preserve the source column ordering rather than
    forcing a remap to the static training grid.
    """
    lon = np.asarray(lon, dtype=np.float32)
    if lon.ndim != 1:
        raise ValueError(f"Expected 1D lon, got {lon.shape}")
    if np.all(np.diff(lon) > 0):
        lo_min = float(lon.min())
        lo_max = float(lon.max())
        if lo_min >= 0.0 and lo_max > 180.0:
            return lon.astype(np.float32, copy=False), "pos0_360"
        return lon.astype(np.float32, copy=False), "neg180_180"
    raise ValueError("Native lon axis must be strictly increasing")


def _squeeze_time(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    raise ValueError(f"Expected 2D or (1,H,W), got {arr.shape}")


def _regrid_to_training_lr(
    field: np.ndarray,  # (H,W) CESM
    src_lat: np.ndarray,  # (H,)
    src_lon: np.ndarray,  # (W,) usually 0..360
    tgt_lat: np.ndarray,  # (180,) -89.5..89.5
    tgt_lon: np.ndarray,  # (360,) -179.5..179.5
) -> np.ndarray:
    """
    2-step linear interpolation: lon then lat, with longitude wrap support.
    """
    src_lat = np.asarray(src_lat, dtype=np.float64)
    src_lon = np.asarray(src_lon, dtype=np.float64)
    tgt_lat = np.asarray(tgt_lat, dtype=np.float64)
    tgt_lon = np.asarray(tgt_lon, dtype=np.float64)
    field = np.asarray(field, dtype=np.float64)

    if src_lat.ndim != 1 or src_lon.ndim != 1 or field.ndim != 2:
        raise ValueError(f"Bad dims: lat{src_lat.shape} lon{src_lon.shape} field{field.shape}")
    if field.shape != (src_lat.shape[0], src_lon.shape[0]):
        raise ValueError(f"Shape mismatch: field{field.shape} vs lat{src_lat.shape} lon{src_lon.shape}")

    # Make latitude ascending.
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        field = field[::-1, :]

    if not np.all(np.diff(src_lon) > 0):
        raise ValueError("src_lon must be strictly increasing for interpolation")

    # Extend longitude for wrap-around.
    src_lon_ext = np.concatenate([src_lon, src_lon + 360.0], axis=0)  # (2W,)
    field_ext = np.concatenate([field, field], axis=1)  # (H, 2W)

    # Map target lon (-180..180) to [0,360) and make it increasing.
    tgt_lon_mod = np.mod(tgt_lon, 360.0)
    tgt_lon_ext = tgt_lon_mod.copy()
    base = float(tgt_lon_ext[0])
    tgt_lon_ext[tgt_lon_ext < base] += 360.0

    tmp = np.empty((field.shape[0], tgt_lon_ext.shape[0]), dtype=np.float64)
    for i in range(field.shape[0]):
        tmp[i, :] = np.interp(tgt_lon_ext, src_lon_ext, field_ext[i, :])

    out = np.empty((tgt_lat.shape[0], tmp.shape[1]), dtype=np.float64)
    for j in range(tmp.shape[1]):
        out[:, j] = np.interp(tgt_lat, src_lat, tmp[:, j])

    return out.astype(np.float32)


def _apply_preprocess(x: np.ndarray, var: str, *, q_scale: float, pre_transform: str) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if var == "PRE":
        if pre_transform == "log1p":
            return np.log1p(np.clip(x, 0.0, None)).astype(np.float32)
        if pre_transform == "none":
            return x
        raise ValueError(pre_transform)
    if var == "Q":
        return (x * float(q_scale)).astype(np.float32)
    return x


def _read_cesm_nc(
    path: Path,
    *,
    lat_var: str,
    lon_var: str,
    var_map: CesmVarMap,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with netCDF4.Dataset(str(path), "r") as f:
        if lat_var not in f.variables or lon_var not in f.variables:
            raise KeyError(f"Missing lat/lon vars in {path.name}: {lat_var!r}/{lon_var!r}")
        src_lat = np.array(f.variables[lat_var][:], dtype=np.float32)
        src_lon = np.array(f.variables[lon_var][:], dtype=np.float32)

        def get(name: str) -> np.ndarray:
            if name not in f.variables:
                raise KeyError(f"Missing variable {name!r} in {path.name}")
            return _squeeze_time(np.array(f.variables[name][:], dtype=np.float32))

        fields = {
            "TAS": get(var_map.tas),
            "PRE": get(var_map.pre),
            "wind10": get(var_map.wind10),
            "Q": get(var_map.q),
            "2M_RH": get(var_map.rh),
            "2M_TMAX": get(var_map.tmax),
            "2M_TMIN": get(var_map.tmin),
            "FSDS": get(var_map.fsds),
        }
    return src_lat, src_lon, fields


def _read_cesm_npy(path: Path) -> dict[str, np.ndarray]:
    d = np.load(str(path), allow_pickle=True).item()
    if not isinstance(d, dict):
        raise TypeError(f"Expected dict in {path.name}, got {type(d)}")
    return d


def _ensure_h5_initialized(
    hf: h5py.File,
    *,
    season: str,
    batch_tag: str,
    gzip_level: int,
    x_tail: tuple[int, int, int],
    static_dir: Path,
    coord_source: str,
    coord_ref: str,
    lr_grid_lats: np.ndarray,
    lr_grid_lons: np.ndarray,
    hr_grid_lats: np.ndarray,
    hr_grid_lons: np.ndarray,
    lr_grid_semantics: str,
    lr_lon_convention: str,
    regrid_enabled: bool,
    pre_transform: str,
    q_scale: float,
) -> None:
    if "data" in hf:
        return

    comp_kw = _data_compression_kwargs(gzip_level)
    vlen_bytes = h5py.special_dtype(vlen=bytes)
    creation_iso = datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")

    g_data = hf.create_group("data")
    g_meta = hf.create_group("metadata")

    g_data.create_dataset(
        "x",
        shape=(0,) + x_tail,
        maxshape=(None,) + x_tail,
        chunks=(1,) + x_tail,
        dtype=np.float16,
        **comp_kw,
    )
    g_data.create_dataset(
        "dates",
        shape=(0,),
        maxshape=(None,),
        dtype=vlen_bytes,
    )

    g_meta.attrs["season"] = season
    g_meta.attrs["batch_tag"] = batch_tag
    g_meta.attrs["variables"] = json.dumps(list(CHANNEL_ORDER), ensure_ascii=False)
    g_meta.attrs["transforms"] = json.dumps(
        {
            "TAS": "none",
            "PRE": "log1p" if pre_transform == "log1p" else "none",
            "wind10": "none",
            "Q": "mul1000" if abs(float(q_scale) - 1000.0) < 1e-6 else f"mul{q_scale:g}",
            "2M_RH": "none",
            "2M_TMAX": "none",
            "2M_TMIN": "none",
            "FSDS": "none",
        },
        ensure_ascii=False,
    )
    g_meta.attrs["norm_domain"] = "after_transform"
    g_meta.attrs["creation_date"] = creation_iso
    g_meta.attrs["source"] = "cesm"
    g_meta.attrs["coord_source"] = coord_source
    g_meta.attrs["coord_ref"] = coord_ref
    g_meta.attrs["regrid_to_training_lr"] = bool(regrid_enabled)
    g_meta.attrs["pre_transform"] = pre_transform
    g_meta.attrs["q_scale"] = float(q_scale)
    g_meta.attrs["lr_grid_semantics"] = lr_grid_semantics
    g_meta.attrs["lr_lon_convention"] = lr_lon_convention

    # record grids for bookkeeping
    g_meta.create_dataset("lr_grid_lats", data=np.asarray(lr_grid_lats, dtype=np.float32), dtype=np.float32)
    g_meta.create_dataset("lr_grid_lons", data=np.asarray(lr_grid_lons, dtype=np.float32), dtype=np.float32)
    g_meta.create_dataset("hr_grid_lats", data=np.asarray(hr_grid_lats, dtype=np.float32), dtype=np.float32)
    g_meta.create_dataset("hr_grid_lons", data=np.asarray(hr_grid_lons, dtype=np.float32), dtype=np.float32)


def _append_one(
    hf: h5py.File,
    *,
    x_stack_fp16: np.ndarray,  # (8,180,360) float16
    date: str,
) -> None:
    g_data = hf["data"]
    ds_x = g_data["x"]
    ds_dates = g_data["dates"]
    n = ds_x.shape[0]
    ds_x.resize((n + 1,) + ds_x.shape[1:])
    ds_dates.resize((n + 1,))
    ds_x[n] = x_stack_fp16
    ds_dates[n] = date.encode("ascii")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert CESM daily data to seasonal HDF5 shards for inference")
    p.add_argument("--cesm_root", type=str, default=str(CESM_ROOT), help="CESM directory")
    p.add_argument("--input_format", choices=["nc", "npy"], default="nc", help="read CESM from daily .nc or dict .npy")
    p.add_argument("--cesm_glob", type=str, default="cesm_1deg_*.nc", help="glob for nc files under cesm_root")
    p.add_argument("--npy_glob", type=str, default="[0-9]*.npy", help="glob for npy files under cesm_root")
    p.add_argument("--coord_ref_nc", type=str, default=None, help="required when input_format=npy; used to load CESM lat/lon")
    p.add_argument("--cesm_lat_var", type=str, default="lat")
    p.add_argument("--cesm_lon_var", type=str, default="lon")

    p.add_argument("--var_tas", type=str, default="t2m")
    p.add_argument("--var_pre", type=str, default="pr")
    p.add_argument("--var_wind10", type=str, default="wind10")
    p.add_argument("--var_q", type=str, default="q")
    p.add_argument("--var_rh", type=str, default="rhmin")
    p.add_argument("--var_tmax", type=str, default="tmax")
    p.add_argument("--var_tmin", type=str, default="tmin")
    p.add_argument("--var_fsds", type=str, default="fsds")

    p.add_argument("--static_dir", type=str, default=str(STATIC_DIR), help="training static dir (lat_lr/lon_lr)")
    p.add_argument("--out_hdf5_root", type=str, required=True, help="output root, will create DJF/MAM/JJA/SON subdirs")
    p.add_argument("--batch_tag", type=str, default="cesm", help="tag used in shard filenames")
    p.add_argument("--samples_per_shard", type=int, default=100)
    p.add_argument("--gzip_level", type=int, default=4)

    p.add_argument("--date_start", type=str, default=None, help="inclusive YYYYMMDD")
    p.add_argument("--date_end", type=str, default=None, help="inclusive YYYYMMDD")
    p.add_argument("--year_min", type=int, default=None)
    p.add_argument("--year_max", type=int, default=None)

    p.add_argument(
        "--grid_semantics",
        choices=["coarse_native", "training_static"],
        default="coarse_native",
        help=(
            "how to interpret/store the LR grid. "
            "coarse_native preserves source 180x360 ordering to mimic coarse.npy semantics; "
            "training_static regrids to static lat_lr/lon_lr."
        ),
    )
    p.add_argument(
        "--no_regrid",
        action="store_true",
        help="deprecated compatibility flag; equivalent to --grid_semantics coarse_native",
    )
    p.add_argument("--pre_transform", choices=["log1p", "none"], default="log1p")
    p.add_argument("--q_scale", type=float, default=1000.0)

    p.add_argument("--overwrite", action="store_true", help="overwrite existing shards (dangerous; default false)")
    p.add_argument("--max_samples", type=int, default=None, help="debug: limit total samples written")
    args = p.parse_args()

    cesm_root = Path(args.cesm_root)
    static_dir = Path(args.static_dir)
    out_root = Path(args.out_hdf5_root)
    out_root.mkdir(parents=True, exist_ok=True)

    var_map = CesmVarMap(
        tas=args.var_tas,
        pre=args.var_pre,
        wind10=args.var_wind10,
        q=args.var_q,
        rh=args.var_rh,
        tmax=args.var_tmax,
        tmin=args.var_tmin,
        fsds=args.var_fsds,
    )

    tgt_lat, tgt_lon = _load_static_lr_grid(static_dir)
    hr_lat_meta, hr_lon_meta = _load_static_hr_grid(static_dir)
    grid_semantics = str(args.grid_semantics)
    if bool(args.no_regrid):
        grid_semantics = "coarse_native"
    regrid_enabled = (grid_semantics == "training_static")

    # Load CESM coord reference if input_format=npy
    src_lat_ref = src_lon_ref = None
    coord_source = "cesm_nc"
    coord_ref = ""
    if args.input_format == "npy":
        if not args.coord_ref_nc:
            raise SystemExit("input_format=npy requires --coord_ref_nc to load CESM lat/lon")
        coord_source = "ref_nc_for_npy"
        coord_ref = args.coord_ref_nc
        with netCDF4.Dataset(str(Path(args.coord_ref_nc)), "r") as f:
            src_lat_ref = np.array(f.variables[args.cesm_lat_var][:], dtype=np.float32)
            src_lon_ref = np.array(f.variables[args.cesm_lon_var][:], dtype=np.float32)

    # Gather input files
    if args.input_format == "nc":
        files = sorted(cesm_root.glob(args.cesm_glob))
        coord_ref = args.cesm_glob
    else:
        files = sorted(cesm_root.glob(args.npy_glob))
        # keep only files with an 8-digit date
        files = [f for f in files if re.fullmatch(r"\d{8}\.npy", f.name)]

    if not files:
        raise SystemExit(f"No input files found under {cesm_root} (format={args.input_format})")

    t0 = time.time()
    n_written_total = 0

    # Per-season shard state
    current_hf: dict[str, h5py.File | None] = {s: None for s in SEASON_MONTHS.keys()}
    current_path: dict[str, Path | None] = {s: None for s in SEASON_MONTHS.keys()}
    current_count: dict[str, int] = {s: 0 for s in SEASON_MONTHS.keys()}
    shard_index: dict[str, int] = {s: 0 for s in SEASON_MONTHS.keys()}

    def close_season(season: str) -> None:
        hf = current_hf[season]
        if hf is None:
            return
        # finalize counts
        try:
            if "metadata" in hf:
                hf["metadata"].attrs["n_samples"] = np.int64(hf["data/x"].shape[0])
                hf["metadata"].attrs["x_shape"] = str(hf["data/x"].shape)
        finally:
            hf.close()
        current_hf[season] = None
        current_path[season] = None
        current_count[season] = 0

    def open_new_shard(season: str) -> None:
        season_dir = out_root / season
        season_dir.mkdir(parents=True, exist_ok=True)
        out_path = season_dir / f"shard_{args.batch_tag}_{shard_index[season]:04d}.h5"
        shard_index[season] += 1

        if out_path.exists():
            if not args.overwrite:
                raise SystemExit(f"Output exists: {out_path} (use --overwrite to replace)")
            out_path.unlink()

        hf = h5py.File(str(out_path), "w")
        current_hf[season] = hf
        current_path[season] = out_path
        current_count[season] = 0

    try:
        for fp in files:
            date = _parse_date_from_name(fp)
            y = int(date[:4])
            if args.year_min is not None and y < int(args.year_min):
                continue
            if args.year_max is not None and y > int(args.year_max):
                continue
            if not _date_in_range(date, args.date_start, args.date_end):
                continue

            season = _season_from_date(date)

            if args.max_samples is not None and n_written_total >= int(args.max_samples):
                break

            hf = current_hf[season]
            if hf is None or current_count[season] >= int(args.samples_per_shard):
                close_season(season)
                open_new_shard(season)
                hf = current_hf[season]
                assert hf is not None

            # read
            if args.input_format == "nc":
                src_lat, src_lon, fields = _read_cesm_nc(
                    fp,
                    lat_var=args.cesm_lat_var,
                    lon_var=args.cesm_lon_var,
                    var_map=var_map,
                )
            else:
                d = _read_cesm_npy(fp)
                assert src_lat_ref is not None and src_lon_ref is not None
                src_lat, src_lon = src_lat_ref, src_lon_ref
                fields = {
                    "TAS": np.asarray(d[var_map.tas], dtype=np.float32),
                    "PRE": np.asarray(d[var_map.pre], dtype=np.float32),
                    "wind10": np.asarray(d[var_map.wind10], dtype=np.float32),
                    "Q": np.asarray(d[var_map.q], dtype=np.float32),
                    "2M_RH": np.asarray(d[var_map.rh], dtype=np.float32),
                    "2M_TMAX": np.asarray(d[var_map.tmax], dtype=np.float32),
                    "2M_TMIN": np.asarray(d[var_map.tmin], dtype=np.float32),
                    "FSDS": np.asarray(d[var_map.fsds], dtype=np.float32),
                }

            if grid_semantics == "training_static":
                lr_lat_meta = tgt_lat
                lr_lon_meta = tgt_lon
                lr_lon_convention = "neg180_180"
            else:
                lr_lat_meta = np.asarray(src_lat, dtype=np.float32)
                lr_lon_meta, lr_lon_convention = _normalize_lon_convention(src_lon)

            # preprocess + optional regrid + stack
            xs: list[np.ndarray] = []
            for v in CHANNEL_ORDER:
                a = fields[v]
                if a.shape != (int(src_lat.shape[0]), int(src_lon.shape[0])):
                    raise ValueError(f"{fp.name} {v}: unexpected shape {a.shape}")
                if regrid_enabled:
                    a = _regrid_to_training_lr(a, src_lat, src_lon, tgt_lat, tgt_lon)
                a = _apply_preprocess(a, v, q_scale=float(args.q_scale), pre_transform=args.pre_transform)
                xs.append(a)

            x_stack = np.stack(xs, axis=0).astype(np.float16)  # (8,180,360)

            _ensure_h5_initialized(
                hf,
                season=season,
                batch_tag=args.batch_tag,
                gzip_level=int(args.gzip_level),
                x_tail=x_stack.shape,
                static_dir=static_dir,
                coord_source=coord_source,
                coord_ref=coord_ref,
                lr_grid_lats=lr_lat_meta,
                lr_grid_lons=lr_lon_meta,
                hr_grid_lats=hr_lat_meta,
                hr_grid_lons=hr_lon_meta,
                lr_grid_semantics=grid_semantics,
                lr_lon_convention=lr_lon_convention,
                regrid_enabled=regrid_enabled,
                pre_transform=args.pre_transform,
                q_scale=float(args.q_scale),
            )

            _append_one(hf, x_stack_fp16=x_stack, date=date)
            current_count[season] += 1
            n_written_total += 1

            if n_written_total % 50 == 0:
                dt = time.time() - t0
                print(f"[{n_written_total}] wrote sample date={date}  elapsed={dt:.1f}s")

    finally:
        for s in SEASON_MONTHS.keys():
            close_season(s)

    dt = time.time() - t0
    print(f"Done. total_written={n_written_total}  out_root={out_root}  elapsed={dt:.1f}s")


if __name__ == "__main__":
    main()

