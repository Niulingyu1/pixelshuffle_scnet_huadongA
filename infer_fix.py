r"""
过期副本，不要用。当前推理入口是同目录的 infer.py（含 --use_ema、--norm_type、
--auto_model_cfg 推断 GroupNorm）。本文件保留仅供对照旧行为，示例路径来自其它工作区。

PixelShuffleDownscaleNet 推理脚本。

主要目标：
  - 在测试集（含 y）及正式评测集（无 y）上执行推理。
  - 支持通过 YYYYMMDD 格式的起止日期进行包含端点的数据筛选。
  - 高效写出预测结果（默认：每个输入分片对应一个输出）。

已知异常测试日期（仅 hdf5_test 默认剔除；CESM HDF5 不剔除）：
  在 /public/share/acqmesjai0/nly/hdf5_test/SON/shard_2020-2024_0003.h5 中，
  以下 5 个日期的 data/x、data/y 存在物理不合理值（如 wind10 恒为 150、
  Q 恒为 0 或 100、TAS 与 TMAX/TMIN 明显不一致等），判断为源数据异常/损坏样本，
  而非模型或分析脚本问题（详见 nc_analyze_logs/hdf5_x_anomaly_scan.csv、
  hdf5_y_anomaly_spotcheck.csv 的排查记录）：
    20240903, 20240926, 20240927, 20241001, 20241004
  开关：--skip_bad_dates / --no_skip_bad_dates
    - 测试集 HDF5：默认开启剔除
    - CESM HDF5（路径含 cesm，或 shard metadata.source=cesm）：默认关闭剔除
    - 也可显式 --exclude_dates "" 关闭（兼容旧用法）

本仓库训练/推理约定：
  - HDF5 文件中的 x/y 存储 8 个气候变量；PRE 已做 log1p 变换；Q 已为 g/kg 单位。
  - “标准化”使用 global_stats_state.json（仅针对 8 个变量），即 z-score。
  - 静态特征与 cos(SZA) 不做归一化或标准化。
  - 模型输入：x_lr（15通道）和 hr_aux（7通道）；模型输出为 z-score 空间下的 8 通道。

正式数据集（CESM daily NetCDF）说明：
  - 文件示例：/root/data/cesm/cesm_1deg_YYYYMMDD.nc（time=1, lat=180, lon=360）
  - 变量示例：t2m/tmax/tmin/pr/fsds/wind10/rhmin/q
  - 注意：CESM 的 lat/lon 往往与训练静态 LR 网格（static/lat_lr.npy, static/lon_lr.npy）不一致，
    推理前需要先重网格到训练 LR 网格；脚本默认会做线性重网格（可用 --no_cesm_regrid 关闭）。

比湿 Q 的 physical 输出改为 kg/kg
output_space=physical 下，Q 默认会在反标准化后 除以 1000（从训练域的 g/kg 转回 kg/kg）。
如需保持旧行为（g/kg），可用：--q_output_unit gkg。

输入物理约束（z-score 之前，针对训练域 x：PRE 已 log1p、Q 为 g/kg）：
  PRE>=0（log1p 空间）、Q>=0、wind10>=0、2M_RH 裁到 [0,100]、FSDS>=0
  开关：--input_physical_constraints / --no_input_physical_constraints
    - CESM HDF5 / cesm_nc：默认开启（避免 RH>100 等越界输入把温度通道打崩）
    - 测试/训练 HDF5：默认关闭（训练域 RH 已在 [0,100]）

输出物理约束（不包含 TMAX>=TMIN），仅在 output_space=physical 生效：
PRE >= 0
Q >= 0
wind10 >= 0
2M_RH 裁剪到 [0, 100]
FSDS >= 0
不做 TMAX>=TMIN
默认 开启约束；如需关闭：
--no_physical_constraints

诊断：每次推理结束写 out_dir/diagnostics.json（输入越界比例、输出约束命中、
温度顺序违反、min/max/mean/std 与逐日空间分位数均值）。



  CUDA_VISIBLE_DEVICES=1 conda run -n pytorch_downscale python /root/work4/infer.py \
  --ckpt /root/work4/runs/exp01_no_cbam_hr_aux/checkpoints/best.pt \
  --hdf5_root /root/data/hdf5_test \
  --seasons DJF \
  --out_dir /root/data/infer_out_test_DJF_no_cbam_hr_aux \
  --output_mode per_sample \      # 输出模式，per_shard 表示每个输入分片生成一个输出文件（也可设为 per_sample，逐样本输出）
  --output_space physical \      # 输出空间，physical 表示解码成物理量（如实际单位），"zscore" 表示输出标准分，"denorm" 表示还原但不变单位
  --output_dtype float16 \       # 输出的数据类型，float16 为半精度浮点数，节省空间
  --compression lzf \            # HDF5 压缩方式，lzf 一种轻量无损压缩算法
  --amp_bf16 \                   # 启用 bfloat16 自动混合精度推理（如支持可提升速度与显存效率）
  --auto_model_cfg               # 自动从 checkpoint 跟踪的 config 恢复网络结构参数（无需手动输入 base_ch、resblocks 等）


nohup python /work/home/nly_2026/work/infer.py \
  --ckpt /work/home/nly_2026/work/runs/exp01_hr/checkpoints/best.pt \
  --hdf5_root /public/share/acqmesjai0/nly/hdf5_cesm_1deg_no_mbc_fp32_sample \
  --out_dir /work/home/nly_2026/work/infer_out_hr_cesm_no_mbc_fp32_sample \
  --output_mode per_sample \
  --output_format nc \
  --output_space physical \
  --no_skip_bad_dates \
  --output_dtype float32 \
  --amp_bf16 \
  --q_output_unit gkg \
  --auto_model_cfg \
  --coord_ref_nc "/public/share/acqmesjai0/nly/cra_1801*3600_nc/obs_20000101.nc" \
  > infer_hr_cesm_no_mbc_fp32.log 2>&1 & echo "作业PID: $!" >> infer_hr_cesm_no_mbc_fp32.log

最好选择q单位为g/kg,否则很小的误差会导致效果很差

  CUDA_VISIBLE_DEVICES=7 nohup conda run -n pytorch_downscale python /root/work4/infer.py \
  --ckpt /root/work4/runs/exp01_no_all/checkpoints/best.pt \
  --hdf5_root /root/data/hdf5_test \
  --seasons DJF SON MAM JJA \
  --q_output_unit gkg \
  --date_start 20200101 --date_end 20240228 \
  --out_dir /root/data/infer_out_test_no_all \
  --output_mode per_sample \
  --output_format nc \
  --output_space physical \
  --amp_bf16 \
  --auto_model_cfg > infer_test_no_all.log 2>&1 & echo "作业PID: $!" >> infer_test_no_all.log

CUDA_VISIBLE_DEVICES=7 nohup conda run -n pytorch_downscale python /root/work4/infer.py \
  --ckpt /root/work4/runs/exp01_no_all/checkpoints/best.pt \
  --hdf5_root /root/data/hdf5_test \
  --seasons DJF \
  --date_start 20200101 --date_end 20240228 \
  --out_dir /root/data/infer_out_test_no_all_no_physical_constraints \
  --no_physical_constraints \
  --output_mode per_sample \
  --output_format nc \
  --output_space physical \
  --amp_bf16 \
  --auto_model_cfg > infer_test_no_all_no_physical_constraints.log 2>&1 & echo "作业PID: $!" >> infer_test_no_all_no_physical_constraints.log


  # 推荐 CESM 推理流程：先离线准备成 HDF5，再走与测试集一致的 HDF5 推理链路
  conda run -n pytorch_downscale python /root/work4/prepare_hdf5_cesm.py \
    --input_format nc \
    --cesm_root /root/data/cesm \
    --out_hdf5_root /root/data/hdf5_cesm \
    --date_start 20000101 --date_end 20000102

  CUDA_VISIBLE_DEVICES=0 conda run -n pytorch_downscale python /root/work4/infer.py \
    --input_source hdf5 \
    --hdf5_root /root/data/hdf5_cesm_com \
    --seasons DJF MAM JJA SON\
    --ckpt /root/work4/runs/exp01_no_all/checkpoints/best.pt \
    --out_dir /root/data/infer_out_cesm_com \
    --no_skip_bad_dates \
    --output_mode per_sample \
    --output_format nc \
    --output_space physical \
    --amp_bf16 \
    --auto_model_cfg

物理约束：
 CUDA_VISIBLE_DEVICES=7 nohup conda run -n pytorch_downscale python /root/work4/infer.py \
  --input_source hdf5 \
  --hdf5_root /root/data/hdf5_cesm \
  --seasons DJF MAM JJA SON \
  --ckpt /root/work4/runs/exp01_no_all/checkpoints/best.pt \
  --out_dir /root/data/infer_out_cesm \
  --no_skip_bad_dates \
  --output_mode per_sample \
  --output_format nc \
  --output_space physical \
  --amp_bf16 \
  --auto_model_cfg > infer_cesm_all.log 2>&1 & echo "PID: $!" >> infer_cesm_all.log

 CUDA_VISIBLE_DEVICES=7 nohup conda run -n pytorch_downscale python /root/work4/infer.py \
  --input_source hdf5 \
  --hdf5_root /root/data/hdf5_cesm_no_mbc \
  --seasons DJF \
  --ckpt /root/work4/runs/exp01_no_all/checkpoints/best.pt \
  --out_dir /root/data/infer_out_cesm_no_mbc \
  --no_skip_bad_dates \
  --output_mode per_sample \
  --output_format nc \
  --output_space physical \
  --amp_bf16 \
  --auto_model_cfg > infer_cesm_all.log 2>&1 & echo "PID: $!" >> infer_cesm_all.log
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import h5py
import netCDF4
import numpy as np
import torch

import dataset as ds
from model import PixelShuffleDownscaleNet
from paths import CESM_ROOT, EXAMPLE_NC, HDF5_ROOT, STATIC_DIR, STATS_FILE


OutputSpace = Literal["zscore", "denorm", "physical"]
OutputMode = Literal["per_shard", "per_sample"]
OutputFormat = Literal["h5", "nc"]
InputSource = Literal["hdf5", "cesm_nc"]
LonConvention = Literal["neg180_180", "pos0_360"]
QOutputUnit = Literal["kgkg", "gkg"]


# 已知源数据异常日期（见文件头说明）。测试集 HDF5 默认剔除；CESM HDF5 默认保留。
KNOWN_BAD_TEST_DATES: tuple[str, ...] = (
    "20240903",
    "20240926",
    "20240927",
    "20241001",
    "20241004",
)


VARIABLE_UNITS: dict[str, str] = {
    "TAS": "K",
    "PRE": "mm/day",
    "wind10": "m/s",
    "Q": "kg/kg",  # physical output default; see --q_output_unit
    "2M_RH": "%",
    "2M_TMAX": "K",
    "2M_TMIN": "K",
    "FSDS": "W/m^2",
}


@dataclass(frozen=True)
class ModelConfig:
    base_ch: int
    num_resblocks: int
    use_cbam: bool
    hr_aux_mode: str


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference for PixelShuffleDownscaleNet")

    # inputs
    p.add_argument("--ckpt", required=True, type=str, help="path to checkpoint .pt (e.g. runs/.../checkpoints/best.pt)")
    p.add_argument(
        "--input_source",
        default="hdf5",
        choices=["hdf5", "cesm_nc"],
        help="input source type. Recommended: hdf5 shards aligned with train/test format; cesm_nc is experimental only.",
    )
    p.add_argument("--hdf5_root", default=str(HDF5_ROOT), type=str, help="input root containing season subfolders")
    p.add_argument("--seasons", nargs="+", default=["MAM", "JJA", "SON", "DJF"])
    p.add_argument("--cesm_root", default=str(CESM_ROOT), type=str, help="official CESM directory (contains cesm_1deg_YYYYMMDD.nc)")
    p.add_argument("--cesm_glob", default="cesm_1deg_*.nc", type=str, help="glob pattern under cesm_root (default: cesm_1deg_*.nc)")
    p.add_argument("--cesm_lat_var", default="lat", type=str, help="latitude variable name in CESM nc (default: lat)")
    p.add_argument("--cesm_lon_var", default="lon", type=str, help="longitude variable name in CESM nc (default: lon)")
    p.add_argument("--cesm_pr_var", default="pr", type=str, help="precip variable name in CESM nc (default: pr)")
    p.add_argument("--cesm_q_var", default="q", type=str, help="specific humidity variable name in CESM nc (default: q)")
    p.add_argument("--cesm_t2m_var", default="t2m", type=str, help="2m temperature variable name in CESM nc (default: t2m)")
    p.add_argument("--cesm_tmax_var", default="tmax", type=str, help="2m tmax variable name in CESM nc (default: tmax)")
    p.add_argument("--cesm_tmin_var", default="tmin", type=str, help="2m tmin variable name in CESM nc (default: tmin)")
    p.add_argument("--cesm_wind10_var", default="wind10", type=str, help="10m wind variable name in CESM nc (default: wind10)")
    p.add_argument("--cesm_rhmin_var", default="rhmin", type=str, help="2m RH variable name in CESM nc (default: rhmin)")
    p.add_argument("--cesm_fsds_var", default="fsds", type=str, help="downwelling shortwave variable name in CESM nc (default: fsds)")
    p.add_argument(
        "--no_cesm_regrid",
        action="store_true",
        help="disable regridding CESM (lat/lon) to the static LR grid; only use if CESM files already match static lat_lr/lon_lr",
    )
    p.add_argument(
        "--cesm_pr_transform",
        default="log1p",
        choices=["log1p", "none"],
        help="transform to apply to CESM precipitation before z-score (default: log1p to match HDF5 preprocessing)",
    )
    p.add_argument(
        "--cesm_q_scale",
        default=1000.0,
        type=float,
        help="scale factor applied to CESM q before z-score (default: 1000 to convert kg/kg -> g/kg)",
    )
    p.add_argument("--static_dir", default=str(STATIC_DIR), type=str)
    p.add_argument("--stats_file", default=str(STATS_FILE), type=str)
    p.add_argument("--date_start", default=None, type=str, help="inclusive YYYYMMDD; optional")
    p.add_argument("--date_end", default=None, type=str, help="inclusive YYYYMMDD; optional")
    p.add_argument(
        "--skip_bad_dates",
        dest="skip_bad_dates",
        action="store_true",
        default=None,
        help=(
            "skip known-bad hdf5_test dates "
            f"({','.join(KNOWN_BAD_TEST_DATES)}). "
            "Default: on for test HDF5, off for CESM HDF5."
        ),
    )
    p.add_argument(
        "--no_skip_bad_dates",
        dest="skip_bad_dates",
        action="store_false",
        help="do not skip known-bad hdf5_test dates (recommended / default for CESM HDF5).",
    )
    p.add_argument(
        "--exclude_dates",
        default=None,
        type=str,
        help=(
            "comma-separated extra YYYYMMDD dates to skip during hdf5 inference. "
            "Known-bad hdf5_test dates are controlled by --skip_bad_dates/--no_skip_bad_dates. "
            "Pass an empty string to disable extra exclusions (also disables known-bad skip "
            "unless --skip_bad_dates is set)."
        ),
    )
    p.add_argument(
        "--coord_ref_nc",
        default=None,
        type=str,
        help="optional reference NetCDF file to source lat/lon coordinates (default: paths.EXAMPLE_NC)",
    )
    p.add_argument("--coord_lat_var", default="lat", type=str, help="latitude variable name in --coord_ref_nc (default: lat)")
    p.add_argument("--coord_lon_var", default="lon", type=str, help="longitude variable name in --coord_ref_nc (default: lon)")

    # output
    p.add_argument("--out_dir", required=True, type=str, help="output directory")
    p.add_argument("--output_mode", default="per_shard", choices=["per_shard", "per_sample"])
    p.add_argument("--output_format", default="h5", choices=["h5", "nc"], help="output file format for predictions")
    p.add_argument(
        "--lon_convention",
        default="neg180_180",
        choices=["neg180_180", "pos0_360"],
        help=(
            "Longitude axis convention for NetCDF outputs. "
            "neg180_180 (default) keeps the native column ordering (verified "
            "against real training HDF5 data) and writes lon -180..180; "
            "pos0_360 rolls by W//2 and writes lon 0..360."
        ),
    )
    p.add_argument("--output_space", default="physical", choices=["zscore", "denorm", "physical"])
    p.add_argument("--output_dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--compression", default="lzf", choices=["none", "lzf", "gzip"])
    p.add_argument("--gzip_level", default=1, type=int, help="only used when --compression=gzip")
    p.add_argument("--overwrite", action="store_true", help="overwrite existing outputs")
    p.add_argument("--nc_zlib_level", default=1, type=int, help="NetCDF zlib compression level (0-9)")
    p.add_argument(
        "--q_output_unit",
        default="kgkg",
        choices=["kgkg", "gkg"],
        help=(
            "specific humidity unit for output_space=physical. "
            "Training/stats use g/kg (Q multiplied by 1000). "
            "kgkg divides by 1000 after denorm; gkg keeps the training unit."
        ),
    )
    p.add_argument(
        "--enforce_physical_constraints",
        action="store_true",
        help=(
            "apply simple physical constraints in output_space=physical: "
            "PRE>=0, Q>=0, wind10>=0, 2M_RH in [0,100], FSDS>=0. "
            "Does NOT enforce TMAX>=TMIN."
        ),
    )
    p.add_argument(
        "--no_physical_constraints",
        dest="enforce_physical_constraints",
        action="store_false",
        help="disable physical constraints for output_space=physical",
    )
    p.set_defaults(enforce_physical_constraints=True)
    p.add_argument(
        "--input_physical_constraints",
        dest="input_physical_constraints",
        action="store_true",
        default=None,
        help=(
            "clip LR inputs before z-score: PRE>=0 (log1p space), Q>=0, wind10>=0, "
            "2M_RH in [0,100], FSDS>=0. Default: on for CESM, off for hdf5_test."
        ),
    )
    p.add_argument(
        "--no_input_physical_constraints",
        dest="input_physical_constraints",
        action="store_false",
        help="do not clip LR inputs before z-score (not recommended for CESM).",
    )

    # runtime
    p.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="torch device string, e.g. 'cuda', 'cuda:1', or 'cpu' (default: cuda)",
    )
    p.add_argument("--amp_bf16", action="store_true", help="use bf16 autocast on cuda (recommended)")
    p.add_argument("--batch_size", default=1, type=int, help="micro-batch per forward (keep 1 unless you have lots of VRAM)")
    p.add_argument("--max_samples", default=None, type=int, help="debug: stop after N samples (after filtering)")
    p.add_argument(
        "--allow_experimental_cesm_nc",
        action="store_true",
        help=(
            "allow input_source=cesm_nc. "
            "For stable production inference, prefer nc/npy -> prepare_hdf5_cesm.py -> input_source=hdf5."
        ),
    )

    # model cfg (MUST match the training run)
    p.add_argument(
        "--auto_model_cfg",
        action="store_true",
        help=(
            "Infer model architecture (base_ch/cbam/hr_aux_mode/resblocks) from checkpoint. "
            "Recommended to avoid mismatches across ablations."
        ),
    )
    p.add_argument("--base_ch", default=256, type=int)
    p.add_argument("--num_resblocks", default=2, type=int)
    p.add_argument("--no_cbam", dest="use_cbam", action="store_false")
    p.set_defaults(use_cbam=True)
    p.add_argument("--hr_aux_mode", default="all", choices=["all", "stage1", "none"])

    return p.parse_args()


def _normalize_date_yyyymmdd(s: str) -> str:
    # Keep digits only, so '2020-01-02' -> '20200102'.
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits if len(digits) >= 8 else s


def _date_in_range(date_yyyymmdd: str, start: str | None, end: str | None) -> bool:
    # YYYYMMDD lexicographic order matches chronological order.
    date_yyyymmdd = _normalize_date_yyyymmdd(date_yyyymmdd)[:8]
    start = _normalize_date_yyyymmdd(start)[:8] if start is not None else None
    end = _normalize_date_yyyymmdd(end)[:8] if end is not None else None
    if start is not None and date_yyyymmdd < start:
        return False
    if end is not None and date_yyyymmdd > end:
        return False
    return True


def _as_date_str(x: bytes | np.bytes_ | str) -> str:
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode()
    return str(x)


def _list_shards(hdf5_root: Path, seasons: Iterable[str]) -> list[Path]:
    shards: list[Path] = []
    for season in seasons:
        d = hdf5_root / season
        if not d.exists():
            continue
        shards.extend(sorted(d.glob("shard_*.h5")))
    return shards


def _attr_to_py(v: object) -> object:
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    return v


def _validate_hdf5_shard_contract(
    h5path: Path,
    expected_lat_lr: np.ndarray,
    expected_lon_lr: np.ndarray,
) -> dict[str, object]:
    """
    Validate one shard against the stable HDF5 inference contract.

    Hard checks:
      - data/x and data/dates must exist
      - data/x shape must be (N,8,180,360)
      - data/dates shape must be (N,)
      - if metadata/lr_grid_lats,lons exist, they must either:
          * match static lat_lr/lon_lr, or
          * explicitly declare coarse-native semantics
    """
    info: dict[str, object] = {
        "path": str(h5path),
        "lr_grid_checked": False,
        "metadata_present": False,
        "lr_grid_semantics": "unknown",
    }
    with h5py.File(h5path, "r") as f:
        if "data/x" not in f or "data/dates" not in f:
            raise ValueError(f"{h5path}: expected datasets data/x and data/dates")
        x_ds = f["data/x"]
        dates_ds = f["data/dates"]
        if x_ds.ndim != 4:
            raise ValueError(f"{h5path}: data/x must be 4D, got shape={x_ds.shape}")
        if x_ds.shape[1:] != (8, 180, 360):
            raise ValueError(
                f"{h5path}: data/x tail must be (8,180,360), got {x_ds.shape[1:]}. "
                "Please rebuild CESM HDF5 with prepare_hdf5_cesm.py."
            )
        if dates_ds.ndim != 1:
            raise ValueError(f"{h5path}: data/dates must be 1D, got shape={dates_ds.shape}")
        if int(dates_ds.shape[0]) != int(x_ds.shape[0]):
            raise ValueError(f"{h5path}: data/dates length {dates_ds.shape[0]} != data/x N {x_ds.shape[0]}")

        info["x_shape"] = tuple(int(x) for x in x_ds.shape)
        info["has_y"] = ("data/y" in f)

        if "metadata" in f and isinstance(f["metadata"], h5py.Group):
            info["metadata_present"] = True
            g_meta = f["metadata"]
            if "lr_grid_semantics" in g_meta.attrs:
                info["lr_grid_semantics"] = str(_attr_to_py(g_meta.attrs["lr_grid_semantics"]))
            if "lr_lon_convention" in g_meta.attrs:
                info["lr_lon_convention"] = str(_attr_to_py(g_meta.attrs["lr_lon_convention"]))
            if "lr_grid_lats" in g_meta and "lr_grid_lons" in g_meta:
                lat = np.array(g_meta["lr_grid_lats"][:], dtype=np.float32)
                lon = np.array(g_meta["lr_grid_lons"][:], dtype=np.float32)
                if lat.shape != expected_lat_lr.shape or lon.shape != expected_lon_lr.shape:
                    raise ValueError(
                        f"{h5path}: metadata LR grid shape mismatch. "
                        f"got lat{lat.shape}/lon{lon.shape}, expected lat{expected_lat_lr.shape}/lon{expected_lon_lr.shape}"
                    )
                lat_ok = np.allclose(lat, expected_lat_lr, atol=1e-5)
                lon_ok = np.allclose(lon, expected_lon_lr, atol=1e-5)
                if lat_ok and lon_ok:
                    info["lr_grid_checked"] = True
                elif info["lr_grid_semantics"] == "coarse_native":
                    # Intentional: preserve coarse/native 180x360 ordering rather than
                    # forcing metadata to mimic the static training LR grid.
                    info["lr_grid_checked"] = False
                    info["native_lr_grid_first_last"] = {
                        "lat_first": float(lat[0]),
                        "lat_last": float(lat[-1]),
                        "lon_first": float(lon[0]),
                        "lon_last": float(lon[-1]),
                    }
                else:
                    raise ValueError(
                        f"{h5path}: metadata LR grid does not match static lat_lr/lon_lr. "
                        "This can cause geo mismatch; rebuild CESM HDF5 with the correct static_dir "
                        "or mark it as coarse_native semantics."
                    )
            if "source" in g_meta.attrs:
                info["source"] = str(_attr_to_py(g_meta.attrs["source"]))
            if "pre_transform" in g_meta.attrs:
                info["pre_transform"] = _attr_to_py(g_meta.attrs["pre_transform"])
            if "q_scale" in g_meta.attrs:
                info["q_scale"] = float(g_meta.attrs["q_scale"])
            if "regrid_to_training_lr" in g_meta.attrs:
                info["regrid_to_training_lr"] = bool(g_meta.attrs["regrid_to_training_lr"])
    return info


def _parse_exclude_date_list(s: str | None) -> set[str]:
    if not s:
        return set()
    return {_normalize_date_yyyymmdd(d)[:8] for d in s.split(",") if d.strip()}


def _hdf5_root_looks_like_cesm(hdf5_root: Path) -> bool:
    return "cesm" in str(hdf5_root).lower()


def _shards_look_like_cesm(shard_infos: Iterable[dict[str, object]]) -> bool:
    for info in shard_infos:
        if str(info.get("source", "")).lower() == "cesm":
            return True
    return False


def _resolve_skip_bad_dates(
    skip_bad_dates_arg: bool | None,
    exclude_dates_arg: str | None,
    is_cesm_hdf5: bool,
) -> tuple[bool, str]:
    """Decide whether to skip KNOWN_BAD_TEST_DATES. Returns (enabled, reason)."""
    if skip_bad_dates_arg is True:
        return True, "flag --skip_bad_dates"
    if skip_bad_dates_arg is False:
        return False, "flag --no_skip_bad_dates"
    if exclude_dates_arg is not None and exclude_dates_arg.strip() == "":
        return False, "--exclude_dates empty"
    if is_cesm_hdf5:
        return False, "cesm hdf5 auto"
    return True, "hdf5_test auto"


def _build_exclude_dates_set(extra: set[str], skip_known_bad: bool) -> set[str]:
    out = set(extra)
    if skip_known_bad:
        out.update(_normalize_date_yyyymmdd(d)[:8] for d in KNOWN_BAD_TEST_DATES)
    return out


def _resolve_input_physical_constraints(
    arg: bool | None,
    is_cesm: bool,
) -> tuple[bool, str]:
    """CESM inputs may violate training-domain bounds (especially RH>100)."""
    if arg is True:
        return True, "flag --input_physical_constraints"
    if arg is False:
        return False, "flag --no_input_physical_constraints"
    if is_cesm:
        return True, "cesm auto"
    return False, "hdf5_test auto"


def _collect_indices_for_shard(
    h5path: Path,
    date_start: str | None,
    date_end: str | None,
    exclude_dates: set[str] | None = None,
) -> tuple[list[int], list[str], bool, int]:
    """Return (keep_i, keep_dates, has_y, n_excluded).

    n_excluded counts samples dropped solely because their date is in
    exclude_dates (used to report/verify known-bad-date filtering).
    """
    with h5py.File(h5path, "r") as f:
        dates = f["data/dates"][:]
        has_y = "data/y" in f

    exclude_dates = exclude_dates or set()
    keep_i: list[int] = []
    keep_dates: list[str] = []
    n_excluded = 0
    for i, d in enumerate(dates):
        ds_ = _as_date_str(d)[:8]
        if not _date_in_range(ds_, date_start, date_end):
            continue
        if ds_ in exclude_dates:
            n_excluded += 1
            continue
        keep_i.append(i)
        keep_dates.append(ds_)
    return keep_i, keep_dates, has_y, n_excluded


def _parse_date_from_filename(path: Path) -> str:
    m = re.search(r"(\d{8})", path.name)
    if not m:
        raise ValueError(f"Could not parse YYYYMMDD from filename: {path.name}")
    return m.group(1)


def _list_cesm_files(cesm_root: Path, pattern: str) -> list[Path]:
    files = sorted(cesm_root.glob(pattern))
    return files


def _squeeze_time(arr: np.ndarray) -> np.ndarray:
    # Accept (lat,lon) or (time,lat,lon) with time=1.
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    raise ValueError(f"Expected 2D or (1,H,W) array, got shape {arr.shape}")


def _regrid_to_static_lr(
    field: np.ndarray,  # (H, W) on CESM grid
    src_lat: np.ndarray,  # (H,)
    src_lon: np.ndarray,  # (W,) expected 0..360
    tgt_lat: np.ndarray,  # (180,) from static lat_lr.npy
    tgt_lon: np.ndarray,  # (360,) from static lon_lr.npy (likely -179.5..179.5)
) -> np.ndarray:
    """
    Linear regrid from CESM regular lat/lon to the static LR grid used in training.

    Why needed:
      - CESM files often use lon in [0,360) and lat endpoints [-90,90],
        while training static grid uses centers lat=-89.5..89.5 and lon=-179.5..179.5.
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

    # Ensure ascending latitude for np.interp
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        field = field[::-1, :]

    # ---- lon interpolation with wrap handling ----
    if not np.all(np.diff(src_lon) > 0):
        raise ValueError("src_lon must be strictly increasing for interpolation")

    src_lon_ext = np.concatenate([src_lon, src_lon + 360.0], axis=0)  # (2W,)
    field_ext = np.concatenate([field, field], axis=1)  # (H, 2W)

    tgt_lon_mod = np.mod(tgt_lon, 360.0)  # map [-180,180) -> [0,360)
    tgt_lon_ext = tgt_lon_mod.copy()
    base = float(tgt_lon_ext[0])
    tgt_lon_ext[tgt_lon_ext < base] += 360.0  # make increasing

    tmp = np.empty((field.shape[0], tgt_lon_ext.shape[0]), dtype=np.float64)
    for i in range(field.shape[0]):
        tmp[i, :] = np.interp(tgt_lon_ext, src_lon_ext, field_ext[i, :])

    # ---- lat interpolation ----
    out = np.empty((tgt_lat.shape[0], tmp.shape[1]), dtype=np.float64)
    for j in range(tmp.shape[1]):
        out[:, j] = np.interp(tgt_lat, src_lat, tmp[:, j])

    return out.astype(np.float32)


def _read_cesm_x_from_nc(
    nc_path: Path,
    *,
    lat_var: str,
    lon_var: str,
    var_map: dict[str, str],
    tgt_lat: np.ndarray,
    tgt_lon: np.ndarray,
    do_regrid: bool,
    pr_transform: str,
    q_scale: float,
) -> tuple[str, np.ndarray]:
    """
    Read a CESM daily NetCDF file and return (date_str, x_raw_preprocessed).

    Output x_raw_preprocessed matches training preprocessing BEFORE z-score:
      - PRE: log1p if pr_transform='log1p'
      - Q:   scaled to g/kg by default (q_scale=1000)
    Shape: (8, 180, 360) float32 on the static LR grid.
    """
    date_str = _parse_date_from_filename(nc_path)
    with netCDF4.Dataset(str(nc_path), "r") as f:
        if lat_var not in f.variables or lon_var not in f.variables:
            raise KeyError(f"Missing lat/lon vars in {nc_path.name}: lat={lat_var!r} lon={lon_var!r}")
        src_lat = np.array(f.variables[lat_var][:], dtype=np.float32)
        src_lon = np.array(f.variables[lon_var][:], dtype=np.float32)

        vals: dict[str, np.ndarray] = {}
        for out_name, in_name in var_map.items():
            if in_name not in f.variables:
                raise KeyError(f"Missing variable {in_name!r} in {nc_path.name} (needed for {out_name})")
            a = np.array(f.variables[in_name][:], dtype=np.float32)
            a2 = _squeeze_time(a)
            if do_regrid:
                a2 = _regrid_to_static_lr(a2, src_lat, src_lon, tgt_lat, tgt_lon)
            vals[out_name] = a2.astype(np.float32, copy=False)

    # apply preprocessing to match HDF5 convention
    pre = vals["PRE"]
    if pr_transform == "log1p":
        pre = np.log1p(np.clip(pre, 0.0, None)).astype(np.float32)
    vals["PRE"] = pre

    q = vals["Q"].astype(np.float32) * float(q_scale)
    vals["Q"] = q

    x = np.stack(
        [
            vals["TAS"],
            vals["PRE"],
            vals["wind10"],
            vals["Q"],
            vals["2M_RH"],
            vals["2M_TMAX"],
            vals["2M_TMIN"],
            vals["FSDS"],
        ],
        axis=0,
    ).astype(np.float32)

    if x.shape != (8, int(tgt_lat.shape[0]), int(tgt_lon.shape[0])):
        raise ValueError(f"Unexpected CESM x shape {x.shape}, expected (8,{tgt_lat.shape[0]},{tgt_lon.shape[0]})")
    return date_str, x


def _torch_load_weights(ckpt_path: Path, device: torch.device) -> object:
    # Prefer weights_only=True when available to reduce pickle surface.
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


def _load_ckpt_state_dict(ckpt_path: Path, device: torch.device) -> dict:
    ckpt = _torch_load_weights(ckpt_path, device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict):
        # tolerate "pure state_dict" checkpoints
        return ckpt
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def _infer_base_ch(sd: dict) -> int:
    # init_conv is always 1×1: init_conv.0.weight = (base_ch, in_ch, 1, 1)
    w0 = sd.get("init_conv.0.weight", None)
    if w0 is None:
        raise KeyError("init_conv.0.weight not found in checkpoint state_dict")
    shape0 = tuple(int(x) for x in w0.shape)
    if len(shape0) != 4:
        raise ValueError(f"Unexpected init_conv.0.weight shape: {shape0}")
    return shape0[0]  # out_channels = base_ch


def _infer_num_resblocks(sd: dict) -> int:
    # stage1.res_blocks is a nn.Sequential of ResBlock; indices are 0..N-1
    # We count how many "stage1.res_blocks.{i}.block.0.weight" keys exist.
    n = 0
    while f"stage1.res_blocks.{n}.block.0.weight" in sd:
        n += 1
    return max(1, n)


def _infer_use_cbam(sd: dict) -> bool:
    return any(".cbam." in k for k in sd.keys())


def _infer_hr_aux_mode(sd: dict, base_ch: int) -> str:
    # Determine if head concatenates hr_aux by looking at head.0.weight in_channels.
    w_head0 = sd.get("head.0.weight", None)
    head_uses_aux = False
    if w_head0 is not None and len(w_head0.shape) == 4:
        in_ch = int(w_head0.shape[1])
        head_uses_aux = (in_ch == base_ch + 7)

    has_s1_inject = "stage1.inject_conv.weight" in sd
    has_s2_inject = "stage2.inject_conv.weight" in sd
    has_s3_inject = "stage3.inject_conv.weight" in sd

    if head_uses_aux or has_s2_inject or has_s3_inject:
        return "all"
    if has_s1_inject:
        return "stage1"
    return "none"


def _infer_model_cfg_from_state_dict(sd: dict) -> ModelConfig:
    base_ch = _infer_base_ch(sd)
    num_resblocks = _infer_num_resblocks(sd)
    use_cbam = _infer_use_cbam(sd)
    hr_aux_mode = _infer_hr_aux_mode(sd, base_ch)
    return ModelConfig(
        base_ch=base_ch,
        num_resblocks=num_resblocks,
        use_cbam=use_cbam,
        hr_aux_mode=hr_aux_mode,
    )


def _build_model(cfg: ModelConfig, device: torch.device) -> torch.nn.Module:
    model = PixelShuffleDownscaleNet(
        in_ch=8,
        hr_aux_ch=7,
        base_ch=cfg.base_ch,
        num_resblocks=cfg.num_resblocks,
        out_ch=8,
        target_h=1801,
        target_w=3600,
        use_cbam=cfg.use_cbam,
        hr_aux_mode=cfg.hr_aux_mode,
        use_checkpoint=False,          # 推理阶段不需要 gradient checkpointing
        stage4_shuffle_conv_k=1,       # 与训练保持一致，加载权重时形状匹配
    ).to(device)
    model.eval()
    return model


def _prep_norm_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean_np, std_np = ds._load_norm_stats()  # (8,)
    mean = torch.from_numpy(mean_np).to(device=device, dtype=torch.float32)[None, :, None, None]
    std = torch.from_numpy(std_np).to(device=device, dtype=torch.float32)[None, :, None, None]
    return mean, std


def _prep_static_tensors(device: torch.device) -> torch.Tensor:
    """加载 HR 静态特征到 GPU（LR 静态已不再拼入模型输入）。"""
    hr_static = ds._load_hr_static()  # (6,1801,3600)
    hr_t = torch.from_numpy(hr_static).to(device=device, dtype=torch.float32)[None, ...]  # (1,6,1801,3600)
    return hr_t


def _to_output_space(
    pred_z: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    space: OutputSpace,
    q_output_unit: QOutputUnit,
    enforce_constraints: bool,
) -> torch.Tensor:
    if space == "zscore":
        return pred_z
    pred_dn = pred_z.float() * std + mean
    if space == "denorm":
        return pred_dn

    # physical:
    #   - PRE is log1p in data -> expm1 inverse
    #   - Q is stored as g/kg (mul1000) in training/stats -> optionally convert to kg/kg
    # Variables order: ["TAS", "PRE", "wind10", "Q", "2M_RH", "2M_TMAX", "2M_TMIN", "FSDS"]
    pred_phy = pred_dn.clone()

    # PRE inverse-transform
    pred_phy[:, 1:2, :, :] = torch.expm1(pred_phy[:, 1:2, :, :])

    # Q unit conversion
    if q_output_unit == "kgkg":
        pred_phy[:, 3:4, :, :] = pred_phy[:, 3:4, :, :] / 1000.0
    elif q_output_unit == "gkg":
        pass
    else:
        raise ValueError(q_output_unit)

    if enforce_constraints:
        # PRE >= 0
        pred_phy[:, 1:2, :, :] = pred_phy[:, 1:2, :, :].clamp_min(0.0)
        # wind10 >= 0
        pred_phy[:, 2:3, :, :] = pred_phy[:, 2:3, :, :].clamp_min(0.0)
        # Q >= 0
        pred_phy[:, 3:4, :, :] = pred_phy[:, 3:4, :, :].clamp_min(0.0)
        # RH in [0,100]
        pred_phy[:, 4:5, :, :] = pred_phy[:, 4:5, :, :].clamp(0.0, 100.0)
        # FSDS >= 0
        pred_phy[:, 7:8, :, :] = pred_phy[:, 7:8, :, :].clamp_min(0.0)

    return pred_phy


# Channel indices: TAS PRE wind10 Q 2M_RH 2M_TMAX 2M_TMIN FSDS
_CH_TAS, _CH_PRE, _CH_WIND, _CH_Q, _CH_RH, _CH_TMAX, _CH_TMIN, _CH_FSDS = range(8)
_INPUT_CLIP_KEYS = ("PRE_lt_0", "wind10_lt_0", "Q_lt_0", "2M_RH_lt_0", "2M_RH_gt_100", "FSDS_lt_0")
_OUTPUT_HIT_KEYS = ("PRE_lt_0", "wind10_lt_0", "Q_lt_0", "2M_RH_lt_0", "2M_RH_gt_100", "FSDS_lt_0")
_TEMP_ORDER_KEYS = (
    "TMAX_lt_TMIN",
    "TAS_gt_TMAX",
    "TAS_lt_TMIN",
    "TAS_lt_180K",
    "TMAX_lt_180K",
    "TMIN_lt_180K",
    "TAS_gt_340K",
)


def _input_violation_counts(x: np.ndarray) -> dict[str, int]:
    """Count training-domain bound violations. x: (8,H,W), PRE=log1p, Q=g/kg."""
    return {
        "PRE_lt_0": int(np.count_nonzero(x[_CH_PRE] < 0.0)),
        "wind10_lt_0": int(np.count_nonzero(x[_CH_WIND] < 0.0)),
        "Q_lt_0": int(np.count_nonzero(x[_CH_Q] < 0.0)),
        "2M_RH_lt_0": int(np.count_nonzero(x[_CH_RH] < 0.0)),
        "2M_RH_gt_100": int(np.count_nonzero(x[_CH_RH] > 100.0)),
        "FSDS_lt_0": int(np.count_nonzero(x[_CH_FSDS] < 0.0)),
        "n_pixels": int(x.shape[-2] * x.shape[-1]),
    }


def _apply_input_physical_constraints_np(x: np.ndarray) -> np.ndarray:
    """Clip LR inputs in training domain. Returns a copy."""
    y = np.array(x, dtype=np.float32, copy=True)
    y[_CH_PRE] = np.clip(y[_CH_PRE], 0.0, None)
    y[_CH_WIND] = np.clip(y[_CH_WIND], 0.0, None)
    y[_CH_Q] = np.clip(y[_CH_Q], 0.0, None)
    y[_CH_RH] = np.clip(y[_CH_RH], 0.0, 100.0)
    y[_CH_FSDS] = np.clip(y[_CH_FSDS], 0.0, None)
    return y


def _output_violation_counts(phy: np.ndarray) -> dict[str, int]:
    """Count physical-space bound violations before output clamp. phy: (8,H,W)."""
    return {
        "PRE_lt_0": int(np.count_nonzero(phy[_CH_PRE] < 0.0)),
        "wind10_lt_0": int(np.count_nonzero(phy[_CH_WIND] < 0.0)),
        "Q_lt_0": int(np.count_nonzero(phy[_CH_Q] < 0.0)),
        "2M_RH_lt_0": int(np.count_nonzero(phy[_CH_RH] < 0.0)),
        "2M_RH_gt_100": int(np.count_nonzero(phy[_CH_RH] > 100.0)),
        "FSDS_lt_0": int(np.count_nonzero(phy[_CH_FSDS] < 0.0)),
        "n_pixels": int(phy.shape[-2] * phy.shape[-1]),
    }


def _apply_output_physical_constraints_np(phy: np.ndarray) -> np.ndarray:
    y = np.array(phy, dtype=np.float32, copy=True)
    y[_CH_PRE] = np.clip(y[_CH_PRE], 0.0, None)
    y[_CH_WIND] = np.clip(y[_CH_WIND], 0.0, None)
    y[_CH_Q] = np.clip(y[_CH_Q], 0.0, None)
    y[_CH_RH] = np.clip(y[_CH_RH], 0.0, 100.0)
    y[_CH_FSDS] = np.clip(y[_CH_FSDS], 0.0, None)
    return y


def _temp_order_counts(phy: np.ndarray) -> dict[str, int]:
    tas = phy[_CH_TAS]
    tmax = phy[_CH_TMAX]
    tmin = phy[_CH_TMIN]
    ok = np.isfinite(tas) & np.isfinite(tmax) & np.isfinite(tmin)
    n_ok = int(ok.sum())
    return {
        "n_finite_triplets": n_ok,
        "TMAX_lt_TMIN": int(np.count_nonzero((tmax < tmin) & ok)),
        "TAS_gt_TMAX": int(np.count_nonzero((tas > tmax) & ok)),
        "TAS_lt_TMIN": int(np.count_nonzero((tas < tmin) & ok)),
        "TAS_lt_180K": int(np.count_nonzero((tas < 180.0) & ok)),
        "TMAX_lt_180K": int(np.count_nonzero((tmax < 180.0) & ok)),
        "TMIN_lt_180K": int(np.count_nonzero((tmin < 180.0) & ok)),
        "TAS_gt_340K": int(np.count_nonzero((tas > 340.0) & ok)),
    }


def _update_field_stats(store: dict[str, dict[str, float]], chw: np.ndarray, variables: list[str]) -> None:
    """Accumulate min/max/sum/sumsq and mean of per-sample spatial quantiles."""
    for i, name in enumerate(variables):
        a = np.asarray(chw[i], dtype=np.float64)
        finite = np.isfinite(a)
        s = store[name]
        s["n_nonfinite"] += float((~finite).sum())
        z = a[finite]
        if z.size == 0:
            continue
        s["n"] += float(z.size)
        s["sum"] += float(z.sum())
        s["sumsq"] += float(np.square(z).sum())
        zmin = float(z.min())
        zmax = float(z.max())
        if zmin < s["min"]:
            s["min"] = zmin
        if zmax > s["max"]:
            s["max"] = zmax
        stride = max(1, z.size // 200000)
        samp = z.reshape(-1)[::stride]
        q01, q50, q99 = np.quantile(samp, [0.01, 0.50, 0.99])
        s["q01_sum"] += float(q01)
        s["q50_sum"] += float(q50)
        s["q99_sum"] += float(q99)
        s["n_samples"] += 1.0


def _empty_field_store(variables: list[str]) -> dict[str, dict[str, float]]:
    return {
        v: {
            "n": 0.0,
            "sum": 0.0,
            "sumsq": 0.0,
            "min": float("inf"),
            "max": float("-inf"),
            "n_nonfinite": 0.0,
            "q01_sum": 0.0,
            "q50_sum": 0.0,
            "q99_sum": 0.0,
            "n_samples": 0.0,
        }
        for v in variables
    }


def _json_safe(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        return _json_safe(obj.item())
    return obj


def _finalize_field_store(store: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, s in store.items():
        n = s["n"]
        ns = s["n_samples"]
        mean = (s["sum"] / n) if n else float("nan")
        var = (s["sumsq"] / n - mean * mean) if n else float("nan")
        std = float(var ** 0.5) if n and var > 0 else (0.0 if n else float("nan"))
        out[name] = {
            "n_pixels": int(n),
            "n_nonfinite": int(s["n_nonfinite"]),
            "min": None if s["min"] == float("inf") else s["min"],
            "max": None if s["max"] == float("-inf") else s["max"],
            "mean": mean,
            "std": std,
            "daily_spatial_q01_mean": (s["q01_sum"] / ns) if ns else float("nan"),
            "daily_spatial_q50_mean": (s["q50_sum"] / ns) if ns else float("nan"),
            "daily_spatial_q99_mean": (s["q99_sum"] / ns) if ns else float("nan"),
            "quantile_note": "q01/q50/q99 are means of per-sample spatial quantiles (subsampled)",
        }
    return out


class InferDiagnostics:
    """Streaming, low-memory diagnostics for one inference run."""

    def __init__(self, variables: list[str]) -> None:
        self.variables = list(variables)
        self.n_samples = 0
        self.input_raw = _empty_field_store(self.variables)
        self.output_raw = _empty_field_store(self.variables)
        self.output_constrained = _empty_field_store(self.variables)
        self.input_viol = {k: 0 for k in _INPUT_CLIP_KEYS}
        self.input_pixels = 0
        self.output_viol = {k: 0 for k in _OUTPUT_HIT_KEYS}
        self.output_pixels = 0
        self.temp_viol = {k: 0 for k in _TEMP_ORDER_KEYS}
        self.temp_triplets = 0

    def update_sample(
        self,
        x_raw: np.ndarray,
        phy_raw: np.ndarray,
        phy_constrained: np.ndarray,
        in_viol: dict[str, int],
        out_viol: dict[str, int],
        temp_viol: dict[str, int],
    ) -> None:
        self.n_samples += 1
        _update_field_stats(self.input_raw, x_raw, self.variables)
        _update_field_stats(self.output_raw, phy_raw, self.variables)
        _update_field_stats(self.output_constrained, phy_constrained, self.variables)
        self.input_pixels += int(in_viol["n_pixels"])
        for k in _INPUT_CLIP_KEYS:
            self.input_viol[k] += int(in_viol[k])
        self.output_pixels += int(out_viol["n_pixels"])
        for k in _OUTPUT_HIT_KEYS:
            self.output_viol[k] += int(out_viol[k])
        self.temp_triplets += int(temp_viol["n_finite_triplets"])
        for k in _TEMP_ORDER_KEYS:
            self.temp_viol[k] += int(temp_viol[k])

    def to_dict(self, *, input_clip_applied: bool, output_clip_applied: bool) -> dict[str, object]:
        def _rate(num: int, den: int) -> float:
            return (num / den) if den else float("nan")

        return {
            "n_samples": self.n_samples,
            "input_clip_applied": bool(input_clip_applied),
            "output_clip_applied": bool(output_clip_applied),
            "input_raw": _finalize_field_store(self.input_raw),
            "output_physical_raw": _finalize_field_store(self.output_raw),
            "output_physical_constrained": _finalize_field_store(self.output_constrained),
            "input_violation_counts": dict(self.input_viol),
            "input_violation_rates": {k: _rate(self.input_viol[k], self.input_pixels) for k in _INPUT_CLIP_KEYS},
            "output_violation_counts": dict(self.output_viol),
            "output_violation_rates": {k: _rate(self.output_viol[k], self.output_pixels) for k in _OUTPUT_HIT_KEYS},
            "temperature_order_counts": dict(self.temp_viol),
            "temperature_order_rates": {k: _rate(self.temp_viol[k], self.temp_triplets) for k in _TEMP_ORDER_KEYS},
        }


def _infer_one_sample(
    *,
    model: torch.nn.Module,
    x_raw: np.ndarray,
    dstr: str,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    hr_static_t: torch.Tensor,
    use_amp: bool,
    out_space: OutputSpace,
    q_output_unit: QOutputUnit,
    enforce_out: bool,
    apply_in: bool,
    out_np_dtype: np.dtype,
    diag: InferDiagnostics | None,
) -> tuple[np.ndarray, torch.Tensor]:
    """Forward one sample. Returns (pred_cpu in requested output space, pred_z)."""
    in_viol = _input_violation_counts(x_raw)
    x_use = _apply_input_physical_constraints_np(x_raw) if apply_in else x_raw

    x_t = torch.from_numpy(np.ascontiguousarray(x_use)).to(device=device, dtype=torch.float32)[None, ...]
    x_norm = (x_t - mean) / std
    sza_hr = torch.from_numpy(np.array(ds._cos_sza_hr(dstr), copy=True)).to(
        device=device, dtype=torch.float32
    )[None, ...]
    hr_aux = torch.cat([hr_static_t, sza_hr], dim=1)

    with torch.no_grad():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_z = model(x_norm, hr_aux)
        else:
            pred_z = model(x_norm, hr_aux)

    phy_raw = (
        _to_output_space(pred_z, mean, std, "physical", q_output_unit, False)
        .squeeze(0)
        .detach()
        .to("cpu")
        .float()
        .numpy()
    )
    out_viol = _output_violation_counts(phy_raw)
    phy_c = _apply_output_physical_constraints_np(phy_raw) if enforce_out else phy_raw
    temp_viol = _temp_order_counts(phy_c)
    if diag is not None:
        diag.update_sample(x_raw, phy_raw, phy_c, in_viol, out_viol, temp_viol)

    if out_space == "physical":
        pred_cpu = phy_c.astype(out_np_dtype, copy=False)
    elif out_space == "zscore":
        pred_cpu = pred_z.squeeze(0).detach().to("cpu").numpy().astype(out_np_dtype, copy=False)
    else:
        pred_cpu = (
            _to_output_space(pred_z, mean, std, "denorm", q_output_unit, False)
            .squeeze(0)
            .detach()
            .to("cpu")
            .numpy()
            .astype(out_np_dtype, copy=False)
        )
    return pred_cpu, pred_z


def _metric_mae(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a,b: (B,8,H,W) float32 on CPU or GPU
    return (a - b).abs().mean(dim=(0, 2, 3))  # (8,)


def _h5_create_dataset(
    f: h5py.File,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
    compression: str,
    gzip_level: int,
    chunks: tuple[int, ...] | None,
) -> h5py.Dataset:
    if compression == "none":
        return f.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks)
    if compression == "lzf":
        return f.create_dataset(name, shape=shape, dtype=dtype, compression="lzf", chunks=chunks)
    return f.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        compression="gzip",
        compression_opts=int(gzip_level),
        chunks=chunks,
    )


def _np_dtype(dtype_str: str) -> np.dtype:
    if dtype_str == "float16":
        return np.float16
    if dtype_str == "float32":
        return np.float32
    raise ValueError(dtype_str)


def _load_lat_lon_hr(static_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    lat = np.load(static_dir / "lat_hr.npy").astype(np.float32)  # (1801,)
    lon = np.load(static_dir / "lon_hr.npy").astype(np.float32)  # (3600,)
    return lat, lon


def _load_lat_lon_from_ref_nc(
    ref_nc_path: Path,
    lat_var: str = "lat",
    lon_var: str = "lon",
) -> tuple[np.ndarray, np.ndarray]:
    with netCDF4.Dataset(str(ref_nc_path), "r") as f:
        if lat_var not in f.variables or lon_var not in f.variables:
            raise KeyError(
                f"Could not find coord vars in ref nc. "
                f"lat_var={lat_var!r} lon_var={lon_var!r} available={list(f.variables.keys())[:50]}"
            )
        lat = np.array(f.variables[lat_var][:], dtype=np.float32)
        lon = np.array(f.variables[lon_var][:], dtype=np.float32)

    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError(f"Expected 1D lat/lon in ref nc, got lat{lat.shape} lon{lon.shape}")
    return lat, lon


def _write_sample_netcdf(
    out_path: Path,
    pred: np.ndarray,  # (8, 1801, 3600)
    variables: list[str],
    lat: np.ndarray,  # (1801,)
    lon: np.ndarray,  # (3600,)
    units: dict[str, str],
    date: str,
    meta_json: str,
    zlib_level: int,
    overwrite: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if not overwrite:
            return
        out_path.unlink()

    # NETCDF4 is required for chunked+compressed 2D fields.
    with netCDF4.Dataset(str(out_path), "w", format="NETCDF4") as nc:
        nc.setncattr("date", date)
        nc.setncattr("meta_json", meta_json)

        nc.createDimension("lat", int(lat.shape[0]))
        nc.createDimension("lon", int(lon.shape[0]))

        vlat = nc.createVariable("lat", "f4", ("lat",))
        vlon = nc.createVariable("lon", "f4", ("lon",))
        vlat[:] = lat
        vlon[:] = lon
        vlat.setncattr("units", "degrees_north")
        vlon.setncattr("units", "degrees_east")

        lvl = int(max(0, min(9, zlib_level)))
        for ci, name in enumerate(variables):
            v = nc.createVariable(
                name,
                "f4",
                ("lat", "lon"),
                zlib=True,
                complevel=lvl,
                shuffle=True,
            )
            if name in units:
                v.setncattr("units", str(units[name]))
            # Note: float16 is not reliably supported across NetCDF readers; write float32.
            v[:, :] = pred[ci].astype(np.float32, copy=False)


def _lon_axis_neg180_180(n_lon: int) -> np.ndarray:
    # Match training static HR grid convention: -180.0 .. (-180 + d*(n_lon-1))
    d = 360.0 / float(n_lon)
    return (-180.0 + d * np.arange(n_lon, dtype=np.float32)).astype(np.float32)


def _lon_axis_pos0_360(n_lon: int) -> np.ndarray:
    # 0.0 .. (d*(n_lon-1)) where d=360/n_lon
    d = 360.0 / float(n_lon)
    return (d * np.arange(n_lon, dtype=np.float32)).astype(np.float32)


def _apply_lon_convention_to_pred(
    pred: np.ndarray,  # (C,H,W)
    lon_convention: LonConvention,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (pred_aligned, lon_axis_to_write).

    Important (verified against real training HDF5 data on 2026-09-09, see
    analysis notes: at 60N, TAS matches Siberian cold pole around lon label
    +50..+110 and the North Atlantic warm anomaly around lon label -40..0
    under the -180..180 labeling -- this pins down the true native ordering):
      - The native column ordering used by training HDF5 `data/x`/`data/y`
        (and hence by the trained model) matches lon=-180..180 monotonic
        indexing -- i.e. it is IDENTICAL to `static/lon_lr.npy` /
        `static/lon_hr.npy` / the `metadata/lr_grid_lons` /
        `metadata/hr_grid_lons` written by prepare_hdf5_seasonal.py.
      - Therefore:
          neg180_180  : keep data as-is (native), write lon -180..180
          pos0_360    : roll by W//2, write lon 0..360
      - This operation is a pure reindexing (no interpolation), so it does NOT
        introduce numeric error; it only changes column ordering to match lon.
        Note W//2 == -W//2 (mod W) for even W, so a single roll magnitude
        works for both directions.
    """
    if pred.ndim != 3:
        raise ValueError(f"Expected pred (C,H,W), got {pred.shape}")
    w = int(pred.shape[-1])
    if lon_convention == "neg180_180":
        return pred, _lon_axis_neg180_180(w)
    if lon_convention == "pos0_360":
        shift = w // 2
        pred2 = np.roll(pred, shift=shift, axis=-1)
        return pred2, _lon_axis_pos0_360(w)
    raise ValueError(lon_convention)

def main() -> None:
    args = _parse_args()

    # route dataset module globals to user-specified paths
    ds.STATIC_DIR = Path(args.static_dir)
    ds.STATS_FILE = Path(args.stats_file)

    hdf5_root = Path(args.hdf5_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Allow passing 'cuda:1' etc. Fall back to CPU when CUDA is unavailable.
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    use_amp = bool(args.amp_bf16 and device.type == "cuda")
    print(f"Using device: {device}  (amp_bf16={use_amp})")

    ckpt_path = Path(args.ckpt)
    sd = _load_ckpt_state_dict(ckpt_path, device)

    if args.auto_model_cfg:
        cfg = _infer_model_cfg_from_state_dict(sd)
        print(f"[auto_model_cfg] inferred: {cfg}")
    else:
        cfg = ModelConfig(
            base_ch=args.base_ch,
            num_resblocks=args.num_resblocks,
            use_cbam=bool(args.use_cbam),
            hr_aux_mode=args.hr_aux_mode,
        )

    model = _build_model(cfg, device)
    model.load_state_dict(sd, strict=True)

    # Prepare constants on device
    mean, std = _prep_norm_tensors(device)
    hr_static_t = _prep_static_tensors(device)
    input_source: InputSource = args.input_source
    if input_source == "cesm_nc" and not bool(args.allow_experimental_cesm_nc):
        raise SystemExit(
            "input_source=cesm_nc is experimental and disabled by default in the stable CESM path.\n"
            "Recommended workflow:\n"
            "  1) prepare_hdf5_cesm.py (nc/npy -> HDF5)\n"
            "  2) infer.py --input_source hdf5 --hdf5_root /path/to/hdf5_cesm\n"
            "If you still need the old direct-NC path for diagnostics, add --allow_experimental_cesm_nc."
        )
    if input_source == "cesm_nc":
        print(
            "[warn] Using experimental cesm_nc path. "
            "For stable production inference, use HDF5 prepared by prepare_hdf5_cesm.py."
        )

    extra_exclude_dates = _parse_exclude_date_list(args.exclude_dates)
    is_cesm_hdf5 = (input_source == "cesm_nc") or _hdf5_root_looks_like_cesm(hdf5_root)
    skip_bad_dates, skip_bad_reason = _resolve_skip_bad_dates(
        args.skip_bad_dates, args.exclude_dates, is_cesm_hdf5
    )
    exclude_dates_set = _build_exclude_dates_set(extra_exclude_dates, skip_bad_dates)
    print(
        f"[skip_bad_dates] {'on' if skip_bad_dates else 'off'} ({skip_bad_reason}); "
        f"exclude_dates={sorted(exclude_dates_set) if exclude_dates_set else []}"
    )
    apply_input_constraints, input_clip_reason = _resolve_input_physical_constraints(
        args.input_physical_constraints, is_cesm_hdf5
    )
    print(
        f"[input_physical_constraints] {'on' if apply_input_constraints else 'off'} ({input_clip_reason})"
    )

    # Output metadata
    meta = {
        "input_source": input_source,
        "ckpt": str(ckpt_path),
        "hdf5_root": str(hdf5_root),
        "seasons": list(args.seasons),
        "allow_experimental_cesm_nc": bool(args.allow_experimental_cesm_nc),
        "cesm_root": str(Path(args.cesm_root)),
        "cesm_glob": args.cesm_glob,
        "cesm_lat_var": args.cesm_lat_var,
        "cesm_lon_var": args.cesm_lon_var,
        "cesm_var_map": {
            "TAS": args.cesm_t2m_var,
            "PRE": args.cesm_pr_var,
            "wind10": args.cesm_wind10_var,
            "Q": args.cesm_q_var,
            "2M_RH": args.cesm_rhmin_var,
            "2M_TMAX": args.cesm_tmax_var,
            "2M_TMIN": args.cesm_tmin_var,
            "FSDS": args.cesm_fsds_var,
        },
        "cesm_regrid_to_static_lr": (not bool(args.no_cesm_regrid)),
        "cesm_pr_transform": args.cesm_pr_transform,
        "cesm_q_scale": float(args.cesm_q_scale),
        "date_start": args.date_start,
        "date_end": args.date_end,
        "skip_bad_dates": bool(skip_bad_dates),
        "skip_bad_dates_reason": skip_bad_reason,
        "is_cesm_hdf5": bool(is_cesm_hdf5),
        "exclude_dates": sorted(exclude_dates_set),
        "exclude_dates_extra": sorted(extra_exclude_dates),
        "coord_ref_nc": args.coord_ref_nc,
        "coord_lat_var": args.coord_lat_var,
        "coord_lon_var": args.coord_lon_var,
        "output_mode": args.output_mode,
        "output_format": args.output_format,
        "lon_convention": args.lon_convention,
        "output_space": args.output_space,
        "output_dtype": args.output_dtype,
        "compression": args.compression,
        "q_output_unit": args.q_output_unit,
        "enforce_physical_constraints": bool(args.enforce_physical_constraints),
        "input_physical_constraints": bool(apply_input_constraints),
        "input_physical_constraints_reason": input_clip_reason,
        "model_cfg": cfg.__dict__,
        "variables": ds.VARIABLES,
    }
    (out_dir / "infer_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    # Running metrics (only when y exists)
    sum_mae_dn = torch.zeros(len(ds.VARIABLES), dtype=torch.float64)
    sum_mae_phy = torch.zeros(len(ds.VARIABLES), dtype=torch.float64)
    n_metric_samples = 0

    out_space: OutputSpace = args.output_space
    out_mode: OutputMode = args.output_mode
    out_fmt: OutputFormat = args.output_format
    out_np_dtype = _np_dtype(args.output_dtype)
    q_output_unit: QOutputUnit = args.q_output_unit
    enforce_constraints = bool(args.enforce_physical_constraints)
    diag = InferDiagnostics(list(ds.VARIABLES))
    per_sample_dirname_h5 = "per_sample" if input_source == "hdf5" else "cesm_per_sample"
    per_sample_dirname_nc = "per_sample_nc" if input_source == "hdf5" else "cesm_per_sample_nc"

    n_done = 0
    coord_source = "static_npy"
    coord_ref_path: str | None = args.coord_ref_nc

    # For NetCDF output, default to the pre-HDF5 reference file when not provided.
    if out_fmt == "nc" and not coord_ref_path:
        default_ref = EXAMPLE_NC
        if default_ref.exists():
            coord_ref_path = str(default_ref)
        else:
            raise SystemExit(
                "output_format=nc requires a reference file for coordinates. "
                "Pass --coord_ref_nc /path/to/obs_YYYYMMDD.nc"
            )

    if coord_ref_path:
        lat_hr, lon_hr = _load_lat_lon_from_ref_nc(
            Path(coord_ref_path),
            lat_var=args.coord_lat_var,
            lon_var=args.coord_lon_var,
        )
        coord_source = "ref_nc"
        print(f"Loaded lat/lon from ref nc: {coord_ref_path}")
    else:
        lat_hr, lon_hr = _load_lat_lon_hr(ds.STATIC_DIR)

    # Always print coord ranges to avoid silent mismatches.
    print(
        f"Coord source: {coord_source} | "
        f"lat[{lat_hr.shape[0]}] min={float(lat_hr.min()):.6f} max={float(lat_hr.max()):.6f} "
        f"first={float(lat_hr[0]):.6f} last={float(lat_hr[-1]):.6f} | "
        f"lon[{lon_hr.shape[0]}] min={float(lon_hr.min()):.6f} max={float(lon_hr.max()):.6f} "
        f"first={float(lon_hr[0]):.6f} last={float(lon_hr[-1]):.6f}"
    )

    # Write coord info back to meta file (overwrite with coord details).
    meta_path = out_dir / "infer_meta.json"
    try:
        meta_obj = json.loads(meta_path.read_text())
        meta_obj["coord_source"] = coord_source
        meta_obj["coord_ref_nc_effective"] = coord_ref_path
        meta_obj["coord_lat_stats"] = {
            "n": int(lat_hr.shape[0]),
            "min": float(lat_hr.min()),
            "max": float(lat_hr.max()),
            "first": float(lat_hr[0]),
            "last": float(lat_hr[-1]),
        }
        meta_obj["coord_lon_stats"] = {
            "n": int(lon_hr.shape[0]),
            "min": float(lon_hr.min()),
            "max": float(lon_hr.max()),
            "first": float(lon_hr[0]),
            "last": float(lon_hr[-1]),
        }
        meta_path.write_text(json.dumps(meta_obj, ensure_ascii=False, indent=2))
    except Exception:
        # Non-fatal: meta is for bookkeeping only.
        pass

    if input_source == "cesm_nc":
        # ---- CESM official daily NetCDF inference (no GT) ----
        if out_mode != "per_sample":
            raise SystemExit("input_source=cesm_nc only supports --output_mode per_sample")

        cesm_root = Path(args.cesm_root)
        cesm_files = _list_cesm_files(cesm_root, args.cesm_glob)
        if not cesm_files:
            raise SystemExit(f"No CESM files found under {cesm_root} with glob={args.cesm_glob!r}")

        # static LR grid (training grid)
        tgt_lat = np.load(ds.STATIC_DIR / "lat_lr.npy").astype(np.float32)
        tgt_lon = np.load(ds.STATIC_DIR / "lon_lr.npy").astype(np.float32)
        do_regrid = not bool(args.no_cesm_regrid)

        var_map = {
            "TAS": args.cesm_t2m_var,
            "PRE": args.cesm_pr_var,
            "wind10": args.cesm_wind10_var,
            "Q": args.cesm_q_var,
            "2M_RH": args.cesm_rhmin_var,
            "2M_TMAX": args.cesm_tmax_var,
            "2M_TMIN": args.cesm_tmin_var,
            "FSDS": args.cesm_fsds_var,
        }

        for nc_path in cesm_files:
            dstr = _parse_date_from_filename(nc_path)
            if not _date_in_range(dstr, args.date_start, args.date_end):
                continue

            if args.max_samples is not None and n_done >= args.max_samples:
                break

            dstr, x_raw = _read_cesm_x_from_nc(
                nc_path,
                lat_var=args.cesm_lat_var,
                lon_var=args.cesm_lon_var,
                var_map=var_map,
                tgt_lat=tgt_lat,
                tgt_lon=tgt_lon,
                do_regrid=do_regrid,
                pr_transform=args.cesm_pr_transform,
                q_scale=float(args.cesm_q_scale),
            )

            pred_cpu, pred_z = _infer_one_sample(
                model=model,
                x_raw=x_raw,
                dstr=dstr,
                device=device,
                mean=mean,
                std=std,
                hr_static_t=hr_static_t,
                use_amp=use_amp,
                out_space=out_space,
                q_output_unit=q_output_unit,
                enforce_out=enforce_constraints,
                apply_in=apply_input_constraints,
                out_np_dtype=out_np_dtype,
                diag=diag,
            )

            if out_fmt == "nc":
                out_path = out_dir / per_sample_dirname_nc / dstr[:4] / f"{dstr}.nc"
                pred_nc, lon_nc = _apply_lon_convention_to_pred(pred_cpu, args.lon_convention)
                _write_sample_netcdf(
                    out_path=out_path,
                    pred=pred_nc,
                    variables=list(ds.VARIABLES),
                    lat=lat_hr,
                    lon=lon_nc,
                    units=VARIABLE_UNITS | ({"Q": "kg/kg"} if q_output_unit == "kgkg" else {"Q": "g/kg"}),
                    date=dstr,
                    meta_json=json.dumps(meta, ensure_ascii=False),
                    zlib_level=args.nc_zlib_level,
                    overwrite=bool(args.overwrite),
                )
            else:
                out_path = out_dir / per_sample_dirname_h5 / dstr[:4] / f"{dstr}.h5"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.exists() and not args.overwrite:
                    print(f"[skip] exists: {out_path}")
                else:
                    with h5py.File(out_path, "w") as fo:
                        fo.attrs["meta_json"] = json.dumps(meta, ensure_ascii=False)
                        fo.attrs["date"] = dstr
                        fo.create_dataset("pred", data=pred_cpu)
                        fo.create_dataset("variables", data=np.array(ds.VARIABLES, dtype="S16"))

            n_done += 1
            if n_done % 10 == 0:
                print(f"[{n_done}] CESM  date={dstr}  file={nc_path.name}")

    else:
        # ---- HDF5 shard inference (test/train style) ----
        shards = _list_shards(hdf5_root, args.seasons)
        if not shards:
            raise SystemExit(f"No shards found under {hdf5_root} for seasons={args.seasons}")

        expected_lat_lr = np.load(ds.STATIC_DIR / "lat_lr.npy").astype(np.float32)
        expected_lon_lr = np.load(ds.STATIC_DIR / "lon_lr.npy").astype(np.float32)
        pre_transform_set: set[str] = set()
        q_scale_set: set[float] = set()
        lr_semantics_set: set[str] = set()
        source_set: set[str] = set()
        contract_infos: list[dict[str, object]] = []
        grid_checked_count = 0
        metadata_present_count = 0
        for h5path in shards:
            ci = _validate_hdf5_shard_contract(
                h5path,
                expected_lat_lr=expected_lat_lr,
                expected_lon_lr=expected_lon_lr,
            )
            contract_infos.append(ci)
            if ci.get("metadata_present"):
                metadata_present_count += 1
            if ci.get("lr_grid_checked"):
                grid_checked_count += 1
            if "lr_grid_semantics" in ci:
                lr_semantics_set.add(str(ci["lr_grid_semantics"]))
            if "source" in ci:
                source_set.add(str(ci["source"]))
            if "pre_transform" in ci:
                pre_transform_set.add(str(ci["pre_transform"]))
            if "q_scale" in ci:
                q_scale_set.add(float(ci["q_scale"]))
        print(
            f"[hdf5_contract] validated {len(shards)} shards | "
            f"metadata={metadata_present_count} | lr_grid_checked={grid_checked_count}"
        )
        if source_set:
            print(f"[hdf5_contract] source seen: {sorted(source_set)}")
        if lr_semantics_set:
            print(f"[hdf5_contract] lr_grid_semantics seen: {sorted(lr_semantics_set)}")
        if pre_transform_set:
            print(f"[hdf5_contract] pre_transform seen: {sorted(pre_transform_set)}")
        if q_scale_set:
            print(f"[hdf5_contract] q_scale seen: {sorted(q_scale_set)}")

        # Refine CESM detection with shard metadata (covers CESM HDF5 not named *cesm*).
        if _shards_look_like_cesm(contract_infos) and not is_cesm_hdf5:
            is_cesm_hdf5 = True
            skip_bad_dates, skip_bad_reason = _resolve_skip_bad_dates(
                args.skip_bad_dates, args.exclude_dates, is_cesm_hdf5
            )
            exclude_dates_set = _build_exclude_dates_set(extra_exclude_dates, skip_bad_dates)
            apply_input_constraints, input_clip_reason = _resolve_input_physical_constraints(
                args.input_physical_constraints, is_cesm_hdf5
            )
            meta["skip_bad_dates"] = bool(skip_bad_dates)
            meta["skip_bad_dates_reason"] = skip_bad_reason
            meta["is_cesm_hdf5"] = bool(is_cesm_hdf5)
            meta["exclude_dates"] = sorted(exclude_dates_set)
            meta["input_physical_constraints"] = bool(apply_input_constraints)
            meta["input_physical_constraints_reason"] = input_clip_reason
            print(
                f"[skip_bad_dates] re-resolved after metadata: "
                f"{'on' if skip_bad_dates else 'off'} ({skip_bad_reason}); "
                f"exclude_dates={sorted(exclude_dates_set) if exclude_dates_set else []}"
            )
            print(
                f"[input_physical_constraints] re-resolved after metadata: "
                f"{'on' if apply_input_constraints else 'off'} ({input_clip_reason})"
            )

        try:
            meta_obj = json.loads((out_dir / "infer_meta.json").read_text())
            meta_obj["skip_bad_dates"] = bool(skip_bad_dates)
            meta_obj["skip_bad_dates_reason"] = skip_bad_reason
            meta_obj["is_cesm_hdf5"] = bool(is_cesm_hdf5)
            meta_obj["exclude_dates"] = sorted(exclude_dates_set)
            meta_obj["input_physical_constraints"] = bool(apply_input_constraints)
            meta_obj["input_physical_constraints_reason"] = input_clip_reason
            meta_obj["hdf5_contract"] = {
                "validated_shards": int(len(shards)),
                "metadata_present_shards": int(metadata_present_count),
                "lr_grid_checked_shards": int(grid_checked_count),
                "source_seen": sorted(source_set),
                "lr_grid_semantics_seen": sorted(lr_semantics_set),
                "pre_transform_seen": sorted(pre_transform_set),
                "q_scale_seen": sorted(q_scale_set),
                "required_data_x_tail": [8, 180, 360],
            }
            (out_dir / "infer_meta.json").write_text(json.dumps(meta_obj, ensure_ascii=False, indent=2))
        except Exception:
            pass

        total_excluded = 0
        for h5path in shards:
            keep_i, keep_dates, has_y, n_excluded = _collect_indices_for_shard(
                h5path, args.date_start, args.date_end, exclude_dates_set
            )
            total_excluded += n_excluded
            if n_excluded:
                print(f"[exclude_dates] skipped {n_excluded} sample(s) in {h5path.name}")
            if not keep_i:
                continue

            # Debug stop after max_samples (global across shards)
            if args.max_samples is not None:
                remain = args.max_samples - n_done
                if remain <= 0:
                    break
                if len(keep_i) > remain:
                    keep_i = keep_i[:remain]
                    keep_dates = keep_dates[:remain]

            rel = h5path.relative_to(hdf5_root)
            if out_mode == "per_shard":
                out_path = out_dir / rel.parent / f"{rel.stem}__pred.h5"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.exists() and not args.overwrite:
                    print(f"[skip] exists: {out_path}")
                    n_done += len(keep_i)
                    continue

                with h5py.File(out_path, "w") as fo:
                    fo.attrs["meta_json"] = json.dumps(meta, ensure_ascii=False)
                    fo.create_dataset("dates", data=np.array(keep_dates, dtype="S8"))

                    # chunk by sample; allow streaming writes
                    chunks = (1, 1, 1801, 3600) if args.compression != "none" else None
                    d_pred = _h5_create_dataset(
                        fo,
                        "pred",
                        shape=(len(keep_i), 8, 1801, 3600),
                        dtype=out_np_dtype,
                        compression=args.compression,
                        gzip_level=args.gzip_level,
                        chunks=chunks,
                    )

                    # optional: keep per-sample MAE arrays when y exists
                    d_mae_dn = None
                    d_mae_phy = None
                    if has_y:
                        d_mae_dn = fo.create_dataset("mae_denorm", shape=(len(keep_i), 8), dtype=np.float32)
                        d_mae_phy = fo.create_dataset("mae_physical", shape=(len(keep_i), 8), dtype=np.float32)

                    with h5py.File(h5path, "r") as fi:
                        x_ds = fi["data/x"]
                        y_ds = fi["data/y"] if has_y else None

                        for j, (si, dstr) in enumerate(zip(keep_i, keep_dates)):
                            x_raw = x_ds[si].astype(np.float32)  # (8,180,360)
                            pred_cpu, pred_z = _infer_one_sample(
                                model=model,
                                x_raw=x_raw,
                                dstr=dstr,
                                device=device,
                                mean=mean,
                                std=std,
                                hr_static_t=hr_static_t,
                                use_amp=use_amp,
                                out_space=out_space,
                                q_output_unit=q_output_unit,
                                enforce_out=enforce_constraints,
                                apply_in=apply_input_constraints,
                                out_np_dtype=out_np_dtype,
                                diag=diag,
                            )
                            d_pred[j] = pred_cpu

                            if has_y and y_ds is not None:
                                y_raw = y_ds[si].astype(np.float32)  # (8,1801,3600)
                                y_t = torch.from_numpy(y_raw).to(device=device, dtype=torch.float32)[None, ...]
                                y_z = (y_t - mean) / std
                                y_dn = y_z.float() * std + mean

                                pred_dn = _to_output_space(pred_z, mean, std, "denorm", q_output_unit, enforce_constraints)
                                mae_dn = _metric_mae(pred_dn, y_dn).detach().to("cpu").float()
                                d_mae_dn[j] = mae_dn.numpy()

                                pred_phy = _to_output_space(pred_z, mean, std, "physical", q_output_unit, enforce_constraints)
                                # apply the same physical postprocessing to y (PRE inverse + optional Q unit + optional constraints)
                                y_phy = _to_output_space(y_z, mean, std, "physical", q_output_unit, enforce_constraints)
                                mae_phy = _metric_mae(pred_phy, y_phy).detach().to("cpu").float()
                                d_mae_phy[j] = mae_phy.numpy()

                                sum_mae_dn += mae_dn.double()
                                sum_mae_phy += mae_phy.double()
                                n_metric_samples += 1

                            n_done += 1
                            if n_done % 10 == 0:
                                print(f"[{n_done}] {rel}  sample={si}  date={dstr}")

            else:  # per_sample
                with h5py.File(h5path, "r") as fi:
                    x_ds = fi["data/x"]
                    y_ds = fi["data/y"] if has_y else None

                    for si, dstr in zip(keep_i, keep_dates):
                        if out_fmt == "nc":
                            out_path = out_dir / per_sample_dirname_nc / dstr[:4] / f"{dstr}.nc"
                        else:
                            out_path = out_dir / per_sample_dirname_h5 / dstr[:4] / f"{dstr}.h5"
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        if out_path.exists() and not args.overwrite:
                            print(f"[skip] exists: {out_path}")
                            n_done += 1
                            continue

                        x_raw = x_ds[si].astype(np.float32)
                        pred_cpu, pred_z = _infer_one_sample(
                            model=model,
                            x_raw=x_raw,
                            dstr=dstr,
                            device=device,
                            mean=mean,
                            std=std,
                            hr_static_t=hr_static_t,
                            use_amp=use_amp,
                            out_space=out_space,
                            q_output_unit=q_output_unit,
                            enforce_out=enforce_constraints,
                            apply_in=apply_input_constraints,
                            out_np_dtype=out_np_dtype,
                            diag=diag,
                        )

                        if out_fmt == "nc":
                            pred_nc, lon_nc = _apply_lon_convention_to_pred(pred_cpu, args.lon_convention)
                            _write_sample_netcdf(
                                out_path=out_path,
                                pred=pred_nc,
                                variables=list(ds.VARIABLES),
                                lat=lat_hr,
                                lon=lon_nc,
                                units=VARIABLE_UNITS | ({"Q": "kg/kg"} if q_output_unit == "kgkg" else {"Q": "g/kg"}),
                                date=dstr,
                                meta_json=json.dumps(meta, ensure_ascii=False),
                                zlib_level=args.nc_zlib_level,
                                overwrite=bool(args.overwrite),
                            )
                        else:
                            with h5py.File(out_path, "w") as fo:
                                fo.attrs["meta_json"] = json.dumps(meta, ensure_ascii=False)
                                fo.attrs["date"] = dstr
                                fo.create_dataset("pred", data=pred_cpu)
                                fo.create_dataset("variables", data=np.array(ds.VARIABLES, dtype="S16"))

                        if has_y and y_ds is not None:
                            y_raw = y_ds[si].astype(np.float32)
                            y_t = torch.from_numpy(y_raw).to(device=device, dtype=torch.float32)[None, ...]
                            y_z = (y_t - mean) / std
                            y_dn = y_z.float() * std + mean
                            pred_dn = _to_output_space(pred_z, mean, std, "denorm", q_output_unit, enforce_constraints)
                            mae_dn = _metric_mae(pred_dn, y_dn).detach().to("cpu").float()
                            pred_phy = _to_output_space(pred_z, mean, std, "physical", q_output_unit, enforce_constraints)
                            y_phy = _to_output_space(y_z, mean, std, "physical", q_output_unit, enforce_constraints)
                            mae_phy = _metric_mae(pred_phy, y_phy).detach().to("cpu").float()

                            sum_mae_dn += mae_dn.double()
                            sum_mae_phy += mae_phy.double()
                            n_metric_samples += 1

                        n_done += 1
                        if args.max_samples is not None and n_done >= args.max_samples:
                            break

            if args.max_samples is not None and n_done >= args.max_samples:
                break

    # Final metrics summary
    if n_metric_samples > 0:
        mae_dn = (sum_mae_dn / n_metric_samples).cpu().numpy()
        mae_phy = (sum_mae_phy / n_metric_samples).cpu().numpy()
        metrics = {
            "n_samples": int(n_metric_samples),
            "mae_denorm": {v: float(mae_dn[i]) for i, v in enumerate(ds.VARIABLES)},
            "mae_physical": {v: float(mae_phy[i]) for i, v in enumerate(ds.VARIABLES)},
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        print("Metrics written:", out_dir / "metrics.json")
        for v in ds.VARIABLES:
            print(f"{v:>10s}  mae_denorm={metrics['mae_denorm'][v]:.4f}  mae_physical={metrics['mae_physical'][v]:.4f}")
    else:
        print("No ground-truth y found; metrics skipped.")

    diag_obj = _json_safe(
        diag.to_dict(
            input_clip_applied=apply_input_constraints,
            output_clip_applied=bool(enforce_constraints and out_space == "physical"),
        )
    )
    (out_dir / "diagnostics.json").write_text(json.dumps(diag_obj, ensure_ascii=False, indent=2))
    print("Diagnostics written:", out_dir / "diagnostics.json")
    in_rates = diag_obj.get("input_violation_rates", {})
    out_rates = diag_obj.get("output_violation_rates", {})
    t_rates = diag_obj.get("temperature_order_rates", {})
    print(
        f"[diagnostics] n={diag.n_samples}  "
        f"input_RH>100={in_rates.get('2M_RH_gt_100')}  "
        f"out_RH>100={out_rates.get('2M_RH_gt_100')}  "
        f"TAS<180K={t_rates.get('TAS_lt_180K')}  "
        f"TMAX<TMIN={t_rates.get('TMAX_lt_TMIN')}"
    )

    if input_source == "hdf5" and total_excluded:
        print(f"[exclude_dates] total skipped across all shards: {total_excluded}")
        try:
            meta_obj = json.loads((out_dir / "infer_meta.json").read_text())
            meta_obj["excluded_samples_total"] = int(total_excluded)
            (out_dir / "infer_meta.json").write_text(json.dumps(meta_obj, ensure_ascii=False, indent=2))
        except Exception:
            pass

    print(f"Done. total_written={n_done}  out_dir={out_dir}")


if __name__ == "__main__":
    main()

