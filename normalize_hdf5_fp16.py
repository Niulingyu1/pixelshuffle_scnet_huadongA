"""离线将训练 HDF5（fp32、未 z-score）转为预标准化 fp16 副本。

对每个样本：
  1. 用 global_stats_state.json 的 var_all mean/std 做与 dataset.py 一致的 z-score
  2. 转 float16 写入新 shard（chunks=(1,)+shape, gzip-4）
  3. 在 metadata 写入 normalized=True 等追溯字段

逐样本读写，不整 shard 载入内存。写 *.h5.tmp 完成后 os.replace 原子替换。
目标已存在且样本数一致、已标记 normalized 时默认跳过（可续跑）。

示例：
  python normalize_hdf5_fp16.py --dry_run
  python normalize_hdf5_fp16.py --seasons MAM --only_shard shard_cra1p5_full_0038.h5
  python normalize_hdf5_fp16.py --seasons MAM JJA SON DJF
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from paths import HDF5_ROOT_RAW, HDF5_ROOT_NORM, STATS_FILE

VARIABLES = ["TAS", "PRE", "wind10", "Q", "2M_RH", "2M_TMAX", "2M_TMIN", "FSDS"]
SEASONS = ["MAM", "JJA", "SON", "DJF"]
PREPROCESSING_TAG = "zscore_fp16_v1"


def _load_norm_stats(stats_file: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(stats_file) as f:
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
            raise KeyError(f"stats file {stats_file} missing var_all/lr for {v}")
        means.append(mn)
        stds.append(std)
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


def _list_src_shards(src_root: Path, seasons: list[str], only_shard: str | None) -> list[Path]:
    out: list[Path] = []
    for season in seasons:
        season_dir = src_root / season
        if not season_dir.is_dir():
            continue
        files = sorted(season_dir.glob("shard_*.h5"))
        if only_shard:
            files = [p for p in files if p.name == only_shard]
        out.extend(files)
    return out


def _dest_is_complete(dst: Path, n_src: int) -> bool:
    if not dst.is_file():
        return False
    try:
        with h5py.File(dst, "r") as f:
            if "data/x" not in f or "data/y" not in f:
                return False
            n_ok = int(f["data/x"].shape[0]) == int(n_src) and int(f["data/y"].shape[0]) == int(n_src)
            normalized = False
            if "metadata" in f:
                normalized = bool(f["metadata"].attrs.get("normalized", False))
            return n_ok and normalized
    except OSError:
        return False


def _copy_metadata_group(src_f: h5py.File, dst_f: h5py.File) -> h5py.Group:
    if "metadata" in src_f:
        src_f.copy("metadata", dst_f, name="metadata")
        g = dst_f["metadata"]
    else:
        g = dst_f.create_group("metadata")
    return g


def _zscore_to_fp16(
    arr: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    name: str,
    sample_i: int,
    src: Path,
) -> np.ndarray:
    z = (arr.astype(np.float32, copy=False) - mean) / std
    out = z.astype(np.float16)
    if not np.isfinite(out).all():
        n_nan = int(np.isnan(out).sum())
        n_inf = int(np.isinf(out).sum())
        zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
        raise RuntimeError(
            f"{src} sample={sample_i} {name}: fp16 出现 NaN/Inf "
            f"(nan={n_nan}, inf={n_inf}, z_min={zmin:.4g}, z_max={zmax:.4g})"
        )
    return out


def _convert_one_shard(
    src: Path,
    dst: Path,
    *,
    mean: np.ndarray,
    std: np.ndarray,
    stats_file: Path,
    gzip_level: int,
    overwrite: bool,
) -> str:
    with h5py.File(src, "r") as sf:
        n = int(sf["data/x"].shape[0])
        x_tail = tuple(int(s) for s in sf["data/x"].shape[1:])
        y_tail = tuple(int(s) for s in sf["data/y"].shape[1:])
        if x_tail != (8, 180, 360) or y_tail != (8, 1801, 3600):
            raise ValueError(f"{src}: unexpected shapes x{x_tail} y{y_tail}")

        if _dest_is_complete(dst, n) and not overwrite:
            return "skip"

        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        if tmp.exists():
            tmp.unlink()

        mean_b = mean[:, None, None]
        std_b = std[:, None, None]
        comp_kw = {"compression": "gzip", "compression_opts": int(gzip_level)}
        vlen_bytes = h5py.special_dtype(vlen=bytes)
        stats_mtime = datetime.fromtimestamp(stats_file.stat().st_mtime, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        with h5py.File(tmp, "w") as df:
            g_meta = _copy_metadata_group(sf, df)
            g_meta.attrs["normalized"] = True
            g_meta.attrs["preprocessing"] = PREPROCESSING_TAG
            g_meta.attrs["stats_file"] = str(stats_file)
            g_meta.attrs["stats_mtime"] = stats_mtime
            g_meta.attrs["source_shard"] = src.name
            g_meta.attrs["storage_dtype"] = "float16"
            g_meta.attrs["n_samples"] = np.int64(n)
            g_meta.attrs["x_shape"] = ",".join(str(s) for s in (n,) + x_tail)
            g_meta.attrs["y_shape"] = ",".join(str(s) for s in (n,) + y_tail)

            g_data = df.create_group("data") if "data" not in df else df["data"]
            for name in list(g_data.keys()):
                del g_data[name]

            ds_x = g_data.create_dataset(
                "x",
                shape=(n,) + x_tail,
                chunks=(1,) + x_tail,
                dtype=np.float16,
                **comp_kw,
            )
            ds_y = g_data.create_dataset(
                "y",
                shape=(n,) + y_tail,
                chunks=(1,) + y_tail,
                dtype=np.float16,
                **comp_kw,
            )
            ds_dates = g_data.create_dataset("dates", shape=(n,), dtype=vlen_bytes)

            src_x = sf["data/x"]
            src_y = sf["data/y"]
            src_dates = sf["data/dates"]
            for i in range(n):
                ds_x[i] = _zscore_to_fp16(src_x[i], mean_b, std_b, name="x", sample_i=i, src=src)
                ds_y[i] = _zscore_to_fp16(src_y[i], mean_b, std_b, name="y", sample_i=i, src=src)
                date = src_dates[i]
                if isinstance(date, (bytes, np.bytes_)):
                    ds_dates[i] = bytes(date)
                else:
                    ds_dates[i] = str(date).encode("ascii")

        os.replace(tmp, dst)
        return "wrote"


def main() -> None:
    p = argparse.ArgumentParser(description="Normalize training HDF5 to fp16 z-score shards")
    p.add_argument("--src_hdf5_root", type=str, default=str(HDF5_ROOT_RAW))
    p.add_argument("--dst_hdf5_root", type=str, default=str(HDF5_ROOT_NORM))
    p.add_argument("--stats_file", type=str, default=str(STATS_FILE))
    p.add_argument("--seasons", nargs="+", default=SEASONS)
    p.add_argument("--only_shard", type=str, default=None, help="只转该文件名，如 shard_cra1p5_full_0038.h5")
    p.add_argument("--gzip_level", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--check_complete",
        action="store_true",
        help="只检查 dst 是否覆盖 src 的全部 shard/样本，不写文件",
    )
    args = p.parse_args()

    src_root = Path(args.src_hdf5_root)
    dst_root = Path(args.dst_hdf5_root)
    stats_file = Path(args.stats_file)
    shards = _list_src_shards(src_root, list(args.seasons), args.only_shard)
    if not shards:
        raise SystemExit(f"No shard_*.h5 found under {src_root} seasons={args.seasons}")

    n_total = 0
    rows = []
    for sp in shards:
        with h5py.File(sp, "r") as f:
            n = int(f["data/x"].shape[0])
        n_total += n
        rel = sp.relative_to(src_root)
        rows.append((sp, dst_root / rel, n))

    print(f"src={src_root}")
    print(f"dst={dst_root}")
    print(f"stats={stats_file}")
    print(f"shards={len(shards)}  samples={n_total}")
    if args.dry_run:
        for sp, dp, n in rows:
            print(f"  {sp.relative_to(src_root)}  n={n}  -> {dp}")
        return
    if args.check_complete:
        missing = []
        bad = []
        tmp_left = list(dst_root.glob("*/*.h5.tmp"))
        for sp, dp, n in rows:
            if not dp.is_file():
                missing.append(str(dp))
                continue
            if not _dest_is_complete(dp, n):
                bad.append(str(dp))
        print(f"tmp_leftover={len(tmp_left)} missing={len(missing)} incomplete={len(bad)}")
        for p in tmp_left[:10]:
            print(f"  tmp {p}")
        for p in missing[:10]:
            print(f"  missing {p}")
        for p in bad[:10]:
            print(f"  incomplete {p}")
        if missing or bad or tmp_left:
            raise SystemExit("completeness check failed")
        print("completeness OK")
        return

    mean, std = _load_norm_stats(stats_file)
    t0 = time.time()
    n_wrote = n_skip = n_done_samples = 0
    for k, (sp, dp, n) in enumerate(rows, start=1):
        t_shard = time.time()
        status = _convert_one_shard(
            sp,
            dp,
            mean=mean,
            std=std,
            stats_file=stats_file,
            gzip_level=int(args.gzip_level),
            overwrite=bool(args.overwrite),
        )
        dt = time.time() - t_shard
        if status == "skip":
            n_skip += 1
            print(f"[{k}/{len(rows)}] skip {sp.relative_to(src_root)} n={n}")
        else:
            n_wrote += 1
            n_done_samples += n
            elapsed = time.time() - t0
            remain = ""
            if n_done_samples > 0:
                rate = n_done_samples / max(elapsed, 1e-6)
                remaining_samples = sum(nn for _, _, nn in rows[k:])
                eta = remaining_samples / max(rate, 1e-6)
                remain = f"  eta={eta/60:.1f}min"
            size_mb = dp.stat().st_size / 1e6 if dp.exists() else 0.0
            print(
                f"[{k}/{len(rows)}] wrote {sp.relative_to(src_root)} n={n} "
                f"dt={dt:.1f}s size={size_mb:.1f}MB{remain}"
            )
            sys.stdout.flush()

    print(
        f"Done. wrote={n_wrote} skip={n_skip} elapsed={time.time()-t0:.1f}s dst={dst_root}"
    )


if __name__ == "__main__":
    main()
