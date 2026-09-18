"""Compare a source fp32 shard vs its pre-normalized fp16 copy through DownscaleDataset.

Usage:
  python scripts/check_hdf5_norm_fp16.py \\
      --src /public/share/acd7koea4a/hdf5 \\
      --dst /public/share/acd7koea4a/hdf5_norm_fp16 \\
      --seasons MAM --manifests cra1p5_full --n_check 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from dataset import DownscaleDataset, VARIABLES  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--seasons", nargs="+", default=["MAM"])
    p.add_argument("--manifests", nargs="+", default=["cra1p5_full"])
    p.add_argument("--n_check", type=int, default=8)
    args = p.parse_args()

    ds_src = DownscaleDataset(hdf5_root=args.src, seasons=args.seasons, manifests=args.manifests)
    ds_dst = DownscaleDataset(hdf5_root=args.dst, seasons=args.seasons, manifests=args.manifests)
    if not ds_src.pre_normalized:
        print(f"[src] pre_normalized={ds_src.pre_normalized}  n={len(ds_src)}")
    else:
        raise SystemExit("src dataset is already pre-normalized; pass the original fp32 root as --src")
    if not ds_dst.pre_normalized:
        raise SystemExit("dst dataset is not pre-normalized; conversion may have failed")
    print(f"[dst] pre_normalized={ds_dst.pre_normalized}  n={len(ds_dst)}")

    n = min(int(args.n_check), len(ds_dst))
    src_map = {(Path(p).name, si): j for j, (p, si) in enumerate(ds_src.index)}
    max_x = 0.0
    max_y = 0.0
    for i in range(n):
        p, si = ds_dst.index[i]
        key = (Path(p).name, si)
        if key not in src_map:
            raise SystemExit(f"src missing matching sample {key}")
        x0, aux0, y0 = ds_src[src_map[key]]
        x1, aux1, y1 = ds_dst[i]
        for name, t in ("x_src", x0), ("y_src", y0), ("x_dst", x1), ("y_dst", y1):
            if t.dtype != torch.bfloat16:
                raise SystemExit(f"{name} dtype={t.dtype}, expected bfloat16")
        if aux0.dtype != torch.float32 or aux1.dtype != torch.float32:
            raise SystemExit(f"hr_aux dtype src={aux0.dtype} dst={aux1.dtype}, expected float32")
        dx = (x0.float() - x1.float()).abs().max().item()
        dy = (y0.float() - y1.float()).abs().max().item()
        da = (aux0.float() - aux1.float()).abs().max().item()
        max_x = max(max_x, dx)
        max_y = max(max_y, dy)
        print(f"  i={i}  max|dx|={dx:.6g}  max|dy|={dy:.6g}  max|d_aux|={da:.6g}")
        if da > 1e-6:
            raise SystemExit("hr_aux should be identical (not converted)")

    print(f"checked n={n}  max|dx|={max_x:.6g}  max|dy|={max_y:.6g}")
    print(f"variables={VARIABLES}")
    # fp32->fp16->bf16 vs fp32->bf16: diffs should stay near bf16 ULP (~0.01 at |z|~2)
    if max_x > 0.05 or max_y > 0.05:
        raise SystemExit("diff exceeds 0.05; inspect conversion")
    print("OK")


if __name__ == "__main__":
    main()
