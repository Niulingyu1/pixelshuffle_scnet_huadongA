"""One-step GPU/DCU smoke: load pre-normalized fp16 shard, forward+backward under autocast bf16.

BW1000（海光 DCU）上代码仍走 ``torch.cuda``；必须用 Notebook/训练镜像自带的 DTK PyTorch。
不要 ``source env/activate.sh``（家目录 conda 是 NVIDIA 版，``cuda.is_available()`` 会变成 False）。

  python scripts/smoke_norm_fp16_step.py \\
      --hdf5_root /public/home/acd7koea4a/hdf5_norm_fp16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from dataset import DownscaleDataset  # noqa: E402
from model import PixelShuffleDownscaleNet  # noqa: E402
from paths import STATIC_DIR, STATS_FILE  # noqa: E402
from train import CombinedLoss  # noqa: E402


def _torch_env_line() -> str:
    hip = getattr(torch.version, "hip", None)
    return (
        f"python={sys.executable}  torch={torch.__version__}  "
        f"cuda_build={getattr(torch.version, 'cuda', None)}  hip_build={hip}  "
        f"cuda_available={torch.cuda.is_available()}  count={torch.cuda.device_count()}"
    )


def _require_accelerator() -> torch.device:
    print(_torch_env_line())
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"device=cuda:0  name={name}")
        return torch.device("cuda", 0)

    hip = getattr(torch.version, "hip", None)
    cuda_build = getattr(torch.version, "cuda", None)
    hint = (
        "当前 Python 看不见加速卡。BW1000 必须用镜像自带的 DTK PyTorch"
        "（``which python`` 不应是 ~/.conda/envs/pytorch_downscale）。"
        "不要 source env/activate.sh。"
    )
    if cuda_build and not hip:
        hint = (
            f"当前是 NVIDIA CUDA 版 PyTorch（cuda={cuda_build}），"
            "在 BW1000（海光 DCU/DTK）上会落到 CPU。"
            "请用 Notebook 镜像自带 python，不要 conda activate pytorch_downscale。"
        )
    raise SystemExit(hint)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5_root", required=True)
    p.add_argument("--seasons", nargs="+", default=["MAM"])
    p.add_argument("--manifests", nargs="+", default=["cra1p5_full"])
    p.add_argument("--static_dir", default=str(STATIC_DIR))
    p.add_argument("--stats_file", default=str(STATS_FILE))
    args = p.parse_args()
    print(f"static_dir={args.static_dir}")
    print(f"stats_file={args.stats_file}")

    ds = DownscaleDataset(
        hdf5_root=args.hdf5_root,
        seasons=args.seasons,
        manifests=args.manifests,
        static_dir=args.static_dir,
        stats_file=args.stats_file,
    )
    x, hr_aux, y = ds[0]
    print(f"x={tuple(x.shape)} {x.dtype}  hr_aux={tuple(hr_aux.shape)} {hr_aux.dtype}  y={tuple(y.shape)} {y.dtype}")
    if x.dtype != torch.bfloat16 or y.dtype != torch.bfloat16:
        raise SystemExit("x/y must be bfloat16")
    if hr_aux.dtype != torch.float32:
        raise SystemExit("hr_aux must stay float32")

    device = _require_accelerator()
    model = PixelShuffleDownscaleNet(
        in_ch=8,
        hr_aux_ch=7,
        base_ch=256,
        num_resblocks=2,
        out_ch=8,
        target_h=1801,
        target_w=3600,
        use_cbam=False,
        hr_aux_mode="stage1",
        use_checkpoint=True,
        stage4_shuffle_conv_k=1,
        norm_type="group",
        interp_chunk_channels=16,
    ).to(device)
    crit = CombinedLoss(
        gamma=0.5,
        lambda_patch_extreme=0.0,
        lambda_wps=0.0,
        lambda_phys=0.0,
    ).to(device)
    x = x.unsqueeze(0).to(device, non_blocking=True)
    hr_aux = hr_aux.unsqueeze(0).to(device, non_blocking=True)
    y = y.unsqueeze(0).to(device, non_blocking=True)
    model.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = model(x, hr_aux)
        loss, sub = crit(pred, y)
    loss.backward()
    print(f"pred={tuple(pred.shape)} {pred.dtype}  loss={float(loss.detach())} {loss.dtype}  sub={sub}")
    print("forward+backward OK")


if __name__ == "__main__":
    main()
