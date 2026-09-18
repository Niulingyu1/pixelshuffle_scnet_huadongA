"""
PixelShuffleDownscaleNet 模型训练脚本。

  - 损失函数（v2，"参考目标 + 风光增量项"解耦设计，见 DOWNSCALE_README.md 第 5 节，
    组合损失，可逐项开关）：
      面积加权 TailWeightedMAE（参考目标）+ PatchExtremeLoss + WindPowerSensitivityProxy
      + PhysicalConsistencyLoss（+ 旧版 SpatialExtremeLoss/FFT/Gradient Loss，默认关闭，供对照）
      · TailWeightedMAE：以 |y_zscore| 为权重的 MAE，提升极端值学习（高/低分位数），
        γ=0 时退化为纯 MAE（--loss_gamma 控制）。v2 默认叠加 cos(latitude) 球面积
        权重（--no_area_weight 可关闭），修正规则经纬网格高纬度像素等权带来的
        面积高估；--var_weights 默认改为全 1（8 通道等权"参考目标"，不再做通道间
        零和式偏置加权，风光资源改进改由下方独立增量项负责）
      · PatchExtremeLoss（--lambda_patch_extreme，默认 0.1）：对 --patch_extreme_vars
        指定通道（默认 wind10、FSDS），把 HR 场划分为局部网格块（默认 180×360，
        与 LR 输入同分辨率近似对齐），约束局部 max/min/mean 一致——取代旧版
        SpatialExtremeLoss 的全球单一极值约束。patch-mean 只约束一阶矩，抑制为
        凑 max 而整体平移的退化解，**不约束极值位置**；确定性回归下可能以抬高
        整块换极值统计（见 DOWNSCALE_README.md 第 5 节）
      · WindPowerSensitivityProxy（--lambda_wps，默认 0.05）：仅对 wind10，在物理
        量纲的风机爬坡段（--wind_cutin 3–--wind_rated 12 m/s）内约束归一化功率
        代理一致。mask 不含额定以上高风速（由 tail + PatchExtreme 监督）。
        **重要局限性**：不是能量产出对齐（10m 非轮毂高度风速、日均非瞬时值）
      · PhysicalConsistencyLoss（--lambda_phys，默认 0.02）：仅依赖预测本身的安全
        网；**先反标准化**再约束 TMIN≤TAS≤TMAX、wind10/FSDS/Q/PRE(log1p) 非负、
        RH∈[0,100]，再按训练期标准差无量纲化组合（禁止在 z 空间直接 hinge）
      · 旧版机制（--lambda_extreme SpatialExtremeLoss、--lambda_freq FFT Loss、
        --lambda_grad Gradient Loss）默认权重为 0（关闭），仅保留供历史对照消融
      · 验证/早停始终使用纯 MAE、像素等权（与历史 run 可比较，不叠加 area_weight）；
        同时在验证集上用与训练相同的 criterion 计算完整组合损失各子项，避免用
        对双方都不公平的整体纯 MAE 去比较不同损失配置
  - 优化器：AdamW + 线性 warmup + 余弦退火
  - 梯度累积（默认 accum_steps=2）
  - bf16 自动混合精度（无需 GradScaler）
  - Gradient Checkpointing（默认开启，全部 4 个 UpStage，--no_checkpoint 可关闭）
  - TensorBoard 记录：Loss/train、Loss/val（纯 MAE，早停/top-K 判据）、LR、
                      MAE_val/<variable>（整体 MAE，物理量纲）
                      Loss/tail_w、Loss/patch_extreme、Loss/wps、Loss/phys、
                      Loss/spatial_extreme、Loss/freq、Loss/grad（训练时各子项，调试用；
                      后三项仅在对应旧版 λ>0 时出现）
                      Loss/val_combined、Loss/val_tail_w 等对应的验证集版本
                      （验证集上用与训练相同 criterion 计算的组合损失各子项，
                       用于判断该 run 的损失配置是否真的改善了其自身优化目标）
                      MAE_val_extreme/<variable>、MAE_val_extreme/mean
                      （仅 |y_zscore| > --val_extreme_z_thresh 的极端像素 MAE，
                       物理量纲；避免被大量"平静"像素稀释，是组合损失 vs 纯 MAE
                       在极端值表现上唯一公平的对比口径）
                      MAE_val_area_weighted/<variable>
                      （与训练损失一致的 cos(latitude) 面积加权 MAE 诊断，仅供
                       论文报告，不参与模型选择，Loss/val 本身不受影响）
                      Skill_val/<variable>、Skill_val_resource/mean
                      （--resource_vars 指定变量相对"LR 双线性插值" naive baseline
                       的 skill score，1 − MAE_model/MAE_baseline，>0 优于该
                       baseline；修复了早期版本把不同物理量纲 MAE 直接算术平均、
                       无量纲可解释性的问题，见 compute_baseline_resource_mae()）
                      MAE_val_physical/PRE
                      （PRE 在写入 HDF5 前做过 log1p 变换，MAE_val/PRE 反标准化后
                       仍是 log1p(mm/day) 空间误差，容易被误读为"物理精度"；本项
                       额外对 PRE 通道做一次 expm1，是唯一真实 mm/day 量纲的降水
                       误差指标，不影响/不替代其余既有指标口径）
  - 检查点保存/恢复（含 epoch、优化器、调度器、早停计数、best_resource 判据）：
    `best.pt`（标准判据，按全部 8 变量纯 MAE 选取，早停/top-K 均以此为准）与
    `best_resource.pt`（并行、非标准判据，按 --resource_vars 默认 wind10/FSDS
    相对 naive baseline 的 skill score 均值选取，越大越好，仅供业务侧参考，
    不影响 best.pt/早停）。`--resume` 会恢复 early_stop_streak / best_resource_val，
    并从磁盘重建 top-K 列表，避免平台中断后续训时早停被重置或覆盖更优的
    best_resource.pt
  - 早停：默认监控验证集 Loss/val（纯 MAE），连续若干次验证无显著改进则结束（--early_stop_patience 为 0 可关闭）
  - 非有限 loss/梯度：optimizer.step 前检测 NaN/Inf；任一 rank 非有限则全体跳过该步
    （不更新参数/EMA/scheduler），避免长跑静默写出坏权重

─────────────────────────────────────────────────────────────
快速调试（hdf5_mini，少量 epoch）：
    python train.py \\
        --hdf5_root /public/share/acd7koea4a/hdf5_mini \\
        --epochs 3 --val_interval 1 \\
        --run_dir runs/debug

─────────────────────────────────────────────────────────────
消融实验对照组（每组独立 run_dir，TensorBoard 中对比）。
A–D 只改架构，损失仍走 v2 默认；E–G 在正式架构上拆损失。不传损失参数 ≠ 纯 MAE。

  # A. 代码默认架构（CBAM 开 + HR-aux 全注入 + BatchNorm；非正式训练）
  python train.py --base_ch 256 \\
      --run_dir runs/ablation/A_code_defaults

  # B. 无 CBAM
  python train.py --base_ch 256 --no_cbam \\
      --run_dir runs/ablation/B_no_cbam

  # C. 无 HR-aux（全程不注入）
  python train.py --base_ch 256 --hr_aux_mode none \\
      --run_dir runs/ablation/C_no_hr_aux

  # D. 仅 Stage1 注入 HR-aux
  python train.py --base_ch 256 --hr_aux_mode stage1 \\
      --run_dir runs/ablation/D_hr_aux_stage1_only

  # E. 纯 MAE（须显式关掉面积加权与全部增量项，才与 nn.L1Loss 数值等价）
  python train.py --base_ch 256 --no_cbam --hr_aux_mode stage1 --norm_type group \\
      --loss_gamma 0.0 --var_weights "" --no_area_weight \\
      --lambda_extreme 0.0 --lambda_patch_extreme 0.0 --lambda_wps 0.0 --lambda_phys 0.0 \\
      --lambda_freq 0.0 --lambda_grad 0.0 \\
      --run_dir runs/ablation/E_pure_mae

  # F. 仅面积加权尾部 MAE（关 v2 增量项与旧版 FFT/Grad/SpatialExtreme）
  python train.py --base_ch 256 --no_cbam --hr_aux_mode stage1 --norm_type group \\
      --loss_gamma 0.5 --lambda_patch_extreme 0.0 --lambda_wps 0.0 --lambda_phys 0.0 \\
      --lambda_extreme 0.0 --lambda_freq 0.0 --lambda_grad 0.0 \\
      --run_dir runs/ablation/F_tail_area_only

  # G. 旧版 FFT/Grad 组合（仅对照；须关掉 v2 增量项，再显式打开旧项）
  python train.py --base_ch 256 --no_cbam --hr_aux_mode stage1 --norm_type group \\
      --lambda_patch_extreme 0.0 --lambda_wps 0.0 --lambda_phys 0.0 \\
      --loss_gamma 0.5 --lambda_freq 0.1 --lambda_grad 0.05 \\
      --run_dir runs/ablation/G_legacy_fft_grad

  # H. v2 正式训练（推荐：架构锁定 + 损失默认值，与 launch_platform_train.sh 一致）
  python train.py --base_ch 256 --no_cbam --hr_aux_mode stage1 --norm_type group \\
      --ema_decay 0.999 --warmup_ratio 0.03 \\
      --run_dir runs/ablation/H_loss_v2_official

  # 同时打开所有 runs 的 TensorBoard：
  tensorboard --logdir runs/ablation

─────────────────────────────────────────────────────────────
"""

# 训练要点：v2 组合损失（面积加权 TailWeightedMAE + PatchExtreme + WPS + Phys；旧版 FFT/Grad/SpatialExtreme 默认关）、AdamW、线性 warmup + 余弦退火、梯度累积（默认 accum=2）、bf16 autocast、gradient checkpoint、TensorBoard 与 top-k 存盘。

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import re
import time
from pathlib import Path

# 必须在任何 CUDA 操作前设置，允许 PyTorch 内存池将"已释放但未归还"的显存
# 及时 cudaFree 回 CUDA，避免大 im2col workspace 释放后滞留造成 OOM。
# 用户可用环境变量覆盖（setdefault 不覆盖已有值）。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from dataset import DownscaleDataset, VARIABLES

# PRE 在写入 HDF5 前做了 log1p 变换（见 dataset.py / DOWNSCALE_README.md 第 2 节），
# 因此 z-score 反标准化后的 PRE 仍处于 log1p(mm/day) 空间，并非真实降水量纲。
# MAE_val/PRE 实际是 log1p 空间的误差，容易被误读为"物理精度"；validate() 中额外
# 对 PRE 通道做一次 expm1，计算 MAE_val_physical/PRE（mm/day，真实物理量纲）作为
# 补充诊断指标，不替换/不影响原有 Loss/val、MAE_val/* 等既有判据口径。
_PRE_IDX = VARIABLES.index("PRE")
from model import PixelShuffleDownscaleNet
from paths import HDF5_ROOT, STATIC_DIR, STATS_FILE


# ---------------------------------------------------------------------------
# Distributed training helpers (torchrun / srun + torch.distributed，NCCL/RCCL 后端)
# ---------------------------------------------------------------------------
#
# 平台参考：SCNet 用户手册《RDMA：使用高性能网络进行分布式训练》
# https://www.scnet.cn/help/docs/mainsite/ai/model-training/rdma/
#
# 设计原则：单卡运行行为完全不变（不依赖任何分布式环境变量时，下列函数均退化为
# no-op / 恒等操作），分布式仅在通过 torchrun（或 srun --ntasks-per-node=N 配合
# torchrun / torch.distributed.run）启动、环境变量 RANK/WORLD_SIZE 存在时才激活。

def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed() -> tuple[bool, int, int, int]:
    """若检测到 torchrun/torch.distributed.launch 注入的环境变量（RANK、WORLD_SIZE、
    LOCAL_RANK），则初始化分布式进程组；否则保持单进程运行不变。

    后端选择：有 GPU/DCU 时用 nccl（Hygon DCU 上由 DTK/RCCL 提供 NCCL 兼容 API，
    对应平台文档中 RDMA/IB 走 NET/IB 通道）；无加速卡时回退 gloo（仅用于 CPU 调试）。

    Returns:
        (是否处于分布式模式, rank, world_size, local_rank)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://",
                                 rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        print(f"[dist] rank={rank}/{world_size} local_rank={local_rank} "
              f"backend={backend} 初始化完成")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


def all_reduce_sum_(tensor: torch.Tensor) -> torch.Tensor:
    """就地对张量在所有进程间求和；非分布式模式下原样返回（no-op）。"""
    if is_dist_avail_and_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def unwrap_model(model: nn.Module) -> nn.Module:
    """返回底层（未被 torch.compile / DDP 包装的）模型，用于 state_dict 保存/加载/EMA。"""
    m = getattr(model, "_orig_mod", model)   # torch.compile 的 OptimizedModule → 原始 nn.Module
    return m.module if isinstance(m, DDP) else m


# ---------------------------------------------------------------------------
# EMA（模型权重指数滑动平均）
# ---------------------------------------------------------------------------

class ModelEMA:
    """指数滑动平均（EMA）模型权重，提升小 batch / BatchNorm 统计噪声下的验证与交付稳定性。

    维护一份与 unwrap_model(model) 结构相同的影子 state_dict（含参数与 buffer）。
    每次 optimizer.step() 后调用 update()：
        shadow = decay * shadow + (1 - decay) * online
    非浮点 buffer（如 BatchNorm 的 num_batches_tracked）直接拷贝，不参与滑动平均。

    验证/推理时用 apply_to() 上下文管理器临时把 EMA 权重换入模型（原地 copy_，DDP 下同样
    安全，因为不改变参数张量本身的对象引用），退出时自动还原为训练用的"在线"权重——
    不影响 optimizer/scheduler 状态，也不改变训练轨迹。
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, device: torch.device | None = None):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            k: (v.detach().clone().to(device) if device is not None else v.detach().clone())
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        for k, shadow_v in self.shadow.items():
            model_v = msd[k].detach()
            if shadow_v.dtype.is_floating_point:
                shadow_v.mul_(d).add_(model_v.to(shadow_v.device, shadow_v.dtype), alpha=1.0 - d)
            else:
                shadow_v.copy_(model_v)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        for k, v in self.shadow.items():
            if k in sd:
                v.copy_(sd[k])

    @contextlib.contextmanager
    def apply_to(self, model: nn.Module):
        """临时把 EMA 权重加载进 model，退出上下文时还原为进入前的在线权重。"""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=True)


class _NoOpWriter:
    """非主进程使用的空 SummaryWriter 替身，避免在训练循环中到处判断 rank。"""

    def add_scalar(self, *args, **kwargs) -> None:
        pass

    def add_text(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _load_area_weight_hr(static_dir: str | Path, target_h: int = 1801) -> torch.Tensor:
    """按纬度计算 cos(lat) 球面积权重，归一化到均值=1，用于训练损失的面积校正。

    背景（面向发表级评估的已知问题，见 DOWNSCALE_README.md 第 5 节）：
    规则经纬网格在高纬度单位像素代表的真实地表面积远小于赤道附近，像素等权的
    损失/评估会系统性放大高纬度（尤其极区）的权重。归一化到均值=1，保证叠加
    该权重后训练损失的整体量级不变（只改变像素间的相对权重分布），不影响与
    现有各项 λ 默认值的量级校准。

    仅作用于**训练损失**（TailWeightedMAE），不影响 Loss/val（早停/best.pt 判据，
    始终保持像素等权的纯 MAE，以维持与历史 run 的可比性，见 validate() 文档）。

    Args:
        static_dir: 包含 lat_hr.npy 的目录（与 dataset.py 的 STATIC_DIR 语义一致）
        target_h:   HR 目标网格高度（默认 1801），用于校验 lat_hr.npy 长度一致

    Returns:
        (1, 1, H, 1) float32 张量，可直接与 (B, C, H, W) 广播相乘
    """
    lat = np.load(Path(static_dir) / "lat_hr.npy").astype(np.float64)   # (H,)
    if lat.shape[0] != target_h:
        raise ValueError(
            f"lat_hr.npy 长度 {lat.shape[0]} 与 target_h={target_h} 不一致，"
            f"请检查 --static_dir 是否与模型 target_h 匹配"
        )
    w = np.cos(np.deg2rad(lat))
    w = np.clip(w, 1e-6, None)   # 极点 cos(90°)=0，避免出现权重恒为 0 的像素行
    w = w / w.mean()
    return torch.from_numpy(w.astype(np.float32)).view(1, 1, -1, 1)


class TailWeightedMAE(nn.Module):
    """振幅自适应 MAE：以 z-score 绝对值为权重，提升极端值像素的梯度贡献。

    当 γ=0 且 channel_weight、area_weight 均为 None（或全 1）时退化为标准 MAE
    （nn.L1Loss 等价），保底不退步。

    Args:
        gamma:  权重斜率（默认 0.5）。w = 1 + γ·clamp(|y|, 0, z_max)
                γ=0.5, z_max=3 → 极端值像素（|z|=3）权重 ×2.5
        z_max:  权重上限对应的 z-score（默认 3.0），避免少数超极端值主导梯度
        channel_weight: 逐变量（通道）固定权重，形状 (C,)，与 per-pixel 的 z-score
                权重相乘叠加。用于面向具体应用（如风光资源评估）让特定变量
                （如 wind10、FSDS）在尾部加权之外获得额外的梯度预算倾斜。
                None 或全 1 时不产生影响。
        area_weight: 球面积权重（cos(latitude)，归一化均值=1），形状可广播到
                (B, C, H, W)（典型为 `_load_area_weight_hr()` 返回的 (1,1,H,1)）。
                None 时不产生影响（等价于像素等权，与历史版本行为一致）。

    Note:
        权重 w 由 target（y_hr）计算并 detach()，不参与反向传播方向，
        仅改变各像素的梯度幅度。
    """

    def __init__(
        self,
        gamma: float = 0.5,
        z_max: float = 3.0,
        channel_weight: torch.Tensor | None = None,
        area_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.z_max = z_max
        if channel_weight is not None:
            self.register_buffer("channel_weight", channel_weight.float())
        else:
            self.channel_weight = None
        if area_weight is not None:
            self.register_buffer("area_weight", area_weight.float())
        else:
            self.area_weight = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.gamma == 0.0 and self.channel_weight is None and self.area_weight is None:
            return (pred - target).abs().mean()
        w = (1.0 + self.gamma * target.abs().clamp(max=self.z_max)).detach()
        if self.channel_weight is not None:
            w = w * self.channel_weight[None, :, None, None]
        if self.area_weight is not None:
            w = w * self.area_weight
        return (w * (pred - target).abs()).mean()


class FFTLoss(nn.Module):
    """2D rfft 幅度谱 L1 损失，直接监督频率域能量分布，改善过平滑。

    在幅度谱上施加 L1（而非 L2），对离群频率分量更鲁棒。
    ortho 归一化使幅度尺度与分辨率无关。

    计算在 float32 下进行（bf16 的 rfft2 精度不足），不影响主路径 bf16 训练。

    内存估算（B=1, C=8, H=1801, W=3600）：
        rfft2 输出 (1, 8, 1801, 1801) complex64 ≈ 417 MB（两个张量合计 ~834 MB），
        计算完毕后立即释放，不常驻显存。
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.float()
        t = target.float()
        pred_amp = torch.fft.rfft2(p, norm="ortho").abs()
        tgt_amp  = torch.fft.rfft2(t, norm="ortho").abs()
        return (pred_amp - tgt_amp).abs().mean()


class GradientLoss(nn.Module):
    """Sobel 梯度 L1 损失，强化空间边缘/锋面的局部一致性，改善过平滑。

    计算在 float32 下进行。Sobel 核以 buffer 注册，随模型一起 .to(device)。

    内存估算（B=1, C=8, H=1801, W=3600）：
        reshape 后 (8,1,1801,3600) float32 ×2（pred/target）×2（gx/gy）≈ 4×200 MB = 800 MB，
        计算完毕后立即释放，不常驻显存。
    """

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1., 0., -1.],
                            [2., 0., -2.],
                            [1., 0., -1.]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[1.,  2.,  1.],
                            [0.,  0.,  0.],
                            [-1., -2., -1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape
        p = pred.float().reshape(B * C, 1, H, W)
        t = target.float().reshape(B * C, 1, H, W)
        gx_p = F.conv2d(p, self.kx, padding=1)
        gy_p = F.conv2d(p, self.ky, padding=1)
        gx_t = F.conv2d(t, self.kx, padding=1)
        gy_t = F.conv2d(t, self.ky, padding=1)
        return (gx_p - gx_t).abs().mean() + (gy_p - gy_t).abs().mean()


class SpatialExtremeLoss(nn.Module):
    """区域空间极值一致性损失：对指定通道分别计算区域内最大值/最小值的 MAE。

    面向风光资源评估场景（参考 NREL Sup3rWind 等风资源超分辨率工作的做法）：
    逐像素 MAE/加权 MAE 只约束"点对点"误差，并不保证样本区域内的峰值（如风速
    极大值、辐照度晴空峰值）或低谷（如辐照度云遮骤降）被准确还原——这些恰恰是
    风光资源评估最关心的极值特征。本损失直接对预测/目标在 (H, W) 维度上的
    max/min 做 MAE，强制模型学习"不要把区域极值抹平"。

    计算在归一化（z-score）空间进行，与其余损失子项在同一量纲下可比。
    仅对 channel_indices 指定的通道生效（默认只有 wind10、FSDS），避免对
    不需要强调极值的变量引入不必要的约束。

    Args:
        channel_indices: 参与计算的通道下标列表（对应 VARIABLES 中的位置）
    """

    def __init__(self, channel_indices: list[int]):
        super().__init__()
        self.register_buffer(
            "idx", torch.tensor(channel_indices, dtype=torch.long)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.index_select(1, self.idx)      # (B, k, H, W)
        t = target.index_select(1, self.idx)
        p_max = p.amax(dim=(2, 3))               # (B, k)
        t_max = t.amax(dim=(2, 3))
        p_min = p.amin(dim=(2, 3))
        t_min = t.amin(dim=(2, 3))
        return (p_max - t_max).abs().mean() + (p_min - t_min).abs().mean()


class PatchExtremeLoss(nn.Module):
    """局部 patch 极值一致性损失（升级版 SpatialExtremeLoss，见 DOWNSCALE_README.md 第 5 节）。

    把"全球单一 max/min 标量"换成滑动网格局部 max/min/mean 场的一致性约束：
    对 --patch_extreme_vars 指定通道，把 HR 场 (H, W) 划分为 (grid_h, grid_w) 个
    自适应网格块（F.adaptive_max_pool2d / adaptive_avg_pool2d），逐块比较预测/
    目标的局部 max、min（用 `-adaptive_max_pool2d(-x)` 技巧获得，避免单独实现
    min_pool）、mean，三者 L1 损失之和。

    动机（相对旧版 SpatialExtremeLoss）：
      1. 监督密度从每图每通道 2 个非零梯度像素，提升到约 3×grid_h×grid_w 个，
         缓解"全球唯一极值点随机落在南极/局地地形异常点、与区域性风光资源
         特征无关"的问题；
      2. 新增的 patch-mean 项约束 patch **一阶矩**，抑制"为命中局部 max 而把
         整块预测整体抬升/压低"的粗糙退化解。成本极低，予以保留。
         **不声称缓解位置错位**：max+min+mean 仍完全不约束极值落在哪个像素；
         与主 MAE 也高度相关（patch 平均误差是逐点 MAE 的低通版本）。
      3. 确定性回归的内在张力：模型不知道次网格极值在哪，降低 L1(patch_max)
         的省力解往往是抬高整块，可能以逐点 MAE/偏差换极值统计。λ 偏大时
         冒烟应看 wind10/FSDS 的 patch 均值偏差（见 DOWNSCALE_README.md 第 5 节）。

    网格大小默认 (180, 360)：与 LR 输入网格**同分辨率近似对齐**（措辞：
    `adaptive_max_pool2d` 按张量下标均匀切分，不是真实球面 lat/lon 单元的
    严格逐格点映射，不做"严格一一对应"这类过度精确的表述）。

    计算在归一化（z-score）空间进行。

    Args:
        channel_indices: 参与计算的通道下标列表
        grid_h, grid_w:  局部网格大小（默认 180, 360，对齐 LR 分辨率）
    """

    def __init__(self, channel_indices: list[int], grid_h: int = 180, grid_w: int = 360):
        super().__init__()
        self.register_buffer("idx", torch.tensor(channel_indices, dtype=torch.long))
        self.grid_h = grid_h
        self.grid_w = grid_w

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred.index_select(1, self.idx)      # (B, k, H, W)
        t = target.index_select(1, self.idx)
        size = (self.grid_h, self.grid_w)

        p_max = F.adaptive_max_pool2d(p, size)
        t_max = F.adaptive_max_pool2d(t, size)
        p_min = -F.adaptive_max_pool2d(-p, size)
        t_min = -F.adaptive_max_pool2d(-t, size)
        p_mean = F.adaptive_avg_pool2d(p, size)
        t_mean = F.adaptive_avg_pool2d(t, size)

        return (
            (p_max - t_max).abs().mean()
            + (p_min - t_min).abs().mean()
            + (p_mean - t_mean).abs().mean()
        )


class WindPowerSensitivityProxy(nn.Module):
    """风功率敏感区代理损失（非能量产出对齐，命名与局限性务必对照
    DOWNSCALE_README.md 第 5 节阅读，不要在论文中单独脱离这些
    局限性引用本损失得出发电量/容量因子结论）。

    仅对 wind10 通道，在物理量纲（反标准化后）的风机"爬坡段"
    （v_cutin ≤ v ≤ v_rated，默认 3–12 m/s，IEC 典型三段式风机功率曲线的
    爬坡区间）内，对下式定义的归一化功率代理做 L1：

        P(v) = clip((v³ − v_cutin³) / (v_rated³ − v_cutin³), 0, 1)

    直接把风速误差按"对发电量的三次方非线性敏感度"重新分配梯度权重——
    这是逐点 MAE / z-score 尾部加权完全捕捉不到的非线性放大关系。

    **局限性声明（必须与该损失的任何结果一起报告）**：
      - `wind10` 是 10 米风速，不是风机轮毂高度（通常 80–120m）风速，二者之间
        存在随大气稳定度/地表粗糙度变化的换算关系，本损失未建模；
      - 训练样本是日尺度平均场，而功率曲线是瞬时非线性映射，根据 Jensen
        不等式 P(日均v) ≠ 日均P(瞬时v)，本损失不能被解释为直接优化容量因子
        或发电量误差；
      - v_cutin/v_rated 取 IEC 典型代理值，未针对具体机型标定。
      本损失仅作为"在气象学习目标中显式引入风速—功率非线性敏感度"的训练
      技巧，其下游能量代表性需要未来用轮毂高度/亚日数据做专门验证。

    mask 由**真值**风速判定（`.detach()`，避免用预测值自选择偏差，与
    TailWeightedMAE 用 target 计算权重的既有模式一致）。**mask 不含额定以上
    高风速**（真值 v>v_rated 时本项完全沉默）：立方段外 P(v) 为常数，高风速
    极值由 TailWeightedMAE 与 PatchExtremeLoss 监督。这是敏感区代理的设计
    取舍，不是"高风速不重要"。全球日均 10m 风速大量落在 3–12 m/s，覆盖率
    更可能过密；冒烟必须看 wps_mask_ratio，过高时考虑降低 λ_wps。

    数值稳定性：全程 float32 计算（含反标准化、v³、边界比较），不用 bf16
    （与 FFTLoss/GradientLoss 的既有约定一致，避免 bf16 精度不足导致 v³ 附近
    数值误差被放大）。

    诊断口径：forward 额外返回 mask 覆盖像素比例。损失值量级不能代表该项
    对参数的梯度贡献（dP/dv ∝ 3v²，爬坡段内可差一个量级以上）；隔离梯度
    范数不是训练默认行为，仅建议在冒烟阶段对 head 末层另算（见
    DOWNSCALE_README.md 第 5 节）。

    Args:
        wind_idx:  wind10 在 VARIABLES 中的通道下标
        wind_mean, wind_std: wind10 通道的 z-score 反标准化统计量（标量）
        v_cutin, v_rated:    功率代理曲线的爬坡段边界（m/s，默认 IEC 典型值 3, 12）
    """

    def __init__(
        self,
        wind_idx: int,
        wind_mean: float,
        wind_std: float,
        v_cutin: float = 3.0,
        v_rated: float = 12.0,
    ):
        super().__init__()
        self.wind_idx = wind_idx
        self.register_buffer("wind_mean", torch.tensor(float(wind_mean)))
        self.register_buffer("wind_std", torch.tensor(float(wind_std)))
        self.v_cutin = float(v_cutin)
        self.v_rated = float(v_rated)

    def _power_proxy(self, v_phys: torch.Tensor) -> torch.Tensor:
        num = v_phys.pow(3) - self.v_cutin ** 3
        den = self.v_rated ** 3 - self.v_cutin ** 3
        return (num / den).clamp(0.0, 1.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = pred[:, self.wind_idx:self.wind_idx + 1].float()
        t = target[:, self.wind_idx:self.wind_idx + 1].float()
        p_phys = p * self.wind_std + self.wind_mean
        t_phys = t * self.wind_std + self.wind_mean

        mask = ((t_phys >= self.v_cutin) & (t_phys <= self.v_rated)).float().detach()
        diff = (self._power_proxy(p_phys) - self._power_proxy(t_phys)).abs()
        mask_sum = mask.sum().clamp(min=1.0)
        loss = (diff * mask).sum() / mask_sum
        mask_ratio = mask.mean().detach()
        return loss, mask_ratio


class PhysicalConsistencyLoss(nn.Module):
    """物理一致性安全网损失：仅依赖预测本身（不需要真值）。

    **必须先反标准化，禁止在 z 空间直接写 hinge**（失败模式：TMIN_z≤TAS_z
    与物理顺序不等价；ReLU(−wind_z) 会惩罚低于全球平均风速的像素；RH−100
    会把物理常数混进 z 空间）。当前实现顺序：

      1. x_phys = pred · σ + μ（各通道独立；PRE 还原后仍是 log1p 空间，
         log1p(PRE)≥0 等价于物理降水 ≥0）
      2. 在该量纲上算 hinge
      3. hinge.mean() / σ_k 无量纲化后再对约束取平均
         ——这是"物理量纲 hinge / 该变量训练期 std"，不是 z 空间算完再除一次

    约束清单（均作用在 denorm(pred) 上）：
      - 温度顺序：TMIN ≤ TAS ≤ TMAX（物理 K）
      - 非负性：wind10, FSDS, Q, PRE（log1p 空间）
      - 相对湿度边界：RH ∈ [0, 100]（物理 %）

    权重应设得很小（安全网，不追求强约束）；训练收敛后各项违反率应自然趋近 0。
    多数像素 hinge=0 时损失均值很小，但违反像素上梯度是常数 1/σ，损失值
    ±3× 不能代表该项梯度贡献（见 DOWNSCALE_README.md 第 5 节）。

    Args:
        norm_mean, norm_std: 全 8 通道的 z-score 反标准化统计量，形状 (8,)

    Returns（forward，仅接收 pred，不需要 target）:
        (loss, violation_rates): violation_rates 是 dict[str, Tensor]，每条
        约束下 hinge>0 的像素占比（detach 标量），用于论文报告物理一致性
        改善程度，不参与训练目标本身。
    """

    def __init__(self, norm_mean: torch.Tensor, norm_std: torch.Tensor):
        super().__init__()
        self.register_buffer("norm_mean", norm_mean.float())
        self.register_buffer("norm_std", norm_std.float())
        self._idx = {v: VARIABLES.index(v) for v in VARIABLES}

    def forward(self, pred: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mean = self.norm_mean[None, :, None, None]
        std  = self.norm_std[None, :, None, None]
        p = pred.float() * std + mean   # (B, 8, H, W) 物理量纲

        i = self._idx
        tas  = p[:, i["TAS"]:i["TAS"] + 1]
        tmax = p[:, i["2M_TMAX"]:i["2M_TMAX"] + 1]
        tmin = p[:, i["2M_TMIN"]:i["2M_TMIN"] + 1]
        wind = p[:, i["wind10"]:i["wind10"] + 1]
        fsds = p[:, i["FSDS"]:i["FSDS"] + 1]
        q    = p[:, i["Q"]:i["Q"] + 1]
        pre  = p[:, i["PRE"]:i["PRE"] + 1]
        rh   = p[:, i["2M_RH"]:i["2M_RH"] + 1]

        std_ = self.norm_std
        eps = 1e-6

        hinges = {
            "tas_ge_tmin": (F.relu(tmin - tas), std_[i["TAS"]]),
            "tas_le_tmax": (F.relu(tas - tmax), std_[i["TAS"]]),
            "wind_nonneg": (F.relu(-wind), std_[i["wind10"]]),
            "fsds_nonneg": (F.relu(-fsds), std_[i["FSDS"]]),
            "q_nonneg":    (F.relu(-q),    std_[i["Q"]]),
            "pre_nonneg":  (F.relu(-pre),  std_[i["PRE"]]),
            "rh_upper":    (F.relu(rh - 100.0), std_[i["2M_RH"]]),
            "rh_lower":    (F.relu(-rh),        std_[i["2M_RH"]]),
        }

        terms = []
        violation_rates: dict[str, torch.Tensor] = {}
        for name, (hinge, scale) in hinges.items():
            terms.append(hinge.mean() / (scale + eps))
            violation_rates[name] = (hinge > 0).float().mean().detach()

        loss = torch.stack(terms).mean()
        return loss, violation_rates


class CombinedLoss(nn.Module):
    """训练用组合损失（v2，见 DOWNSCALE_README.md 第 5 节）：面积加权
    TailWeightedMAE（参考目标） + λ_pe·PatchExtremeLoss + λ_wps·
    WindPowerSensitivityProxy + λ_phys·PhysicalConsistencyLoss，另保留旧版
    λ_e·SpatialExtremeLoss + λ_f·FFTLoss + λ_g·GradientLoss（默认关闭，供对照
    消融）。

    当所有附加项权重为 0（lambda_freq=lambda_grad=lambda_extreme=
    lambda_patch_extreme=lambda_wps=lambda_phys=0，gamma=0，channel_weight/
    area_weight 均为 None）时，与 nn.L1Loss() 完全等价。

    **退化说明（措辞已按外部评审修正，见 DOWNSCALE_README.md 第 5 节）**：这只
    保证损失函数**数值**的退化，不构成"训练出的模型在其余变量上必然不退步"
    的形式化证明——是否退步需要看实际验证集结果。

    v2 默认配方（"参考目标 + 风光增量项"解耦设计）：
      - 参考目标：TailWeightedMAE（γ=0.5, z_max=3, channel_weight 默认全 1，
        area_weight 默认开启 cos(latitude) 面积加权）——8 通道等权，不存在
        通道间零和抢梯度，只有 z-score 尾部权重（按各像素自身极端程度计算）
        和面积权重（球面积校正）；
      - PatchExtremeLoss（默认 λ=0.1，wind10/FSDS）：局部 max/min 与 patch 一阶矩
        一致性（不约束极值位置；确定性回归下可能抬高整块，见该类文档）；
      - WindPowerSensitivityProxy（默认 λ=0.05，wind10）：仅 3–12 m/s 立方敏感区
        代理，**不是**能量产出对齐，额定以上本项沉默；
      - PhysicalConsistencyLoss（默认 λ=0.02）：先反标准化再无量纲化的物理安全网。

    旧版机制（SpatialExtremeLoss 全局极值 + 全通道/风光通道 FFT/Grad）默认
    权重为 0（关闭），仅保留供历史对照消融，不建议与 PatchExtremeLoss 同时
    对同一批变量启用（会重复施加极值监督）。

    各子项损失在 TensorBoard 中单独记录（Loss/tail_w、Loss/spatial_extreme、
    Loss/patch_extreme、Loss/freq、Loss/grad、Loss/wps、Loss/phys 等，通过
    forward 返回的 sub 字典由调用方通用记录，无需为新增键改动训练/验证循环）。

    验证/早停使用独立的 nn.L1Loss()（像素等权，不叠加 area_weight），保持与
    历史 run 的可比性，见 validate() 文档。

    Args:
        gamma:          TailWeightedMAE 的权重斜率（默认 0.5；0 = 不做 z-score 尾部加权）
        z_max:          权重截断 z-score（默认 3.0）
        channel_weight: TailWeightedMAE 的逐变量通道权重，形状 (C,)（默认 None = 全 1）
        area_weight:    TailWeightedMAE 的球面积权重（默认 None = 不加权；
                        典型用 `_load_area_weight_hr()` 生成）
        lambda_extreme: 旧版 SpatialExtremeLoss 权重（默认 0；0 = 关闭，仅供对照）
        extreme_channel_indices: SpatialExtremeLoss 参与的通道下标列表
        lambda_freq:    FFT Loss 权重（默认 0；0 = 关闭）
        lambda_grad:    Gradient Loss 权重（默认 0；0 = 关闭）
        lambda_patch_extreme: PatchExtremeLoss 权重（默认 0）
        patch_channel_indices: PatchExtremeLoss 参与的通道下标列表
        patch_grid:     PatchExtremeLoss 的局部网格大小 (grid_h, grid_w)
        lambda_wps:     WindPowerSensitivityProxy 权重（默认 0）
        wind_idx:       wind10 在 VARIABLES 中的通道下标
        wind_mean, wind_std: wind10 通道的反标准化统计量（标量）
        wind_cutin, wind_rated: 功率代理曲线的爬坡段边界（m/s）
        lambda_phys:    PhysicalConsistencyLoss 权重（默认 0）
        norm_mean, norm_std: 全 8 通道反标准化统计量，形状 (8,)（PhysicalConsistencyLoss 用）
    """

    def __init__(
        self,
        gamma: float = 0.5,
        z_max: float = 3.0,
        channel_weight: torch.Tensor | None = None,
        area_weight: torch.Tensor | None = None,
        lambda_extreme: float = 0.0,
        extreme_channel_indices: list[int] | None = None,
        lambda_freq: float = 0.0,
        lambda_grad: float = 0.0,
        lambda_patch_extreme: float = 0.0,
        patch_channel_indices: list[int] | None = None,
        patch_grid: tuple[int, int] = (180, 360),
        lambda_wps: float = 0.0,
        wind_idx: int | None = None,
        wind_mean: float | None = None,
        wind_std: float | None = None,
        wind_cutin: float = 3.0,
        wind_rated: float = 12.0,
        lambda_phys: float = 0.0,
        norm_mean: torch.Tensor | None = None,
        norm_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.tail_loss  = TailWeightedMAE(
            gamma=gamma, z_max=z_max, channel_weight=channel_weight, area_weight=area_weight
        )
        self.fft_loss   = FFTLoss()   if lambda_freq > 0 else None
        self.grad_loss  = GradientLoss() if lambda_grad > 0 else None
        self.extreme_loss = (
            SpatialExtremeLoss(extreme_channel_indices)
            if lambda_extreme > 0 and extreme_channel_indices
            else None
        )
        self.patch_extreme_loss = (
            PatchExtremeLoss(patch_channel_indices, grid_h=patch_grid[0], grid_w=patch_grid[1])
            if lambda_patch_extreme > 0 and patch_channel_indices
            else None
        )
        self.wps_loss = (
            WindPowerSensitivityProxy(
                wind_idx, wind_mean, wind_std, v_cutin=wind_cutin, v_rated=wind_rated
            )
            if lambda_wps > 0 and wind_idx is not None
            else None
        )
        self.phys_loss = (
            PhysicalConsistencyLoss(norm_mean, norm_std)
            if lambda_phys > 0 and norm_mean is not None and norm_std is not None
            else None
        )

        self.lambda_freq          = lambda_freq
        self.lambda_grad          = lambda_grad
        self.lambda_extreme       = lambda_extreme
        self.lambda_patch_extreme = lambda_patch_extreme
        self.lambda_wps           = lambda_wps
        self.lambda_phys          = lambda_phys

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """返回 (total_loss, sub_losses_dict)。

        sub_losses_dict 键：'tail_w'、'spatial_extreme'/'patch_extreme'/'freq'/
        'grad'/'wps'/'wps_mask_ratio'/'phys'/'phys_violation_<约束名>'（均可选，
        取决于对应 λ 是否 >0）。TensorBoard 记录由调用方完成（避免在 loss
        forward 中直接写 writer），且调用方已用通用的 `for k, v in sub.items()`
        循环记录，新增键无需改动训练/验证主循环代码。
        """
        l_tail = self.tail_loss(pred, target)
        sub = {"tail_w": l_tail.item()}
        total = l_tail

        if self.extreme_loss is not None:
            l_extreme = self.extreme_loss(pred, target)
            total     = total + self.lambda_extreme * l_extreme
            sub["spatial_extreme"] = l_extreme.item()

        if self.patch_extreme_loss is not None:
            l_patch = self.patch_extreme_loss(pred, target)
            total   = total + self.lambda_patch_extreme * l_patch
            sub["patch_extreme"] = l_patch.item()

        if self.fft_loss is not None:
            l_freq = self.fft_loss(pred, target)
            total  = total + self.lambda_freq * l_freq
            sub["freq"] = l_freq.item()

        if self.grad_loss is not None:
            l_grad = self.grad_loss(pred, target)
            total  = total + self.lambda_grad * l_grad
            sub["grad"] = l_grad.item()

        if self.wps_loss is not None:
            l_wps, wps_mask_ratio = self.wps_loss(pred, target)
            total = total + self.lambda_wps * l_wps
            sub["wps"] = l_wps.item()
            sub["wps_mask_ratio"] = wps_mask_ratio.item()

        if self.phys_loss is not None:
            l_phys, phys_violation = self.phys_loss(pred)
            total = total + self.lambda_phys * l_phys
            sub["phys"] = l_phys.item()
            for name, rate in phys_violation.items():
                sub[f"phys_violation_{name}"] = rate.item()

        return total, sub


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """固定 Python/NumPy/PyTorch 种子，便于数据划分与训练复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    warmup_start_ratio: float = 0.01,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay.

    Warmup (scheduler steps ``0 .. warmup_steps-1``):
        multiplier goes linearly from ``warmup_start_ratio`` → ``1.0``

    Cosine (``warmup_steps .. total_steps-1``):
        multiplier goes from ``1.0`` → ``min_lr_ratio``
        i.e. ``eta_min + (1 - eta_min) * 0.5 * (1 + cos(pi * progress))``.
    """
    # 学习率倍率：先线性从 warmup_start_ratio 升到 1，再余弦降到 min_lr_ratio

    ws = max(0.0, min(1.0, warmup_start_ratio))
    em = max(0.0, min(1.0, min_lr_ratio))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            if warmup_steps <= 1:
                return 1.0
            t = current_step / (warmup_steps - 1)
            return ws + (1.0 - ws) * t
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return em + (1.0 - em) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Resource checkpoint baseline（修复 best_resource.pt 的量纲问题，见
# DOWNSCALE_README.md 第 6 节）
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_baseline_resource_mae(
    loader: DataLoader,
    device: torch.device,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    resource_var_indices: list[int],
) -> dict[str, float]:
    """计算"naive baseline"（LR 输入双线性插值到 HR 分辨率）在 --resource_vars
    指定变量上的物理量纲 MAE，用作 best_resource.pt 的 skill score 归一化基准。

    修复的问题：原实现把 wind10（m/s）与 FSDS（W/m²）的物理量纲 MAE 直接算术
    平均作为"资源最优"判据——两个不同物理量纲的数值不可加，且量纲更大的一方
    会系统性主导结果，不能解释为"资源最优"（外部评审指出，核实后确认是当前
    代码库既有缺陷，见 DOWNSCALE_README.md 第 6 节）。

    x_lr 与 y_hr 用同一组全局 mean/std 做 z-score（dataset.py 已确认一致），
    因此 baseline 反标准化用与验证时相同的 norm_mean/std 即可，不需要额外统计量。

    只需在训练开始前算一次（不随 epoch 变化），因此不缓存进 checkpoint（每次
    进程启动都重算一次的成本等价于一次额外的验证轮次，相对全量训练总成本可
    忽略，换来实现更简单、不需要改动 save_checkpoint/load_checkpoint 字段）。

    Returns:
        {变量名: 物理量纲 MAE}，仅包含 resource_var_indices 对应的变量；
        resource_var_indices 为空时返回空字典。
    """
    if not resource_var_indices:
        return {}

    mn = norm_mean.to(device)[None, :, None, None]
    sd = norm_std.to(device)[None, :, None, None]

    sums = torch.zeros(len(resource_var_indices), dtype=torch.float64, device=device)
    n = 0
    for x_lr, _hr_aux, y_hr in loader:
        x_lr = x_lr.to(device, non_blocking=True)
        y_hr = y_hr.to(device, non_blocking=True)

        # naive baseline：LR 输入直接双线性插值到 HR 分辨率，与模型输出同形状，
        # 对齐方式（align_corners=True）与模型内部的双线性插值约定一致
        baseline_pred = F.interpolate(
            x_lr, size=y_hr.shape[-2:], mode="bilinear", align_corners=True
        )

        pred_dn = baseline_pred.float() * sd + mn
        y_dn    = y_hr.float()          * sd + mn
        mae = (pred_dn - y_dn).abs().mean(dim=(0, 2, 3))   # (8,)

        for j, idx in enumerate(resource_var_indices):
            sums[j] += mae[idx].double() * x_lr.size(0)
        n += x_lr.size(0)

    n_tensor = torch.tensor([float(n)], dtype=torch.float64, device=device)
    all_reduce_sum_(sums)
    all_reduce_sum_(n_tensor)
    n = n_tensor.item()

    result = {
        VARIABLES[idx]: (sums[j] / max(1.0, n)).item()
        for j, idx in enumerate(resource_var_indices)
    }
    if is_main_process():
        base_str = ", ".join(f"{k}={v:.4f}" for k, v in result.items())
        print(f"[resource baseline] 双线性插值 naive baseline 物理量纲 MAE：{base_str}"
              f"（用于 skill score 归一化，n_samples={int(n)}）")
    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    writer: SummaryWriter,
    global_step: int,
    epoch: int,
    criterion: "CombinedLoss | None" = None,
    extreme_z_thresh: float = 1.5,
    resource_var_indices: list[int] | None = None,
    baseline_resource_mae: dict[str, float] | None = None,
    area_weight: torch.Tensor | None = None,
) -> tuple[float, float | None]:
    """验证模型并记录指标。返回 (mean_loss, resource_metric)。

    mean_loss：平均纯 MAE 验证损失（与历史 run 可比较）。
    始终使用纯 L1Loss 作为 Loss/val（早停 / top-K / best.pt 的判据），
    与训练用组合损失解耦，保证跨 run（包括纯 MAE / 组合损失等不同配置）的可比性。
    这是模型选择的**唯一**判据，不受 --resource_vars 等业务侧配置影响，也
    **不**叠加 area_weight（外部评审建议过"验证指标同步面积加权"，但与本项目
    "Loss/val 必须与历史 run 严格可比"的既有原则冲突，权衡后保留 Loss/val 不变，
    面积加权版本作为下方的新增诊断指标 MAE_val_area_weighted/* 单独提供，
    见 DOWNSCALE_README.md 第 6 节）。

    resource_metric：`resource_var_indices` 指定变量的 **skill score** 均值
    （`1 − MAE_model_var / MAE_baseline_var`，`MAE_baseline_var` 来自
    `compute_baseline_resource_mae()` 的双线性插值 naive baseline，无量纲、
    可跨变量合成，修复了原实现把 wind10 的 m/s MAE 与 FSDS 的 W/m² MAE 直接
    算术平均、无物理可解释性的问题）。**仅作为并行的业务侧参考指标**（用于
    额外维护一份 best_resource.pt，见 main() 中的用法），不参与、也不替代
    上面的早停/top-K/best.pt 判据。`baseline_resource_mae` 为空或
    `resource_var_indices` 为空时返回 None。

    在此基础上，额外记录两类"完整对比"指标，用于评估组合损失是否真的比纯 MAE
    在其设计目标上更优（而不是仅比较对两者都不公平的整体 MAE）：

      1. 组合损失各子项在验证集上的取值（Loss/val_combined、Loss/val_tail_w、
         Loss/val_spatial_extreme、Loss/val_patch_extreme、Loss/val_freq、
         Loss/val_grad、Loss/val_wps、Loss/val_phys 等）—— 用与训练完全相同的
         criterion 计算，直接反映该 run 的损失配置在其自身优化目标上的验证
         表现。对应权重为 0 的子项会退化/缺失，但仍可跨 run 对比，判断风光
         资源极值/功率敏感区/物理一致性是否真的因新损失机制而改善。

      2. 极端值 MAE（MAE_val_extreme/<var>，物理量纲）—— 仅在 |y_zscore| >
         extreme_z_thresh 的像素上统计 MAE，直接衡量"极端方面"的表现，
         这正是 TailWeightedMAE 设计要优化的区域，也是纯 MAE 整体指标
         无法体现、容易被大量"平静"像素稀释掉的部分。

      3.（新增）面积加权 MAE 诊断（MAE_val_area_weighted/<var>）—— 与训练损失
         的 cos(latitude) 面积权重口径一致，供论文报告用，不参与模型选择。
    """
    val_criterion = nn.L1Loss()
    model.eval()
    total_loss = 0.0
    per_var_mae = torch.zeros(len(VARIABLES), dtype=torch.float64)
    n_samples = 0

    # 组合损失子项（与训练时完全相同的 criterion，用同一份 gamma/z_max/lambda 计算）
    track_combined = criterion is not None
    combined_total = 0.0
    sub_totals: dict[str, float] = {}

    # 极端值 MAE：按变量分别累计 sum 和像素计数（避免不同 batch 极端像素占比不同带来偏差）
    extreme_mae_sum   = torch.zeros(len(VARIABLES), dtype=torch.float64)
    extreme_pix_count = torch.zeros(len(VARIABLES), dtype=torch.float64)

    # PRE 物理量纲（mm/day，expm1 还原后）补充 MAE，见上方 _PRE_IDX 注释
    pre_physical_mae_sum = 0.0

    # 面积加权 MAE 诊断（cos(latitude) 口径，与训练损失一致；仅诊断，不参与模型选择）
    area_weighted_mae_sum = torch.zeros(len(VARIABLES), dtype=torch.float64)
    _aw = area_weight.to(device) if area_weight is not None else None

    for x_lr, hr_aux, y_hr in loader:
        x_lr   = x_lr.to(device,  non_blocking=True)
        hr_aux = hr_aux.to(device, non_blocking=True)
        y_hr   = y_hr.to(device,  non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(x_lr, hr_aux)
            loss = val_criterion(pred, y_hr)
            if track_combined:
                c_loss, c_sub = criterion(pred, y_hr)

        total_loss += loss.item() * x_lr.size(0)
        if track_combined:
            combined_total += c_loss.item() * x_lr.size(0)
            for k, v in c_sub.items():
                sub_totals[k] = sub_totals.get(k, 0.0) + v * x_lr.size(0)

        # 验证集上反标准化，按变量统计 MAE（物理量纲，便于解读）
        mn = norm_mean.to(device)[None, :, None, None]   # (1, 8, 1, 1)
        sd = norm_std.to(device)[None, :, None, None]
        pred_dn = pred.float() * sd + mn
        y_dn    = y_hr.float()  * sd + mn
        mae = (pred_dn - y_dn).abs().mean(dim=(0, 2, 3))   # (8,)
        per_var_mae += mae.cpu().double() * x_lr.size(0)
        n_samples   += x_lr.size(0)

        # PRE 物理量纲（mm/day）补充指标：z-score 反标准化后 PRE 仍在 log1p 空间，
        # 需再做一次 expm1 才是真实降水量纲；clamp 防止早期训练不稳定预测导致
        # expm1 数值溢出（log1p(mm/day) 正常范围远小于 30，30 对应 ~1e13 mm/day）。
        pre_pred_phys = torch.expm1(pred_dn[:, _PRE_IDX:_PRE_IDX + 1].clamp(max=30.0))
        pre_y_phys    = torch.expm1(y_dn[:,    _PRE_IDX:_PRE_IDX + 1].clamp(max=30.0))
        pre_physical_mae_sum += (pre_pred_phys - pre_y_phys).abs().mean().item() * x_lr.size(0)

        # 极端值掩码基于 z-score（与 TailWeightedMAE / loss_z_max 同一空间），
        # 逐变量统计掩码内像素的绝对误差（物理量纲）与像素数
        abs_err = (pred_dn - y_dn).abs()                       # (B, 8, H, W)
        extreme_mask = y_hr.abs() > extreme_z_thresh            # (B, 8, H, W)
        masked_err = torch.where(extreme_mask, abs_err, torch.zeros_like(abs_err))
        extreme_mae_sum   += masked_err.sum(dim=(0, 2, 3)).cpu().double()
        extreme_pix_count += extreme_mask.sum(dim=(0, 2, 3)).cpu().double()

        # 面积加权 MAE 诊断（仅诊断指标，不参与早停/best.pt；area_weight 已归一化
        # 均值=1，因此 (abs_err * aw).mean() 是对该口径下的加权平均的有效近似，
        # 与训练损失 TailWeightedMAE 里叠加 area_weight 后仍取 .mean() 的做法一致）
        if _aw is not None:
            area_weighted_mae_sum += (abs_err * _aw).mean(dim=(0, 2, 3)).cpu().double() * x_lr.size(0)

    # ------------------------------------------------------------------
    # 分布式聚合：各 rank 用 DistributedSampler 分到互不重叠的验证子集，
    # 因此各累计量需先跨进程求和，再统一在其上计算均值/比例；否则各 rank
    # 各自算出的"局部均值"再简单平均会因各 rank 样本数/极端像素数不同而有偏。
    # 非分布式模式下 all_reduce_sum_ 为 no-op，数值与聚合前完全一致。
    # ------------------------------------------------------------------
    _agg_device = device if torch.cuda.is_available() else torch.device("cpu")
    scalar_keys = sorted(sub_totals.keys())  # 固定顺序，便于与 all_reduce 后的张量对应
    scalars = torch.tensor(
        [total_loss, combined_total, float(n_samples), pre_physical_mae_sum]
        + [sub_totals[k] for k in scalar_keys],
        dtype=torch.float64, device=_agg_device,
    )
    all_reduce_sum_(scalars)
    total_loss, combined_total, n_samples_f, pre_physical_mae_sum = (
        scalars[0].item(), scalars[1].item(), scalars[2].item(), scalars[3].item()
    )
    n_samples = int(round(n_samples_f))
    sub_totals = {k: scalars[4 + i].item() for i, k in enumerate(scalar_keys)}

    per_var_mae = per_var_mae.to(_agg_device)
    extreme_mae_sum = extreme_mae_sum.to(_agg_device)
    extreme_pix_count = extreme_pix_count.to(_agg_device)
    area_weighted_mae_sum = area_weighted_mae_sum.to(_agg_device)
    all_reduce_sum_(per_var_mae)
    all_reduce_sum_(extreme_mae_sum)
    all_reduce_sum_(extreme_pix_count)
    all_reduce_sum_(area_weighted_mae_sum)
    per_var_mae = per_var_mae.cpu()
    extreme_mae_sum = extreme_mae_sum.cpu()
    extreme_pix_count = extreme_pix_count.cpu()
    area_weighted_mae_sum = area_weighted_mae_sum.cpu()

    mean_loss    = total_loss / max(1, n_samples_f)
    per_var_mae /= max(1, n_samples_f)
    area_weighted_mae = area_weighted_mae_sum / max(1, n_samples_f)
    mean_pre_physical_mae = pre_physical_mae_sum / max(1, n_samples_f)

    writer.add_scalar("Loss/val", mean_loss, global_step)
    writer.add_scalar("MAE_val_physical/PRE", mean_pre_physical_mae, global_step)
    if is_main_process():
        print(f"  [epoch {epoch:03d} val]  loss(pure_mae)={mean_loss:.6f}  "
              f"(n_samples={n_samples}, world_size={get_world_size()})")
        print(f"  [epoch {epoch:03d} val]  PRE physical(mm/day, expm1还原)_MAE="
              f"{mean_pre_physical_mae:.4f}  (对照 MAE_val/PRE={per_var_mae[_PRE_IDX].item():.4f}"
              f" 为 log1p 空间误差)")

    if track_combined:
        mean_combined = combined_total / max(1, n_samples_f)
        writer.add_scalar("Loss/val_combined", mean_combined, global_step)
        sub_str_parts = []
        for k, v_sum in sub_totals.items():
            v_mean = v_sum / max(1, n_samples_f)
            writer.add_scalar(f"Loss/val_{k}", v_mean, global_step)
            sub_str_parts.append(f"{k}={v_mean:.6f}")
        if is_main_process():
            print(f"  [epoch {epoch:03d} val]  loss(combined)={mean_combined:.6f}  "
                  f"({', '.join(sub_str_parts)})")

    per_var_extreme_mae = extreme_mae_sum / extreme_pix_count.clamp(min=1.0)
    valid_extreme = extreme_pix_count > 0
    if valid_extreme.any():
        mean_extreme_mae = per_var_extreme_mae[valid_extreme].mean().item()
    else:
        mean_extreme_mae = float("nan")
    writer.add_scalar("MAE_val_extreme/mean", mean_extreme_mae, global_step)
    if is_main_process():
        print(f"  [epoch {epoch:03d} val]  extreme(|z|>{extreme_z_thresh:g})_MAE_mean="
              f"{mean_extreme_mae:.4f}")

    for i, vname in enumerate(VARIABLES):
        mae_val = per_var_mae[i].item()
        writer.add_scalar(f"MAE_val/{vname}", mae_val, global_step)
        extreme_val = per_var_extreme_mae[i].item()
        n_extreme   = extreme_pix_count[i].item()
        writer.add_scalar(f"MAE_val_extreme/{vname}", extreme_val, global_step)
        if _aw is not None:
            writer.add_scalar(f"MAE_val_area_weighted/{vname}", area_weighted_mae[i].item(), global_step)
        if is_main_process():
            print(f"    {vname:>10s}: mae={mae_val:.4f}  "
                  f"extreme_mae={extreme_val:.4f} (n_pix={int(n_extreme)})")

    # resource_metric：skill score 均值（修复量纲错误算术平均，见
    # compute_baseline_resource_mae() 文档与 DOWNSCALE_README.md 第 6 节）。
    # >0 表示优于"LR 双线性插值"这个 naive baseline；数值越大越好（与旧版
    # "resource_metric 越小越好"的方向相反，main() 中 best_resource_val 的
    # 初值与比较方向已同步反转）。
    resource_metric: float | None = None
    if resource_var_indices and baseline_resource_mae:
        skills = []
        for i in resource_var_indices:
            vname = VARIABLES[i]
            base_mae = baseline_resource_mae.get(vname)
            if base_mae and base_mae > 0:
                skill = 1.0 - per_var_mae[i].item() / base_mae
            else:
                skill = float("nan")
            writer.add_scalar(f"Skill_val/{vname}", skill, global_step)
            skills.append(skill)
        if skills:
            resource_metric = float(np.mean(skills))
            writer.add_scalar("Skill_val_resource/mean", resource_metric, global_step)
            if is_main_process():
                names = ", ".join(VARIABLES[i] for i in resource_var_indices)
                skill_str = ", ".join(f"{VARIABLES[i]}={s:.4f}" for i, s in zip(resource_var_indices, skills))
                print(f"  [epoch {epoch:03d} val]  skill_score({names})_mean={resource_metric:.4f}  "
                      f"({skill_str})  （业务侧参考指标，相对双线性插值 baseline，不参与 best.pt 判据）")

    model.train()
    return mean_loss, resource_metric


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_EPOCH_CKPT_RE = re.compile(r"^epoch_\d+_valloss([0-9.]+)\.pt$")


def _torch_load(path: Path, device: torch.device) -> object:
    """优先 weights_only=True；旧版 PyTorch 或不支持的 checkpoint 类型再回退。"""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception:
        return torch.load(path, map_location=device)


def rebuild_saved_ckpts(ckpt_dir: Path) -> list[tuple[float, Path]]:
    """从磁盘重建 top-K 列表（不含 latest.pt / best.pt / best_resource.pt）。

    `--resume` 后内存中的 saved_ckpts 是空的，若不扫描已有 epoch_*.pt，后续
    top-K 淘汰无法删除 resume 前落盘的文件，磁盘上会越积越多。
    """
    found: list[tuple[float, Path]] = []
    if not ckpt_dir.is_dir():
        return found
    for p in ckpt_dir.glob("epoch_*_valloss*.pt"):
        m = _EPOCH_CKPT_RE.match(p.name)
        if m is None:
            continue
        try:
            found.append((float(m.group(1)), p))
        except ValueError:
            continue
    found.sort(key=lambda t: t[0])
    return found


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    ema: "ModelEMA | None" = None,
    early_stop_streak: int = 0,
    best_resource_val: float = float("-inf"),
) -> None:
    """保存 epoch、全局步数、最优验证损失及优化器/调度器状态，便于断点续训。

    分布式训练下仅 rank0 应调用本函数（由调用方保证）；`model` 若为 torch.compile/DDP
    包装，此处自动解包，保存的 state_dict 与单卡训练完全一致（不含 "module." 前缀、
    不含 compile 包装层），确保 checkpoint 可在单卡 / 任意 world_size / 是否 compile
    下互相加载。

    ema 非 None 时额外保存 "model_ema"（EMA 影子权重，供部署/推理使用；"model" 字段
    仍是训练用的"在线"权重，用于正确恢复训练轨迹，二者不要混淆）。

    early_stop_streak / best_resource_val 必须一并保存：平台中断后从 latest.pt
    续训时，若这两项从初值重新开始，早停耐心会被放大，且第一次验证就会覆盖
    写 best_resource.pt（即使 skill 比历史最优更差）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch":              epoch,
        "global_step":        global_step,
        "best_val_loss":      best_val_loss,
        "early_stop_streak":  int(early_stop_streak),
        "best_resource_val":  float(best_resource_val),
        "model":              unwrap_model(model).state_dict(),
        "optimizer":          optimizer.state_dict(),
        "scheduler":          scheduler.state_dict(),
    }
    if ema is not None:
        ckpt["model_ema"] = ema.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    ema: "ModelEMA | None" = None,
) -> tuple[int, int, float, int, float]:
    """加载检查点。返回 (start_epoch, global_step, best_val_loss, early_stop_streak, best_resource_val)。

    `model` 若为 torch.compile/DDP 包装，state_dict 加载到底层原始模块（unwrap_model），
    与保存时的裸模型 state_dict 格式对应；各 rank 均需调用本函数以保持模型/
    优化器/调度器状态一致（DDP 要求所有进程参数初始值相同）。

    ema 非 None 时尝试恢复 EMA 影子权重；若 checkpoint 是旧版（未启用过 EMA、无
    "model_ema" 字段），则用当前（刚加载完 "model" 的）在线权重重新初始化 EMA，
    而不是报错中断——语义等价于"从这个 epoch 开始才启用 EMA"。

    旧 checkpoint 若无 early_stop_streak / best_resource_val，分别回退为 0 / -inf，
    并打印警告，不中断加载。
    """
    ckpt = _torch_load(path, device)
    unwrap_model(model).load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if ema is not None:
        if "model_ema" in ckpt:
            ema.load_state_dict(ckpt["model_ema"])
        else:
            if is_main_process():
                print("[EMA] checkpoint 中无 model_ema（此前未启用 EMA），"
                      "已用当前在线权重重新初始化 EMA")
            ema.load_state_dict(unwrap_model(model).state_dict())
    early_stop_streak = int(ckpt.get("early_stop_streak", 0))
    best_resource_val = float(ckpt.get("best_resource_val", float("-inf")))
    if is_main_process():
        print(f"Resumed: epoch={ckpt['epoch']+1}, step={ckpt['global_step']}, "
              f"best_val={ckpt['best_val_loss']:.6f}, "
              f"early_stop_streak={early_stop_streak}, "
              f"best_resource_val={best_resource_val:.6f}")
        if "early_stop_streak" not in ckpt or "best_resource_val" not in ckpt:
            print("[resume] 旧 checkpoint 无 early_stop_streak/best_resource_val，"
                  "早停计数从 0 开始，best_resource.pt 判据从 -inf 重新计")
    return (
        ckpt["epoch"] + 1,
        ckpt["global_step"],
        ckpt["best_val_loss"],
        early_stop_streak,
        best_resource_val,
    )


def _optimizer_step_is_finite(
    loss: torch.Tensor,
    grad_norm: float | torch.Tensor,
    device: torch.device,
) -> bool:
    """loss 与梯度范数均有限才允许 optimizer.step。

    DDP 下任一 rank 非有限则全体跳过（MIN 聚合），避免有的 rank step、有的
    rank 跳过导致参数/NCCL 失步。backward 必须仍在所有 rank 上执行完毕，
    不能在 backward 前按本地 loss 单独 return。
    """
    loss_ok = math.isfinite(float(loss.detach().float().cpu()))
    try:
        norm_ok = math.isfinite(float(grad_norm))
    except (TypeError, ValueError):
        norm_ok = False
    flag = torch.tensor([1.0 if (loss_ok and norm_ok) else 0.0], device=device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item() > 0.5)


# ---------------------------------------------------------------------------
# Variable-name helpers (--var_weights / --extreme_vars 解析)
# ---------------------------------------------------------------------------

def parse_var_weights(spec: str) -> torch.Tensor:
    """解析 'VAR=weight,VAR=weight,...' 为按 VARIABLES 顺序排列的 (C,) 权重张量。

    未在 spec 中出现的变量权重默认为 1.0（中性）。变量名必须属于 VARIABLES，
    否则抛出 ValueError 并列出合法变量名，方便排查拼写错误。
    """
    weights = {v: 1.0 for v in VARIABLES}
    spec = spec.strip()
    if not spec:
        return torch.tensor([weights[v] for v in VARIABLES], dtype=torch.float32)

    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"--var_weights 格式错误：'{item}' 缺少 '='（期望 'VAR=weight'）"
            )
        name, val = item.split("=", 1)
        name = name.strip()
        if name not in weights:
            raise ValueError(
                f"--var_weights 中的变量名 '{name}' 不合法，合法变量为：{VARIABLES}"
            )
        weights[name] = float(val.strip())

    return torch.tensor([weights[v] for v in VARIABLES], dtype=torch.float32)


def parse_var_list(spec: str) -> list[int]:
    """解析 'VAR,VAR,...' 为对应 VARIABLES 中的下标列表。空字符串返回空列表。"""
    spec = spec.strip()
    if not spec:
        return []
    indices = []
    for item in spec.split(","):
        name = item.strip()
        if not name:
            continue
        if name not in VARIABLES:
            raise ValueError(
                f"--extreme_vars 中的变量名 '{name}' 不合法，合法变量为：{VARIABLES}"
            )
        indices.append(VARIABLES.index(name))
    return indices


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PixelShuffleDownscaleNet")

    # data
    p.add_argument("--hdf5_root",   default=str(HDF5_ROOT),   type=str)
    p.add_argument("--static_dir",  default=str(STATIC_DIR),  type=str)
    p.add_argument("--stats_file",  default=str(STATS_FILE), type=str)
    p.add_argument("--seasons",     nargs="+", default=["MAM", "JJA", "SON", "DJF"])
    p.add_argument(
        "--manifests", nargs="+", default=None,
        help=(
            "可选，允许的 shard batch_tag 白名单（如 --manifests cra1p5_full）。"
            "shard 文件名须形如 shard_{tag}_{序号}.h5；传入后仅纳入 tag 在白名单内的 "
            "shard，用于混合多批次数据时排除某个坏批次。默认不传 = 不过滤，纳入 "
            "hdf5_root 下所有 shard_*.h5（见 dataset.py _build_index）。"
        ),
    )
    p.add_argument("--val_fraction", default=0.1, type=float,
                   help="fraction of samples reserved for validation (default: 0.1)")
    p.add_argument("--num_workers", default=4, type=int)

    # model architecture
    p.add_argument("--base_ch",       default=256, type=int,
                   help="feature channels throughout all stages (default: 256). "
                        "Stage 4 shuffle_conv uses 1×1 (no im2col workspace), enabling base_ch=256 "
                        "on RTX 6000 Ada (47.51 GiB).")
    p.add_argument("--num_resblocks", default=2,   type=int,
                   help="ResBlocks per UpStage (default: 2)")
    # ablation switches
    p.add_argument("--no_cbam", dest="use_cbam", action="store_false",
                   help="disable CBAM attention in all UpStages")
    p.set_defaults(use_cbam=True)
    p.add_argument("--no_checkpoint", dest="use_checkpoint", action="store_false",
                   help="disable gradient checkpointing (default: enabled for all 4 UpStages)")
    p.set_defaults(use_checkpoint=True)
    p.add_argument(
        "--compile",
        action="store_true",
        help=(
            "用 torch.compile() 包装模型（PyTorch 2.x，当前环境 2.5.1），部分场景下可带来 "
            "10%%~30%% 训练加速。默认关闭：与 gradient checkpointing / DDP / 本平台 "
            "DCU+RCCL 后端的组合兼容性依赖具体 PyTorch/驱动版本，建议先在小规模冒烟测试 "
            "上验证前向/反向/loss 数值与不开启时一致、且确无报错后，再用于正式训练。"
        ),
    )
    p.add_argument(
        "--hr_aux_mode",
        default="all",
        choices=["all", "stage1", "none"],
        help=(
            "HR-aux injection strategy (default: all):\n"
            "  all    — inject at Stage 1-3 + concat at Head (full model)\n"
            "  stage1 — inject only at Stage 1 (360×720), Stage 2-3 & Head skipped\n"
            "  none   — no injection anywhere; hr_aux is passed but ignored"
        ),
    )
    p.add_argument(
        "--norm_type",
        default="batch",
        choices=["batch", "group"],
        help=(
            "ResBlock/InitConv/Head 的归一化层类型（默认 batch=nn.BatchNorm2d，与历史版本"
            "一致）。group=nn.GroupNorm，按通道分组在单样本内部统计，不依赖 batch 维，"
            "对 --batch_size 较小（尤其单卡/单进程 batch=1）场景更稳健，避免 BN 统计噪声"
            "导致的 loss 抖动（见 DOWNSCALE_README.md 第 4 节）；代价是与 'batch' 版本的 "
            "checkpoint 不兼容（层结构不同），需从头训练。分布式下 --sync_bn 仅对 "
            "norm_type=batch 生效（GroupNorm 本身不依赖跨进程同步）。"
        ),
    )
    p.add_argument(
        "--interp_chunk_channels",
        default=32,
        type=int,
        help=(
            "Stage4→Head 前把 256 通道整体做双线性插值时的分块大小（默认 32）。"
            "实测本平台(ROCm/HIP) upsample_bilinear2d 的输出实际按 fp32 分配显存"
            "（不受 autocast(dtype=bf16) 影响），256 通道一次性插值需要单次 ~15.8 GiB "
            "连续显存，在多卡 DDP 叠加通信开销后容易 OOM（2026-09-14 8 卡全量首个 batch "
            "实测触发）。按通道分块插值 + 每块算完立刻转回 bf16 再拼接，与不分块结果"
            "逐元素相同，只是把一次大分配拆小，同时让最终拼接结果也只需 bf16 大小的"
            "显存（约为不转换时的一半）。调小此值(如 16)进一步降低单次分配峰值，"
            "调大（如 256，等价不分块）仅用于对照排查。"
        ),
    )
    p.add_argument(
        "--interp_backend",
        default="interpolate",
        choices=["interpolate", "grid_sample"],
        help=(
            "Stage4→Head 前重采样用的底层算子（默认 'interpolate'，即 F.interpolate 的"
            "分块+bf16回写方案，已验证与不分块结果逐元素相同，是当前推荐的安全默认值）。"
            "'grid_sample' 是 aten::grid_sampler_2d，与 upsample_bilinear2d 是不同 kernel，"
            "某些后端 dtype 支持情况可能不同，值得实测是否能原生规避 fp32 fallback；"
            "与 'interpolate' 数学等价但非逐位相同（~1e-5 量级浮点误差，可忽略）。"
            "仅用于 8 卡上对比显存/吞吐，若实测无优势或同样 fp32 fallback，继续用默认值。"
        ),
    )

    # loss function
    p.add_argument(
        "--loss_gamma",
        default=0.5,
        type=float,
        help=(
            "TailWeightedMAE 权重斜率 γ（默认 0.5）。"
            "w = 1 + γ·clamp(|y_zscore|, 0, loss_z_max)。"
            "0 = 退化为纯 MAE（与旧版完全兼容）。"
        ),
    )
    p.add_argument(
        "--loss_z_max",
        default=3.0,
        type=float,
        help="TailWeightedMAE 权重截断 z-score（默认 3.0）",
    )
    p.add_argument(
        "--lambda_freq",
        default=0.0,
        type=float,
        help=(
            "FFT Loss 权重（默认 0，关闭）。"
            "对 2D rfft 幅度谱施加 L1 监督，抑制过平滑。"
            "是通用图像质量损失，与风光资源极值目标不完全一致，"
            "消融实验显示其收益有限且会拖累部分变量，默认关闭；"
            "仍可显式指定非零值打开。"
        ),
    )
    p.add_argument(
        "--lambda_grad",
        default=0.0,
        type=float,
        help=(
            "Gradient Loss 权重（默认 0，关闭）。"
            "对 Sobel 梯度图施加 L1 监督，强化空间边缘一致性。"
            "是通用图像质量损失，消融实验显示其原始量级远大于 tail_w，"
            "即使权重很小也会显著拖累 TAS/2M_TMAX/2M_TMIN 等连续变量的逐点精度，"
            "默认关闭；仍可显式指定非零值打开。"
        ),
    )
    p.add_argument(
        "--var_weights",
        default="TAS=1.0,PRE=1.0,wind10=1.0,Q=1.0,2M_RH=1.0,2M_TMAX=1.0,2M_TMIN=1.0,FSDS=1.0",
        type=str,
        help=(
            "TailWeightedMAE 的逐变量（通道）固定权重，格式为 'VAR=weight' 逗号分隔，"
            "键必须属于 VARIABLES（见 dataset.py）。与 z-score 尾部权重相乘叠加。"
            "v2 默认改为全 1（等权'参考目标'，见 DOWNSCALE_README.md 第 5 节）："
            "旧版默认对 wind10/FSDS/TAS/2M_RH 做固定偏置加权，本质是在全部 8 通道间"
            "做零和式梯度再分配（抬风光的同时压低其余变量的梯度预算），与"
            "'各变量同时变好、风光资源改进完全由独立增量项负责'的设计原则冲突，"
            "已改为解耦：风光增强改由 --lambda_patch_extreme / --lambda_wps 等"
            "独立增量项负责，不再依赖本参数的通道偏置。仍可显式传入旧版偏置值"
            "用于对照消融（如 'wind10=1.5,FSDS=1.5,TAS=1.2,2M_RH=1.2'）。"
        ),
    )
    p.add_argument(
        "--no_area_weight", dest="use_area_weight", action="store_false",
        help=(
            "关闭训练损失中的 cos(latitude) 球面积权重（默认开启）。规则经纬网格"
            "在高纬度单位像素代表的真实地表面积远小于赤道附近，像素等权的训练"
            "目标会系统性放大高纬度权重；默认叠加归一化到均值=1 的面积权重"
            "（见 _load_area_weight_hr()），只改变像素间相对权重、不改变损失"
            "整体量级。仅影响训练损失（TailWeightedMAE），不影响 Loss/val（早停/"
            "best.pt 判据，始终保持像素等权的纯 MAE，与历史 run 可比）。"
        ),
    )
    p.set_defaults(use_area_weight=True)
    p.add_argument(
        "--extreme_vars",
        default="wind10,FSDS",
        type=str,
        help=(
            "参与旧版 SpatialExtremeLoss（全球单一最大/最小值一致性损失）的变量，"
            "逗号分隔，键必须属于 VARIABLES。v2 默认 --lambda_extreme=0（关闭），"
            "本损失已被 --lambda_patch_extreme 指定的 PatchExtremeLoss（局部网格"
            "极值，监督密度更高）取代为默认机制，仅保留供历史对照消融，不建议"
            "与 PatchExtremeLoss 同时对同一批变量启用（会重复施加极值监督）。"
        ),
    )
    p.add_argument(
        "--lambda_extreme",
        default=0.0,
        type=float,
        help=(
            "旧版 SpatialExtremeLoss 权重（v2 默认 0，关闭）。"
            "对 --extreme_vars 指定的通道，约束预测与目标在**整张全球样本**内的"
            "空间最大值/最小值一致（参考 NREL Sup3rWind 等风资源超分辨率工作的"
            "做法，但该做法原本面向局地训练图块，搬到全球单样本场景后每张图每"
            "通道只有 2 个非零梯度像素，监督信号稀疏，已被 --lambda_patch_extreme"
            "取代为默认机制）。仅保留供历史对照消融，非零时显式打开。"
        ),
    )
    p.add_argument(
        "--patch_extreme_vars",
        default="wind10,FSDS",
        type=str,
        help=(
            "参与 PatchExtremeLoss（局部网格 max/min/mean 一致性损失，v2 新默认"
            "机制，替代旧版全局 SpatialExtremeLoss）的变量，逗号分隔，键必须属于"
            "VARIABLES。默认仅 wind10、FSDS——风光资源评估最核心的两个资源变量。"
            "留空字符串则不启用该损失（等价于 --lambda_patch_extreme 0）。"
        ),
    )
    p.add_argument(
        "--lambda_patch_extreme",
        default=0.1,
        type=float,
        help=(
            "PatchExtremeLoss 权重（v2 默认 0.1，与旧版 --lambda_extreme 的历史"
            "默认值同量级，作为起点，正式训练前建议先过一次数量级校验，见"
            "DOWNSCALE_README.md 第 5 节）。对 --patch_extreme_vars 指定"
            "的通道，把 HR 场划分为 (--patch_grid_h, --patch_grid_w) 个局部网格块，"
            "逐块约束预测/目标的局部 max、min、mean 一致。patch-mean 只约束一阶矩，"
            "抑制为凑 max 而整体平移的退化解，不约束极值位置；λ 偏大时可能用抬高"
            "整块换极值统计。0 = 关闭。"
        ),
    )
    p.add_argument(
        "--patch_grid_h", default=180, type=int,
        help="PatchExtremeLoss 局部网格高度（默认 180，与 LR 输入网格同分辨率对齐）",
    )
    p.add_argument(
        "--patch_grid_w", default=360, type=int,
        help="PatchExtremeLoss 局部网格宽度（默认 360，与 LR 输入网格同分辨率对齐）",
    )
    p.add_argument(
        "--lambda_wps",
        default=0.05,
        type=float,
        help=(
            "WindPowerSensitivityProxy 权重（v2 新增，默认 0.05，小权重）。"
            "仅对 wind10 通道，在物理量纲的风机爬坡段（--wind_cutin 到 "
            "--wind_rated）内约束归一化功率代理一致。mask 按真值判定，额定以上"
            "高风速不参与本项（由 tail 与 PatchExtreme 监督）。全球日均 10m 风速"
            "大量落在该区间，覆盖率可能很高，冒烟须看 Loss/wps_mask_ratio。"
            "**重要局限性**：wind10 是 10m 而非轮毂高度风速，日均而非瞬时"
            "（Jensen 不等式下 P(日均v)≠日均P(瞬时v)），不能解释为容量因子或"
            "发电量误差优化。0 = 关闭。"
        ),
    )
    p.add_argument(
        "--wind_cutin", default=3.0, type=float,
        help="WindPowerSensitivityProxy 爬坡段下界（m/s，默认 3.0，IEC 典型代理值，非机型标定值）",
    )
    p.add_argument(
        "--wind_rated", default=12.0, type=float,
        help="WindPowerSensitivityProxy 爬坡段上界（m/s，默认 12.0，IEC 典型代理值，非机型标定值）",
    )
    p.add_argument(
        "--lambda_phys",
        default=0.02,
        type=float,
        help=(
            "PhysicalConsistencyLoss 权重（v2 新增，默认 0.02，很小，仅作安全网）。"
            "仅依赖预测本身。计算顺序：先用 norm_mean/std 反标准化，再在物理"
            "（PRE 为 log1p）量纲上施加 hinge：TMIN≤TAS≤TMAX、wind10/FSDS/Q/PRE"
            "非负、RH∈[0,100]，再除以各变量训练期 std 无量纲化后平均。"
            "禁止在 z 空间直接比较或写 ReLU(−z)。0 = 关闭。"
        ),
    )
    p.add_argument(
        "--resource_vars",
        default="wind10,FSDS",
        type=str,
        help=(
            "业务侧关注变量（默认 wind10,FSDS，即风光资源核心变量），逗号分隔，"
            "键必须属于 VARIABLES。用于在训练开始前计算一次'LR 双线性插值'"
            "naive baseline 的物理量纲 MAE（见 compute_baseline_resource_mae()），"
            "并在训练过程中额外维护一份 best_resource.pt checkpoint（按这些变量"
            "相对该 baseline 的 skill score 均值取最优——修复了旧版把不同物理"
            "量纲 MAE 直接算术平均、无量纲可解释性的问题），与 best.pt（始终按"
            "全部 8 变量的纯 MAE 选取，早停/top-K 判据不变）并行存在、互不影响。"
            "对应 TensorBoard 标量 Skill_val/<var>、Skill_val_resource/mean。"
            "留空字符串则不启用。"
        ),
    )
    p.add_argument(
        "--val_extreme_z_thresh",
        default=1.5,
        type=float,
        help=(
            "验证集极端值 MAE 的判定阈值（z-score 绝对值，默认 1.5）。"
            "仅统计 |y_zscore| > 该阈值的像素 MAE（MAE_val_extreme/*），"
            "用于评估组合损失在极端值上是否真的优于纯 MAE，"
            "不受大量'平静'像素稀释影响。建议与 --loss_z_max 配合观察 "
            "（阈值越大越接近 loss_z_max 覆盖的极端区间）。"
        ),
    )

    # optimisation
    p.add_argument("--epochs",       default=50,   type=int)
    p.add_argument("--batch_size",   default=1,    type=int)
    p.add_argument("--lr",           default=2e-4, type=float)
    p.add_argument("--weight_decay", default=1e-4, type=float)
    p.add_argument("--accum_steps",  default=2,    type=int,
                   help="gradient accumulation steps (default: 2)")
    p.add_argument("--warmup_steps", default=100,  type=int,
                   help="linear warmup optimizer steps (default: 100). "
                        "若同时设置 --warmup_ratio，本值会被自动覆盖。")
    p.add_argument(
        "--warmup_ratio",
        default=None,
        type=float,
        help=(
            "若设置（如 0.03~0.05），会在算出真实 total_steps = steps_per_epoch×epochs 后，"
            "自动用 warmup_steps = round(warmup_ratio × total_steps) 覆盖 --warmup_steps。"
            "--warmup_steps 的默认值 100 是面向调试/小数据集的经验值，全量数据下 "
            "total_steps 可能达到数万步，固定 100 步 warmup 占比过小；用 --warmup_ratio "
            "可以不必先跑一次看日志里的 total_steps 再手动回填 --warmup_steps。"
            "留空（默认）则完全使用 --warmup_steps 原始值，行为与此前一致。"
        ),
    )
    p.add_argument(
        "--ema_decay",
        default=0.0,
        type=float,
        help=(
            "模型权重指数滑动平均（EMA）衰减率，如 0.999（每个 optimizer 步后 "
            "shadow = 0.999·shadow + 0.001·online）。0 = 关闭（默认，行为与此前完全一致）。"
            "开启后：验证阶段临时换用 EMA 权重（对 --batch_size 较小/BatchNorm 统计噪声更"
            "稳健，通常比最后一步的在线权重泛化更好）；checkpoint 中额外保存 "
            "model_ema 供部署/推理使用，'model' 字段仍是训练用的在线权重（用于正确续训）。"
        ),
    )
    p.add_argument(
        "--warmup_start_ratio",
        default=0.01,
        type=float,
        help="first warmup step: lr = base_lr × this (default: 0.01); linear ramp to 1.0",
    )
    p.add_argument(
        "--min_lr_ratio",
        default=0.01,
        type=float,
        help="cosine floor: final lr = base_lr × this (default: 0.01); use 0 for decay to zero",
    )
    p.add_argument("--clip_grad",    default=1.0,  type=float,
                   help="gradient norm clipping value (default: 1.0)")

    # logging & checkpointing
    p.add_argument("--run_dir",      default="runs/exp01", type=str,
                   help="directory for TensorBoard logs and checkpoints")
    p.add_argument("--log_interval", default=50,  type=int,
                   help="log train scalars every N optimizer steps")
    p.add_argument("--val_interval", default=1,   type=int,
                   help="run validation every N epochs")
    p.add_argument(
        "--early_stop_patience",
        default=20,
        type=int,
        help=(
            "early stopping: stop after this many consecutive validations without "
            "val_loss improvement ≥ min_delta vs previous best before that validation "
            "(counts each validation run; with val_interval>1 one 'wait' spans "
            "multiple epochs). Use 0 to disable."
        ),
    )
    p.add_argument(
        "--early_stop_min_delta",
        default=0.0,
        type=float,
        help=(
            "minimum val_loss decrease (absolute, normalized L1) to reset the "
            "early-stop counter; small values (e.g. 1e-5) reduce jitter from float noise"
        ),
    )
    p.add_argument("--save_top_k",   default=3,   type=int,
                   help="keep top-K checkpoints by val loss")
    p.add_argument("--resume",       default=None, type=str,
                   help="path to checkpoint .pt file to resume training")
    p.add_argument("--seed",         default=42,  type=int)
    p.add_argument(
        "--eval_only",
        action="store_true",
        help=(
            "仅加载 --resume 指定的 checkpoint，在（与训练时相同种子/划分下的）验证集上跑一次 "
            "validate() 记录完整指标（Loss/val、Loss/val_combined 及子项、MAE_val_extreme 等）"
            "后退出，不进行训练。用于给历史 run（其日志未包含新指标）补算完整对比指标，"
            "无需重新训练。必须与 --resume 一起使用。"
        ),
    )

    # distributed training（torchrun / srun 启动时由环境变量 RANK/WORLD_SIZE/LOCAL_RANK
    # 自动检测并生效，无需用户手动传参；下列参数仅用于微调分布式行为，单卡运行时忽略）
    p.add_argument(
        "--no_sync_bn", dest="sync_bn", action="store_false",
        help=(
            "分布式训练时默认将模型中的 BatchNorm 转为 SyncBatchNorm（跨所有进程同步 "
            "batch 统计量），缓解 --batch_size 1 时单进程 BN 统计噪声大的问题（见 "
            "DOWNSCALE_README.md 第 4 节）。传本参数关闭该转换（单卡运行时无影响）。"
        ),
    )
    p.set_defaults(sync_bn=True)
    p.add_argument(
        "--ddp_find_unused_parameters",
        action="store_true",
        help=(
            "DistributedDataParallel 的 find_unused_parameters（默认关闭）。"
            "本模型各消融开关（--no_cbam/--hr_aux_mode 等）只影响初始化时创建哪些子模块，"
            "不改变前向路径中已创建参数的使用，默认 False 即可；"
            "如遇到 DDP 报 'unused parameters' 错误可临时打开排查。"
        ),
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # 分布式初始化必须在任何 CUDA 上下文建立前完成（set_device）。未通过 torchrun/srun
    # 注入 RANK/WORLD_SIZE 时 is_distributed=False，后续所有分布式分支均不生效，
    # 单卡行为与改动前完全一致。
    is_distributed, rank, world_size, local_rank = setup_distributed()

    # 所有 rank 用相同种子，保证数据集 train/val 划分、模型初始参数（DDP 要求）一致；
    # DistributedSampler 内部按 rank 切分数据，不依赖这里的随机状态再做区分。
    seed_everything(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank if is_distributed else 0)
    else:
        device = torch.device("cpu")
    if is_main_process():
        print(f"Using device: {device}  "
              f"(distributed={is_distributed}, world_size={world_size})")

    run_dir  = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    if is_main_process():
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 仅 rank0 写 TensorBoard；其余 rank 用空实现占位，训练循环无需到处判断 rank
    writer: "SummaryWriter | _NoOpWriter"
    writer = SummaryWriter(log_dir=str(run_dir / "tb")) if is_main_process() else _NoOpWriter()

    # ------------------------------------------------------------------
    # Dataset & DataLoaders
    # ------------------------------------------------------------------
    dataset = DownscaleDataset(
        hdf5_root  = args.hdf5_root,
        seasons    = args.seasons,
        static_dir = args.static_dir,
        stats_file = args.stats_file,
        manifests  = args.manifests,
    )
    # 与数据集一致的 8 变量 z-score 参数，验证时反标准化用
    norm_mean, norm_std = dataset.get_norm_stats()   # (8,) float32 tensors

    n_total = len(dataset)
    n_val   = max(1, int(n_total * args.val_fraction))
    n_train = n_total - n_val

    # 在已设种子下打乱索引，按比例划分训练/验证，划分结果可复现；
    # 种子在所有 rank 上相同 ⇒ 划分结果在所有 rank 上完全一致。
    # 这是同分布内插验证（日尺度场存在时间自相关，验证可能偏乐观）；
    # 时间外推评估使用独立测试集（2020-2024），不在本脚本内切分。
    indices = list(range(n_total))
    random.shuffle(indices)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)

    # 分布式：用 DistributedSampler 把训练/验证集按 rank 切分为互不重叠的子集
    # （train 用 shuffle+seed，每个 epoch 用 sampler.set_epoch 重新打乱；
    #  val 用 shuffle=False、drop_last=False，样本不会被丢弃，validate() 内已对
    #  各 rank 的累计量做 all_reduce 求和后再统一算均值，与单卡结果一致）
    train_sampler = (
        DistributedSampler(train_set, num_replicas=world_size, rank=rank,
                            shuffle=True, seed=args.seed)
        if is_distributed else None
    )
    val_sampler = (
        DistributedSampler(val_set, num_replicas=world_size, rank=rank,
                            shuffle=False, drop_last=False)
        if is_distributed else None
    )

    # num_workers>0 时启用 persistent_workers，减少 epoch 间 worker 重启开销
    _persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_set,
        batch_size         = args.batch_size,
        shuffle            = (train_sampler is None),
        sampler            = train_sampler,
        num_workers        = args.num_workers,
        pin_memory         = device.type == "cuda",
        drop_last          = True,
        persistent_workers = _persistent,
    )
    val_loader = DataLoader(
        val_set,
        batch_size         = args.batch_size,
        shuffle            = False,
        sampler            = val_sampler,
        num_workers        = args.num_workers,
        pin_memory         = device.type == "cuda",
        persistent_workers = _persistent,
    )

    if is_main_process():
        print(f"Dataset: {n_total} samples  (train={n_train}, val={n_val})")
        print(f"Train batches/epoch/rank: {len(train_loader)}  "
              f"(optimizer steps/epoch ≈ {len(train_loader) // args.accum_steps}, "
              f"global batch = batch_size×accum_steps×world_size = "
              f"{args.batch_size}×{args.accum_steps}×{world_size} = "
              f"{args.batch_size * args.accum_steps * world_size})")

    # ------------------------------------------------------------------
    # Model, loss, optimiser, scheduler
    # ------------------------------------------------------------------
    model = PixelShuffleDownscaleNet(
        in_ch                 = 8,
        hr_aux_ch             = 7,
        base_ch               = args.base_ch,
        num_resblocks         = args.num_resblocks,
        out_ch                = 8,
        target_h              = 1801,
        target_w              = 3600,
        use_cbam              = args.use_cbam,
        hr_aux_mode           = args.hr_aux_mode,
        use_checkpoint        = args.use_checkpoint,
        stage4_shuffle_conv_k = 1,   # 1×1 消除 Stage4 的 17.80 GiB im2col workspace
        norm_type             = args.norm_type,
        interp_chunk_channels = args.interp_chunk_channels,
        interp_backend        = args.interp_backend,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg_str = (
        f"base_ch={args.base_ch}  in_ch=8  init=1x1  "
        f"use_cbam={args.use_cbam}  hr_aux_mode={args.hr_aux_mode}  "
        f"resblocks={args.num_resblocks}  checkpoint={args.use_checkpoint}  "
        f"norm_type={args.norm_type}  s4_shuf=1x1  params={n_params:,}  "
        f"distributed={is_distributed} world_size={world_size}  "
        f"compile={args.compile}  ema_decay={args.ema_decay}  "
        f"interp_backend={args.interp_backend}  interp_chunk_channels={args.interp_chunk_channels}"
    )
    if is_main_process():
        print(f"Model: {cfg_str}")
    writer.add_text("model/config", cfg_str, 0)

    if is_distributed and args.sync_bn and device.type == "cuda" and args.norm_type == "batch":
        # SyncBatchNorm 仅支持 GPU 模块 + BatchNorm；CPU-only 调试环境或 norm_type=group
        # 自动跳过（GroupNorm 按通道分组在单样本内部统计，不依赖 batch 维，无需跨进程同步）。
        # batch_size=1 时单进程 BN 统计噪声大（见 DOWNSCALE_README.md 第 4 节），
        # SyncBatchNorm 用所有进程的样本联合估计 mean/var，缓解该问题；
        # 转换须在 .to(device) 之后、DDP 包装之前进行
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if is_main_process():
            print("已将 BatchNorm 转换为 SyncBatchNorm（跨进程同步统计量）")

    if is_distributed:
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank}
        model = DDP(model, find_unused_parameters=args.ddp_find_unused_parameters,
                    **ddp_kwargs)

    # EMA 需要在 DDP 包装之后创建（用 unwrap_model 拿到底层裸模块的初始权重作为影子权重起点），
    # 但要在 torch.compile 包装之前创建（compile 后 unwrap_model 会经 _orig_mod 正确解包，
    # 不影响 EMA，这里顺序主要是为了让 EMA 影子权重与 DDP 广播后的初始参数完全一致）
    ema: "ModelEMA | None" = None
    if args.ema_decay > 0:
        ema = ModelEMA(unwrap_model(model), decay=args.ema_decay, device=device)
        if is_main_process():
            print(f"[EMA] 已启用，decay={args.ema_decay}")

    if args.compile:
        # torch.compile 可以直接包装 DDP 模块（PyTorch 2.x 官方支持的用法）；
        # unwrap_model 已通过 getattr(model, "_orig_mod", model) 处理 compile 包装，
        # 后续 save/load_checkpoint、EMA、validate() 均无需感知是否 compile。
        model = torch.compile(model)
        if is_main_process():
            print("[compile] 已用 torch.compile() 包装模型")

    channel_weight = parse_var_weights(args.var_weights)
    extreme_indices = parse_var_list(args.extreme_vars)
    patch_extreme_indices = parse_var_list(args.patch_extreme_vars)
    resource_indices = parse_var_list(args.resource_vars)
    wind_idx = VARIABLES.index("wind10")

    area_weight = (
        _load_area_weight_hr(args.static_dir, target_h=1801) if args.use_area_weight else None
    )

    criterion = CombinedLoss(
        gamma                    = args.loss_gamma,
        z_max                    = args.loss_z_max,
        channel_weight           = channel_weight,
        area_weight              = area_weight,
        lambda_extreme           = args.lambda_extreme,
        extreme_channel_indices  = extreme_indices,
        lambda_freq              = args.lambda_freq,
        lambda_grad              = args.lambda_grad,
        lambda_patch_extreme     = args.lambda_patch_extreme,
        patch_channel_indices    = patch_extreme_indices,
        patch_grid               = (args.patch_grid_h, args.patch_grid_w),
        lambda_wps               = args.lambda_wps,
        wind_idx                 = wind_idx,
        wind_mean                = norm_mean[wind_idx].item(),
        wind_std                 = norm_std[wind_idx].item(),
        wind_cutin               = args.wind_cutin,
        wind_rated               = args.wind_rated,
        lambda_phys              = args.lambda_phys,
        norm_mean                = norm_mean,
        norm_std                 = norm_std,
    ).to(device)
    var_weight_str = ", ".join(f"{v}={w:g}" for v, w in zip(VARIABLES, channel_weight.tolist()))
    extreme_var_str = ", ".join(VARIABLES[i] for i in extreme_indices) or "none"
    patch_extreme_var_str = ", ".join(VARIABLES[i] for i in patch_extreme_indices) or "none"
    loss_cfg = (
        f"loss(v2): TailWeightedMAE(γ={args.loss_gamma}, z_max={args.loss_z_max}, "
        f"channel_weight=[{var_weight_str}], area_weight={'on' if area_weight is not None else 'off'})"
        f" + PatchExtreme×{args.lambda_patch_extreme}(vars=[{patch_extreme_var_str}], "
        f"grid={args.patch_grid_h}x{args.patch_grid_w})"
        f" + WindPowerSensitivityProxy×{args.lambda_wps}"
        f"(cutin={args.wind_cutin},rated={args.wind_rated})"
        f" + PhysicalConsistency×{args.lambda_phys}"
        f" + [legacy] SpatialExtreme×{args.lambda_extreme}(vars=[{extreme_var_str}])"
        f" + FFT×{args.lambda_freq}"
        f" + Grad×{args.lambda_grad}"
    )
    if is_main_process():
        print(f"Loss: {loss_cfg}")
    writer.add_text("loss/config", loss_cfg, 0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # 调度器按「优化器步」计数：每 accum_steps 个 micro-batch 才 step 一次
    steps_per_epoch = max(1, len(train_loader) // args.accum_steps)
    total_steps     = steps_per_epoch * args.epochs

    warmup_steps = args.warmup_steps
    if args.warmup_ratio is not None:
        warmup_steps = max(1, round(args.warmup_ratio * total_steps))
        if is_main_process():
            print(f"[warmup] --warmup_ratio={args.warmup_ratio:g} × total_steps={total_steps} "
                  f"→ warmup_steps={warmup_steps}（覆盖 --warmup_steps={args.warmup_steps}）")

    scheduler = build_scheduler(
        optimizer,
        warmup_steps,
        total_steps,
        warmup_start_ratio=args.warmup_start_ratio,
        min_lr_ratio=args.min_lr_ratio,
    )
    if is_main_process():
        print(
            f"LR schedule: warmup {warmup_steps} steps "
            f"({args.warmup_start_ratio:.0%}→100% of base), "
            f"cosine to {args.min_lr_ratio:.0%} of base "
            f"(total optimizer steps ≈ {total_steps})"
        )

    # ------------------------------------------------------------------
    # Optional resume
    # ------------------------------------------------------------------
    start_epoch   = 0
    global_step   = 0
    best_val_loss = float("inf")
    # 按验证 loss 保留最优的 K 个检查点，超出则删除较差文件
    saved_ckpts: list[tuple[float, Path]] = []
    # 业务侧并行判据（--resource_vars，默认 wind10/FSDS）；仅额外维护 best_resource.pt，
    # 不影响 best_val_loss/early_stop_streak/top-K 等标准判据。
    # 判据已从"resource_metric 越小越好"（物理量纲 MAE）改为"越大越好"
    # （skill score，见 validate()/compute_baseline_resource_mae() 文档），
    # 初值同步由 inf 改为 -inf。
    best_resource_val = float("-inf")
    # 连续多少次「验证」未带来相对历史最优的改进（按验证次数计，非裸 epoch）
    early_stop_streak = 0
    stopped_early = False

    if args.resume:
        start_epoch, global_step, best_val_loss, early_stop_streak, best_resource_val = (
            load_checkpoint(
                Path(args.resume), model, optimizer, scheduler, device, ema=ema
            )
        )
        if is_main_process():
            saved_ckpts = rebuild_saved_ckpts(ckpt_dir)
            while len(saved_ckpts) > args.save_top_k:
                _, old_path = saved_ckpts.pop()
                if old_path.exists():
                    old_path.unlink()
            print(f"[resume] rebuilt top-K from disk: {len(saved_ckpts)} file(s) kept")

    # 训练开始前（或 resume 后）计算一次 naive baseline（LR 双线性插值），用于
    # best_resource.pt 的 skill score 归一化。不随 epoch 变化，因此不缓存进
    # checkpoint；每次进程启动重算一次的成本约等于一次额外验证轮次，相对全量
    # 训练总成本可忽略（见 compute_baseline_resource_mae() 文档）。
    baseline_resource_mae = compute_baseline_resource_mae(
        val_loader, device, norm_mean, norm_std, resource_indices,
    )

    if args.eval_only:
        if not args.resume:
            raise ValueError("--eval_only 必须与 --resume <checkpoint.pt> 一起使用")
        if is_main_process():
            print(f"[eval_only] 补算完整验证指标（checkpoint={args.resume}）...")
        _eval_ctx = (
            ema.apply_to(unwrap_model(model)) if ema is not None else contextlib.nullcontext()
        )
        with _eval_ctx:
            validate(
                model, val_loader, device,
                norm_mean, norm_std, writer, global_step, start_epoch,
                criterion=criterion,
                extreme_z_thresh=args.val_extreme_z_thresh,
                resource_var_indices=resource_indices,
                baseline_resource_mae=baseline_resource_mae,
                area_weight=area_weight,
            )
        writer.close()
        if is_main_process():
            print("[eval_only] 完成，未进行训练。")
        cleanup_distributed()
        return

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.train()
    for epoch in range(start_epoch, args.epochs):
        # DistributedSampler 依赖 set_epoch 重新用 seed+epoch 派生打乱顺序，
        # 否则每个 epoch 各 rank 的样本划分会保持一致（失去跨 epoch 随机性）
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t_epoch = time.time()
        accum_loss = 0.0      # 自上次 optimizer.step 以来各 micro-batch 的 loss 之和（未除 accum）
        micro_count = 0       # 同上段内已累积的 micro-batch 数
        accum_sub: dict[str, float] = {}   # 各子项累积和，每 log_interval 步清零
        sub_count = 0                      # accum_sub 累积覆盖的 micro-batch 数（用于正确求平均）

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x_lr, hr_aux, y_hr) in enumerate(train_loader):
            is_update_step = (batch_idx + 1) % args.accum_steps == 0
            is_last_batch  = (batch_idx + 1) == len(train_loader)

            x_lr   = x_lr.to(device,  non_blocking=True)
            hr_aux = hr_aux.to(device, non_blocking=True)
            y_hr   = y_hr.to(device,  non_blocking=True)

            # 梯度累积的中间 micro-batch 用 model.no_sync() 跳过 DDP 的梯度 all-reduce，
            # 只在真正 optimizer.step 前的最后一个 micro-batch 同步一次，减少通信量
            need_sync = is_update_step or is_last_batch
            sync_ctx = (
                contextlib.nullcontext() if (not is_distributed) or need_sync
                else model.no_sync()
            )
            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = model(x_lr, hr_aux)
                    loss, sub_losses = criterion(pred, y_hr)

                # backward 前除以 accum_steps，使多步梯度均值等价于单大批次
                (loss / args.accum_steps).backward()

            accum_loss  += loss.item()
            micro_count += 1
            # 累积各子项（用于日志平均）
            for k, v in sub_losses.items():
                accum_sub[k] = accum_sub.get(k, 0.0) + v
            sub_count += 1

            if need_sync:
                # 防止梯度过大；最后不足一整组 accum 的 batch 也会触发更新
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad
                )
                step_ok = _optimizer_step_is_finite(loss, grad_norm, device)
                if not step_ok:
                    # 所有 rank 一起跳过：不 step optimizer/scheduler、不更新 EMA、
                    # 不推进 global_step，清掉已污染的梯度后继续。DDP 下 backward
                    # 已完成，不能只让部分 rank return。
                    optimizer.zero_grad(set_to_none=True)
                    accum_loss = 0.0
                    micro_count = 0
                    accum_sub.clear()
                    sub_count = 0
                    writer.add_scalar("Train/nonfinite_skip", 1.0, global_step)
                    if is_main_process():
                        print(
                            f"[nonfinite] skip optimizer.step at epoch {epoch+1} "
                            f"batch {batch_idx} global_step={global_step} "
                            f"loss={float(loss.detach().float().cpu())} "
                            f"grad_norm={float(grad_norm)}"
                        )
                    continue

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if ema is not None:
                    ema.update(unwrap_model(model))

                # 首个 optimizer step 后打印各 rank 的峰值显存占用/保留量，
                # 用于在正式长跑前对比 --interp_chunk_channels 取值的显存余量
                # （峰值只反映到目前为止最紧张的一刻，之后重置计数器不影响训练本身）。
                if global_step == 1 and torch.cuda.is_available():
                    rank_tag = f"rank{dist.get_rank()}" if is_distributed else "rank0"
                    alloc = torch.cuda.max_memory_allocated() / 1024**3
                    reserv = torch.cuda.max_memory_reserved() / 1024**3
                    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
                    print(
                        f"[mem] {rank_tag} 首个 optimizer step 后："
                        f"max_allocated={alloc:.2f} GiB  max_reserved={reserv:.2f} GiB  "
                        f"device_total={total:.2f} GiB  interp_chunk_channels={args.interp_chunk_channels}"
                    )
                    torch.cuda.reset_peak_memory_stats()

                # 日志记录的是本 optimizer 步内各 micro-batch 的平均 loss
                step_loss  = accum_loss / micro_count
                current_lr = scheduler.get_last_lr()[0]
                accum_loss  = 0.0
                micro_count = 0

                if global_step % args.log_interval == 0:
                    writer.add_scalar("Loss/train", step_loss, global_step)
                    writer.add_scalar("LR", current_lr, global_step)
                    # 各子项独立记录（过去 log_interval 步、共 sub_count 个 micro-batch 的平均）。
                    # 注意：sub_count = log_interval * accum_steps（而非 log_interval），
                    # 因为 accum_sub 在每个 micro-batch（每个 batch_idx）都会累加一次，
                    # 而不是每个 optimizer 步累加一次；此前误用 log_interval 做分母，
                    # 会使各子项数值系统性放大 accum_steps 倍（数值间的相对比例不受影响，
                    # 但与 Loss/train 的绝对量级不可比）。
                    n_logged = max(1, sub_count)
                    for k, v in accum_sub.items():
                        writer.add_scalar(f"Loss/{k}", v / n_logged, global_step)
                    accum_sub.clear()
                    sub_count = 0
                    if is_main_process():
                        print(
                            f"Epoch {epoch+1:03d} | step {global_step:6d} | "
                            f"loss={step_loss:.6f} | lr={current_lr:.3e}"
                        )

        elapsed = time.time() - t_epoch
        if is_main_process():
            print(f"Epoch {epoch+1:03d} done in {elapsed:.1f}s")

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        if (epoch + 1) % args.val_interval == 0:
            best_before_val = best_val_loss
            # 启用 EMA 时验证阶段临时换用 EMA 权重（通常比训练轨迹上的"在线"权重更稳健，
            # 尤其在小 batch / BatchNorm 噪声较大的场景），validate() 结束后自动还原为
            # 在线权重继续训练；未启用 EMA 时行为与此前完全一致
            _val_ctx = (
                ema.apply_to(unwrap_model(model)) if ema is not None else contextlib.nullcontext()
            )
            with _val_ctx:
                val_loss, resource_metric = validate(
                    model, val_loader, device,
                    norm_mean, norm_std, writer, global_step, epoch + 1,
                    criterion=criterion,
                    extreme_z_thresh=args.val_extreme_z_thresh,
                    resource_var_indices=resource_indices,
                    baseline_resource_mae=baseline_resource_mae,
                    area_weight=area_weight,
                )
            improved = val_loss < (best_before_val - args.early_stop_min_delta)
            if args.early_stop_patience > 0:
                if improved:
                    early_stop_streak = 0
                else:
                    early_stop_streak += 1
                writer.add_scalar("EarlyStop/streak", early_stop_streak, global_step)

            # 先更新 best_* 再落盘，保证 latest.pt 里的 best_val_loss /
            # best_resource_val / early_stop_streak 与当前进程状态一致，
            # 从 latest.pt resume 时不会用过期的最优值覆盖更好的 best.pt。
            is_new_best = val_loss < best_val_loss
            if is_new_best:
                best_val_loss = val_loss
            is_new_best_resource = (
                resource_metric is not None and resource_metric > best_resource_val
            )
            if is_new_best_resource:
                best_resource_val = resource_metric

            def _save_ckpt(path: Path) -> None:
                save_checkpoint(
                    path, model, optimizer, scheduler,
                    epoch, global_step, best_val_loss, ema=ema,
                    early_stop_streak=early_stop_streak,
                    best_resource_val=best_resource_val,
                )

            # checkpoint 落盘仅由 rank0 执行（val_loss 已在 validate() 内跨 rank 聚合，
            # 所有进程算出的值一致，因此 best_val_loss/early_stop_streak 等状态变量在
            # 各 rank 上天然保持同步，无需额外广播）
            if is_main_process():
                ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}_valloss{val_loss:.6f}.pt"
                _save_ckpt(ckpt_path)
                saved_ckpts.append((val_loss, ckpt_path))

                saved_ckpts.sort(key=lambda t: t[0])
                while len(saved_ckpts) > args.save_top_k:
                    _, old_path = saved_ckpts.pop()   # 去掉验证 loss 最差的一个
                    if old_path.exists():
                        old_path.unlink()

                # latest.pt：始终覆盖为最近一次验证的状态，不参与 top-K 淘汰。
                # 多日长跑（平台墙钩/时长限制导致意外中断）时，最近的 top-K 检查点未必是
                # 最后一个 epoch（可能因 val_loss 不是最优被立即删除），--resume 若只认
                # top-K 文件可能丢失比预期更多的训练进度；latest.pt 保证总能从最近一次
                # 验证点续训，代价仅是多一份 checkpoint 的磁盘占用。
                _save_ckpt(ckpt_dir / "latest.pt")

                if is_new_best:
                    best_path = ckpt_dir / "best.pt"
                    _save_ckpt(best_path)
                    print(f"  New best val_loss={best_val_loss:.6f} → {best_path}")

                if is_new_best_resource:
                    best_resource_path = ckpt_dir / "best_resource.pt"
                    _save_ckpt(best_resource_path)
                    print(f"  New best skill_score={best_resource_val:.6f} "
                          f"(vars={args.resource_vars}) → {best_resource_path}")

            # 各 rank 用相同（已聚合的）val_loss 独立计算，early_stop_streak 与
            # best_val_loss 在所有 rank 上天然保持一致，因此下面的停止条件在所有
            # 进程上会同时触发，无需额外的跨进程广播/同步
            if (
                args.early_stop_patience > 0
                and early_stop_streak >= args.early_stop_patience
            ):
                if is_main_process():
                    print(
                        f"Early stopping: val_loss did not improve over prior best by "
                        f"≥{args.early_stop_min_delta:g} for {args.early_stop_patience} consecutive "
                        f"validation(s). best_val_loss={best_val_loss:.6f}"
                    )
                writer.add_text(
                    "run/early_stop",
                    f"triggered at epoch {epoch+1}, streak={early_stop_streak}, "
                    f"best_val_loss={best_val_loss:.6f}",
                    global_step,
                )
                stopped_early = True
                break

            # 让 rank0 先写完 checkpoint，再让所有 rank 一起进入下一 epoch，
            # 避免（若外部脚本并发读取 checkpoint 目录）读到写入中的文件
            if is_distributed:
                dist.barrier()

    writer.close()
    if is_main_process():
        if stopped_early:
            print("Training finished (early stop).")
        else:
            print("Training complete.")
    cleanup_distributed()


if __name__ == "__main__":
    main()
