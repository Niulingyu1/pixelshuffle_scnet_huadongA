"""
PixelShuffleDownscaleNet 模型训练脚本。

  - 损失函数（面向风光资源评估场景，组合损失，可逐项开关）：
      TailWeightedMAE（+逐变量通道权重） + SpatialExtremeLoss + FFT Loss + Gradient Loss
      · TailWeightedMAE：以 |y_zscore| 为权重的 MAE，提升极端值学习（高/低分位数）
        γ=0 时退化为纯 MAE，保底不退步（--loss_gamma 控制）
      · 逐变量通道权重（--var_weights）：在 z-score 尾部权重之上叠加固定的变量级
        权重，让 wind10/FSDS（风光资源核心变量）、TAS/2M_RH（组合可反映湿球温度
        WBT，与人体舒适度、电力需求相关）获得更多梯度预算；PRE/Q/2M_TMAX/2M_TMIN
        默认中性（不特别加权也不打压）。默认值：wind10=1.5, FSDS=1.5, TAS=1.2,
        2M_RH=1.2, PRE=Q=2M_TMAX=2M_TMIN=1.0。
      · SpatialExtremeLoss：对 --extreme_vars 指定的通道（默认 wind10、FSDS）约束
        预测/目标在样本区域内空间最大值与最小值的一致性，直接保护风速极大值、
        辐照度晴空峰值/云遮骤降等风光资源评估最关心的极值特征（参考 NREL
        Sup3rWind 等风资源超分辨率工作的做法，--lambda_extreme 控制，默认 0.1）
      · FFT Loss：2D rfft 幅度谱 L1，直接监督高频能量，改善过平滑（--lambda_freq
        控制，默认 0 关闭——通用图像质量目标与风光资源极值目标不完全一致，
        消融实验显示收益有限，默认关闭但保留可选）
      · Gradient Loss：Sobel 梯度 L1，强化空间边缘/锋面一致性（--lambda_grad
        控制，默认 0 关闭——原始量级远大于 tail_w，即使权重很小也会显著拖累
        TAS/2M_TMAX/2M_TMIN 等连续变量的逐点精度，默认关闭但保留可选）
      · 验证/早停始终使用纯 MAE（与历史 run 可比较）；同时在验证集上用与训练相同
        的 criterion 计算完整组合损失各子项，避免用对双方都不公平的整体纯 MAE
        去比较不同损失配置
  - 优化器：AdamW + 线性 warmup + 余弦退火
  - 梯度累积（默认 accum_steps=2）
  - bf16 自动混合精度（无需 GradScaler）
  - Gradient Checkpointing（默认开启，全部 4 个 UpStage，--no_checkpoint 可关闭）
  - TensorBoard 记录：Loss/train、Loss/val（纯 MAE，早停/top-K 判据）、LR、
                      MAE_val/<variable>（整体 MAE，物理量纲）
                      Loss/tail_w、Loss/spatial_extreme、Loss/freq、Loss/grad（训练时各子项，调试用）
                      Loss/val_combined、Loss/val_tail_w、Loss/val_spatial_extreme、
                      Loss/val_freq、Loss/val_grad
                      （验证集上用与训练相同 criterion 计算的组合损失各子项，
                       用于判断该 run 的损失配置是否真的改善了其自身优化目标）
                      MAE_val_extreme/<variable>、MAE_val_extreme/mean
                      （仅 |y_zscore| > --val_extreme_z_thresh 的极端像素 MAE，
                       物理量纲；避免被大量"平静"像素稀释，是组合损失 vs 纯 MAE
                       在极端值表现上唯一公平的对比口径）
                      MAE_val_physical/PRE
                      （PRE 在写入 HDF5 前做过 log1p 变换，MAE_val/PRE 反标准化后
                       仍是 log1p(mm/day) 空间误差，容易被误读为"物理精度"；本项
                       额外对 PRE 通道做一次 expm1，是唯一真实 mm/day 量纲的降水
                       误差指标，不影响/不替代其余既有指标口径）
  - 检查点保存/恢复（含 epoch、优化器、调度器状态）：`best.pt`（标准判据，按全部 8 变量
    纯 MAE 选取，早停/top-K 均以此为准）与 `best_resource.pt`（并行、非标准判据，按
    --resource_vars 默认 wind10/FSDS 的物理量纲平均 MAE 选取，仅供业务侧参考，不影响
    best.pt/早停）
  - 早停：默认监控验证集 Loss/val（纯 MAE），连续若干次验证无显著改进则结束（--early_stop_patience 为 0 可关闭）

─────────────────────────────────────────────────────────────
快速调试（hdf5_mini，少量 epoch）：
    python train.py \\
        --hdf5_root /public/share/acd7koea4a/hdf5_mini \\
        --epochs 3 --val_interval 1 \\
        --run_dir runs/debug

─────────────────────────────────────────────────────────────
消融实验对照组（每组独立 run_dir，TensorBoard 中对比）：

  # A. 基准（全量：1×1 init + base_ch=256 + CBAM-M1 + HR-aux 全量注入 + 组合损失默认值）
  python train.py --base_ch 256 \\
      --run_dir runs/ablation/A_baseline

  # B. 无 CBAM
  python train.py --base_ch 256 --no_cbam \\
      --run_dir runs/ablation/B_no_cbam

  # C. 无 HR-aux（全程不注入）
  python train.py --base_ch 256 --hr_aux_mode none \\
      --run_dir runs/ablation/C_no_hr_aux

  # D. 仅 Stage1 注入 HR-aux（360×720 分辨率获得地形/位置引导，高分辨率阶段不注入）
  python train.py --base_ch 256 --hr_aux_mode stage1 \\
      --run_dir runs/ablation/D_hr_aux_stage1_only

  # E. 纯 MAE（γ=0，关闭 FFT/Grad，等价于旧版训练，用作损失函数消融基准）
  python train.py --base_ch 256 \\
      --loss_gamma 0.0 --lambda_freq 0.0 --lambda_grad 0.0 \\
      --run_dir runs/ablation/E_pure_mae

  # F. 仅尾部加权（验证极值权重单独效果）
  python train.py --base_ch 256 \\
      --loss_gamma 0.5 --lambda_freq 0.0 --lambda_grad 0.0 \\
      --run_dir runs/ablation/F_tail_only

  # G. 完整组合损失（旧默认，含 FFT/Grad，用于消融对比新默认是否更优）
  python train.py --base_ch 256 \\
      --loss_gamma 0.5 --lambda_freq 0.1 --lambda_grad 0.05 \\
      --run_dir runs/ablation/G_combined_loss

  # H. 风光资源导向损失（新默认：TailMAE×逐变量通道权重 + SpatialExtreme(wind10,FSDS)×0.1，
  #    关闭 FFT/Grad；架构与 no_all 一致，用于验证新损失机制本身的效果）
  python train.py --base_ch 256 --no_cbam --hr_aux_mode none \\
      --loss_gamma 1.5 --loss_z_max 1.5 \\
      --lambda_freq 0 --lambda_grad 0 \\
      --var_weights "TAS=1.2,PRE=1.0,wind10=1.5,Q=1.0,2M_RH=1.2,2M_TMAX=1.0,2M_TMIN=1.0,FSDS=1.5" \\
      --extreme_vars wind10,FSDS --lambda_extreme 0.1 \\
      --val_extreme_z_thresh 1.5 \\
      --run_dir runs/ablation/H_wind_solar_focus

  # 同时打开所有 runs 的 TensorBoard：
  tensorboard --logdir runs/ablation

─────────────────────────────────────────────────────────────
"""

# 训练要点：组合损失（TailWeightedMAE + FFT + Gradient）、AdamW、线性 warmup + 余弦退火、梯度累积（默认 accum=2）、bf16 autocast、gradient checkpoint、TensorBoard 与 top-k 存盘。

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
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

# PRE 在写入 HDF5 前做了 log1p 变换（见 dataset.py / DOWNSCALE_README.md 第1节），
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

class TailWeightedMAE(nn.Module):
    """振幅自适应 MAE：以 z-score 绝对值为权重，提升极端值像素的梯度贡献。

    当 γ=0 且 channel_weight 全 1（或为 None）时退化为标准 MAE（nn.L1Loss 等价），保底不退步。

    Args:
        gamma:  权重斜率（默认 0.5）。w = 1 + γ·clamp(|y|, 0, z_max)
                γ=0.5, z_max=3 → 极端值像素（|z|=3）权重 ×2.5
        z_max:  权重上限对应的 z-score（默认 3.0），避免少数超极端值主导梯度
        channel_weight: 逐变量（通道）固定权重，形状 (C,)，与 per-pixel 的 z-score
                权重相乘叠加。用于面向具体应用（如风光资源评估）让特定变量
                （如 wind10、FSDS）在尾部加权之外获得额外的梯度预算倾斜。
                None 或全 1 时不产生影响。

    Note:
        权重 w 由 target（y_hr）计算并 detach()，不参与反向传播方向，
        仅改变各像素的梯度幅度。
    """

    def __init__(
        self,
        gamma: float = 0.5,
        z_max: float = 3.0,
        channel_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.z_max = z_max
        if channel_weight is not None:
            self.register_buffer("channel_weight", channel_weight.float())
        else:
            self.channel_weight = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.gamma == 0.0 and self.channel_weight is None:
            return (pred - target).abs().mean()
        w = (1.0 + self.gamma * target.abs().clamp(max=self.z_max)).detach()
        if self.channel_weight is not None:
            w = w * self.channel_weight[None, :, None, None]
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


class CombinedLoss(nn.Module):
    """训练用组合损失：TailWeightedMAE（可带逐变量通道权重） + λ_e·SpatialExtremeLoss
    + λ_f·FFTLoss + λ_g·GradientLoss。

    当所有附加项权重为 0（lambda_freq=0, lambda_grad=0, lambda_extreme=0, gamma=0,
    channel_weight 全 1）时，与 nn.L1Loss() 完全等价，保底不退步。

    默认配置面向风光资源评估场景：TailWeightedMAE 叠加逐变量通道权重（默认对
    wind10/FSDS/TAS/2M_RH/2M_TMIN 加权，PRE 保持中性），并额外用 SpatialExtremeLoss
    直接约束 wind10/FSDS 的区域极值还原（默认开启，λ_extreme=0.1）。FFT/Gradient
    两个通用图像质量损失默认关闭（λ_freq=λ_grad=0）——此前消融实验显示它们会
    拖累 TAS/2M_TMAX/2M_TMIN 等连续变量的逐点精度，且目标（整体频谱/边缘锐度）
    与风光资源极值这一具体目标不一致，容易稀释新机制的效果；两个类仍保留，
    可通过 --lambda_freq / --lambda_grad 显式打开做后续消融。

    各子项损失在 TensorBoard 中单独记录（Loss/tail_w、Loss/spatial_extreme、
    Loss/freq、Loss/grad），便于判断各项收敛情况和权重比例合理性。

    验证/早停使用独立的 nn.L1Loss()，保持与历史 run 的可比性。

    Args:
        gamma:          TailWeightedMAE 的权重斜率（默认 0.5；0 = 不做 z-score 尾部加权）
        z_max:          权重截断 z-score（默认 3.0）
        channel_weight: TailWeightedMAE 的逐变量通道权重，形状 (C,)（默认 None = 全 1）
        lambda_extreme: SpatialExtremeLoss 权重（默认 0；0 = 关闭）
        extreme_channel_indices: SpatialExtremeLoss 参与的通道下标列表
        lambda_freq:    FFT Loss 权重（默认 0；0 = 关闭）
        lambda_grad:    Gradient Loss 权重（默认 0；0 = 关闭）
    """

    def __init__(
        self,
        gamma: float = 0.5,
        z_max: float = 3.0,
        channel_weight: torch.Tensor | None = None,
        lambda_extreme: float = 0.0,
        extreme_channel_indices: list[int] | None = None,
        lambda_freq: float = 0.0,
        lambda_grad: float = 0.0,
    ):
        super().__init__()
        self.tail_loss  = TailWeightedMAE(gamma=gamma, z_max=z_max, channel_weight=channel_weight)
        self.fft_loss   = FFTLoss()   if lambda_freq > 0 else None
        self.grad_loss  = GradientLoss() if lambda_grad > 0 else None
        self.extreme_loss = (
            SpatialExtremeLoss(extreme_channel_indices)
            if lambda_extreme > 0 and extreme_channel_indices
            else None
        )
        self.lambda_freq    = lambda_freq
        self.lambda_grad    = lambda_grad
        self.lambda_extreme = lambda_extreme

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """返回 (total_loss, sub_losses_dict)。

        sub_losses_dict 键：'tail_w'、'spatial_extreme'（可选）、'freq'（可选）、'grad'（可选）。
        TensorBoard 记录由调用方完成（避免在 loss forward 中直接写 writer）。
        """
        l_tail = self.tail_loss(pred, target)
        sub = {"tail_w": l_tail.item()}
        total = l_tail

        if self.extreme_loss is not None:
            l_extreme = self.extreme_loss(pred, target)
            total     = total + self.lambda_extreme * l_extreme
            sub["spatial_extreme"] = l_extreme.item()

        if self.fft_loss is not None:
            l_freq = self.fft_loss(pred, target)
            total  = total + self.lambda_freq * l_freq
            sub["freq"] = l_freq.item()

        if self.grad_loss is not None:
            l_grad = self.grad_loss(pred, target)
            total  = total + self.lambda_grad * l_grad
            sub["grad"] = l_grad.item()

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
) -> tuple[float, float | None]:
    """验证模型并记录指标。返回 (mean_loss, resource_metric)。

    mean_loss：平均纯 MAE 验证损失（与历史 run 可比较）。
    始终使用纯 L1Loss 作为 Loss/val（早停 / top-K / best.pt 的判据），
    与训练用组合损失解耦，保证跨 run（包括纯 MAE / 组合损失等不同配置）的可比性。
    这是模型选择的**唯一**判据，不受 --resource_vars 等业务侧配置影响。

    resource_metric：`resource_var_indices` 指定的变量（默认 wind10/FSDS，见
    --resource_vars）在物理量纲上的平均 MAE，**仅作为并行的业务侧参考指标**
    （用于额外维护一份 best_resource.pt，见 main() 中的用法），不参与、也不
    替代上面的早停/top-K/best.pt 判据——避免"以偏概全"地只根据两个变量的表现
    去挑选对全部 8 变量负责的通用 checkpoint。resource_var_indices 为 None 或
    空列表时返回 None。

    在此基础上，额外记录两类"完整对比"指标，用于评估组合损失是否真的比纯 MAE
    在其设计目标上更优（而不是仅比较对两者都不公平的整体 MAE）：

      1. 组合损失各子项在验证集上的取值（Loss/val_combined、Loss/val_tail_w、
         Loss/val_spatial_extreme、Loss/val_freq、Loss/val_grad）—— 用与训练完全
         相同的 criterion 计算，直接反映该 run 的损失配置在其自身优化目标上的
         验证表现。对应权重为 0 的子项会退化/缺失，但仍可跨 run 对比
         spatial_extreme/freq/grad 子项，判断风光资源极值、频谱、梯度保真度
         是否真的因新损失机制而改善。

      2. 极端值 MAE（MAE_val_extreme/<var>，物理量纲）—— 仅在 |y_zscore| >
         extreme_z_thresh 的像素上统计 MAE，直接衡量"极端方面"的表现，
         这正是 TailWeightedMAE 设计要优化的区域，也是纯 MAE 整体指标
         无法体现、容易被大量"平静"像素稀释掉的部分。
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
    all_reduce_sum_(per_var_mae)
    all_reduce_sum_(extreme_mae_sum)
    all_reduce_sum_(extreme_pix_count)
    per_var_mae = per_var_mae.cpu()
    extreme_mae_sum = extreme_mae_sum.cpu()
    extreme_pix_count = extreme_pix_count.cpu()

    mean_loss    = total_loss / max(1, n_samples_f)
    per_var_mae /= max(1, n_samples_f)
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
        if is_main_process():
            print(f"    {vname:>10s}: mae={mae_val:.4f}  "
                  f"extreme_mae={extreme_val:.4f} (n_pix={int(n_extreme)})")

    resource_metric: float | None = None
    if resource_var_indices:
        resource_metric = per_var_mae[resource_var_indices].mean().item()
        writer.add_scalar("MAE_val_resource/mean", resource_metric, global_step)
        if is_main_process():
            names = ", ".join(VARIABLES[i] for i in resource_var_indices)
            print(f"  [epoch {epoch:03d} val]  resource_mean_mae({names})={resource_metric:.4f}"
                  f"  （业务侧参考指标，不参与 best.pt 判据）")

    model.train()
    return mean_loss, resource_metric


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    ema: "ModelEMA | None" = None,
) -> None:
    """保存 epoch、全局步数、最优验证损失及优化器/调度器状态，便于断点续训。

    分布式训练下仅 rank0 应调用本函数（由调用方保证）；`model` 若为 torch.compile/DDP
    包装，此处自动解包，保存的 state_dict 与单卡训练完全一致（不含 "module." 前缀、
    不含 compile 包装层），确保 checkpoint 可在单卡 / 任意 world_size / 是否 compile
    下互相加载。

    ema 非 None 时额外保存 "model_ema"（EMA 影子权重，供部署/推理使用；"model" 字段
    仍是训练用的"在线"权重，用于正确恢复训练轨迹，二者不要混淆）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch":         epoch,
        "global_step":   global_step,
        "best_val_loss": best_val_loss,
        "model":         unwrap_model(model).state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
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
) -> tuple[int, int, float]:
    """加载检查点。返回 (start_epoch, global_step, best_val_loss)。

    `model` 若为 torch.compile/DDP 包装，state_dict 加载到底层原始模块（unwrap_model），
    与保存时的裸模型 state_dict 格式对应；各 rank 均需调用本函数以保持模型/
    优化器/调度器状态一致（DDP 要求所有进程参数初始值相同）。

    ema 非 None 时尝试恢复 EMA 影子权重；若 checkpoint 是旧版（未启用过 EMA、无
    "model_ema" 字段），则用当前（刚加载完 "model" 的）在线权重重新初始化 EMA，
    而不是报错中断——语义等价于"从这个 epoch 开始才启用 EMA"。
    """
    # map_location 保证在 CPU/GPU 上都能正确加载 state_dict
    ckpt = torch.load(path, map_location=device)
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
    if is_main_process():
        print(f"Resumed: epoch={ckpt['epoch']+1}, step={ckpt['global_step']}, "
              f"best_val={ckpt['best_val_loss']:.6f}")
    return ckpt["epoch"] + 1, ckpt["global_step"], ckpt["best_val_loss"]


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
            "导致的 loss 抖动（见 DOWNSCALE_README.md 第7节）；代价是与 'batch' 版本的 "
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
        default="TAS=1.2,PRE=1.0,wind10=1.5,Q=1.0,2M_RH=1.2,2M_TMAX=1.0,2M_TMIN=1.0,FSDS=1.5",
        type=str,
        help=(
            "TailWeightedMAE 的逐变量（通道）固定权重，格式为 'VAR=weight' 逗号分隔，"
            "键必须属于 VARIABLES（见 dataset.py）。与 z-score 尾部权重相乘叠加，"
            "用于让特定变量在极端值学习上获得更多梯度预算。"
            "默认面向风光资源评估场景：wind10/FSDS（核心资源变量）权重 1.5，"
            "TAS/2M_RH（组合可反映湿球温度 WBT，与人体舒适度、电力需求相关）权重 1.2，"
            "PRE/Q/2M_TMAX/2M_TMIN 保持中性权重 1.0（不特别加权也不打压）。"
        ),
    )
    p.add_argument(
        "--extreme_vars",
        default="wind10,FSDS",
        type=str,
        help=(
            "参与 SpatialExtremeLoss（区域最大/最小值一致性损失）的变量，"
            "逗号分隔，键必须属于 VARIABLES。默认仅 wind10、FSDS——风光资源评估"
            "最核心的两个资源变量。留空字符串则不启用该损失（等价于 --lambda_extreme 0）。"
        ),
    )
    p.add_argument(
        "--lambda_extreme",
        default=0.1,
        type=float,
        help=(
            "SpatialExtremeLoss 权重（默认 0.1）。"
            "对 --extreme_vars 指定的通道，约束预测与目标在样本区域内的空间"
            "最大值/最小值一致（参考 NREL Sup3rWind 等风资源超分辨率工作的做法），"
            "直接保护风速极大值、辐照度晴空峰值/云遮骤降等风光资源评估关心的极值特征。"
            "0 = 关闭。"
        ),
    )
    p.add_argument(
        "--resource_vars",
        default="wind10,FSDS",
        type=str,
        help=(
            "业务侧关注变量（默认 wind10,FSDS，即风光资源核心变量），逗号分隔，"
            "键必须属于 VARIABLES。用于在训练过程中额外维护一份 best_resource.pt "
            "checkpoint（按这些变量的物理量纲平均 MAE 取最优），与 best.pt（始终按"
            "全部 8 变量的纯 MAE 选取，早停/top-K 判据不变）并行存在、互不影响。"
            "对应 TensorBoard 标量 MAE_val_resource/mean。留空字符串则不启用。"
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
            "DOWNSCALE_README.md 第 7 节）。传本参数关闭该转换（单卡运行时无影响）。"
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
    # 种子在所有 rank 上相同 ⇒ 划分结果在所有 rank 上完全一致
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
        # batch_size=1 时单进程 BN 统计噪声大（见 DOWNSCALE_README.md 第 7 节），
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
    resource_indices = parse_var_list(args.resource_vars)

    criterion = CombinedLoss(
        gamma                    = args.loss_gamma,
        z_max                    = args.loss_z_max,
        channel_weight           = channel_weight,
        lambda_extreme           = args.lambda_extreme,
        extreme_channel_indices  = extreme_indices,
        lambda_freq              = args.lambda_freq,
        lambda_grad              = args.lambda_grad,
    ).to(device)
    var_weight_str = ", ".join(f"{v}={w:g}" for v, w in zip(VARIABLES, channel_weight.tolist()))
    extreme_var_str = ", ".join(VARIABLES[i] for i in extreme_indices) or "none"
    loss_cfg = (
        f"loss: TailWeightedMAE(γ={args.loss_gamma}, z_max={args.loss_z_max}, "
        f"channel_weight=[{var_weight_str}])"
        f" + SpatialExtreme×{args.lambda_extreme}(vars=[{extreme_var_str}])"
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
    # 不影响 best_val_loss/early_stop_streak/top-K 等标准判据
    best_resource_val = float("inf")

    if args.resume:
        start_epoch, global_step, best_val_loss = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, device, ema=ema
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
            )
        writer.close()
        if is_main_process():
            print("[eval_only] 完成，未进行训练。")
        cleanup_distributed()
        return

    # 连续多少次「验证」未带来相对历史最优的改进（按验证次数计，非裸 epoch）
    early_stop_streak = 0
    stopped_early = False

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

        optimizer.zero_grad()

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
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
                )
            improved = val_loss < (best_before_val - args.early_stop_min_delta)
            if args.early_stop_patience > 0:
                if improved:
                    early_stop_streak = 0
                else:
                    early_stop_streak += 1
                writer.add_scalar("EarlyStop/streak", early_stop_streak, global_step)

            # checkpoint 落盘仅由 rank0 执行（val_loss 已在 validate() 内跨 rank 聚合，
            # 所有进程算出的值一致，因此 best_val_loss/early_stop_streak 等状态变量在
            # 各 rank 上天然保持同步，无需额外广播）
            if is_main_process():
                ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}_valloss{val_loss:.6f}.pt"
                save_checkpoint(
                    ckpt_path, model, optimizer, scheduler,
                    epoch, global_step, best_val_loss, ema=ema,
                )
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
                save_checkpoint(
                    ckpt_dir / "latest.pt", model, optimizer, scheduler,
                    epoch, global_step, best_val_loss, ema=ema,
                )

            # save/overwrite best.pt（best_val_loss 在所有 rank 上算出的值相同，
            # 但落盘仍只由 rank0 执行）
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if is_main_process():
                    best_path = ckpt_dir / "best.pt"
                    save_checkpoint(
                        best_path, model, optimizer, scheduler,
                        epoch, global_step, best_val_loss, ema=ema,
                    )
                    print(f"  New best val_loss={best_val_loss:.6f} → {best_path}")

            # 并行维护 best_resource.pt（--resource_vars 指定变量的物理量纲平均 MAE
            # 最优时保存），纯粹是给业务侧（本项目关注 wind10/FSDS）多一个可选起点，
            # 与上面 best_val_loss/best.pt 的标准判据完全独立、互不覆盖
            if resource_metric is not None and resource_metric < best_resource_val:
                best_resource_val = resource_metric
                if is_main_process():
                    best_resource_path = ckpt_dir / "best_resource.pt"
                    save_checkpoint(
                        best_resource_path, model, optimizer, scheduler,
                        epoch, global_step, best_val_loss, ema=ema,
                    )
                    print(f"  New best resource_metric={best_resource_val:.6f} "
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
