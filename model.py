"""
基于 PixelShuffle 的气候变量降尺度模型。

- 输入 (LR)：`data/x` → `(B, 8, 180, 360)` float16（仅 8 个气候变量，LR 静态特征移出）
- 目标 (HR)：`data/y` → `(B, 8, 1801, 3600)` float16
- HR 辅助通道（逐 stage 注入，原始分辨率 1801×3600）：
| `dem_hr_norm` | 1 |
| `latlon_sincos_hr` | 4 |
| `land_sea_mask_hr` | 1 |
| `cos(SZA)_hr`（在线计算） | 1 |
| 合计 HR aux | 7 ch |


结构自下而上（bottom-up）：
  InitConv（1×1 单层，in_ch→base_ch @ 180×360；默认 in_ch=8, base_ch=256）
  UpStage × 4（每层通过 PixelShuffle 实现 2× 上采样，Stage1-3 后进行 HR 辅助通道注入）
    Stage 1-3 shuffle_conv: 3×3（保留空间感受野）
    Stage 4   shuffle_conv: 1×1（默认，消除 1440×2880 下 17.80 GiB im2col workspace）
  双线性插值到 1801×3600（align_corners=True）
  Head（base_ch+hr_aux_ch → base_ch → 8 通道）

类定义顺序：
  1. ChannelAttention（通道注意力，M1：reduction=4，仅 GAP）
  2. SpatialAttention（空间注意力，M1：kernel_size=5）
  3. CBAMBlock（通道 + 空间注意力模块）
  4. ResBlock（残差块）
  5. UpStage（单阶段上采样结构）
  6. PixelShuffleDownscaleNet   ← 主模型

消融实验开关（PixelShuffleDownscaleNet 构造参数）：
  use_cbam       : bool — 是否启用每阶段的 CBAM 注意力（默认 True）
  hr_aux_mode    : str  — HR 辅助特征注入策略（默认 'all'）
                   'all'    — Stage 1-3 均注入 + Head 拼接（完整版）
                   'stage1' — 仅 Stage 1 注入，Stage 2-3 及 Head 不拼接
                   'none'   — 全程不注入；forward 仍接受 hr_aux 参数但忽略
        use_checkpoint : bool — 是否对 4 个 UpStage 全部启用 gradient checkpointing（默认 True）
                   节省激活值显存，backward 时重算各 stage；适合 base_ch=256 的大通道设置
  norm_type      : str  — ResBlock/InitConv/Head 的归一化层类型（默认 'batch'）
                   'batch' — nn.BatchNorm2d（默认，与历史版本一致）
                   'group' — nn.GroupNorm（不依赖 batch 维统计量，小 batch/DDP 每卡 batch=1
                              场景更稳健；与 'batch' 版本 checkpoint 不兼容，需从头训练）
  interp_chunk_channels : int — Stage4→Head 前 256 通道整体插值时的分块大小（默认 32）
                   规避 ROCm/HIP 平台 upsample_bilinear2d 实测按 fp32 分配输出（不受
                   autocast 影响）导致的单次 ~15.8 GiB 连续显存分配 OOM 风险；与不分块
                   结果逐元素相同，纯粹是显存分配策略，不改变模型语义/权重
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt


# ---------------------------------------------------------------------------
# 归一化层工厂：BatchNorm2d（默认，与历史版本一致）或 GroupNorm（--norm_type group）
# ---------------------------------------------------------------------------
# BatchNorm2d 依赖 batch 维统计量，--batch_size 1 时统计噪声大（见 DOWNSCALE_README.md
# 第 4 节）；GroupNorm 按通道分组在单样本内部统计，不依赖 batch 维，对小 batch 更稳健，
# 代价是与 BatchNorm 版本的 checkpoint 不兼容（层类型不同），需从头训练。

def _num_groups(channels: int, target: int = 32) -> int:
    """选取能整除 channels 且不超过 target 的最大分组数，保证 GroupNorm 合法。"""
    g = min(target, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return max(g, 1)


def _make_norm(channels: int, norm_type: str) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm2d(channels)
    if norm_type == "group":
        return nn.GroupNorm(_num_groups(channels), channels)
    raise ValueError(f"norm_type must be 'batch' or 'group'; got {norm_type!r}")


# ---------------------------------------------------------------------------
# 1. ChannelAttention  （M1 修复：reduction=4，仅 GAP，去掉 GMP）
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """SE-style channel attention（M1 版本）。

    仅使用 Global Average Pooling，移除 Global Max Pooling 分支。
    GMP 在气候数据中容易被极端值主导，导致注意力权重不稳定（ESCA 2024）。
    reduction 从 16 降至 4，避免高压缩比下的信息瓶颈破坏跨变量物理协变关系。
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 16)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=(2, 3))              # (B, C) — 仅全局平均池化
        attn = torch.sigmoid(self.mlp(avg))
        return x * attn.unsqueeze(-1).unsqueeze(-1)


# ---------------------------------------------------------------------------
# 2. SpatialAttention  （M1 修复：kernel_size=5）
# ---------------------------------------------------------------------------

class SpatialAttention(nn.Module):
    """Spatial attention: channel-wise avg+max concat → conv → sigmoid gate.

    kernel_size 从 7 调整为 5：气候降尺度的特征图分辨率高（180×360 起），
    5×5 在保持足够感受野的同时减少空间注意力图的过度平滑。
    """

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        mx  = x.amax(dim=1, keepdim=True)   # (B, 1, H, W)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


# ---------------------------------------------------------------------------
# 3. CBAMBlock
# ---------------------------------------------------------------------------

class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (CBAM, ECCV 2018).

    Applies channel attention followed by spatial attention in sequence.
    使用 M1 修复后的 ChannelAttention（reduction=4, 仅 GAP）和
    SpatialAttention（kernel_size=5）。
    """

    def __init__(self, channels: int, reduction: int = 4, spatial_kernel: int = 5):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# ---------------------------------------------------------------------------
# 4. ResBlock
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Basic residual block: Conv-Norm-ReLU-Conv-Norm + identity skip。

    norm_type: 'batch'（默认，nn.BatchNorm2d）或 'group'（nn.GroupNorm，见上方 _make_norm）。
    """

    def __init__(self, channels: int, norm_type: str = "batch"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _make_norm(channels, norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _make_norm(channels, norm_type),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


# ---------------------------------------------------------------------------
# 5. UpStage
# ---------------------------------------------------------------------------

class UpStage(nn.Module):
    """单步 2× 上采样模块。

    流程:
        ResBlock × num_resblocks
        → [可选] CBAMBlock 注意力模块
        → Conv(C → 4C) + PixelShuffle(2)   → 空间分辨率 ×2，通道数保持 C
        → [可选] concat HR-aux（hr_aux_ch 个通道） + Conv1×1(C+hr_aux_ch → C)

    参数说明:
        channels:        本阶段特征通道数
        hr_aux_ch:       PixelShuffle 后注入的 HR-aux 通道数；0 = 不注入（Stage 4）
        num_resblocks:   ResBlock 数量（默认 2）
        use_cbam:        是否在 PixelShuffle 前插入 CBAM（默认 True）
        shuffle_conv_k:  shuffle_conv 的卷积核大小（默认 3）
                         Stage 4（1440×2880 输入）使用 1 可消除 ~17.8 GiB im2col workspace，
                         其余 stage 前面 ResBlock 已提供足够空间混合，1×1 亦可；
                         3×3 在低分辨率 stage 有助于捕捉跨像素物理结构，建议保留。
    """

    def __init__(
        self,
        channels: int,
        hr_aux_ch: int = 7,
        num_resblocks: int = 2,
        use_cbam: bool = True,
        shuffle_conv_k: int = 3,
        norm_type: str = "batch",
    ):
        super().__init__()
        self.res_blocks = nn.Sequential(
            *[ResBlock(channels, norm_type=norm_type) for _ in range(num_resblocks)]
        )

        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAMBlock(channels)

        # PixelShuffle(2): 输入通道须为输出通道的 4 倍
        # shuffle_conv_k=1 时无 im2col workspace，适用于高分辨率 stage（Stage 4）
        _pad = shuffle_conv_k // 2
        self.shuffle_conv  = nn.Conv2d(channels, channels * 4, shuffle_conv_k, padding=_pad, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(2)

        self.hr_aux_ch = hr_aux_ch
        if hr_aux_ch > 0:
            self.inject_conv = nn.Conv2d(channels + hr_aux_ch, channels, 1, bias=False)
        else:
            self.inject_conv = None

    def forward(
        self,
        x: torch.Tensor,
        hr_aux_interp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, C, H, W)
            hr_aux_interp:  (B, hr_aux_ch, 2H, 2W)，已插值到本阶段输出分辨率；
                            hr_aux_ch=0 时传 None。
        Returns:
            (B, C, 2H, 2W)
        """
        x = self.res_blocks(x)

        if self.use_cbam:
            x = self.cbam(x)

        x = self.pixel_shuffle(self.shuffle_conv(x))   # (B, C, 2H, 2W)

        if self.inject_conv is not None and hr_aux_interp is not None:
            x = self.inject_conv(torch.cat([x, hr_aux_interp], dim=1))

        return x


# ---------------------------------------------------------------------------
# 6. PixelShuffleDownscaleNet  （主模型，必须为最后一个类）
# ---------------------------------------------------------------------------

class PixelShuffleDownscaleNet(nn.Module):
    """4-stage PixelShuffle 气候变量降尺度网络。

    上采样路径（每 stage 分辨率 ×2）：
        Stage 1: 180×360   → 360×720     （注入 HR-aux，若 use_hr_aux=True）
        Stage 2: 360×720   → 720×1440    （注入 HR-aux，若 use_hr_aux=True）
        Stage 3: 720×1440  → 1440×2880   （注入 HR-aux，若 use_hr_aux=True）
        Stage 4: 1440×2880 → 2880×5760   （不注入，超过目标分辨率）

    再经双线性插值（align_corners=True）缩放到目标分辨率（1801×3600），
    与原生分辨率的 HR-aux 拼接后，经 Head 输出 8 个预测变量。

    基础参数：
        in_ch:         LR 输入通道数（默认 8：仅 8 个气候变量）
        hr_aux_ch:     HR 辅助通道数（默认 7：6 HR-static + 1 cos_sza_hr）
        base_ch:       各阶段统一特征通道数（默认 256）
        num_resblocks: 每阶段 ResBlock 数（默认 2）
        out_ch:        输出通道数（默认 8，与预测变量数一致）
        target_h/w:    目标格点尺寸（默认 1801×3600）
        stage4_shuffle_conv_k: Stage 4 的 shuffle_conv 卷积核大小（默认 1）
                       Stage 4 输入 1440×2880，3×3 时 im2col workspace 高达 17.80 GiB，
                       直接导致 base_ch=256 下 OOM；改为 1×1 消除该 workspace，
                       前置 ResBlock（3×3）已完成空间混合，质量影响可忽略。
                       Stage 1-3 的分辨率较低，保留 3×3 有助于捕捉跨像素物理结构。

    消融实验开关：
        use_cbam       : 是否在每个 UpStage 中使用 CBAM 注意力（默认 True）
        hr_aux_mode    : HR 辅助特征注入策略（默认 'all'）
                         'all'    — Stage 1-3 均注入 + Head 拼接（完整版，默认）
                         'stage1' — 仅 Stage 1（360×720）注入，Stage 2-3 及 Head 不拼接
                         'none'   — 全程不注入，head_in = base_ch
        norm_type      : ResBlock/InitConv/Head 归一化层类型（默认 'batch'，另见 'group'/GroupNorm）
        use_checkpoint : 是否启用 gradient checkpointing（默认 True）
                         覆盖范围：全部 4 个 UpStage + 最终 bilinear interp + Head
                         显存节省：
                           · 4 个 stage 的内部激活值不驻留（已有）
                           · stage4 输出 (2880×5760×base_ch, ~7.9 GiB @256ch bf16) 不驻留（新增）
                           · head 中间张量 ×3 (~9.3 GiB @256ch bf16) 不驻留（新增）
                           · 合计释放 ~17 GiB，base_ch=256 峰值从 >42 GiB 降至 ~25 GiB
                         重计算代价：interp+head 重算开销 ≈ 7,700 GFLOPs（仅 stage4 重算的 0.20×）
    """

    def __init__(
        self,
        in_ch: int = 8,
        hr_aux_ch: int = 7,
        base_ch: int = 256,
        num_resblocks: int = 2,
        out_ch: int = 8,
        target_h: int = 1801,
        target_w: int = 3600,
        use_cbam: bool = True,
        hr_aux_mode: str = "all",
        use_checkpoint: bool = True,
        stage4_shuffle_conv_k: int = 1,
        norm_type: str = "batch",
        interp_chunk_channels: int = 32,
        interp_backend: str = "interpolate",
    ):
        if hr_aux_mode not in ("all", "stage1", "none"):
            raise ValueError(f"hr_aux_mode must be 'all', 'stage1', or 'none'; got {hr_aux_mode!r}")
        if norm_type not in ("batch", "group"):
            raise ValueError(f"norm_type must be 'batch' or 'group'; got {norm_type!r}")
        if interp_backend not in ("interpolate", "grid_sample"):
            raise ValueError(f"interp_backend must be 'interpolate' or 'grid_sample'; got {interp_backend!r}")

        super().__init__()
        self.target_h      = target_h
        self.target_w      = target_w
        self.hr_aux_mode   = hr_aux_mode
        self.hr_aux_ch     = hr_aux_ch
        self.use_checkpoint = use_checkpoint
        self.norm_type     = norm_type
        self.interp_chunk_channels = interp_chunk_channels
        self.interp_backend        = interp_backend

        # 各阶段和 Head 实际使用的辅助通道数
        _s1_aux   = hr_aux_ch if hr_aux_mode in ("all", "stage1") else 0
        _s23_aux  = hr_aux_ch if hr_aux_mode == "all" else 0
        _head_aux = hr_aux_ch if hr_aux_mode == "all" else 0

        # ---- 初始特征提取：1×1 Conv（纯通道映射，无空间感受野） ----
        # 让后续 ResBlock 的 3×3 承担空间特征提取，init_conv 仅做通道扩展
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 1, bias=False),
            _make_norm(base_ch, norm_type),
            nn.ReLU(inplace=True),
        )

        # ---- 4 个上采样阶段 ----
        _stage_kwargs = dict(num_resblocks=num_resblocks, use_cbam=use_cbam, norm_type=norm_type)
        self.stage1 = UpStage(base_ch, hr_aux_ch=_s1_aux,  **_stage_kwargs)
        self.stage2 = UpStage(base_ch, hr_aux_ch=_s23_aux, **_stage_kwargs)
        self.stage3 = UpStage(base_ch, hr_aux_ch=_s23_aux, **_stage_kwargs)
        # Stage 4 输入分辨率 1440×2880，3×3 shuffle_conv 的 im2col workspace 高达 17.80 GiB，
        # 直接导致 base_ch=256 下 OOM；默认使用 1×1（stage4_shuffle_conv_k=1）消除该峰值。
        self.stage4 = UpStage(base_ch, hr_aux_ch=0, shuffle_conv_k=stage4_shuffle_conv_k, **_stage_kwargs)

        # ---- 输出 Head ----
        head_in = base_ch + _head_aux
        self.head = nn.Sequential(
            nn.Conv2d(head_in, base_ch, 3, padding=1, bias=False),
            _make_norm(base_ch, norm_type),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, out_ch, 1),
        )

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def _interp(t: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """地理格点对齐的双线性插值（align_corners=True）。"""
        return F.interpolate(t, size=size, mode="bilinear", align_corners=True)

    @staticmethod
    def _interp_chunked(
        t: torch.Tensor,
        size: tuple[int, int],
        chunk_channels: int = 32,
    ) -> torch.Tensor:
        """按通道分块的双线性插值，数学上与 `_interp` 完全等价（插值按通道独立，
        不存在跨通道混合），用于规避大 Tensor 单次插值在本平台(ROCm/HIP)上的显存问题。

        实测（2026-09-14，8 卡 DDP 全量数据首个 batch）：Stage4 输出(2880×5760×256)
        →Head 前插值到 1801×3600 这一步，`torch.OutOfMemoryError: Tried to allocate
        15.82 GiB` —— 精确等于 `2880×5760×256×4byte(fp32)`，是本该按 bf16 计算的
        `~7.9 GiB` 的整整 2 倍。说明本平台 `upsample_bilinear2d` 算子的输出实际按
        fp32 分配，不受 `torch.autocast(dtype=bf16)` 影响（很可能是 ROCm 该算子缺
        原生 bf16 kernel，内部回退到 fp32）。64 GiB 卡上这一次性 ~15.8 GiB 连续分配
        本就很紧，DDP/NCCL 初始化开销叠加后就会在首个 batch 直接 OOM。

        做法：把 256 通道拆成若干 `chunk_channels` 大小的小块分别插值（每块只需
        ~chunk_channels/256 比例的显存），每块算完立刻 `.to(原 dtype)` 转回 bf16
        再拼接——不仅单次分配变小，拼接后的最终结果也只需 bf16 大小的连续显存
        （约为不转换时的一半），双重降低 OOM 风险，且结果与一次性插值逐元素相同。
        """
        in_dtype = t.dtype
        if t.shape[1] <= chunk_channels:
            return F.interpolate(t, size=size, mode="bilinear", align_corners=True).to(in_dtype)
        chunks = torch.split(t, chunk_channels, dim=1)
        out_chunks = [
            F.interpolate(c, size=size, mode="bilinear", align_corners=True).to(in_dtype)
            for c in chunks
        ]
        return torch.cat(out_chunks, dim=1)

    @staticmethod
    def _interp_grid_sample(t: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """用 `F.grid_sample` 实现与 `_interp`（align_corners=True 双线性）等价的重采样。

        `grid_sample` 走的是 `aten::grid_sampler_2d`，与 `F.interpolate` 的
        `aten::upsample_bilinear2d` 是两套独立 kernel；在某些后端（包括不排除
        ROCm/HIP）两者的 dtype 支持情况可能不同，值得在实测中对比是否能规避
        `_interp` 那个强制 fp32 输出的问题。若实测发现 grid_sample 同样 fp32
        fallback，则放弃此路径，继续用 `_interp_chunked`。

        数学推导：align_corners=True 时，输出坐标 i∈[0,H_out-1] 映射到输入坐标
        `y = i*(H_in-1)/(H_out-1)`；grid_sample 的归一化坐标（align_corners=True）
        为 `y_norm = 2y/(H_in-1) - 1 = 2i/(H_out-1) - 1`，恰好是不依赖 H_in 的
        `linspace(-1, 1, H_out)` —— 与 `_interp` 理论上给出相同结果，但两者是
        不同的浮点实现路径，逐元素会有 ~1e-5 量级的浮点误差（非 bug，是正常的
        浮点重排差异），对训练精度的影响可忽略。
        """
        B, _, H_in, W_in = t.shape
        H_out, W_out = size
        ys = torch.linspace(-1, 1, H_out, device=t.device, dtype=torch.float32)
        xs = torch.linspace(-1, 1, W_out, device=t.device, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1).to(t.dtype)
        return F.grid_sample(t, grid, mode="bilinear", align_corners=True, padding_mode="border")

    @staticmethod
    def _interp_grid_sample_chunked(
        t: torch.Tensor,
        size: tuple[int, int],
        chunk_channels: int = 32,
    ) -> torch.Tensor:
        """`_interp_grid_sample` 的分块版本，与 `_interp_chunked` 同样的显存策略，
        便于在实测中直接对比 grid_sample 与 interpolate 两条路径的显存/吞吐。
        """
        in_dtype = t.dtype
        if t.shape[1] <= chunk_channels:
            return PixelShuffleDownscaleNet._interp_grid_sample(t, size).to(in_dtype)
        chunks = torch.split(t, chunk_channels, dim=1)
        out_chunks = [
            PixelShuffleDownscaleNet._interp_grid_sample(c, size).to(in_dtype)
            for c in chunks
        ]
        return torch.cat(out_chunks, dim=1)

    def _resize_head_input(self, f: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """按 self.interp_backend 分发到 interpolate 或 grid_sample 的分块实现。"""
        if self.interp_backend == "grid_sample":
            return self._interp_grid_sample_chunked(f, size, chunk_channels=self.interp_chunk_channels)
        return self._interp_chunked(f, size, chunk_channels=self.interp_chunk_channels)

    @staticmethod
    def _run_stage(
        stage: UpStage,
        aux: torch.Tensor | None,
    ):
        """返回一个以 x 为唯一 Tensor 参数的闭包，供 gradient checkpoint 包装。

        checkpoint 只支持 Tensor 参数的重算；将 aux 固定在闭包中，
        x 作为唯一的需要梯度的输入。
        """
        def fn(x: torch.Tensor) -> torch.Tensor:
            return stage(x, aux)
        return fn

    def _run_head(self, hr_aux_for_cat: torch.Tensor | None):
        """返回一个用于 checkpoint 包装 bilinear interp + Head 的闭包。

        将 stage4 输出 (2880×5760×base_ch, ~7.9 GiB @256ch bf16) 和
        Head 中间张量 (~9.3 GiB) 从常驻显存中移除，backward 时重算。
        hr_aux_for_cat: hr_aux_mode='all' 时传入原生 HR-aux 张量；否则 None。
        """
        target_h = self.target_h
        target_w = self.target_w
        head     = self.head

        def fn(f: torch.Tensor) -> torch.Tensor:
            f = self._resize_head_input(f, (target_h, target_w))
            if hr_aux_for_cat is not None:
                f = torch.cat([f, hr_aux_for_cat], dim=1)
            return head(f)
        return fn

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x_lr: torch.Tensor,
        hr_aux: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_lr:   (B, 8, 180, 360)   — 归一化 LR 输入（8 个气候变量）
            hr_aux: (B, 7, 1801, 3600) — HR 辅助特征；
                    hr_aux_mode='none' 时传入但被忽略，DataLoader 无需修改。
        Returns:
            (B, 8, 1801, 3600)
        """
        f = self.init_conv(x_lr)   # (B, base_ch, 180, 360)

        _, _, h0, w0 = f.shape
        h1, w1 = h0 * 2, w0 * 2   # Stage 1 输出：360×720
        h2, w2 = h1 * 2, w1 * 2   # Stage 2 输出：720×1440
        h3, w3 = h2 * 2, w2 * 2   # Stage 3 输出：1440×2880

        ckpt = self.use_checkpoint and self.training

        # ------ 4 个上采样 stage（stage-specific HR-aux 注入差异在此处理）------
        if self.hr_aux_mode == "all":
            s1_aux = self._interp(hr_aux, (h1, w1))
            s2_aux = self._interp(hr_aux, (h2, w2))
            s3_aux = self._interp(hr_aux, (h3, w3))
            if ckpt:
                f = grad_ckpt(self._run_stage(self.stage1, s1_aux), f, use_reentrant=False)
                f = grad_ckpt(self._run_stage(self.stage2, s2_aux), f, use_reentrant=False)
                f = grad_ckpt(self._run_stage(self.stage3, s3_aux), f, use_reentrant=False)
                f = grad_ckpt(self._run_stage(self.stage4, None),   f, use_reentrant=False)
            else:
                f = self.stage1(f, s1_aux)
                f = self.stage2(f, s2_aux)
                f = self.stage3(f, s3_aux)
                f = self.stage4(f, None)

        elif self.hr_aux_mode == "stage1":
            s1_aux = self._interp(hr_aux, (h1, w1))
            if ckpt:
                f = grad_ckpt(self._run_stage(self.stage1, s1_aux), f, use_reentrant=False)
                f = grad_ckpt(self._run_stage(self.stage2, None),   f, use_reentrant=False)
                f = grad_ckpt(self._run_stage(self.stage3, None),   f, use_reentrant=False)
                f = grad_ckpt(self._run_stage(self.stage4, None),   f, use_reentrant=False)
            else:
                f = self.stage1(f, s1_aux)
                f = self.stage2(f, None)
                f = self.stage3(f, None)
                f = self.stage4(f, None)

        else:  # "none"
            if ckpt:
                f = grad_ckpt(self._run_stage(self.stage1, None), f, use_reentrant=False)
                f = grad_ckpt(self._run_stage(self.stage2, None), f, use_reentrant=False)
                f = grad_ckpt(self._run_stage(self.stage3, None), f, use_reentrant=False)
                f = grad_ckpt(self._run_stage(self.stage4, None), f, use_reentrant=False)
            else:
                f = self.stage1(f, None)
                f = self.stage2(f, None)
                f = self.stage3(f, None)
                f = self.stage4(f, None)

        # ------ 统一末尾路径：bilinear interp → [cat hr_aux] → Head ------
        # hr_aux_mode='all' 时 Head 拼接原生 HR-aux；其余模式不拼接
        _hr_cat = hr_aux if self.hr_aux_mode == "all" else None
        if ckpt:
            # 将 stage4 输出 (~7.9 GiB) 和 head 中间张量 (~9.3 GiB) 也纳入 checkpoint，
            # backward 时重算，避免常驻显存。额外重计算量仅为 stage4 重算的 0.20×。
            return grad_ckpt(self._run_head(_hr_cat), f, use_reentrant=False)
        else:
            f = self._resize_head_input(f, (self.target_h, self.target_w))
            if _hr_cat is not None:
                f = torch.cat([f, _hr_cat], dim=1)
            return self.head(f)
