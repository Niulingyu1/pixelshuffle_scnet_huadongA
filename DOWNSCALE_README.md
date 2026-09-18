# PixelShuffle 气候降尺度：说明文档

---

## 1. 任务与数据


| 项目   | 说明                                                                                                          |
| ---- | ----------------------------------------------------------------------------------------------------------- |
| 目标   | 全球约 1°（180×360）→ 约 0.1°（1801×3600），8 个变量：`TAS`, `PRE`, `wind10`, `Q`, `2M_RH`, `2M_TMAX`, `2M_TMIN`, `FSDS` |
| HDF5 | 全量训练默认指向 `HDF5_ROOT_NORM`=`/public/home/acd7koea4a/hdf5_norm_fp16`：磁盘 `data/x`/`data/y` 为离线 z-score 后的 **float16**，`metadata.normalized=True`。原始备份仍在 `/public/share/acd7koea4a/hdf5`（`HDF5_ROOT_RAW`，fp32、未 z-score）。`hdf5_mini` / CESM 推理 HDF5 仍为未标准化物理量。`Dataset` 按 metadata 自动识别，出口 `x_lr`/`y_hr` 一律 **bfloat16**；`hr_aux` 保持 fp32。 |
| 预处理  | 写入原始 HDF5 前：`PRE` 为 `log1p`；`Q` 已 ×1000（g/kg）                                                                 |
| 归一化  | `global_stats_state.json` 中 `var_all` 的 mean / `sqrt(M2/n)`。预标准化数据离线完成 z-score；未标准化数据仍在 `dataset.py` 在线做。**静态与 cos(SZA) 不归一化** |

### 1.1 离线 z-score + fp16 副本

全量训练集已离线写成 `/public/home/acd7koea4a/hdf5_norm_fp16`（目录结构与文件名与原始 `hdf5/` 相同）：磁盘上 `data/x`/`data/y` 已是 z-score 后的 **float16**，`metadata.normalized=True`。`dataset.py` 读出后统一转为 **bfloat16** 再返回，与 `train.py` 的 `autocast(dtype=bfloat16)` 对齐；`hr_aux` 仍为在线 fp32。`hdf5_mini` / CESM 推理 HDF5 保持未标准化物理量。

```bash
python normalize_hdf5_fp16.py --dry_run
python normalize_hdf5_fp16.py --check_complete
python scripts/check_hdf5_norm_fp16.py \
    --src /public/share/acd7koea4a/hdf5 \
    --dst /public/home/acd7koea4a/hdf5_norm_fp16 \
    --seasons MAM --manifests cra1p5_full --n_check 8
```

`paths.py` 里 `HDF5_ROOT` 已指向 `HDF5_ROOT_NORM`。`infer.py --input_source hdf5` 若指向预标准化目录会直接报错，避免静默二次标准化。

2026-09-18 已在 BW1000（DTK torch 2.7.1）上跑通 `scripts/smoke_norm_fp16_step.py`：`pred`/`loss` 保持 bfloat16，前向+反向无 dtype 报错。正式训练用 `scripts/launch_platform_train.sh`（镜像自带 python，不要 conda）。

---

## 2. 输入 / 输出张量

- `**x_lr`**：`(B, 8, 180, 360)` **bfloat16** — 仅 8 个气象变量（已 z-score）；LR 静态与 `cos(SZA)_lr` 不再拼入（地形/位置/日射等由 `hr_aux` 在 HR 网格上注入）
- `**hr_aux**`：`(B, 7, 1801, 3600)` **float32** — 6 HR 静态 + 1 `cos(SZA)_hr`；在 `forward` 内按阶段双线性插值到各层输出尺寸
- **模型输出**：`(B, 8, 1801, 3600)`，与 `y_hr` 同形状；**无末端激活**（回归）

### 通道明细

**LR（8）**：与 HDF5 `data/x` 一致的 8 个变量通道（z-score 后）。

**HR 辅助（7）**：`dem_hr_norm`(1) + `latlon_sincos_hr`(4) + `land_sea_mask_hr`(1) + `cos(SZA)_hr`(1)

### cos(SZA)

- 日平均公式：赤纬 δ、日落时角 H₀、对纬向积分得到 `<cos(SZA)>`，clip 到 `[0,1]`；**仅依赖纬度与 DOY**（经向复制成 2D 栅格）
- LRU：`cos(SZA)_lr` `maxsize=400`；`cos(SZA)_hr` `maxsize=50`（控制显存/内存）
- 日期来自 HDF5 `data/dates`（如 `b'19790101'`）

---

## 3. 模型设计（`PixelShuffleDownscaleNet`）

**与代码同步的要点**：默认 `in_ch=8`（仅 8 变量）、`base_ch=256`、`InitConv` 为 1×1、CBAM 为 M1 变体（通道注意力仅 GAP、`reduction=4`、空间卷积核 5）、`Stage4` 的 `shuffle_conv` 默认 **1×1**（控制显存）。下文流程图通道数已与 `model.py` 当前默认（`base_ch=256`）保持一致。

整体思路：**四段 2× PixelShuffle 堆叠 → 空间上累计 16× → 双线性对齐目标格点 → 拼接原生 HR 辅助 → Head 出 8 通道**。残差在 **PixelShuffle 之前** 的卷积特征内（`ResBlock`），**不跨上采样做恒等相加**（通道/分辨率变化由卷积与 shuffle 承担）。

### 3.1 数据流与分辨率

（以下通道数与 `train.py` / `PixelShuffleDownscaleNet` 默认一致：`base_ch=256`。若改小 `base_ch`，图中通道数随之变化。）

```
180×360  ──InitConv──► base_ch（默认 256）
    │
    ├─ Stage1: Res×N + CBAM + PixelShuffle(2) + [concat HR_aux(7) → 1×1] ─► 360×720
    ├─ Stage2: … ─► 720×1440
    ├─ Stage3: … ─► 1440×2880
    └─ Stage4: …（不注 HR_aux）─► 2880×5760

2880×5760 ──bilinear, size=(1801,3600), align_corners=True──► base_ch
    │
    └─ concat 原生 hr_aux (7) ─► (base_ch+7) ──Head──► 8ch
```

- **Stage 1–3**：上采样后将 `hr_aux` 插到**当前层输出分辨率**，与特征拼接，**1×1 Conv** 压回 `base_ch`（默认 256）
- **Stage 4**：不注入 HR 辅助（中间分辨率已超过目标，避免冗余与显存）
- **对齐目标**：`F.interpolate(..., mode='bilinear', align_corners=True)`，使格网**角点**与地理端点对齐，减少整体偏移

### 3.2 单阶段 `UpStage`（内部顺序）

1. `**ResBlock` × `num_resblocks`（默认 2）**
  `Conv3×3` → BN → ReLU → `Conv3×3` → BN，**与输入恒等相加**后再 ReLU。
2. `**CBAMBlock`**（ECCV 2018 风格，自实现，当前为 **M1 修复版**，见 `model.py`）
  - **ChannelAttention**：**仅全局 avg pool**（已去掉 GMP，避免极端值主导注意力权重）→ 共享 MLP → sigmoid，**reduction=4**（`mid = max(C//4, 16)`）
  - **SpatialAttention**：通道维 mean/max 拼成 2 通道 → **5×5 Conv**（从 7×7 调整，减少空间注意力图过度平滑）→ sigmoid
  - 顺序：**先通道、后空间**
3. **上采样**
  `Conv3×3(C → 4C)`（无偏置）+ `**nn.PixelShuffle(2)`**：空间 2×、通道回到 C。  
   数学上等价于子像素卷积：用可学习权重把“高分辨率细节”编进通道维再展开。
4. **（可选）HR 注入**
  `torch.cat([f, hr_aux_interp], dim=1)` → `**Conv1×1(C+7 → C)`**（`hr_aux_ch=0` 时本层关闭）

### 3.3 首层与 Head

- **InitConv（当前 `model.py` 固定为单层 1×1，无其它可选项）**  
**单层 1×1**：`Conv2d(in_ch→base_ch, 1×1)` + BN + ReLU，即在 **不改变空间分辨率** 的前提下，将 `in_ch`（默认 8，仅气象变量，LR 静态已移出）个 LR 通道映射为 `base_ch` 维特征；**默认 `base_ch=256`** 时等价于「8→256」，主干全程保持该通道数。  
注：历史版本曾有 `--init_type 3x3`/`wide3x3` 等可选 init 结构，**当前 `model.py` 已简化为固定单层 1×1**，不再支持该参数；如需恢复请以 `model.py` 实际代码为准。
- **Head**（`hr_aux_mode=all` 时）：`Conv((base_ch+7)→base_ch, 3×3)` + BN + ReLU → `Conv(base_ch→8, 1×1)`。默认 `base_ch=256` 时首层为 **263→256**。`hr_aux_mode` 为 `stage1`/`none` 时首层输入为 `base_ch` 通道（无拼接）。

### 3.4 超参（与 `train.py` / `model.py` 当前默认一致）


| 参数                     | 默认                                      |
| ---------------------- | --------------------------------------- |
| `in_ch`                | 8（仅气象变量，LR 静态特征已移出，见第2节）             |
| `hr_aux_ch`            | 7                                       |
| `base_ch`              | 256                                     |
| `num_resblocks`        | 2                                       |
| `out_ch`               | 8                                       |
| `target_h`, `target_w` | 1801, 3600                              |
| `use_cbam`             | `True`（`--no_cbam` 关闭）                 |
| `hr_aux_mode`          | `all`（另见 `stage1`/`none`，`--hr_aux_mode` 指定） |
| `use_checkpoint`       | `True`（`--no_checkpoint` 关闭）           |
| `stage4_shuffle_conv_k`| 1（消除 Stage4 im2col workspace，固定值，非 CLI 参数） |


---

## 4. 训练损失函数设计（`train.py`）

训练阶段使用 `**CombinedLoss`**（v2，"参考目标 + 风光增量项"解耦设计，完整决策记录见
`loss_plan/final_decision.md`）：在 **z-score 空间** 对 `pred` 与 `y_hr` 计算，由一个恒定
参与的"参考目标"（面积加权 `TailWeightedMAE`）与若干可加性增量项组成（各增量项权重为 0
时自动关闭）；验证与早停仍只用 **纯 MAE**（`nn.L1Loss`，像素等权），与历史实验的 `Loss/val`
口径一致（详见 4.5 节与本文档「验证损失应如何保存才符合标准」一节的完整论证）。

> **本节已按 `train.py` v2 代码同步更新**（此前版本描述的是 v1：`TailWeightedMAE`+逐变量
> 通道偏置权重+`SpatialExtremeLoss`+FFT/Grad 四项固定组合。v2 改为解耦设计：参考目标改为
> 8 通道等权+面积加权，风光资源增强改由独立的 `PatchExtremeLoss`/`WindPowerSensitivityProxy`/
> `PhysicalConsistencyLoss` 负责，不再依赖通道间的零和式偏置加权；旧版机制全部保留、默认
> 关闭，供历史对照消融）。

### 4.1 总损失形式

```
L_train = L_tail(γ, z_max, c, area_weight)
        + λ_pe · L_patch_extreme
        + λ_wps · L_wind_power_sensitivity
        + λ_phys · L_phys_consistency
        + [旧版，默认关闭] λe · L_extreme  +  λf · L_FFT  +  λg · L_grad
```

其中：
- `L_tail`：**参考目标**，面积加权尾部加权 MAE（`TailWeightedMAE`），像素权重
  `w = 1 + γ·clamp(|y|, 0, z_max)`，再乘以逐通道固定权重向量 `c`（`--var_weights`，
  **v2 默认全 1**，不做通道间偏置）与 `cos(latitude)` 球面积权重（`--no_area_weight`
  可关闭）；
- `L_patch_extreme`：**局部网格极值一致性损失**（`PatchExtremeLoss`），对
  `--patch_extreme_vars` 指定通道，把 HR 场划分为局部网格块（默认 180×360），约束局部
  max/min 与 patch 一阶矩（mean）一致；**不约束极值位置**，确定性回归下可能抬高整块；
- `L_wind_power_sensitivity`：**风功率立方敏感区代理**（`WindPowerSensitivityProxy`），
  仅对 `wind10`、仅在真值 3–12 m/s 内约束归一化功率代理；额定以上本项沉默；**不是**
  能量产出对齐（10m 非轮毂高度、日均非瞬时）；
- `L_phys_consistency`：**物理一致性安全网**（`PhysicalConsistencyLoss`），**先反标准化**
  再约束温度顺序、非负性、RH 边界，再按训练期标准差无量纲化组合（禁止在 z 空间 hinge）；
- `L_extreme`/`L_FFT`/`L_grad`：v1 旧版机制（全球极值 / FFT 幅度谱 / Sobel 梯度），
  **v2 默认权重为 0**，仅保留供历史对照消融。

**保底等价**：当上述全部 λ 为 0、`--loss_gamma 0`、`--var_weights` 全 1、
`--no_area_weight` 时，`L_train` 与 `nn.L1Loss()` 完全等价。这只保证损失函数**数值**
的退化，不构成"训练出的模型在其余变量上必然不退步"的形式化证明——实际效果需看验证集结果。

### 4.2 各子项原理（与当前任务的关系）

| 子项 | 含义 | 作用 |
| --- | --- | --- |
| **TailWeightedMAE**（面积加权，v2 通道权重默认全 1） | 像素权重 = z-score 尾部权重 × 球面积权重（× 可选通道偏置，默认不启用） | 尾部权重提升**极端值像素**（任意通道）的梯度贡献；面积权重修正规则经纬网格在高纬度像素等权带来的面积高估；8 通道等权，不存在通道间零和式抢梯度 |
| **PatchExtremeLoss** | 对指定通道划分局部网格，约束局部 max/min 与 patch **一阶矩**（mean） | 提高极值监督密度。patch-mean 抑制为凑 max 而整体平移的退化解，**不约束极值落点**，也不解决位置错位；确定性回归下降低 patch-max 的省力解往往是抬高整块，可能以 MAE/偏差换极值统计（`λ_p` 偏大时冒烟看 patch 均值偏差）。详见 `loss_plan/final_decision.md` 第 11.4、11.5 节 |
| **WindPowerSensitivityProxy** | 仅 `wind10`，真值 3–12 m/s 内约束归一化功率代理（∝v³） | 只覆盖立方敏感区；**额定以上与切出区间本项沉默**，高风速由 tail + PatchExtreme 监督。日均 10m 风速大量落在该区间，mask 更可能过密。**不是**容量因子/发电量优化（10m、日均、Jensen 不等式）。冒烟必须看 `wps_mask_ratio` |
| **PhysicalConsistencyLoss** | **先** `pred·σ+μ` 反标准化，再 hinge，再除 `σ_k` | 约束 `TMIN≤TAS≤TMAX`（K）、非负、`RH∈[0,100]`（%）；PRE 还原后是 log1p，`≥0` 仍等价于物理降水非负。**禁止在 z 空间直接比较或写 `ReLU(−z)`**。损失均值小不代表梯度贡献小 |
| **[旧版] SpatialExtremeLoss** | 对 `--extreme_vars` 指定通道，约束预测/目标在整张全球样本上的 max/min 一致 | v1 机制，每张图每通道仅 2 个非零梯度像素，监督信号稀疏，已被 `PatchExtremeLoss` 取代为默认机制，默认权重 0，仅供对照 |
| **FFTLoss** | `rfft2(..., norm="ortho")` 幅度谱 L1 | 抑制过平滑；消融实验显示收益有限，会拖累部分变量的逐点精度，默认关闭 |
| **GradientLoss** | Sobel 梯度图 L1 | 强化锋面/地形陡坡等空间结构一致性；消融实验显示其原始量级远大于 tail_w，默认关闭 |

**未纳入主损失、可作延伸阅读**：Huber / Charbonnier；纯分位数或 GEV 似然；Log-PSD；概率式
CRPS。若后续要加，建议在 `CombinedLoss` 外单独做分支实验。

### 4.3 命令行参数与默认值（与 `train.py` v2 代码一致）

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--loss_gamma` | `0.5` | 尾部权重斜率 γ；`0` 时该项退化为普通 MAE。**作用于全部 8 个通道**，非逐变量 |
| `--loss_z_max` | `3.0` | 权重中对 `abs(y)` 的截断上界（z-score），抑制极少数超大值主导梯度 |
| `--var_weights` | `TAS=1.0,PRE=1.0,wind10=1.0,Q=1.0,2M_RH=1.0,2M_TMAX=1.0,2M_TMIN=1.0,FSDS=1.0` | `TailWeightedMAE` 的逐变量固定权重，与 z-score 尾部权重相乘叠加；**v2 默认全 1**（不做通道间偏置，风光增强改由下方独立增量项负责），仍可显式传入 v1 式偏置值用于对照消融 |
| `--no_area_weight` | 关闭该标志时默认开启面积加权 | 关闭训练损失中的 `cos(latitude)` 球面积权重；仅影响训练损失，不影响 `Loss/val` |
| `--extreme_vars` | `wind10,FSDS` | 参与旧版 `SpatialExtremeLoss` 的变量（v2 默认关闭，仅供对照） |
| `--lambda_extreme` | `0.0`（v2 默认关闭） | λe；旧版全局极值损失权重，已被 `--lambda_patch_extreme` 取代为默认机制 |
| `--patch_extreme_vars` | `wind10,FSDS` | 参与 `PatchExtremeLoss` 的变量；留空字符串关闭该损失 |
| `--lambda_patch_extreme` | `0.1` | λ_pe；`0` 关闭 `PatchExtremeLoss` |
| `--patch_grid_h` / `--patch_grid_w` | `180` / `360` | `PatchExtremeLoss` 局部网格大小，默认与 LR 输入网格同分辨率对齐 |
| `--lambda_wps` | `0.05` | λ_wps；`0` 关闭 `WindPowerSensitivityProxy` |
| `--wind_cutin` / `--wind_rated` | `3.0` / `12.0` | 功率代理曲线爬坡段边界（m/s，IEC 典型代理值，非机型标定值） |
| `--lambda_phys` | `0.02` | λ_phys；`0` 关闭 `PhysicalConsistencyLoss` |
| `--lambda_freq` | `0.0`（默认关闭） | λf；非零显式打开 FFT 项 |
| `--lambda_grad` | `0.0`（默认关闭） | λg；非零显式打开梯度项 |
| `--resource_vars` | `wind10,FSDS` | 用于计算 `best_resource.pt` 的 skill score（相对 LR 双线性插值 naive baseline），修复了旧版把不同物理量纲 MAE 直接算术平均的问题 |
| `--val_extreme_z_thresh` | `1.5` | 验证集 `MAE_val_extreme/*` 的极端像素判定阈值（z-score） |

### 4.4 使用示例

**默认（v2 推荐起点：面积加权尾部 MAE + PatchExtreme + WindPowerSensitivityProxy + PhysicalConsistency）**：

```bash
python train.py --base_ch 256 --no_cbam --hr_aux_mode stage1 --norm_type group \
  --run_dir runs/exp_loss_v2_default
# 等价于显式写出：
python train.py --base_ch 256 --no_cbam --hr_aux_mode stage1 --norm_type group \
  --loss_gamma 0.5 --loss_z_max 3.0 \
  --var_weights "TAS=1.0,PRE=1.0,wind10=1.0,Q=1.0,2M_RH=1.0,2M_TMAX=1.0,2M_TMIN=1.0,FSDS=1.0" \
  --patch_extreme_vars wind10,FSDS --lambda_patch_extreme 0.1 \
  --lambda_wps 0.05 --wind_cutin 3.0 --wind_rated 12.0 \
  --lambda_phys 0.02 \
  --lambda_extreme 0.0 --lambda_freq 0.0 --lambda_grad 0.0 \
  --run_dir runs/exp_loss_v2_default
```

**与纯 MAE 完全一致（用于对照基线，即已跑完的 `hr` 基线配置）**：

```bash
python train.py --base_ch 256 --no_cbam --hr_aux_mode stage1 --norm_type group \
  --loss_gamma 0.0 --var_weights "" --no_area_weight \
  --lambda_extreme 0.0 --lambda_patch_extreme 0.0 --lambda_wps 0.0 --lambda_phys 0.0 \
  --lambda_freq 0.0 --lambda_grad 0.0 \
  --run_dir runs/exp_loss_pure_mae
```

**调权建议**：`--loss_gamma`/`--loss_z_max` 影响全部 8 个通道，调整前先看 `MAE_val/*` 是否
整体健康；只想突出 `wind10`/`FSDS` 时优先调 `--lambda_patch_extreme`/`--lambda_wps`（仅影响
指定通道），不建议再通过 `--var_weights` 做通道偏置。正式长跑前务必先按
`loss_plan/final_decision.md` 第 4、11 节做一次数量级校验：损失值 ±3× **不能**代替梯度贡献
可比；同时记录 `wps_mask_ratio`（可能过密）、wind10/FSDS 的 patch 均值偏差（是否整体抬升）。
若覆盖率很高或均值系统性偏高，优先降 `λ_wps` / `λ_p`，不改损失结构。

### 4.5 TensorBoard 与验证口径

| 标量 | 含义 |
| --- | --- |
| `Loss/train` | 每个日志步内，**组合损失**在若干 micro-batch 上的平均 |
| `Loss/tail_w`、`Loss/patch_extreme`、`Loss/wps`、`Loss/phys`、`Loss/spatial_extreme`、`Loss/freq`、`Loss/grad` | 各子项在未加权前的量级（每 `log_interval` 步记录一次均值；后三项仅在对应旧版 λ>0 时出现） |
| `Loss/val` | **仅纯 MAE**（z-score 空间，像素等权，不叠加面积权重），**唯一的**早停 / top-K checkpoint / `best.pt` 判据 |
| `Loss/val_combined`、`Loss/val_tail_w` 等 | 验证集上用**与训练相同**的 `CombinedLoss` 计算的完整组合损失及子项，仅用于诊断该 run 自身优化目标是否达成，**不参与**模型选择 |
| `MAE_val/<变量>` | 反标准化后的逐变量 MAE（物理量纲，像素等权；**注意 `PRE` 仍是 log1p(mm/day) 空间误差**，见下方） |
| `MAE_val_area_weighted/<变量>` | 与训练损失一致的 `cos(latitude)` 面积加权 MAE 诊断，仅供论文报告，**不参与**模型选择 |
| `MAE_val_physical/PRE` | `PRE` 通道额外做一次 `expm1` 还原后的真实 mm/day 量纲 MAE，是唯一真实物理量纲的降水误差指标 |
| `MAE_val_extreme/<变量>`、`MAE_val_extreme/mean` | 仅 `abs(y_zscore) > --val_extreme_z_thresh` 的极端像素 MAE（物理量纲） |
| `Skill_val/<变量>`、`Skill_val_resource/mean` | `--resource_vars` 指定变量相对"LR 双线性插值" naive baseline 的 skill score（`1 − MAE_model/MAE_baseline`），>0 优于该 baseline；`best_resource.pt` 按该均值最大时保存（修复了旧版把不同物理量纲 MAE 直接算术平均的问题） |

**为什么 `Loss/val` 必须固定用像素等权的纯 MAE，而不是训练用的组合损失/面积加权版本**：见本文档下方「验证损失应如何保存才符合标准」一节的完整论证，以及 `loss_plan/final_decision.md` 中对该取舍的记录。

### 4.6 显存与数值注意

- FFT、Sobel、`PatchExtremeLoss` 的 pooling 操作、`WindPowerSensitivityProxy`/`PhysicalConsistencyLoss` 的反标准化在 **float32** 上计算，单步会临时占用较大显存（与全图 1801×3600、8 通道有关）；若 OOM，可先关闭对应 λ 做二分排查。
- 正式长跑前建议先做一次 1-epoch 小规模数据的数量级校验（见 `loss_plan/final_decision.md` 第 4、11 节）：损失值 ±3×、`wps_mask_ratio`、patch 均值偏差、`max_memory_allocated`；损失值接近不代表梯度贡献可比。
- 实现见 `train.py` 中 `TailWeightedMAE`、`PatchExtremeLoss`、`WindPowerSensitivityProxy`、`PhysicalConsistencyLoss`（新增）、`SpatialExtremeLoss`、`FFTLoss`、`GradientLoss`（旧版保留）、`CombinedLoss`；`validate()` 内固定 `nn.L1Loss()` 作为 `Loss/val`，不受训练组合损失配置影响。已知表述边界（不改公式）：物理一致性必须先 denorm；WPS 不含额定以上高风速；patch-mean 不约束极值位置。

---

## 5. 训练脚本要点（`train.py`，优化与其它）


| 项目         | 说明                                                                                                        |
| ---------- | --------------------------------------------------------------------------------------------------------- |
| 损失（训练）     | `CombinedLoss`（v2）：面积加权 TailWeightedMAE（参考目标）+ λ_pe·PatchExtreme + λ_wps·WindPowerSensitivityProxy + λ_phys·PhysicalConsistency（见 **第 4 节**）；默认 γ=0.5、`loss_z_max`=3、λ_pe=0.1、λ_wps=0.05、λ_phys=0.02；旧版 SpatialExtreme/FFT/Gradient 默认关闭，仅供对照 |
| 损失（验证）     | 始终 `nn.L1Loss`（MAE），z-score 空间，像素等权；`Loss/val` 与早停以此为准                                                         |
| 优化器        | `AdamW`，默认 `lr=2e-4`，`weight_decay=1e-4`                                                                  |
| 学习率        | 线性 warmup：`warmup_start_ratio`（默认 **0.01**）→ **1.0**；再余弦至 `min_lr_ratio`（默认 **0.01**；可设 `0` 衰减到 0）        |
| 梯度累积       | 默认 `accum_steps=2`                                                                                        |
| 混合精度       | `torch.autocast(..., dtype=torch.bfloat16)`，**不用 GradScaler**                                             |
| Checkpoint | 默认对全部 4 个 `UpStage` 及末尾 interp+Head 启用 gradient checkpointing；`--no_checkpoint` 可关闭                       |
| 日志         | TensorBoard：`Loss/train`、`Loss/val`、`LR`、`MAE_val/*`、各损失子项；运行目录见 `--run_dir`                              |
| 检查点        | `best.pt` + top-k；`--resume` 可恢复优化器与调度器                                                                   |


### 5.1 分布式训练（多卡 / 多节点 DDP）

`train.py` 已支持 `torch.distributed` DistributedDataParallel（DDP），**默认单卡运行行为完全不变**——仅当通过 `torchrun`（或 `srun` + `torchrun`）启动、环境变量 `RANK`/`WORLD_SIZE`/`LOCAL_RANK` 存在时才会激活分布式逻辑。

参考本平台用户手册《RDMA：使用高性能网络进行分布式训练》：
https://www.scnet.cn/help/docs/mainsite/ai/model-training/rdma/

**实现要点**：

| 环节 | 处理方式 |
| --- | --- |
| 进程组初始化 | 检测到 `RANK`/`WORLD_SIZE` 时自动 `dist.init_process_group`；有 GPU/DCU 时用 `nccl` 后端（Hygon DCU 上由 DTK/RCCL 提供 NCCL 兼容 API），否则回退 `gloo`（仅 CPU 调试用） |
| 数据划分 | `DistributedSampler` 按 rank 切分训练/验证集（各 rank 互不重叠）；训练集每 epoch `set_epoch()` 重新打乱 |
| BatchNorm | `--norm_type batch`（默认）时将 `BatchNorm` 转为 `SyncBatchNorm`（跨所有进程同步统计量），缓解 `--batch_size 1` 时单进程 BN 统计噪声大的问题（见第 7 节）；`--no_sync_bn` 关闭。`--norm_type group` 时此转换自动跳过（GroupNorm 不依赖 batch 维统计量） |
| 梯度累积通信优化 | 梯度累积的中间 micro-batch 用 `model.no_sync()` 跳过 DDP 的梯度 all-reduce，只在真正 `optimizer.step()` 前同步一次 |
| 验证指标聚合 | `validate()` 内所有累计量（loss、逐变量 MAE、极端值统计等）先跨进程 `all_reduce` 求和，再统一计算均值，与单卡结果严格一致（不受 world_size 影响） |
| 日志与存盘 | TensorBoard 与 checkpoint 落盘仅由 rank0 执行；其余 rank 用空实现占位。checkpoint 中的 `model` state_dict 与单卡格式完全一致（自动 unwrap DDP 包装），可在任意卡数下互相加载/`--resume` |
| 早停 / top-K | `best_val_loss`、`early_stop_streak` 用聚合后的 `val_loss` 在所有 rank 上独立计算，天然保持一致，无需额外广播 |

### 5.1a 架构/训练机制开关：`--norm_type`、`--compile`、`--ema_decay`、`--warmup_ratio`

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--norm_type {batch,group}` | `batch` | `ResBlock`/`InitConv`/`Head` 的归一化层。`group`=`nn.GroupNorm`（按通道分组、单样本内部统计，不依赖 batch 维），对 `--batch_size` 较小场景更稳健，但与 `batch` 版本的 checkpoint **不兼容**（层结构不同，需从头训练）。**正式训练已确定使用 `group`**（见第 9 节） |
| `--compile` | 关闭 | 用 `torch.compile()` 包装模型（PyTorch 2.5.1 支持），可能带来 10%~30% 加速；与 gradient checkpointing / DDP / 本平台 DCU+RCCL 的组合兼容性未经充分验证，**默认关闭**，建议先在 `slurm/train_smoke_test_ddp.slurm` 上单独加一次验证再用于正式训练 |
| `--ema_decay` | `0.0`（关闭） | 模型权重指数滑动平均（如 `0.999`）。开启后验证阶段临时换用 EMA 权重（对小 batch/BN 噪声更稳健），checkpoint 额外保存 `model_ema`（部署/推理用）；`model` 字段仍是训练用的在线权重（用于正确 `--resume`） |
| `--warmup_ratio` | 空（不生效） | 设置后自动用 `warmup_steps = round(warmup_ratio × total_steps)` 覆盖 `--warmup_steps`，不必先跑一次看日志里的 `total_steps` 再手动回填；建议 `0.03~0.05` |

**启动方式**（正式长跑只走 SCNet 控制台，见第 5.1.1 / 第 9 节）：

```bash
bash /public/home/acd7koea4a/work/scripts/launch_platform_train.sh
```

等价的 `torchrun` 写法见第 5.1.1 节。`slurm/*.slurm` 已归档，本仓库不用 `sbatch` 提交训练。

**注意事项**：

- `NPROC_PER_NODE` 必须等于控制台「每实例加速卡数量」；8 卡保持 `--batch_size 1`。
- 架构冒烟用 `scripts/launch_platform_smoke_single.sh` / `smoke_ddp.sh`；全量 1 epoch 探测用 `launch_platform_train_copy.sh`。
- `--eval_only --resume <ckpt>` 同样支持分布式启动（用多卡加速大验证集的补算指标过程）。
- 梯度累积 `--accum_steps` 与分布式 `world_size` 会同时放大有效 batch；调 `--lr`/`--warmup_steps` 时请按新的 global batch 重新核对。

### 5.1.1 在 SCNet「模型训练」控制台任务中启动（容器/vcjob，非 Slurm sbatch）

若不走 `sbatch`，而是在控制台 人工智能服务 → 模型训练 → 创建训练任务 里配置资源并提交
（参考 https://www.scnet.cn/help/docs/mainsite/ai/model-training/ ），启动方式与 Slurm 不同：

- 平台按「实例数」「每实例加速卡数量」调度容器，并自动为每个实例注入分布式环境变量
  （详见《环境变量列表》https://www.scnet.cn/help/docs/mainsite/ai/model-training/environment-variable/ ）：
  `WORLD_SIZE`（实例数）、`RANK`（当前实例序号，0 起始）、`MASTER_ADDR`（worker-0 hostname）、
  `MASTER_PORT`（默认 23456）；若开启 RDMA，`NCCL_IB_*` 等变量也会自动注入，无需手动 `export`。
- `train.py` 的分布式逻辑通过检测 **torchrun 为每个子进程设置**的 `RANK`/`WORLD_SIZE`/`LOCAL_RANK`
  来激活 DDP，因此**代码本身无需任何改动**，只需在「启动命令」里用平台注入的变量正确调用
  `torchrun`（写法与官方《模型训练最佳实践》示例一致）：

```bash
# 不要手抄旧消融参数。正式配置已封装：
bash /public/home/acd7koea4a/work/scripts/launch_platform_train.sh
```

等价的 `torchrun` 写法（`NPROC_PER_NODE` 必须等于控制台「每实例加速卡数量」）：

```bash
torchrun \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=${NPROC_PER_NODE:-8} \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --hdf5_root "$(python -c 'from paths import HDF5_ROOT; print(HDF5_ROOT)')" \
    --epochs 100 --batch_size 1 --accum_steps 2 --val_fraction 0.2 --val_interval 2 \
    --no_cbam --hr_aux_mode stage1 --norm_type group --ema_decay 0.999 \
    --warmup_ratio 0.03 --early_stop_patience 20 \
    --run_dir runs/exp_prod_ddp_platform
```

  其中 `--nproc_per_node` 需与控制台「每实例加速卡数量」一致（如 2 卡实例填 2）。
  架构冒烟用 `scripts/launch_platform_smoke_single.sh` / `smoke_ddp.sh`；
  8 卡短跑用 `scripts/launch_platform_short_8gpu.sh`。

- **提交前请确认**（容器环境与 Slurm 登录节点不同，以下几点最容易踩坑）：
  1. **不要** `source env/activate.sh`——该脚本是为 Slurm 登录节点的 `module` + `conda`
     环境写的，容器镜像（如 `jupyterlab-pytorch:2.7.1-ubuntu22.04-dtk26.04-py3.11-devel`）
     里没有对应的 module 系统，直接用镜像自带 `python`/`torch` 即可。
  2. **依赖**：确认镜像里已有 `h5py`、`netCDF4`、`tensorboard`（`requirements.txt` /
     `environment.yml`），没有则在自定义镜像里预装好（避免每次启动都现装浪费时间）。
  3. **路径挂载**：数据根目录为 `/public/share/acd7koea4a`（见 `paths.py` 中 `DATA_ROOT`）；
     代码与训练产物在 `/public/home/acd7koea4a/work`。若容器未挂 share，`paths.py` 会回退到
     `/public/home/acd7koea4a/local_data`（`static/` + `states/global_stats_state.json` 真实副本）。
     家目录下的 `~/static`、`~/states` 是指向 share 的符号链接，share 未挂时不可用。
     也可 `--static_dir`/`--stats_file` 显式指定。
  4. 首次运行建议先用「SSH」进容器手动跑通一小段（如 `hdf5_mini` + 1 epoch）确认路径/依赖
     无误，再提交正式多卡任务。

### 5.2 PixelShuffle baseline（仅 8 变量，无静态场 / 无 HR-aux）

相关代码集中在目录 `**pixelshuffle_baseline/`**：`pixelshuffle.py`（`UpsampleModel`）、`dataset_dynamic_only.py`（仅 `data/x`、`data/y` + 与主项目相同的 8 变量 z-score）、`train_pixelshuffle_baseline.py`（训练入口）。也可 `from pixelshuffle_baseline import UpsampleModel, DynamicOnlyDataset`。


| 项目            | 说明                                                                                                                                                                   |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 目录            | `pixelshuffle_baseline/`                                                                                                                                             |
| 脚本            | `pixelshuffle_baseline/train_pixelshuffle_baseline.py`                                                                                                               |
| 损失 / 优化 / AMP | 与上表 `train.py` 一致（MAE、AdamW、warmup+cosine、`accum_steps=4`、CUDA 上 bf16）                                                                                               |
| 学习率调度         | 与 `train.py` 相同：优先 `from train import build_scheduler`；失败则用同公式本地实现；`LambdaLR` 按**优化器步**（每 `accum_steps` 个 micro-batch 一步）                                            |
| 早停            | `--early_stop_patience`（默认 **15**，按验证周期计；**0** 关闭）、`--early_stop_min_delta`；TensorBoard：`EarlyStopping/bad_epochs`；checkpoint 含 `early_stop_bad_epochs` 供 `--resume` |
| 默认 epoch      | **100**（可用 `--epochs` 覆盖）                                                                                                                                            |
| 日志目录          | `--run_dir` 下 `tb/` 与 `checkpoints/`（含 `best.pt`）                                                                                                                    |


**环境**：先激活 `pytorch_downscale`，再运行 Python（与主项目一致）。

**GPU1 + hdf5_half + 100 epoch 示例**（训练前用 `nvidia-smi` 确认 GPU1 空闲）：

```bash
conda activate pytorch_downscale
CUDA_VISIBLE_DEVICES=1 nohup python pixelshuffle_baseline/train_pixelshuffle_baseline.py \
    --hdf5_root /public/share/acd7koea4a/hdf5_half \
    --epochs 100 \
    --batch_size 1 --accum_steps 4 --val_fraction 0.1 \
    --num_workers 4 \
    --run_dir runs/pixelshuffle_baseline \
    > logs/train_pixelshuffle_baseline.log 2>&1 & echo "作业PID: $!" >> logs/train_pixelshuffle_baseline.log
```

不手动 activate 时可用：

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n pytorch_downscale python pixelshuffle_baseline/train_pixelshuffle_baseline.py \
    --hdf5_root /public/share/acd7koea4a/hdf5_half --epochs 100 --run_dir runs/pixelshuffle_baseline
```

或在项目根目录：`python -m pixelshuffle_baseline.train_pixelshuffle_baseline ...`

**TensorBoard 与主项目对比**：若主实验与 baseline 的 `tb` 均在 `runs/` 下不同子目录，可一次加载父目录：

```bash
tensorboard --logdir runs
```

或在 TensorBoard UI 中勾选/筛选 `pixelshuffle_baseline` 与 `exp01_*` 等 run。

**历史 debug/消融 run 命令**（曾用于早期架构探索，其中部分参数如 `--no_hr_aux`、`--init_type` 在当前 `model.py`/`train.py` 中已不存在，直接照抄会报参数错误）已整理、校正并迁移至 **第 6 节「调试与消融实验速查」**，以当前实际 CLI 参数为准。

---

## 6. 调试与消融实验速查（已按当前 CLI 校正）

> 以下命令均已核对，使用 `train.py` **当前真实存在**的参数（不再含已废弃的 `--init_type`、`--no_hr_aux`）。

**调试用 mini 数据集**（`--hdf5_root` 切换即可，无需改代码；mini 数据集共 971 个样本：DJF 264 + JJA 207 + MAM 213 + SON 287，几分钟内跑完 1 个 epoch，足以验证 loss 下降、checkpoint 保存是否正常）：

```bash
nohup python train.py \
    --hdf5_root /public/share/acd7koea4a/hdf5_mini \
    --epochs 3 --val_interval 1 \
    --num_workers 2 \
    --seasons DJF \
    --run_dir runs/debug > train_debug.log 2>&1 & echo "作业PID: $!" >> train_debug.log
```

**消融实验开关速查表**（`model.py` + `train.py`，与当前代码一致）：

| CLI 参数 | 作用 | 模型内部变化 |
| --- | --- | --- |
| 默认（无额外参数） | 全量模型 | CBAM ✓、HR-aux `all` ✓、gradient checkpoint ✓ |
| `--no_cbam` | 去掉每阶段 CBAM | `UpStage.cbam` 不创建，forward 跳过 |
| `--hr_aux_mode none` | 全程不注入 HR-aux | 所有 `inject_conv` 不创建，`head_in=base_ch`；DataLoader 不变，`hr_aux` 仍被传入但模型忽略 |
| `--hr_aux_mode stage1` | 仅 Stage1 注入 HR-aux | Stage2-3 与 Head 不拼接 HR-aux |
| `--no_checkpoint` | 关闭 gradient checkpointing | 显存占用上升，速度加快 |
| `--manifests <tag...>` | 限定 shard 的 batch_tag 白名单 | 见 `dataset.py` `_build_index`，用于排除坏批次 |

多个开关可以叠加使用（如 `--no_cbam --hr_aux_mode none`）。架构消融示例见
`train.py` 顶部 A–D；损失函数消融（含纯 MAE 必须关掉的全部 λ）见顶部 E–H。
**不要只传 `--loss_gamma 0` 就当成纯 MAE。** 本文档不再重复维护完整命令列表。

---

## 7. 小 batch 与 BatchNorm 说明

`ResBlock` 与 Init/Head 默认使用 **BatchNorm2d**（`--norm_type batch`）。`batch_size=1` 时 BN 的 batch 统计噪声较大；若 loss 抖动明显，可尝试增大有效 batch（累积步数不变时提高 `batch_size`，或增加 `--accum_steps`/多卡 DDP 扩大 global batch，注意只有 `batch_size` 才能真正改善 BN 统计，`accum_steps` 只平滑梯度），或直接用 `--norm_type group` 切到 **GroupNorm**（已实现，不依赖 batch 维统计量，代价是与 `batch` 版本 checkpoint 不兼容，需从头训练；正式训练已确定使用该开关，见第 9 节）。

---

## 8. CESM 推理推荐流程（稳定版）

为避免 CESM 直接推理时可能出现的局部空间错配，推荐将 CESM 原始数据先离线转换为与测试集一致语义的 HDF5，再用 `infer.py` 的 HDF5 分支推理。
正式权重请加 `--auto_model_cfg --use_ema`（GroupNorm + EMA）。

推荐流程：

1. `nc/npy -> HDF5`（离线准备）

```bash
conda run -n pytorch_downscale python /public/home/acd7koea4a/work/prepare_hdf5_cesm.py \
  --input_format nc \
  --cesm_root /public/share/acd7koea4a/cesm \
  --out_hdf5_root /public/share/acd7koea4a/hdf5_cesm \
  --date_start 20000101 --date_end 20000101 \
  --overwrite
```

1. 只走 HDF5 推理链路（与测试集同路径）

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n pytorch_downscale python /public/home/acd7koea4a/work/infer.py \
  --input_source hdf5 \
  --hdf5_root /public/share/acd7koea4a/hdf5_cesm \
  --seasons DJF \
  --ckpt /public/home/acd7koea4a/work/runs/exp01_no_all/checkpoints/best.pt \
  --auto_model_cfg --use_ema \
  --out_dir /public/home/acd7koea4a/work/infer_out_cesm_stable \
  --output_mode per_sample \
  --output_format nc \
  --lon_convention neg180_180 \
  --output_space physical \
  --amp_bf16
```

说明：

- `infer.py` 会校验 HDF5 输入契约（`data/x`, `data/dates`, `x` 形状）；
- 写 NetCDF 必须显式传 `--lon_convention`（`neg180_180` 或 `pos0_360`），没有隐式默认，避免坐标系被静默翻转；
- 若 HDF5 含 `metadata/lr_grid_lats/lr_grid_lons`，会与 `static/lat_lr.npy/lon_lr.npy` 强一致校验，避免静默坐标偏差；
- `input_source=cesm_nc` 仅保留为实验诊断入口，默认不启用。

---

## 9. 正式训练配置（已确定）

正式长跑入口：**SCNet「模型训练」控制台** →
`bash /public/home/acd7koea4a/work/scripts/launch_platform_train.sh`。
本仓库不用 `sbatch` / `slurm/*.slurm`。

架构：**不开 CBAM（`--no_cbam`）+ HR 辅助仅在 Stage1 注入（`--hr_aux_mode stage1`）+
`--norm_type group` + `--ema_decay 0.999` + `--warmup_ratio 0.03`**（见第 5.1a 节）。
**损失函数为第 4 节 v2 默认**（面积加权 `TailWeightedMAE` + `PatchExtremeLoss` +
`WindPowerSensitivityProxy` + `PhysicalConsistencyLoss`）；脚本不传损失参数，与
`train.py` v2 默认值一致。8 卡保持 `--batch_size 1`，不要提到 2。
数据：`paths.HDF5_ROOT`（`/public/home/acd7koea4a/hdf5_norm_fp16`）+ `--manifests cra1p5_full`。

```bash
# 控制台「启动命令」填这一行（NPROC_PER_NODE 默认 8，须等于「每实例加速卡数量」）
bash /public/home/acd7koea4a/work/scripts/launch_platform_train.sh
```

脚本内部等价命令：

```bash
torchrun --nnodes="${WORLD_SIZE:-1}" --nproc_per_node="${NPROC_PER_NODE:-8}" \
    --node_rank="${RANK:-0}" --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-23456}" \
    train.py \
    --hdf5_root /public/home/acd7koea4a/hdf5_norm_fp16 \
    --seasons MAM JJA SON DJF --manifests cra1p5_full \
    --epochs 100 --batch_size 1 --accum_steps 2 --val_fraction 0.2 --val_interval 2 \
    --num_workers 4 \
    --no_cbam --hr_aux_mode stage1 \
    --norm_type group --ema_decay 0.999 \
    --warmup_ratio 0.03 \
    --early_stop_patience 20 \
    --run_dir runs/exp_prod_ddp_platform
```

提交 100 epoch 前，建议先跑通：

1. `scripts/launch_platform_smoke_single.sh`（单卡）/ `smoke_ddp.sh`（多卡）——16 样本管线冒烟
2. `scripts/launch_platform_train_copy.sh`（全量 1 epoch）——确认读盘 / 显存 / `validate()` / 落盘
3. 可选：`scripts/launch_platform_short_8gpu.sh`（8 卡、2 epoch）

裸跑 `python train.py` 会落到另一套 CLI 默认架构（CBAM 开、`hr_aux=all`、BatchNorm、无 EMA），不要当正式训练。
测试集尚未就绪，推理脚本本轮不改；不要把 `infer.py --hdf5_root` 指到 `hdf5_norm_fp16`。

更完整的分析、调参依据与逐步操作清单见 `TRAINING_ANALYSIS.md`。

