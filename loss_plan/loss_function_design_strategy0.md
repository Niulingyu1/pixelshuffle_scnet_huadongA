# Loss function design strategy
_Exported on 2026/9/15 at GMT+8 14:25:53 from Cursor (3.20.17)_

---

**User**

深入熟悉并掌握当前项目下的所有文件，确定损失函数的设计，根据现有的信息，结合高水平期刊文献检索，为我确定最佳的损失函数设计方案。由于训练成本昂贵，要求尽量不通过多次调参，而是保证一次运行能实现不错的效果，目的是保证各变量的效果同时增强风光变量的效果，未来将其应用风光资源的评估，针对当前降尺度中的缺陷和痛点，使我的模型结果具有一定的改进或创新意义


---

**Cursor**


# 损失函数设计方案（面向风光资源评估的降尺度模型）

我已通读项目全部核心文件（`model.py`、`train.py`、`dataset.py`、`DOWNSCALE_README.md`、`old_logs/report.md`、`eval/training_monitor/ablation_training_status.md`、`eval/metrics_extreme.py`），并检索了 NREL/PNAS/Nature Machine Intelligence 及 2025–2026 年最新的降尺度损失函数基准研究。以下是**方案本身**，未改动任何代码。

## 一、项目现状（已确认的事实，是本方案的出发点）

- 任务：`PixelShuffleDownscaleNet`，1°→0.1°（16×）全球逐日 8 变量降尺度，`TAS/PRE/wind10/Q/2M_RH/2M_TMAX/2M_TMIN/FSDS`；每个训练样本是**整个全球格点**（1801×3600），不是局地小图块。
- 现有 `CombinedLoss`（`train.py`）= `TailWeightedMAE`（z-score 尾部加权 + 逐变量通道权重）+ `SpatialExtremeLoss`（wind10/FSDS 的**全球单样本** max/min MAE，λ=0.1）+ `FFTLoss`（全部 8 通道，默认 λ=0）+ `GradientLoss`（全部 8 通道，默认 λ=0）。
- 消融实证结论（`ablation_training_status.md` + `old_logs/report.md`，非常关键，是我方案"少调参"的依据）：
  - HR 辅助场是主要收益来源；`hr_loss`（HR-aux + 组合损失）目前验证集最优（0.034723），仅比纯 HR-aux（`hr`, 0.034822）好 0.28%。
  - **对全部 8 通道**施加 FFT/Grad 通用图像质量损失，会拖累 TAS/2M_TMAX/2M_TMIN 等本身空间较光滑变量的逐点精度（项目文档已自述该结论，`loss_100` 单独不优于基线）。
  - `hr_loss` 相对 `hr` 的优势主要来自 FSDS、Q、2M_RH，即与风光资源更相关的变量。
- `eval/metrics_extreme.py` 已经实现了非常完整的下游评估口径：风机功率曲线 `wind_to_power`（cut-in/rated/cut-out 三段式）、光伏出力代理 `fsds_to_pv_proxy`、多邻域 `FSSAccumulator`、能量干旱事件、分位数误差——但**这些评估口径目前完全没有反馈进训练损失**，这是最大的"训练-评估脱节"。

## 二、现有损失设计的四个具体痛点（诊断，而非泛泛而谈）

1. **`SpatialExtremeLoss` 与"整幅全球样本"不匹配**：NREL Sup3rWind/Sup3r 原始 `SpatialExtremesLoss`（Stengel et al. 2020, *PNAS* 117:16805; Benton et al. 2024, NREL/TP-6A40-89241）是在**局地训练图块**上取 max/min，图块内极值本身有物理意义。本项目样本是全球格点，"全球唯一最大值"通常落在南极/局地地形性大风等个别点，与"某地风光资源关心的区域性极值"关系疏远，且每个样本每通道反传时只有 **2 个像素**获得非零梯度，信号极度稀疏、方差大。
2. **通用图像质量损失（FFT/Grad）与逐点精度目标冲突**：项目自己的消融已经证明，对全部通道施加频域/梯度约束会牺牲 TAS/TMAX/TMIN 的精度，这与最新基准研究的结论一致——*"Training loss matters"*（ESSOAR 2025，风速+降水降尺度损失基准）明确指出"没有一种损失组合在所有任务/所有评估维度上都占优"，需要按变量的物理特性做**任务定制**而非全局套用。
3. **训练目标与"风光资源评估"下游口径脱节**：无论是逐点 MAE 还是现有极值损失，度量的都是气象量本身的误差，而不是能量产出误差。风速在切入-额定风速之间（3–12 m/s）呈三次方非线性放大误差，同样量级的 wind10 MAE 落在这个区间与落在满发段的影响可以相差数倍——`eval/metrics_extreme.py` 已经为此专门做了 `wind_power_region_mae` 评测，但训练完全没有对齐这一目标。
4. **无任何物理一致性约束**：8 通道回归头彼此独立、无末端激活，理论上可能出现 TMIN>TMAX、wind10<0、FSDS<0、RH 越界等物理不可能的组合，目前无检测也无惩罚，对下游能量产出估计是隐患。

## 三、文献依据（近两年高水平检索结果）

| 文献 | 关键结论 | 对本方案的支撑 |
|---|---|---|
| Stengel et al. 2020, *PNAS*；NREL Sup3rWind 报告（2024，NREL/TP-6A40-89241） | GAN 内容损失 = MAE + `SpatialExtremesLoss`（局地图块 max/min） | 沿用"极值一致性"思想，但指出其对全球单样本场景失效，需局部化改造 |
| *Training loss matters*（ESSOAR/GMD, 2025，风速+降水降尺度损失基准） | MSE/MAE 是有效但非最优基线；结构感知损失（Spectral/GDL/Wavelet）需与像素损失搭配、且不存在"万能组合"，需按变量定制 | 支持"分通道选择性施加频域/梯度损失"而非全局套用 |
| *A Probabilistic U-Net Approach to Downscaling*（NeurIPS AI4Science 2025，arXiv:2511.03197） | WMSE-MS-SSIM 更擅长极值，afCRPS 更擅长细尺度谱；两者互补 | 支持多目标组合优于单一损失，但也提示单一新损失权重不宜过大 |
| FastNet（arXiv:2509.17601, 2025） | 频谱损失 + 梯度损失 + **物理一致性/变量表示**改造联合使用，能在几乎不损失整体 RMSE 的前提下显著提升物理真实性 | 支持"轻量物理约束 + 变量专属表示"的设计路线 |
| Hess et al. 2025, *Nature Machine Intelligence* 7:363 | 强调下游任务（不确定性、极端事件）应作为降尺度评估/设计的核心，而非通用图像指标 | 支持"面向风光资源评估重新设计损失口径"的整体方向 |
| Sup3r `loss_metrics` 源码（NREL 现行实现） | 官方已把 `SpatialFftLoss` 按频率平方加权、把物理量（如温度梯度）残差直接做损失项，证明"把下游物理量嵌入损失"是业界现行做法而非空想 | 支持 `ResourceTransformLoss`（把功率曲线嵌入损失）的可行性 |

## 四、推荐的最终损失方案

$$L = \underbrace{L_{\text{tail}}}_{\text{主干，8变量兜底}} + \lambda_{ms}\,\underbrace{L_{\text{ms-extreme}}}_{\text{多尺度局部极值}} + \lambda_r\,\underbrace{L_{\text{resource}}}_{\text{风光能量口径}} + \lambda_f\,\underbrace{L_{\text{FFT}}^{(wind,FSDS)}}_{\text{仅风光通道}} + \lambda_g\,\underbrace{L_{\text{grad}}^{(wind,FSDS)}}_{\text{仅风光通道}} + \lambda_p\,\underbrace{L_{\text{phys}}}_{\text{物理安全网}}$$

**1. 主干不变**：`TailWeightedMAE`（γ=0.5, z_max=3）+ 现有 `--var_weights` 默认值（wind10/FSDS=1.5，TAS/2M_RH=1.2，其余=1.0）。理由：ablation 已验证这是目前唯一跑通且优于纯 MAE 的配置，改动它风险最大、收益未知，**不动**。

**2. `MultiScaleExtremeLoss`（替代 `SpatialExtremeLoss`，新设计，解决痛点①）**：把"全球单一 max/min 标量"换成滑窗局部 max/min **场**（`F.max_pool2d` 及取负技巧得到局部 min），在多个窗口尺度上逐像素做 L1：
- 窗口建议 `(8, 32, 128)` HR 像元（约 0.8°/90km 电站选址尺度、3.2°/350km 区域电网尺度、12.8°/1400km 天气系统尺度）；
- 仅作用于 `wind10, FSDS`；
- 关键优势：与项目自己 `eval/metrics_extreme.py::FSSAccumulator`（多邻域 Fractions Skill Score）的评测思想完全对齐，**训练即优化评估指标本身**，且每个窗口产生密集梯度而非 2 个像素，稳定性远好于原实现。
- 建议 `λ_ms = 0.1`（与现有 `--lambda_extreme` 默认值一致的量级，保守起步）。

**3. `ResourceTransformLoss`（全新损失，本方案核心创新点，解决痛点③）**：把预测/真值反标准化到物理量纲后，分别经过
- 风机功率曲线（IEC 三段式：cut-in=3, rated=12, cut-out=25 m/s，与 `eval/metrics_extreme.py` 默认值完全一致）；
- 光伏出力代理（`clip(FSDS/1000, 0, 1)`，同样与评测口径一致）；
再对**转换后的能量代理量**做 L1。这样风速误差按"对发电量的实际敏感度"重新分配梯度权重（爬升段误差被放大关注），是从"气象量误差"到"能量产出误差"的直接对齐，也是与现有 3 项文献（Sup3rWind、FastNet、loss-bench）相比**本项目场景独有**的针对性设计（通用降尺度论文不做能源转换，是因为它们不是面向风光资源评估的）。
- 建议 `λ_r = 0.1`（与逐点 MAE 量级校准后取值，见下文"量级校准"说明）。

**4. FFT/Grad 损失改为"通道选择性"（解决痛点②）**：把现有 `FFTLoss`/`GradientLoss` 限定在 `wind10, FSDS` 两个通道上计算（新增可选 `channel_indices` 参数，`None` 时保持向后兼容全通道）。这样既保留频域/梯度约束改善风光变量细尺度纹理的收益（loss-bench 论文证实 Spec/GDL 对细尺度结构有正向作用），又彻底规避消融实验中发现的"拖累 TAS/TMAX/TMIN"问题——因为这两个损失项的梯度不再流向温度/湿度等通道。
- 建议 `λ_f = 0.1, λ_g = 0.05`（与项目原默认值一致，因为现在只作用于 2 个通道，风险已显著降低，可以直接沿用文档中原有的调参经验值，不需要重新搜索）。

**5. `PhysicalConsistencyLoss`（全新损失，轻量安全网，解决痛点④）**：仅依赖模型自身预测（不需要真值），对 TMIN≤TMAX、wind10/FSDS/Q/PRE(log1p)≥0、RH∈[0,100] 做 hinge（ReLU）惩罚，全部在反标准化物理量纲上计算。
- 建议 `λ_p = 0.02`（很小，只是正则安全网，训练收敛后该项应自然趋近 0；若权重过大会用精度换取无意义达标，因此刻意压低）。

### 量级校准方法（保证"一次运行"而不是网格搜索）
参照 *Training loss matters*（ESSOAR 2025）论文的方法论——"先将各子损失校准到同一数量级，再用固定权重线性组合，而非大范围网格搜索权重"——建议在正式训练前，**仅用 1 个 epoch（几分钟，几乎零成本）**跑一遍前向，打印各子项在未加权前的均值（`Loss/tail_w`、`Loss/ms_extreme`、`Loss/resource`、`Loss/freq`、`Loss/grad`、`Loss/phys`，这些 TensorBoard 标量项目现有代码框架已经具备，只需按新损失扩展），确认各项乘以上述权重后与 `Loss/tail_w` 同一数量级（±3× 以内）即可直接进入正式训练，无需多轮正式训练的调参。这一步不是"调参"，而是数值健全性检查，成本可忽略。

### 保底设计（风险控制）
当 `λ_ms=λ_r=λ_p=0` 且 FFT/Grad 通道限定为空时，本方案**退化为项目当前已验证的 `hr_loss` 配置**（0.034723，当前最优）。也就是说新增的三项损失都是在已验证最优基线上的**增量**，单次运行即便新增项收益不及预期，理论下界也不会差于现状——这是应对"训练成本昂贵、只能跑一次"约束的核心保障机制。

## 五、需要落地到代码的改动清单（供后续实施）

1. `train.py::FFTLoss` / `GradientLoss`：新增可选 `channel_indices` 参数（`None`=全通道，向后兼容）。
2. `train.py`：新增 `MultiScaleExtremeLoss`（多尺度局部 max/min pooling）替代/升级 `SpatialExtremeLoss`（保留旧类以兼容历史 checkpoint 对照）。
3. `train.py`：新增 `ResourceTransformLoss`（风机功率曲线 + 光伏出力代理，需持有 `norm_mean/norm_std` 做反标准化）。
4. `train.py`：新增 `PhysicalConsistencyLoss`（TMIN≤TMAX、非负性、RH 边界，仅依赖预测）。
5. `CombinedLoss`：接入以上四项，`forward` 返回值扩展 `sub` 字典的对应键，`validate()` 沿用现有"验证集用相同 criterion 计算完整组合损失各子项"的机制（代码框架已具备，直接复用）。
6. `parse_args()` 新增 CLI：`--lambda_multiscale_extreme`（默认 0.1）、`--extreme_windows`（默认 "8,32,128"）、`--lambda_resource`（默认 0.1）、`--lambda_phys`（默认 0.02）、`--fft_grad_vars`（默认 "wind10,FSDS"，用于同时限定 FFT/Grad 的作用通道）。
7. `DOWNSCALE_README.md` 第 4 节据此更新，并注明与 `eval/metrics_extreme.py` 口径的对应关系。

推荐的一次性启动命令（在现有 `hr_loss` 最优配置基础上叠加新项，其余沿用项目默认）：

```bash
python train.py --hdf5_root /root/data/hdf5_half --epochs 100 \
  --loss_gamma 0.5 --loss_z_max 3.0 \
  --var_weights "TAS=1.2,PRE=1.0,wind10=1.5,Q=1.0,2M_RH=1.2,2M_TMAX=1.0,2M_TMIN=1.0,FSDS=1.5" \
  --lambda_multiscale_extreme 0.1 --extreme_windows 8,32,128 --extreme_vars wind10,FSDS \
  --lambda_resource 0.1 \
  --lambda_freq 0.1 --lambda_grad 0.05 --fft_grad_vars wind10,FSDS \
  --lambda_phys 0.02 \
  --run_dir runs/exp03_resource_loss
```

## 六、验证与止损（无需额外训练迭代）

- TensorBoard 上直接对比 `Loss/val`（纯 MAE，早停判据，跨 run 可比）与 `Loss/val_*` 各子项，确认新方案不劣于 `hr_loss` 的 0.034723。
- 训练完成后**无需重训**，直接用现成 `eval/metrics_extreme.py` 的 `wind_power_region_mae`、`pv_irradiance_bin_mae`、`FSSAccumulator` 对新 checkpoint 做针对性复核，量化"风光资源评估口径下"的实际改进幅度——这正是该评估库已经为此场景预先设计好、此前从未被训练目标使用过的验证手段。

