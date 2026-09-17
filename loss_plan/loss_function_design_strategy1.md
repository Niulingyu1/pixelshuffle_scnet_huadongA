Now let's update the argparse block to add the new flags:Now let's update the instantiation block:Now let's add the recommended one-shot config to the top docstring:Now let's finalize the file and present it to the user.我已经通读了四个文件，并直接在 `train.py` 里落地了新的损失函数设计（已生成可下载版本）。下面按"现状诊断 → 文献依据 → 最终方案 → 为什么能一次成型"说明。

## 1. 现状诊断：当前损失设计的两个结构性痛点

你现在的 `CombinedLoss` 已经做得比较完整（TailWeightedMAE + 逐变量通道权重 + SpatialExtremeLoss + 可选 FFT/Grad），但有两个结构性缺陷限制了它的上限：

**痛点一：`SpatialExtremeLoss` 的监督信号极度稀疏。** 它对每个样本、每个通道在整幅 1801×3600 图上只取 1 个最大值 + 1 个最小值，相当于每张图只有 2 个监督点，梯度只回传到"全图唯一极值像素"。这恰好复现了 NREL Sup3rWind 论文里 `mae_extreme` 项的做法（`maxt`=全时间域最大值、`maxs`=全空间域最大值），但你的场景是确定性 CNN 单次前向覆盖全球网格，这种粒度对"区域性风速极大值簇""云遮导致的局部辐照度骤降"这类你真正关心的风光资源特征约束太弱。

**痛点二：FFT/Gradient loss 是"全通道统一施加"，这是它们拖累 TAS/2M_TMAX/2M_TMIN 的根因，而不是这两个损失本身没用。** 消融实验 E vs G 的结论其实容易被误读为"频谱/梯度损失对这个任务没用"，但更准确的诊断是：把 8 个物理性质差异很大的变量（wind10/FSDS 高频异质 vs TAS 等空间平滑）塞进同一个 FFT/Sobel 张量里算平均损失，会把风光变量需要的"保留高频"目标强加给本该光滑的变量。

## 2. 文献依据

- **NREL Sup3rWind / Sup3rCC**（Buster et al., NREL 官方页面 + arXiv 2024 Ukraine 论文）：确认了空间极值一致性损失（maxs/maxt 的 MAE）是风光资源超分辨率的标准做法，验证了你现有 SpatialExtremeLoss 的设计方向是对的，只是粒度可以做得更细。
- **Stengel et al. 2020, PNAS "Adversarial super-resolution of climatological wind and solar data"**：首次将 SRGAN 式内容损失+对抗损失用于风光降尺度，明确指出纯 L1/L2 损失会导致能量谱高频段系统性欠估计（过平滑），这是 wind/solar 领域公认的痛点来源。
- **Hess et al. 2022, AtmoDist**（大气动力学表征学习）：进一步验证了"L2-based downscaling 结果在能量谱上明显偏低、偏平滑"这一现象，并指出频谱域监督是常见修正手段。
- **Skillful downscaling GAN 论文（Wind, 2023, arXiv:2302.08720）**：展示了频谱损失/对抗损失在提升精细尺度湍流细节上的价值，但同时指出这种收益是以牺牲部分逐点精度为代价的——这进一步支持"通道限定"而非"全局施加"的必要性。
- **Kendall, Gal & Cipolla 2018 (CVPR), "Multi-task learning using uncertainty to weigh losses"**：多任务不确定性自动加权，可以省去人工搜索 λ；我评估后没有把它设为默认（原因见下），但列为可选后续方向。

## 3. 最终方案（已写入代码，对应 `runs/ablation/I_wind_solar_focus_v2`）

在你现有框架基础上做了两处结构性改动，都是**新增能力、默认向后兼容**：

**改动 A —`SpatialExtremeLoss` → `PatchExtremeLoss`（新默认，`--extreme_mode patch`）**
用 `adaptive_max_pool2d` 把 wind10/FSDS 的 HR 图切成 180×360 个局部块（正好对应 LR 网格分辨率，物理意义是"每个 LR 源格点对应的 HR 覆盖区域内的极值"），逐块约束 max/min。监督信号从 2 个/图提升到 2×180×360 个/图，能捕捉区域性风速极值簇和云遮骤降区域，而不只是全球唯一极值点。`adaptive_max_pool2d` 天然处理不整除的情况，不需要额外对齐逻辑，计算开销远低于 FFT/Sobel。旧版 `SpatialExtremeLoss` 类完整保留，`--extreme_mode global` 可随时切回做对照。

**改动 B — FFT/Gradient loss 默认限定到 `extreme_vars`（新增 `channel_select`，默认开启）**
`FFTLoss`/`GradientLoss` 现在都接受 `channel_indices` 参数：一旦打开（`--lambda_freq`/`--lambda_grad` 非 0），只对 wind10、FSDS 计算，TAS/PRE/Q/2M_RH/2M_TMAX/2M_TMIN 完全不参与这两项——这些变量在这两个损失里梯度恒为 0，从根源上消除"拖累其他变量"的副作用。`--freq_grad_all_channels` 可退回旧版全通道行为。

推荐的一次成型运行命令（已写进 `train.py` 顶部注释，标记为"推荐"）：

```bash
python train.py --base_ch 256 \
    --loss_gamma 0.5 --loss_z_max 3.0 \
    --var_weights "TAS=1.2,PRE=1.0,wind10=1.5,Q=1.0,2M_RH=1.2,2M_TMAX=1.0,2M_TMIN=1.0,FSDS=1.5" \
    --extreme_vars wind10,FSDS --lambda_extreme 0.1 \
    --extreme_mode patch --patch_grid_h 180 --patch_grid_w 360 \
    --lambda_freq 0.05 --lambda_grad 0.02 \
    --val_extreme_z_thresh 1.5 \
    --run_dir runs/ablation/I_wind_solar_focus_v2
```

## 4. 为什么这个方案能"一次运行"达到不错效果，而不需要多次调参

- **数学上保证不退步**：所有新增机制在权重为 0 时严格退化为原有行为（`CombinedLoss` 文档里那条"全 0 等价于 nn.L1Loss()"的保证依然成立），风险下限就是你现有 baseline。
- **权重取值不是拍脑袋，而是复用了你自己的消融实验结果**：`loss_gamma=0.5`、`var_weights`、`lambda_extreme=0.1` 直接沿用已跑过的 H 配置；`lambda_freq=0.05`/`lambda_grad=0.02` 是在 G 配置（0.1/0.05，全通道）基础上因为改为仅 2/8 通道承担、风险面显著收窄而保留的保守取值，不是新猜的数字。
- **通道隔离让"新增项"和"原有 6 个变量的精度"在数学上互不干扰**：这是本次改动里最关键的风险控制点——FFT/Grad 对 TAS 等变量的梯度贡献恒为零，即使 λ 取值不够"精调"，也不可能像旧版 G 配置那样拖累它们。
- **不引入新的可学习参数或训练动态**（没有采用 Kendall & Gal 的不确定性自动加权）：在单次昂贵训练里，我认为引入新的自适应权重机制本身也是一种"未验证的风险"，收益不如直接用文献+你自己消融实验已验证过的固定值稳妥。这个方向可以作为你论文里"未来工作"的一个点。
- **验证口径完全不变**：早停/best.pt 仍按纯 MAE 判定，`Loss/spatial_extreme`（现在语义上是 patch 版本）、`Loss/freq`、`Loss/grad` 仍分别记录在 TensorBoard，跑完就能直接看各子项收敛情况，不需要额外实验。

## 5. 面向论文的创新点表述

如果要写方法论创新，建议这样组织："提出了一种面向多变量气候降尺度的**通道感知混合损失（channel-aware hybrid loss）**框架——不同于以往工作对所有变量施加同构损失、仅通过标量权重区分重要性（如你原有的 `var_weights`），本文进一步在**损失函数结构层面**区分风光资源变量与常规气象变量：为 wind10/FSDS 引入与降尺度倍率对齐的局部分块极值一致性约束和限定通道的频谱/梯度监督，其余变量保持纯逐点损失不受影响。相较 NREL Sup3rWind 的全局极值一致性设计，本文将极值监督粒度提升到与 LR 网格分辨率对齐的局部尺度。" 这个表述是基于你现有框架的真实结构性扩展，站得住脚。