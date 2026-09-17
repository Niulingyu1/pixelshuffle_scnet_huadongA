# Loss function design strategy
_Exported on 2026/9/15 at GMT+8 13:24:10 from Cursor (3.10.20)_

---

**User**

深入熟悉并掌握当前项目下的所有文件，确定损失函数的设计，根据现有的信息，结合高水平期刊文献检索，为我确定最佳损失函数设计方案，由于训练成本昂贵，要求尽量不通过多次调参，而是保证一次运行能实现不错的效果，目的是保证各变量的效果同时增强风光变量的效果，未来将其应用风光资源的评估，针对当前降尺度中的缺陷和痛点，使我的模型结果具有一定的改进或创新意义

---

**Cursor**

先全面摸清项目里的损失函数与降尺度设定，再对照高水平文献给出一次训练就能稳住、并强化风光变量的方案。

项目已能看出一套组合损失，我接着读实现与实验记录，并检索高水平降尺度/风光评估文献。

接着核对默认超参、已有训练结果，并检索气候降尺度与风光评估相关的高水平损失设计。

历史

纯 MAE 跑次里 FSDS 明显落后。我继续核对最终指标，并补齐 NREL 极值损失与风光物理量相关文献。

证据已经齐了：纯 MAE 会低估风光极值，全图 `SpatialExtremeLoss` 约束过弱。接下来把一次训练就能落地的方案做成可对照的分析页。

**一次训练应使用「共享 Tail-MAE 主干 + 仅作用于风光的加性头」**，不要再用通道权重去抢温度/降水的梯度，也不要开全通道 FFT/Grad。完整对照与冻结配方在 [风光损失方案](/public/home/acd7koea4a/.cursor/projects/public-home-acd7koea4a-work/canvases/loss-design-scheme.canvas.tsx)（可在聊天旁打开的分析页）。

## 结论

仓库里现有的 `CombinedLoss` 方向对（尾部加权 + 风光极值），但有两处会在**一次长跑**里失效：

1. **`SpatialExtremeLoss` 对整张 1801×3600 全球图取 max/min**，只约束 2 个标量。NREL Sup3rWind 的同类损失是在**小 patch** 上算的；搬到全球场后，极大值往往落在南大洋/高原单点，管不到山地急流、海岸大风、云隙辐照峰值。
2. **`wind10=1.5, FSDS=1.5` 是零和再分配**。总梯度预算固定时，抬风光就会压 TAS/PRE/Q，和「各变量同时变好」冲突。

A800 纯 MAE 日志已经把瓶颈写死：epoch 60 的 `best.pt` 上，TAS/Q 误差降到 epoch 1 的约 **23%**，`wind10` 仍剩 **53%**，`FSDS` 仍剩 **75%**。MAE 最小化的是中位数，大风被系统性压低；而风电功率密度 \(\propto v^3\)，约 4% 的风速误差会变成约 **12%** 的功率误差。

## 冻结配方（RC-ACL，λ 不再搜）

\[
L = L_{\text{pixel}} + 0.10\,L_{\text{patch}} + 0.15\,L_{\text{hp}} + 0.10\,L_{\text{wpd}} + 1.0\,L_{\text{phys}}
\]

| 项 | 设定 | 作用 |
| --- | --- | --- |
| \(L_{\text{pixel}}\) | \(\gamma=0.5,\,z_{\max}=3\)，**8 通道权重全为 1** | 保全部变量；FSDS 仅按 `cos(SZA)` **再分配**像素权重（除均值后期望仍为 1，不增加该通道总预算） |
| \(L_{\text{patch}}\) | 只对 wind10/FSDS，64×64 `max_pool` 的局地 max/min MAE | 把 NREL 极值损失的语义修回来 |
| \(L_{\text{hp}}\) | 只对 wind10/FSDS，\(x-\mathrm{avgpool}_{5}\) 的 L1 | 补过平滑；**不要**全图 rfft / 全通道 Sobel（仓库消融已证明伤温度） |
| \(L_{\text{wpd}}\) | 物理空间相对误差 \(\lvert\hat v^3-v^3\rvert/(\lvert v^3\rvert+1)\) | 训练目标对齐风资源评估 |
| \(L_{\text{phys}}\) | TMAX≥TAS≥TMIN、风速/辐照非负、RH∈[0,100] 的铰链 | 满足后梯度≈0，不干扰主项 |

\(\lambda\in\{0.10,0.15\}\) 能直接冻住，是因为前两项与 z-score MAE 同量纲，\(L_{\text{wpd}}\) 用相对误差自归一化。早停和 `best.pt` **继续只看 8 变量纯 MAE**；风光看 `best_resource.pt` 以及 WPD / 白天 FSDS / 分块极值。

## 明确不要做的

- 消融 H 的 `--loss_gamma 1.5 --loss_z_max 1.5`：约 87% 像素顶到权重上限，尾部加权名存实亡。
- 把 wind/FSDS 通道权重拉到 2.0：继续零和，温度会退。
- GAN / 扩散 / NLL-Bernoulli-Gamma：文献里高频更好，但训练成本和调参面超出「一次跑通」。
- 在损失搜索的同时改 CBAM / `hr_aux` / GroupNorm。

创新点可以写成四条可检验命题：**全球极值损失的语义坍缩与分块修复**、**通道选择性高频（温度不回退）**、**WPD 进入训练目标**、**加性特化优于通道再加权**。这些对齐 PNAS 2020 / Energies 2025 Sup3rWind、2025 loss-bench、以及 Nature Communications 2025 的风电功率密度结论，又避开了本项目已经踩过的 FFT/Grad 坑。

确认后我可以把 RC-ACL 直接写进 `CombinedLoss`（改 `forward` 以接收 `hr_aux` 和反标准化统计量），架构与优化器不动。

