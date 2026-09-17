> **已过期（历史讨论稿）。** 定案与当前代码以 `loss_plan/final_decision.md` 和 `train.py` v2 为准；不要按本文 CLI 训练。

# Loss function design strategy
_Exported on 2026/9/15 at GMT+8 13:23:43 from Cursor (3.10.20)_

---

**User**

深入熟悉并掌握当前项目下的所有文件，确定损失函数的设计，根据现有的信息，结合高水平期刊文献检索，为我确定最佳损失函数设计方案，由于训练成本昂贵，要求尽量不通过多次调参，而是保证一次运行能实现不错的效果，目的是保证各变量的效果同时增强风光变量的效果，未来将其应用风光资源的评估，针对当前降尺度中的缺陷和痛点，使我的模型结果具有一定的改进或创新意义

---

**Cursor**

我会先把项目结构和现有损失函数设计摸清楚，再对照高水平文献，给出一次训练就能稳住各变量、同时加强风光效果的方案。

核心代码和训练日志还在核对，我会同时把当前损失实现、评估痛点和已有实验记录对齐。

接下来会把损失实现、已有训练结果和评估报告对齐，并检索气候降尺度与风光资源相关文献。

已有消融显示组合损失单独收益有限，我会继续对照测试集痛点和文献，再给出一次训练可用的方案。

测试集指标和既往讨论还在核对，同时检索气候降尺度、风光超分辨率和多任务损失的高水平文献。

一次训练的最优方案，不是把现有 `CombinedLoss` 的权重再拧一遍，而是：**全变量护栏用纯 MAE，风光增强只用通道隔离的辅助项**。这样温度/降水不会被拖累，风速和辐射又能对准资源评估痛点。

---

## 1. 项目里损失实际是什么

损失全部写在 `train.py`，不在 `model.py`。z-score 空间上：

\[
\mathcal{L}_{\text{现在默认}}
=
\underbrace{\text{TailMAE}(\gamma=0.5,\ z_{\max}=3)\times w_c}_{\text{8 通道一起加权}}
+
0.1\,\mathcal{L}_{\text{global max/min}}(\text{wind10, FSDS})
\]

FFT / Grad 默认已关。通道权重是 `wind10=FSDS=1.5`，`TAS=2M_RH=1.2`。

这套**默认配置从未做过全量训练**。真正跑完并上过测试集的是：

| 实验 | 损失 | 架构 | 测试集 overall MAE |
|---|---|---|---|
| `hr` | **纯 MAE** | base_ch=256，无 CBAM，**HR-aux stage1** | **1.081（最好）** |
| `no_all_100` | 纯 MAE | 无 HR-aux | 1.211 |
| `loss_combined` | TailMAE + FFT×0.1 + Grad×0.05 | 无 HR-aux | 1.232（更差） |
| `hr_loss` | 同上组合损失 | HR-aux stage1 | 验证略优于 `hr`，**没有进测试集主表** |

验证集分变量上，`hr_loss` 相对 `hr`：FSDS / Q / RH 略好，**PRE / TMAX / TMIN 变差**，wind10 几乎持平。测试集上 `loss_combined` 的 FSDS 偏差是 **−2.11 W/m²**，比纯 MAE 的 −1.28 更低估。

结论已经很硬：

1. **HR-aux 才是主收益**；损失不是。
2. **全局 FFT / Grad 会伤连续变量**，且加重 FSDS 低估。
3. 当前默认把尾部权重和通道权重打在全部 8 个变量上，和“各变量同时变好”是冲突的。
4. `SpatialExtremeLoss` 对整张 1801×3600 取全球 max/min，梯度几乎只打在一两个格点上，对区域风光资源几乎无效。NREL Sup3r 是在**小块**上做这件事。

---

## 2. 降尺度痛点（文献 + 你自己的评测）

| 痛点 | 证据 | 对风光评估的后果 |
|---|---|---|
| 对称损失系统性压低极值 | ExtremeCast（Xu et al., 2024）：MSE/MAE 在极值分布下会偏向低估；Rastogi et al., GRL 2025；Lopez-Gomez et al., AIES 2023 | 峰值风速、晴空辐照被抹平 |
| 过平滑、高频丢失 | Stengel et al., **PNAS 2020**（风光超分用 GAN）；你评测里的 RAPSD / grad-RMSE | 地形加速、云影结构变糊 |
| 全球像素 MAE 被“平常天气”淹没 | 你已写进 `TailWeightedMAE` 注释 | 能量干旱、切入风速附近误差被稀释 |
| 多变量抢梯度 | DSAF（Peng et al.）；Kendall et al., CVPR 2018；Atmosphere 2022 Laplace 多任务加权 | 给风光加权，温度变差 |
| 像素误差 ≠ 资源误差 | 你已在 `metrics_extreme.py` 里评功率曲线 / PV bin / 干旱 CSI；风电爬坡段功率 \(\propto v^3\) | MAE 下降，容量因子仍可能差 |
| 日均辐射被极夜/高纬低值稀释 | 日均 `FSDS`；`hr_aux` 已有日均 \(\cos(\mathrm{SZA})\) | 热带/夏季高潜力区学不够 |

GAN / 扩散能补高频，但不适合你这次：**训练贵、8 变量联合 GAN 不稳定，且你已证明全局频谱项会伤温度**。Stengel 的 GAN 只做风和辐射两个变量。

---

## 3. 明确不要做的（一次训练里排除）

- 不要再开全局 `--lambda_freq / --lambda_grad`。测试集已经否决。
- 不要用当前默认 `var_weights`（1.5 / 1.2）去改 bulk 损失。这是从温度偷梯度。
- 不要 \(\gamma=1.5\) 打在 8 个通道上（脚本里的 H 组）。风险高、无全量证据。
- 不要上 GAN、CRPS 分布头、GEV、新晴空场 `CSDSF`。一次训练、不引入新数据。
- 不要对全球场做 max/min 当作“区域极值”。

---

## 4. 一次训练的推荐损失（核心）

架构锁定已验证最优：**`base_ch=256`，`--no_cbam`，`--hr_aux_mode stage1`**。只改损失，归因才干净。

设计原则就一句：

> **8 个变量的 bulk 目标与 `hr` 完全相同（纯 MAE）；风光只通过“只作用于 wind10 / FSDS”的辅助项多拿梯度。**

共享骨干仍会有轻微耦合，但这是在不伤护栏的前提下，文献里最稳的一次训练策略。

\[
\mathcal{L}
=
\underbrace{\frac{1}{8}\sum_{c=1}^{8}\mathrm{MAE}(\hat y_c,y_c)}_{\text{全变量护栏，等于已验证的 hr}}
+
\mathcal{L}_{\text{tw-asym}}^{\{\text{wind},R\}}
+
0.1\,\mathcal{L}_{\text{patch-ext}}^{\{\text{wind},R\}}
+
0.1\,\mathcal{L}_{\text{sza}}^{R}
+
0.05\,\mathcal{L}_{\text{power}}^{\text{wind}}
\]

全部在 z-score 空间算（功率项先反标准化再算，见下）。系数全部来自已有实验或物理尺度，不新开网格搜索。

### （1）全变量护栏：逐通道平均纯 MAE

现在的 `.mean()` 已对 8 通道等权。保持 \(\gamma=0\)、通道权重全 1。这就是 `hr` 能同时把 TAS / TMAX / TMIN / PRE / Q 拉起来的原因。

### （2）风光专用、双向不对称尾部加权（主创新）

只对 `wind10`、`FSDS`：

\[
w = \bigl(1 + 0.5\cdot\mathrm{clip}(|y|,0,1.5)\bigr)\cdot a
\]

\(a\) 是 ExtremeCast 思想的简化版，但按**能源评估**改成双尾（他们的 Exloss 主要纠高值低估；你的评测同时看 P99 和低值干旱）：

- \(y > 1.5\) 且 \(\hat y < y\)：\(a=1.5\)（峰值低估：晴空辐照、大风）
- \(y < -1.5\) 且 \(\hat y > y\)：\(a=1.5\)（低值高估：切入以下、弱辐照被抹平）
- 其余：\(a=1\)

\(\gamma=0.5\)、阈值 \(1.5\) 直接复用你已经在用的 `loss_gamma` 和 `val_extreme_z_thresh`，不是新超参。\(z_{\max}\) 用 1.5 而不是 3，避免极少数全球风暴像素绑架梯度。

相对当前 `TailWeightedMAE` 的改进：后者对 8 变量、且对残差符号对称，所以压不住 FSDS 的系统性低估。

### （3）把全球极值改成补丁极值（纠正现有实现）

NREL Sup3rWind / Benton et al., *Energies* 2025：内容损失里加空间（和时间）max/min 的 MAE。他们在补丁上训练；你是全球一张图，必须改成局部：

对 `wind10`、`FSDS` 做 `adaptive_max_pool2d / min_pool2d` 到 \((H/16,\ W/16)=(112,225)\)，再对池化后的 max/min 做 MAE。

- 16 对应 16× 降尺度，不是调参。
- 每个补丁一个极值，梯度变密，才对应“区域资源峰值 / 云遮低谷”。
- \(\lambda_e=0.1\) 保持现有默认。

### （4）日均太阳几何加权（不增加数据）

`hr_aux` 最后一通道已经是日均 \(\langle\cos\theta\rangle\in[0,1]\)。日均 `FSDS` 不能用瞬时白天 mask，但可以用**当日可获得能量**加权：

\[
\mathcal{L}_{\text{sza}}
=
\frac{\sum m\,|\hat R - R|}{\sum m+\varepsilon},\quad
m=\mathrm{ReLU}(\langle\cos\theta\rangle)
\]

极夜自动接近 0，热带/夏季正午潜力区权重大。这是你现有数据能做的、最贴近辐射物理的一项，无需 `CSDSF`。

### （5）可微风电功率曲线（对准已有评测）

IEC 典型值已在 `metrics_extreme.py`：切入 3、额定 12、切出 25 m/s。只在爬坡段（对误差最敏感）监督：

\[
P(v)=\mathrm{clip}\Bigl(\frac{v^3-v_{\mathrm{in}}^3}{v_{\mathrm{r}}^3-v_{\mathrm{in}}^3},0,1\Bigr)
\quad(v_{\mathrm{in}}\le v\le v_{\mathrm{out}})
\]

\(\mathcal{L}_{\text{power}}=\mathrm{MAE}(P(\hat v),P(v))\)，\(v\) 由 z-score 反标准化。\(\lambda_p=0.05\) 很小，避免把风速主 MAE 带偏。这直接对准你已经在画的功率曲线分区 MAE。

---

## 5. 创新点怎么写才站得住

不是“又加了几个 loss 权重”，而是针对统计降尺度用于风光资源评估的三个缺口：

1. **任务解耦的多变量损失**：全变量 bulk MAE 与风光辅助项分离。避免 Rastogi/Lopez-Gomez 式极值损失伤均值、也避免你自己 `loss_combined` 伤 TAS/TMAX。
2. **能源双尾不对称**：同时打“峰值低估”和“干旱高估”，对应容量因子和能量干旱，而不是只抄 Exloss 的单侧高值。
3. **全球 0.1° 场上的补丁极值 + 日均太阳几何 + 功率曲线**：把 Sup3r 的极值项从补丁训练正确迁移到全球图，并把训练目标和 `compare_models_test.py` 里已经在用的资源指标对齐。

架构贡献继续写 HR-aux；损失贡献写“在 `hr` 护栏不退步的前提下，专门改善 wind10/FSDS 的尾部、干旱和功率曲线”。

---

## 6. 一次运行的配置（建议原样用）

```bash
python train.py \
  --base_ch 256 --no_cbam --hr_aux_mode stage1 \
  --loss_gamma 0.0 \
  --var_weights "TAS=1.0,PRE=1.0,wind10=1.0,Q=1.0,2M_RH=1.0,2M_TMAX=1.0,2M_TMIN=1.0,FSDS=1.0" \
  --lambda_freq 0 --lambda_grad 0 \
  --extreme_vars wind10,FSDS --lambda_extreme 0.1 \
  --val_extreme_z_thresh 1.5 \
  --epochs 100 \
  --run_dir runs/exp_resource_decoupled
```

上面是**护栏部分**（与 `hr` 对齐）。实现后，资源项作为 `CombinedLoss` 内部固定模块，不要再暴露一堆要扫的 λ。建议写死：

| 项 | 固定值 | 理由 |
|---|---|---|
| 资源尾部 \(\gamma\) | 0.5 | 已在 `hr_loss` 用过，不爆炸 |
| 双尾阈值 / \(z_{\max}\) | 1.5 | 与现有验证极端阈值一致 |
| 不对称倍率 | 1.5 | 温和，避免指数爆炸（Lopez-Gomez 的风险） |
| 补丁极值 \(\lambda_e\) | 0.1 | 现有默认 + NREL 量级 |
| 补丁尺度 | \(H/16,W/16\) | 等于上采样倍率 |
| SZA 项 \(\lambda_s\) | 0.1 | 与极值项同量级 |
| 功率项 \(\lambda_p\) | 0.05 | 更小，只微调爬坡段 |
| FFT / Grad | 0 | 测试集已否决 |

早停、存盘继续：**`best_mae.pt` 看整体（对标 `hr`）**，**`best_extreme.pt` 看 wind10/FSDS 极端像素**。论文主结果用 `best_mae`；资源表同时报 `best_extreme`。

验证时除原有 `MAE_val/*` 外，必须看：`MAE_val_extreme/wind10`、`FSDS`，以及反标准化后的功率爬坡段 MAE、FSDS 的 `qbias_mean`（你现在 FSDS 是系统性负偏差）。

---

## 7. 和现有默认的差别（为什么现默认不宜直接开跑）

当前 `train.py` 默认是：8 通道 TailMAE(\(\gamma=0.5\)) × 风光 1.5 倍 + 全球 max/min。这会：

- 把尾部压力加到 TAS/TMAX/TMIN 上（`hr_loss` 已出现温度变差）；
- 用通道权重从护栏里抢梯度；
- 全球极值项几乎不提供区域资源信号。

一次贵训练不该拿这套未验证默认去赌。推荐方案的 bulk 与测试集冠军 `hr` 同目标，风光增强是**加项而不是改主项**。

---

当前代码还实现不了第（2）（3）的通道隔离、补丁池化、SZA 和功率项，需要改 `CombinedLoss`。如果你确认按这个方案落地，我可以直接改 `train.py`，并保持 `γ=0` + 全 1 权重时与旧 `hr` 训练目标数值等价，方便对照。

