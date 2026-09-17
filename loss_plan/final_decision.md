# 损失函数设计——最终方案 v2（已结合外部评审修正）

日期：2026-09-15（v2，替换 v1）；2026-09-17 已按第 9 节清单完成 `train.py` 代码实施
状态：方案已与用户逐项确认定案，**代码已实施完成**（`train.py` 新增/修改的类、CLI 参数、
`main()`/`validate()` 改动均已落地；`DOWNSCALE_README.md` 第 4/5/9 节已同步更新）。
因当前登录节点无 GPU/torch 环境，仅完成静态语法/字节码编译检查与逐段人工复核，**尚未在真实
GPU 上跑过 1-epoch 数量级校验（第 4 节）**，正式长跑前必须先完成该校验。

2026-09-17 补充：实施后对照 `loss_plan/ques.md` 复核了 5 条风险。**损失公式不改**；其中第 1 条
（物理一致性误在 z 空间算 hinge）经核对代码**不成立**（已先反标准化），但第 2.4 节原文容易误读，
已改写。第 2–5 条成立，已在第 2、4、5、8 节及第 11 节标注，并下调相关论文措辞。

---

## 0. 修订说明（v1 → v2）

v1 定案后，用户提供了一份独立评审（`loss-design-review.canvas.tsx`），对 v1 做了系统性审查。逐条核实后（对照 `train.py`/`dataset.py` 实际代码），**评审中的绝大多数"必须修改"项成立**，已在 v2 中采纳；其中两项（`best_resource` 量纲错误算术平均、验证集随机切分的时间泄漏）是**当前代码库本来就存在的缺陷**，v1 沿用而未质疑，属于本次审查的额外收获，不是评审"挑错"挑出来的新问题。

### 0.1 客观核实结论（逐条）

| 评审意见 | 核实结果 | 处理 |
|---|---|---|
| `best_resource` 把 m/s 与 W/m² 直接算术平均，无物理可解释性 | **属实**（`validate()` 现有代码），且与本次损失方案无关，是既有缺陷 | **采纳，一并修复**（用户已确认 q7=立即修） |
| 验证集随机逐样本切分存在时间泄漏 | **属实**（`train.py` 现有代码 `random.shuffle`），`TRAINING_ANALYSIS.md` 早已自知此问题但列为"以后再做" | **本次不改**，论文中明确标注为局限性（用户已确认 q8=暂保留） |
| RampPowerCurve 套用轮毂高度功率曲线到 10m 日均风速，违反 Jensen 不等式 | **完全成立**，是 v1 最大的过度承诺 | **采纳**：重新定位为"功率敏感区代理"，去掉"对齐能量产出"的表述 |
| PatchExtreme 只约束 max/min 数值、不约束位置，可能诱导伪极值 | **成立**，是此类损失的通用理论弱点 | **采纳**：增加 patch-mean 项，抑制"为凑 max 而整体平移 patch"的退化解；**不声称缓解位置错位**（max+mean 仍不约束极值落点，见第 2.2、11.4 节） |
| "护栏不退步"表述过度承诺（共享网络下新增项仍会间接影响其余变量） | **成立**，是措辞层面的严谨性问题 | **采纳**：改称"参考目标（reference objective）"，不再声称"从根本上避免冲突" |
| 物理一致性损失把不同量纲 hinge 直接相加，且漏了 TAS 上下界 | **成立**，是可低成本修复的设计 bug | **采纳**：先反标准化到物理/log1p 量纲再算 hinge，再除以训练期 `σ_k` 无量纲化；补齐 `TMIN≤TAS≤TMAX`（切勿在 z 空间直接比较或写 `ReLU(−z)`，见第 2.4、11.1 节） |
| Ramp 项 clip 外无梯度、需 float32 计算 | **成立** | **采纳**：显式 float32（与 `FFTLoss`/`GradientLoss` 先例一致） |
| 全球像素等权评价，规则经纬网高估高纬度权重 | **属实**（当前 `validate()`/`TailWeightedMAE` 均无面积加权），且是既有代码普遍缺陷 | **采纳**：`TailWeightedMAE` 像素权重叠加 `cos(latitude)`，验证指标同步改为面积加权 |
| Tail 权重应改为 grid-cell×month 局地气候异常，而非全球 z-score | 方向正确，但需要新增一整套训练期逐格点逐月分位数预计算管线 | **本次不做**（用户已确认 q9=保留全球 z-score），论文中标注为局限性/未来工作 |
| 按连续年份分块训练/验证/测试 | 方法论上正确，但会使本次所有历史 run（含已确认可信的 `hr` 基线）不可比 | **本次不做**（同 q8），标注为局限性 |
| 最小消融矩阵 E0–E5（含多种子） | 方法论正确，是发表级论文的标准要求，但与"训练成本昂贵、一次运行"直接冲突 | **不完整采纳**：本次只跑"完整新方案"1 次（纯 MAE 对照已在其他工作区跑完，见 q10），不做完整消融矩阵，论文中如实说明这一局限 |
| 升级为概率/CRPS 或至少报告谱、方差比等诊断 | 训练范式建议，超出本次损失重设计范围；但**评估层面**诊断成本很低 | **部分采纳**：不改训练范式，但建议交付后用现成 checkpoint 补充谱/方差诊断（不需要重训） |
| 基线定义应写出完整 CLI，不要用消融字母引用 | 成立（`train.py` 文档里 "F" 配置实际未关闭默认权重/极值项，引用不精确） | **采纳**，见第 5 节 |
| 计算/显存：多次 adaptive max-pool 反向可能增加显存峰值 | 成立，需要实测 | **采纳**：新增项全部走 float32 支路，冒烟阶段记录 `max_memory_allocated` |

### 0.2 我不完全认同评审的部分（保留判断依据）

- 评审要求"逐变量非劣约束 + 多种子置信区间才能证明不退步"——标准过严，与本项目"一次训练"的硬约束冲突。改为**如实报告验证集各变量 MAE 相对基线的实测变化**，不做形式化统计检验，但也不再使用"从根本上避免冲突"这类无法证明的措辞。
- 评审建议 PatchExtreme 完整实现可微 top-k/exceedance frequency——理论更优，但工程成本远超"最小改动"约束，改用成本更低的 patch-mean 折中方案。patch-mean **不**解决极值位置错位，只抑制整体平移退化解（见第 11.4 节）。

---

## 1. 最终总损失公式（v2）

$$L = \underbrace{L_{\text{tail}}^{\text{area-weighted}}(\gamma=0.5,\ z_{\max}=3,\ c\equiv 1)}_{\text{参考目标，8 通道等权 + cos(lat) 面积加权}} \;+\; \lambda_{p}\,\underbrace{L_{\text{patch}}^{\{wind10,\,FSDS\}}}_{\text{局部极值+均值一致性}} \;+\; \lambda_{r}\,\underbrace{L_{\text{wps}}^{wind10}}_{\text{风功率敏感区代理（非能量对齐）}} \;+\; \lambda_{h}\,\underbrace{L_{\text{phys}}^{\text{无量纲}}}_{\text{物理一致性安全网}}$$

**退化说明（措辞已按评审修正）**：当 `λ_p=λ_r=λ_h=0` 时，`L` 的**函数值**严格等于面积加权版 `TailWeightedMAE(γ=0.5, z_max=3, channel_weight=1)`。这只保证损失函数本身的退化，**不构成"训练出的模型在其余变量上必然不退步"的证明**——是否退步需要看实际验证集结果，训练完成后如实报告。

---

## 2. 各子项详细设计（v2 修订版）

### 2.1 参考目标：面积加权 `TailWeightedMAE(γ=0.5, z_max=3.0, channel_weight=None)`

- 在现有 `TailWeightedMAE` 的像素权重 `w = 1 + γ·clamp(|y|, 0, z_max)` 基础上，**再乘以 `cos(latitude)` 归一化面积权重**（`dataset.py` 中已有 `_LAT_HR` 静态纬度数组，可直接生成 `(1801,1)` 的面积权重图并广播，成本极低）。
- `--var_weights` 默认改为全 1（同 v1），偏置权重仍可显式传参用于对照。
- 验证指标 `MAE_val/*` 同步改为面积加权版本，与训练目标口径一致（避免"训练面积加权、评估像素等权"的口径不一致）。

### 2.2 `PatchExtremeLoss`（v2：新增 patch-mean 项）

- 网格 `(180, 360)`，与 LR 网格**同分辨率近似对齐**（措辞修正：不再声称"严格一一对应"，因为 `adaptive_max_pool2d` 在真实球面上并非精确的 lat/lon 单元映射）。
- `L_patch = L1(patch_max) + L1(patch_min) + L1(patch_mean)`
  - `patch_mean`（`F.adaptive_avg_pool2d`）约束 patch **一阶矩**，用于抑制"为命中局部 max 而把整块预测整体抬升/压低"的粗糙退化解。成本极低，予以保留。
  - **不要写成"缓解了位置错位"**：`max + min + mean` 仍然完全不约束极值落在哪个像素；patch 内挪动极值位置、同时保持 max 与 mean 正确，仍然轻易能做到。该项与主 MAE 也高度相关（patch 平均误差是逐点 MAE 的低通版本），信息增量有限。
- **确定性回归的内在张力**（不是 bug，是预期 trade-off）：模型无法知道 patch 内次网格极值出现在哪个像素，降低 `L1(patch_max)` 的省力解往往是整体抬高该 patch。结果可能是 wind10/FSDS 的极值统计改善、逐点 MAE 和整体偏差变差。若 `λ_p=0.1` 偏大，可能出现"MAE 明显退步换来极值小幅改善"的不划算结果。冒烟阶段除损失量级外，应记录 wind10/FSDS 的 patch 均值偏差作为早期预警（见第 4、11.5 节）。
- 默认 `λ_p = 0.1`，仍需过第 4 节数量级校验；若冒烟已见系统性正偏差，优先下调 `λ_p`，不改损失结构。

### 2.3 `WindPowerSensitivityProxy`（v2：重新定位，替代 v1 的 `RampPowerCurveLoss`）

**命名与表述变更**：不再称为"功率曲线损失"或声称"对齐能量产出误差"，改称**风功率敏感区代理目标**，论文中需明确声明局限性：

> 本项使用 10m 风速（而非风机轮毂高度风速）和日尺度平均值（而非亚日/瞬时值）作为代理；根据 Jensen 不等式，`P(日均v) ≠ 日均P(v)`，因此该项不能被解释为直接优化容量因子或发电量误差，仅作为"在气象学习目标中显式引入风速—功率非线性敏感度"的训练技巧，其下游能量代表性需要未来用轮毂高度/亚日数据做专门验证。

数学形式（不变，仅重新定位表述）：

$$P(v)=\mathrm{clip}\!\left(\frac{v^3-v_{\text{in}}^3}{v_{\text{r}}^3-v_{\text{in}}^3},\ 0,\ 1\right), \quad v_{\text{in}}=3,\ v_{\text{r}}=12\ \text{m/s（IEC 典型值，仅作代理，非机型校准值）}$$

$$L_{\text{wps}} = \frac{\sum_{\text{mask}} \bigl|P(\hat v) - P(v)\bigr|}{\sum \text{mask} + \varepsilon}, \quad \text{mask} = \mathbb{1}[v_{\text{in}} \le v \le v_{\text{r}}]\ \text{（由真值判定，}.detach()\text{）}$$

- **mask 范围是设计取舍，不是实现错误**：额定风速以上（及切出区间）`P(v)` 被 clip 成常数，本项对真值 `v>12` 的像素完全沉默。高风速极值由 `TailWeightedMAE` + `PatchExtremeLoss` 监督，不由本项承担。论文必须写清这一点，避免被理解成"功率项覆盖了容量因子贡献最大的区域"。若把 mask 扩到 `v_true ≥ v_{\text{in}}`，真值已超额定而预测仍在爬坡段时会产生"把预测往额定功率推"的信号，但那就不再是纯敏感区代理，本次**不改 mask**。
- **覆盖率更可能过密而不是过稀**：全球日均 10m 风速大量落在 3–12 m/s。`λ_r=0.05` 乘上高覆盖率后，实际影响可能比权重字面值更大。第 4 节必须记录 `wps_mask_ratio`；若覆盖率很高（例如 >40%），优先考虑压低 `λ_r`，而不是先改曲线。
- **数值稳定性修正**：全程 float32 计算（含反标准化、v³、边界比较），不用 bf16。
- 训练/验证时额外记录：`mask 覆盖像素比例`（诊断用，不参与模型选择）。隔离梯度范数不是训练默认行为：损失值 ±3× 不能代表梯度贡献可比，冒烟阶段建议另记各子项对 head 末层权重的 `grad.norm()`（见第 4、11.2 节）。
- 默认 `λ_r = 0.05`（不变）。
- 工程实现：构造时传入 wind10 的 `norm_mean/std`（buffer），`forward(pred, target)` 签名不变。

### 2.4 `PhysicalConsistencyLoss`（v2：先反标准化，再无量纲化；补齐 TAS 边界）

**计算顺序（必须按此理解，切勿在 z 空间直接写 hinge）**：

1. 反标准化：`x_{\text{phys}} = \hat y \odot \sigma + \mu`（`μ, σ` 为各通道训练期 `norm_mean/std`）。TAS/TMIN/TMAX 为 K，wind10 为 m/s，FSDS 为 W/m²，RH 为 %；**PRE 还原后仍是 log1p 空间**（与 `dataset.py` 写入约定一致），`log1p(\text{PRE})\ge 0` 等价于物理降水 `≥0`。
2. 在该量纲上算 hinge。
3. 再除以该约束相关变量的训练期 `σ_k`（即 `norm_std`）做无量纲化后平均——这是"物理量纲 hinge / 该变量 std"，**不是**在 z 空间算完再除一次。

$$L_{\text{phys}} = \frac{1}{K}\sum_{k=1}^{K} \frac{\mathrm{mean}\bigl(\mathrm{hinge}_k(\mathrm{denorm}(\hat y))\bigr)}{\sigma_k}$$

若跳过第 1 步、在 z 空间写 `ReLU(TMIN_z-TAS_z)` 或 `ReLU(-wind_z)`、`ReLU(RH_z-100)`，约束会系统性施加错误：不同温度通道的 mean/std 不同，顺序在 z 空间与物理空间不等价；`ReLU(-wind_z)` 会惩罚所有低于全球平均风速的像素。`train.py` 已按上述顺序实现，第 1 条 ques 所担心的 bug **代码中不存在**；此前本节公式未写出 `denorm`，容易误读。

**约束清单**（均作用在 `denorm(pred)` 上；v1 遗漏 TAS 上下界，已补齐）：

- 温度顺序：`ReLU(TMIN − TAS) + ReLU(TAS − TMAX)`（物理 K）
- 非负性：`ReLU(−wind10)`、`ReLU(−FSDS)`、`ReLU(−Q)`、`ReLU(−PRE_{\log1p})`
- RH 边界：`ReLU(RH−100) + ReLU(−RH)`（物理 %）

**额外记录违反率**：`Loss/phys_violation_<约束名>`（hinge>0 的像素占比），用于论文报告物理一致性，不参与训练目标本身。

默认 `λ_h = 0.02`（安全网，训练收敛后应自然趋近 0）。

---

## 3. 并行修复：`best_resource.pt` 量纲问题（独立于损失函数，用户已确认 q7=立即修）

**问题**：现有 `resource_metric = per_var_mae[resource_var_indices].mean().item()` 把 wind10（m/s）和 FSDS（W/m²）直接算术平均，无物理意义。

**修复方案**：

1. 训练开始前，对验证集做一次**基线通道**推理：把 `x_lr` 直接双线性插值（`F.interpolate`）到 HR 分辨率作为"naive baseline"预测，计算该 baseline 在 wind10、FSDS 上的物理量纲 MAE（`MAE_baseline_wind10`、`MAE_baseline_FSDS`），只需算一次，训练全程复用（不随 epoch 变化）。
2. 每次验证时，对每个资源变量计算 **skill score**：`SS_var = 1 − MAE_model_var / MAE_baseline_var`（无量纲，>0 表示优于双线性插值基线）。
3. `MAE_val_resource/mean` 改为 `SS_wind10` 与 `SS_FSDS` 的**预先约定的等权平均**（而不是物理量纲 MAE 平均），同时**分别记录** `Skill_val/wind10`、`Skill_val/FSDS`，供论文报告两个独立分数或 Pareto 结果，不只看合成后的单一数字。
4. `best_resource.pt` 判据从"resource_metric 最小"改为"skill score 均值最大"。

这一修复不涉及 `CombinedLoss`，只涉及 `validate()`/`main()`，与第 2 节的损失函数改动相互独立，可分别实施验证。

---

## 4. 一次性启动前的数量级校验（数值健全性检查，不是调参）

正式长跑前，用 1 个 epoch 的小规模数据跑一遍前向，打印各子项未加权前的均值：`Loss/tail_w`（面积加权后）、`Loss/patch_extreme`（含 mean 项）、`Loss/wps`、`Loss/phys`（无量纲化后）。**损失值互相在 ±3× 以内只是必要的量级检查，不能保证各项对参数的梯度贡献可比**（见第 11.2 节）：`PhysicalConsistencyLoss` 多数像素 hinge=0、均值很小，违反像素上梯度却是常数 `1/σ`；`WindPowerSensitivityProxy` 的 `dP/dv ∝ 3v²`，切入口附近与额定附近可差一个量级以上。

同时记录：

- `wps` 支路的 mask 覆盖比例：既要排除"过稀疏导致该项基本不生效"，也要警惕"过密导致 `λ_r=0.05` 实际影响偏大"（日均 10m 风速大量落在 3–12 m/s）
- wind10 / FSDS 的 patch 均值偏差（预测 patch-mean − 真值 patch-mean）：预警 `PatchExtremeLoss` 是否在用整体抬升换极值（第 11.5 节）
- 冒烟阶段建议（非正式长跑默认）：各子项对 head 末层权重的 `grad.norm()`（`torch.autograd.grad(L_i, head[-1].weight, retain_graph=True)`），用于发现"某项几乎没梯度"或"某项抢走 head"
- `max_memory_allocated`：判断新增 pooling/hinge 是否显著推高显存峰值

---

## 5. 明确的实验计划（用户已确认 q10：不做完整消融矩阵）

| 实验 | 状态 | 说明 |
|---|---|---|
| 纯 MAE 基线（对应"hr"配置：HR-aux stage1，无通道偏置，无 tail-weight） | **已在其他工作区跑完**，结果可信（用户已确认） | 作为本次唯一对照 |
| 本方案完整版（v2：面积加权 tail-weight + PatchExtreme + WindPowerSensitivityProxy + 无量纲 PhysicalConsistency） | **本次唯一新增训练**，架构固定为 `--no_cbam --hr_aux_mode stage1 --norm_type group`（`DOWNSCALE_README.md` 第 9 节已确定的正式架构） | 与纯 MAE 基线做验证集/测试集全指标对比 |

**明确写入论文的局限性声明**（不回避、不夸大）：

1. 本研究只做了"纯 MAE 基线 vs. 本方案完整版"的单次对照，未做逐机制的独立消融（如"只加 PatchExtreme 不加 WindPowerSensitivityProxy"），无法精确拆解各子项的独立贡献，仅能报告组合效果；受限于单次全量训练的高昂算力成本。
2. 验证/测试集切分为随机逐样本切分，存在潜在时间自相关导致的乐观偏差，可能不足以支撑对"未来气候""跨年份泛化""CESM 域迁移"等外推场景的强结论。
3. Tail 权重基于全球 z-score，未做逐格点/逐月的局地气候异常校正，可能把气候带差异部分误学为"极端事件"。
4. `WindPowerSensitivityProxy` 使用 10m 日均风速代理轮毂高度瞬时风速，不能直接解释为发电量/容量因子误差的优化；mask 仅覆盖真值 3–12 m/s 的立方敏感区，额定以上与切出区间由尾部加权和局部极值项监督，本项沉默。
5. `PatchExtremeLoss` 的 patch-mean 只约束一阶矩，不约束极值位置；确定性回归下降低 patch-max 误差的省力解往往是整体抬高该 patch，可能以逐点 MAE / 偏差换极值统计。
6. 训练目标仍是确定性回归（条件均值/中位数），不保证亚网格方差、空间相关结构、极端事件持续性等分布性质被正确恢复；建议在交付后补充谱分析、方差比等诊断（不需要重新训练）。
7. 冒烟阶段用损失值 ±3× 做量级检查，不等于各项梯度贡献已校准；正式训练默认不计算隔离梯度范数。

---

## 6. 明确不做的部分（v1 已定 + v2 追加）

- ❌ PV 辐照代理转换损失、SZA 日均能量加权、通道限定 FFT/Grad、不确定性自动加权/GAN/CRPS（v1 已定）
- ❌ 完整可微 top-k/exceedance frequency 版本的 PatchExtreme（评审建议，成本过高，用 patch-mean 折中）
- ❌ 按年份重新划分训练/验证/测试（评审建议，成本高且破坏历史可比性，列为局限性）
- ❌ grid-cell×month 局地气候异常版 tail 权重（评审建议，需新预计算管线，列为局限性）
- ❌ 完整消融矩阵 E0–E5 多种子（评审建议，与"一次训练"约束冲突，只做单次完整方案 vs. 已有纯 MAE 基线对照）

---

## 7. 一次性启动前的数量级校验清单（汇总，供实施时对照）

- [ ] `Loss/tail_w`（面积加权后）、`Loss/patch_extreme`（max+min+mean）、`Loss/wps`、`Loss/phys`（无量纲化后）损失值量级互相在 ±3× 以内（**只检查损失值，不代表梯度贡献可比**）
- [ ] `wps` 支路 mask 覆盖比例记录在案：非零；若很高（例如 >40%）则评估是否下调 `λ_r`
- [ ] wind10 / FSDS 的 patch 均值偏差无系统性大幅抬升（否则评估是否下调 `λ_p`）
- [ ] （可选，仅冒烟）各子项对 head 末层 `grad.norm()` 均非零、无一项目数量级碾压其余项
- [ ] `max_memory_allocated` 相对现有基线的增幅在可接受范围
- [ ] `best_resource.pt` 判据已切换为 skill score 均值，baseline（双线性插值 LR）MAE 已预计算并记录

---

## 8. 面向论文的方法论表述（v2 修订版，措辞已降级）

> 针对面向风光资源评估的气候降尺度任务，本文提出**参考目标与风光增量项解耦的混合损失设计**：将"全部变量的面积加权尾部加权 MAE"作为参考目标，与仅作用于风光通道的增量项（局部极值与 patch 一阶矩一致性、风功率立方敏感区代理、反标准化后的无量纲物理一致性安全网）在结构上分离；增量项权重为 0 时损失函数值可退化为参考目标（不构成模型效果不退步的形式化证明，具体效果以实测验证集为准）。相较面向局地训练图块的全局极值一致性，本文将极值监督下沉到与 LR 网格同分辨率近似对齐的局部尺度，并增加 patch-mean 以抑制为命中 max 而整体平移的退化解（**不声称约束了极值位置**）。风功率项仅在真值 3–12 m/s 的立方敏感区生效，额定以上由尾部加权与局部极值项监督；该代理使用 10m 日均风速，不能解释为容量因子或发电量误差优化。研究局限性详见第 5、11 节。

**已删除的过度表述**：不再使用"首次""从根本上避免冲突""直接实现能量产出对齐""严格一一对应""缓解了位置错位/伪极值位置错位""功率项覆盖高风速/容量因子贡献最大区域"等无法证明或不准确的措辞。

---

## 9. 待实施的代码改动清单（v2，供下一步执行，本文件本身不修改代码）

1. `train.py` 新增 `PatchExtremeLoss`（含 mean 项，替代/升级 `SpatialExtremeLoss`，旧类保留供对照）。
2. `train.py` 新增 `WindPowerSensitivityProxy`（构造时接收 wind10 的 `norm_mean/std`，float32 计算，记录 mask 覆盖比例与梯度范数诊断）。
3. `train.py` 新增 `PhysicalConsistencyLoss`（构造时接收全 8 通道 `norm_mean/std`，无量纲化组合，补齐 TAS 上下界，记录逐约束违反率）。
4. `TailWeightedMAE` 增加可选 `area_weight`（`(H,1)` 或 `(H,W)` 张量，构造时传入，来自 `cos(latitude)` 预计算），像素权重叠加该项；`None` 时向后兼容（不加面积权重）。
5. `CombinedLoss.__init__` 接入以上改动；`forward(pred, target)` 对外签名不变。
6. `main()`：
   - 构造 `criterion` 时传入 `norm_mean/std`（已有变量）与面积权重（新增，从 `_LAT_HR` 计算）；
   - 训练开始前，用现有 val_loader 跑一次双线性插值 baseline，计算 `MAE_baseline_wind10/FSDS`，用于 skill score；
   - `validate()` 内 `resource_metric` 计算逻辑改为 skill score 均值，`best_resource.pt` 判据同步更新。
7. `parse_args()` 新增 CLI（在 v1 基础上不变，另加）：`--baseline_resource_mae_recompute`（可选，跳过重算直接传入已知基线值，避免每次 resume 都重跑一次全量验证集插值）。
8. `DOWNSCALE_README.md` 第 3、4 节同步更新（当前文档仍描述旧的通道偏置 + 全局极值 + 量纲错误的 resource 指标方案）。

---

## 10. 后续行动

- 代码已按第 9 节实施；文档措辞已按第 11 节（`ques.md` 复核）修订。
- 正式长跑前先跑第 4 节的 1-epoch 数量级校验（损失值 ±3×、`wps_mask_ratio`、patch 均值偏差、显存；可选 head 末层 `grad.norm()`）。
- 正式训练完成后，与已有纯 MAE 基线做验证集/测试集全指标对比（含 skill score、面积加权 MAE、物理违反率等新增诊断），如实报告，不做多机制消融拆解。

---

## 11. 实施后复核（`loss_plan/ques.md`，2026-09-17）

损失公式与默认 λ **不改**。下表是复核结论，供论文表述与冒烟诊断对照。

| # | 疑问 | 结论 | 文档处理 |
|---|---|---|---|
| 11.1 | 物理一致性是否在 z 空间算 hinge（`TMIN_z≤TAS_z`、`ReLU(−wind_z)`、`RH_z−100`） | **代码不成立**：`PhysicalConsistencyLoss` 先 `pred·σ+μ` 再算 hinge，再除 `σ_k` 无量纲化。PRE 还原后是 log1p 空间，`≥0` 仍等价于物理降水非负。 | 第 2.4 节公式已补上 `denorm`；禁止在论文/注释里写成对 z 空间直接 hinge |
| 11.2 | ±3× 校验的是损失值，不是梯度贡献 | **成立**。hinge 项均值可很小但局部梯度陡；WPS 的 `dP/dv` 在爬坡段内可差约 15×。 | 第 4、7 节已标注；冒烟可选记 head 末层 `grad.norm()`，正式长跑默认不算 |
| 11.3 | WPS 的 `mask=1[3≤v_true≤12]` 排除额定以上高风速 | **成立，但是设计取舍**。立方敏感区之外 `P(v)` 为常数；高风速由 tail + PatchExtreme 监督。覆盖率更可能过密。 | 第 2.3、5.4、8 节已写清；不改 mask；覆盖率过高则考虑降 `λ_r` |
| 11.4 | `patch_mean` 并未解决极值位置错位 | **成立**。`max+mean` 不约束落点；能抑制的是整块平移去凑 max。与主 MAE 高度相关。 | **保留该项**；删除"缓解位置错位/伪极值位置错位"；改为"约束 patch 一阶矩，抑制整体平移退化解" |
| 11.5 | PatchExtreme 与确定性回归的张力：省力解是抬高整块 | **成立，是预期 trade-off**。`λ_p=0.1` 若偏大，可能 MAE 退步换极值小幅改善。 | 第 2.2、4、5.5 节已标注；冒烟记录 patch 均值偏差，必要时降 `λ_p` |
