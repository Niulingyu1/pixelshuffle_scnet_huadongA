# 降尺度训练：分析梳理与下一步行动

日期：2026-09-12  
范围：当前仓库脚本与任务逻辑、迁移遗留问题、损失函数/验证口径、训练优化路径  
数据现状：正式全量 HDF5 仍在上传；现有可用样本为 `/public/share/acd7koea4a/hdf5/MAM/shard_cra1p5_full_0038.h5`（16 个样本，float32）

---

## 1. 任务与整体逻辑（先对齐再训练）

### 1.1 任务

把全球约 1°（180×360）的 8 个气象变量降尺度到约 0.1°（1801×3600）：

`TAS`, `PRE`, `wind10`, `Q`, `2M_RH`, `2M_TMAX`, `2M_TMIN`, `FSDS`

预处理约定：

- `PRE` 写入 HDF5 前已做 `log1p`（验证时反标准化后仍是 log1p 空间，真实 mm/day 需再做 `expm1`）
- `Q` 已 ×1000（g/kg）
- 8 个动态变量用 `global_stats_state.json` 的 `var_all` 做 z-score
- 静态场与 `cos(SZA)` 不参与该 z-score

### 1.2 数据流

```
HDF5 shard_*.h5
  → 读 data/x, data/y, data/dates
  → z-score（同一组 mean/std）
  → x_lr = 8 个归一化气象变量
  → hr_aux = 6 个 HR 静态 + 1 个在线计算的 cos(SZA)_hr
  → 模型输出 8 通道 HR 预测（无末端激活，回归）
```

当前实测 shard：

| 项 | 值 |
| --- | --- |
| 文件 | `hdf5/MAM/shard_cra1p5_full_0038.h5` |
| `data/x` | `(16, 8, 180, 360)` float32 |
| `data/y` | `(16, 8, 1801, 3600)` float32 |
| 压缩 | gzip-4，按样本分块 |
| 质量 | 无 NaN / Inf，数值范围正常 |
| 注释过期点 | 代码注释仍写 float16，实际是 float32（正确性不受影响，磁盘/带宽约为原先预期的 2 倍） |

### 1.3 模型

`PixelShuffleDownscaleNet`：

1. InitConv：1×1，`8 → base_ch`（默认 256）
2. 4 个 UpStage：ResBlock × 2 → CBAM（M1）→ PixelShuffle(2×) → 按需注入 HR-aux
3. Stage4 的 shuffle_conv 用 1×1，避免 1440×2880 上 3×3 的巨大 im2col workspace
4. 双线性插值对齐到 1801×3600（`align_corners=True`）
5. Head 输出 8 通道

默认消融开关：`use_cbam=True`，`hr_aux_mode=all`，gradient checkpoint 开启。

### 1.4 训练回路

- 损失：`CombinedLoss`（见第 3 节）
- 优化器：AdamW（`lr=2e-4`，`weight_decay=1e-4`）
- 调度：线性 warmup → 余弦退火
- 梯度累积：默认 2（正式脚本目前写 4）
- 精度：bf16 autocast，不用 GradScaler
- 分布式：检测到 `RANK`/`WORLD_SIZE` 才启用 DDP；单卡行为不变
- 早停 / top-K / `best.pt`：**始终按验证集纯 MAE（`Loss/val`）**

---

## 2. 已完成的代码与文档修复

这些改动已经落地，下一步训练应基于当前代码，不要再照抄旧命令。

### 2.1 `dataset.py`：`manifests` 真正生效

原先 `_build_index()` 接收 `manifests` 但完全不用，会把季节目录下全部 `shard_*.h5` 读进来。正式数据上传过程中目录里会混入未完成/坏批次文件，这会直接污染训练。

现在的规则：

- 不传 `--manifests`：纳入全部 `shard_*.h5`（当前只有单一批次 `cra1p5_full` 时可以这样）
- 传入 `--manifests cra1p5_full`：只保留 `shard_{tag}_{序号}.h5` 且 `tag` 在白名单内的文件
- 文件名不符合约定：打印警告并跳过，避免静默漏读/误读

`train.py` 已增加对应 CLI：`--manifests`。

隔离测试目录（软链接，不复制数据）：

```
/public/home/acd7koea4a/work/smoke_test_data/MAM/shard_cra1p5_full_0038.h5
```

登录节点已验证：`--manifests cra1p5_full` 只加载该 shard，样本数 = 16。

### 2.2 `train.py`：PRE 物理量纲补充指标

`MAE_val/PRE` 反 z-score 后仍是 `log1p(mm/day)` 误差，容易被误读成“降水物理精度差”。

新增：

- TensorBoard：`MAE_val_physical/PRE`（`expm1` 还原后的真实 mm/day MAE）
- 验证日志同时打印对照
- 不替换、不影响 `Loss/val`、`MAE_val/*`、早停、top-K
- DDP 下已纳入 `all_reduce` 聚合
- `expm1` 前 `clamp(max=30)`，防止早期预测发散溢出

### 2.3 文档与正式训练脚本与代码对齐

此前三处“顺带发现”的过期问题已修：

| 问题 | 现状 |
| --- | --- |
| README 第 4 节仍写 `lambda_freq=0.1`、`lambda_grad=0.05`，且未写 `var_weights` / `SpatialExtremeLoss` | 已按当前 `train.py` 重写第 4 节 |
| README 仍出现 `--no_hr_aux`、`--init_type`、`in_ch=15`、`base_ch=128` 等旧参数 | 已校正第 3 节通道数；新增第 6 节速查，旧命令不再作为可执行示例 |
| `slurm/train_gpu.slurm` 锁死旧消融基线（`base_ch=128 --no_cbam --hr_aux_mode none` + 纯 MAE） | 已改为当前默认全量配置，输出目录 `runs/exp_full_default` |

权威消融命令以 `train.py` 文件顶部 docstring 为准。

### 2.4 验证保存：标准判据 + 业务并行判据

在不改变标准模型选择规则的前提下，新增：

- `--resource_vars`（默认 `wind10,FSDS`）
- TensorBoard：`MAE_val_resource/mean`
- 并行 checkpoint：`best_resource.pt`

`best.pt` / 早停 / top-K **仍然只看全部 8 变量的纯 MAE**。`best_resource.pt` 只给风光资源侧多一个可选起点，二者互不覆盖。

---

## 3. 损失函数：当前设计与如何加强风、光

### 3.1 当前默认（推荐作为第一版正式训练）

```
L_train = L_tail(γ=0.5, z_max=3.0, c)
        + λe · L_extreme(wind10, FSDS)     # 默认 0.1
        + λf · L_FFT                       # 默认 0，关闭
        + λg · L_grad                      # 默认 0，关闭
```

| 子项 | 默认 | 作用范围 | 对 wind10 / FSDS 的针对性 |
| --- | --- | --- | --- |
| 尾部权重 `γ·clamp(\|y\|,0,3)` | 开，γ=0.5 | 全部 8 通道、逐像素 | 无针对性；\|z\|=3 时权重最高约 2.5× |
| 逐变量权重 `--var_weights` | 开 | 指定通道、全部像素 | 已倾斜：wind10=1.5，FSDS=1.5，TAS/2M_RH=1.2，其余=1.0 |
| `SpatialExtremeLoss` | 开，λ=0.1 | 仅 `--extreme_vars` | 专门约束区域 max/min，保护风速峰值、辐照度晴空/云遮极值 |
| FFT / Grad | 关 | 全部通道 | 历史消融显示会拖累 TAS / TMAX / TMIN，默认不要开 |

结论：**当前默认已经是风光资源优先设计**。第一版正式训练不要再叠 FFT/Grad，也不要一上来就把权重拉到极端。

### 3.2 想让辐射和风学得更好：三个杠杆，从精准到粗放

只改命令行，不必改代码。

| 优先级 | 参数 | 建议 | 为什么先调它 |
| --- | --- | --- | --- |
| 1 | `--lambda_extreme` | 0.1 → 0.2（最多试到 0.3） | 只影响 wind10/FSDS，直接保护区域极值，最不影响其它变量 |
| 2 | `--var_weights` 中 wind10/FSDS | 1.5 → 2.0 | 只抬这两个通道的全场梯度预算；同时把 TAS/2M_RH 从 1.2 降回 1.0，避免它们继续抢梯度 |
| 3 | `--loss_gamma` | 先不动 | 作用在全部 8 通道，对风/光没有针对性 |

推荐加强配置（待默认全量跑通后再做对照，不要和第一版绑在一起）：

```bash
python train.py --base_ch 256 \
  --var_weights "TAS=1.0,PRE=1.0,wind10=2.0,Q=1.0,2M_RH=1.0,2M_TMAX=1.0,2M_TMIN=1.0,FSDS=2.0" \
  --extreme_vars wind10,FSDS --lambda_extreme 0.2 \
  --run_dir runs/exp_loss_wind_solar_boost
```

观察：

- 应下降：`MAE_val/wind10`、`MAE_val/FSDS`、`MAE_val_extreme/wind10`、`MAE_val_extreme/FSDS`、`MAE_val_resource/mean`
- 不应明显变差：`MAE_val/TAS`、`MAE_val/2M_RH`、`MAE_val_physical/PRE`
- 这是多目标权衡，建议 2～3 组小规模消融后再定案

数据预处理层面（收益可能更大，但不要现在做）：wind10 近似 Weibull、右偏，FSDS 有昼夜/云遮跳变。二者目前只用与温度相同的线性 z-score。若下一版重做 HDF5，可考虑给 wind10 做 `sqrt` 或 `log1p`，让回归目标更接近正态。这需要重生成数据，不适合冒烟阶段改。

---

## 4. 组合损失下，验证损失应如何保存

### 4.1 标准做法（当前代码已实现，不要改掉）

**模型选择判据必须与训练损失配置解耦。**

| 产物 / 标量 | 口径 | 用途 |
| --- | --- | --- |
| `Loss/val` | 全部 8 变量、z-score 空间纯 MAE | **唯一**早停 / top-K / `best.pt` 判据 |
| `Loss/val_combined` 及各子项 | 与训练相同的 `CombinedLoss` | 诊断：该 run 自己的优化目标在验证集上是否达成 |
| `MAE_val/<变量>` | 反 z-score 后的物理量纲 MAE | 解读各变量精度；注意 PRE 仍是 log1p 空间 |
| `MAE_val_physical/PRE` | `expm1` 后的 mm/day | PRE 唯一真实物理量纲误差 |
| `MAE_val_extreme/*` | 仅 \|z\| > 阈值的像素 | 看极值是否真的比纯 MAE 训练更好 |
| `MAE_val_resource/mean` | 默认 wind10+FSDS 物理 MAE 平均 | 业务并行指标 |
| `best.pt` | `Loss/val` 最优 | 标准交付起点 |
| `best_resource.pt` | `MAE_val_resource/mean` 最优 | 风光资源侧可选起点 |

### 4.2 为什么不能用组合损失挑 checkpoint

1. **跨 run 不可比**：`λ_extreme=0.1` 和 `0.3` 的组合损失值不是同一个函数，数字不能直接比大小。
2. **自我评分**：用训练目标挑模型，容易选出对该损失项过拟合、整体点误差反而更差的权重。
3. **历史可比**：所有旧 run 都用纯 MAE 这把尺子；改掉就无法和历史实验对齐。

正确用法：

- 选模型、早停、写论文主表：看 `Loss/val` 和 `best.pt`
- 判断“风光加强是否生效”：看 `MAE_val_extreme/wind10`、`MAE_val_extreme/FSDS`、`Loss/val_spatial_extreme`、`best_resource.pt`
- 最终交付：两个 checkpoint 都做一次推理对比，再决定用哪一个

---

## 5. 还可优化的地方（按成本 / 收益排序）

### A. 零代码，正式训练前就该核对（优先）

1. **有效 batch 与 BN 噪声**  
   现在 `batch_size=1`，`ResBlock` / Init / Head 全是 BatchNorm2d，单样本统计噪声大。`accum_steps` 只平滑梯度，**不改善 BN 统计**。有多卡时优先把 `batch_size` 提到 2～4，并保持默认 `--sync_bn`。

2. **按真实步数重算 `--warmup_steps`**  
   默认 100 是调试量级。冒烟或半量跑起来后，看日志里 `total optimizer steps ≈ N`，把 warmup 设为约 3%～5% 的 N。

3. **全量数据后放宽 `--val_interval`**  
   每个验证样本都是 1801×3600 全图前向，很贵。数据上来后可用 `--val_interval 2~5`。早停按“验证次数”计，不会因此失效。

### B. 小改代码，基线跑通后再做

4. **BatchNorm → GroupNorm**：彻底摆脱小 batch BN 噪声，但与旧 checkpoint 不兼容，应做成开关消融，不要直接改默认。
5. **`torch.compile()`**：当前环境是 PyTorch 2.5.1，可能有 10%～30% 加速；先在冒烟规模验证与 checkpoint / DDP 的兼容性。
6. **EMA 权重**：对小 batch 很有效，实现成本低，适合作为交付权重。

### C. 数据协议，收益可能最大，但不要和第一版训练绑死

7. **wind10 非线性变换**：见第 3.2 节，需重做 HDF5。
8. **验证集按年份切，而不是随机打乱样本**：当前 `random.shuffle(indices)` 会让相邻日期同时出现在 train/val，验证可能偏乐观。若最终要推未见年份或 CESM，应改成按年分组。
9. **分季节归一化**：四季合并的 `var_all` 会拉大温度/辐射的 z-score 尺度。可作为后续实验，不是本次必须项。

原则：**第一版只做 A 类核对 + 当前默认损失/架构；B、C 一次只改一类，否则无法定位是数据、损失还是结构的问题。**

---

## 6. 下一步怎么做

登录节点不要直接跑训练。下面按顺序执行。

### 第 0 步：确认分区（现在就能做）

`slurm/*.slurm` 里的 `--partition=qdagnormal` 是占位。登录节点此前查不到本账号可见分区，提交前必须改成真实分区：

```bash
sinfo
sacctmgr show assoc user=$USER format=account,partition%20
```

两处都要改：

- `slurm/train_smoke_test.slurm`
- `slurm/train_gpu.slurm`

### 第 1 步：计算节点冒烟（数据未齐时的唯一训练动作）

目的：在真实 GPU 上验证前向/反向/存盘/`manifests`/`MAE_val_physical/PRE`/`best_resource.pt` 都通，而不是出有意义的精度。

```bash
cd /public/home/acd7koea4a/work
sbatch slurm/train_smoke_test.slurm
```

等价命令（脚本内部实际执行的是）：

```bash
python -u train.py \
    --hdf5_root /public/home/acd7koea4a/work/smoke_test_data \
    --seasons MAM \
    --manifests cra1p5_full \
    --val_fraction 0.1 \
    --epochs 3 --val_interval 1 \
    --batch_size 1 --accum_steps 2 --warmup_steps 5 \
    --num_workers 2 \
    --early_stop_patience 0 --save_top_k 1 \
    --run_dir runs/smoke_cra1p5_full_0038
```

通过标准：

- 3 个 epoch 正常结束，无 OOM
- 出现 `Loss/train`、`Loss/val`
- 出现 `MAE_val_physical/PRE`，量级应是 mm/day，不应只是个位数 log1p 误差被误当成物理误差
- `checkpoints/` 下有 `best.pt`；若 wind10/FSDS 有改进，还会有 `best_resource.pt`

查看：

```bash
tail -f logs/slurm_smoke_<jobid>.out
tensorboard --logdir runs/smoke_cra1p5_full_0038
```

冒烟失败先修管线，不要开始全量。

### 第 2 步：等正式数据上传完毕后做完整性检查

上传结束后先不要立刻开 100 epoch。建议在计算节点或低负载环境做轻量检查（读元数据，不要在登录节点扫全部大文件）：

1. 四季目录是否齐全：`MAM/JJA/SON/DJF`
2. 是否还有 `*.raysync.uploading` 残留
3. 每个 `shard_*.h5` 能否打开，且含 `data/x`、`data/y`、`data/dates`
4. 样本总数是否与预期年份/季节大致相符
5. 决定是否加 `--manifests cra1p5_full`（目录里若可能混入其它 tag 或半成品，建议加上）

### 第 3 步：第一版正式训练（默认全量配置，不要同时改损失）

数据齐、冒烟通过后：

```bash
cd /public/home/acd7koea4a/work
# 先改好 --partition
sbatch slurm/train_gpu.slurm
```

这版使用当前代码默认：

- `base_ch=256`，CBAM 开，`hr_aux_mode=all`
- 组合损失默认值（Tail + 通道权重 + SpatialExtreme×0.1，FFT/Grad 关）
- `epochs=100`，`batch_size=1`，`accum_steps=4`，`val_fraction=0.2`
- 早停 patience=20
- 输出：`runs/exp_full_default`

有多卡时，优先改用 `slurm/train_ddp_single_node.slurm`，并把 `batch_size` 提到 2～4，而不是只加 `accum_steps`。

开跑后立刻根据日志重算 warmup：

```
LR schedule: ... (total optimizer steps ≈ N)
```

若 N 很大而 warmup 仍是 100，下一版（或 resume 前重启）改为约 `0.03N ~ 0.05N`。

### 第 4 步：第一版跑稳后再做风光加强对照

不要和第一版并行改架构。单独开一个 run：

```bash
python -u train.py \
    --hdf5_root /public/share/acd7koea4a/hdf5 \
    --manifests cra1p5_full \
    --epochs 100 --batch_size 1 --accum_steps 4 --val_fraction 0.2 \
    --num_workers 4 --early_stop_patience 20 \
    --var_weights "TAS=1.0,PRE=1.0,wind10=2.0,Q=1.0,2M_RH=1.0,2M_TMAX=1.0,2M_TMIN=1.0,FSDS=2.0" \
    --extreme_vars wind10,FSDS --lambda_extreme 0.2 \
    --run_dir runs/exp_loss_wind_solar_boost
```

对比口径：

- 公平比整体点误差：两边的 `Loss/val`
- 比风光是否真的更好：`MAE_val_extreme/wind10`、`MAE_val_extreme/FSDS`、`MAE_val_resource/mean`
- 交付候选：`best.pt` 与 `best_resource.pt` 都推理一次

### 第 5 步：基线有数之后再排优化队列

顺序建议：

1. 多卡 + 提高真实 `batch_size`（解决 BN）
2. 按年份划分验证集（若目标是外推到未见年 / CESM）
3. EMA
4. GroupNorm 开关消融
5. wind10 预处理变换（需重做数据）

---

## 7. 提交前检查清单

- [ ] 已把 slurm 的 `--partition` 改成账号真实可用分区
- [ ] 不在登录节点跑 `train.py`
- [ ] 冒烟使用 `smoke_test_data` + `--manifests cra1p5_full`，不要指向正在上传的整个 `hdf5/MAM`
- [ ] 正式训练前确认没有 `.raysync.uploading` 半成品被 glob 进来
- [ ] 第一版正式训练使用默认损失，不叠加 FFT/Grad，不和新的 `var_weights` 绑在同一 run
- [ ] 选模型看 `best.pt` / `Loss/val`；看风光看 `best_resource.pt` / `MAE_val_extreme/*`
- [ ] 看降水精度时看 `MAE_val_physical/PRE`，不要只看 `MAE_val/PRE`

---

## 8. 方案 A/B 落地情况（已完成，本次改动）

用户决策：同意方案 B（GroupNorm / torch.compile / EMA）+ 方案 A 调参；暂不做方案 C；
**正式训练确定**：`--no_cbam`（不开 CBAM）+ `--hr_aux_mode stage1`（仅 Stage1 注入 HR）+
分布式训练（DDP）。

### 8.1 方案 B：新增的架构/训练机制开关

| 开关 | 实现位置 | 说明 |
| --- | --- | --- |
| `--norm_type {batch,group}` | `model.py`（`_make_norm`/`ResBlock`/`UpStage`/`PixelShuffleDownscaleNet`）+ `train.py` CLI | `group`=GroupNorm，不依赖 batch 维统计量；分布式 `--sync_bn` 仅在 `norm_type=batch` 时生效；**正式训练用 `group`** |
| `--compile` | `train.py`（DDP 包装之后 `torch.compile(model)`） | 默认关闭；`unwrap_model()` 已同时处理 `torch.compile` 的 `_orig_mod` 和 DDP 的 `.module`，checkpoint/EMA 均不受影响 |
| `--ema_decay` | `train.py`（新增 `ModelEMA` 类） | 默认 `0.0` 关闭；开启后验证阶段临时换用 EMA 权重（`ema.apply_to()` 上下文管理器，结束自动还原在线权重），checkpoint 新增 `model_ema` 字段；`--resume` 时若旧 checkpoint 无该字段会自动用当前在线权重重新初始化，不报错 |
| `--warmup_ratio` | `train.py`（调度器创建前） | 设置后用 `round(ratio × total_steps)` 自动覆盖 `--warmup_steps`，不必先跑一次看日志再回填 |

`infer.py` 同步更新：`--norm_type`（手动指定）+ `--auto_model_cfg` 自动从 state_dict 判断
（`init_conv.1.running_mean` 是否存在）+ `--use_ema`（加载 checkpoint 的 `model_ema` 而非
`model`，无该字段自动回退并警告）。

**验证**：已在登录节点用 CPU（`base_ch=4` 极小模型）跑通单卡与 2 进程 `gloo` DDP 全流程
（数据加载→前向反向→EMA 更新→验证→`best.pt`/`best_resource.pt` 落盘含 `model_ema`→
`--resume` 恢复 EMA→`infer.py --auto_model_cfg`/`--use_ema` 正确识别 `norm_type=group` 并
加载），2 进程 DDP 在第 1 个 epoch 验证阶段被 OOM-kill（预期内——登录节点本就不适合跑
这类内存较大的任务，这印证了之前的约束，不是新引入的问题）。真实 GPU/DCU 上的数值与
显存表现仍需用户在计算节点上验证。

### 8.2 方案 A：调参（零代码，体现在下面的正式启动命令里）

| 调整 | 落地方式 |
| --- | --- |
| 有效 batch 与 BN 噪声 | 已切 `--norm_type group`（不再依赖 batch 维统计量），同时多卡 DDP 下 `--batch_size` 由 1 提到 2 |
| 按真实步数重算 warmup | `--warmup_ratio 0.03` |
| 放宽验证频率 | `--val_interval 2` |

### 8.3 正式训练命令（已更新到 slurm 脚本，用户自行在计算节点提交）

单节点多卡（首选，`slurm/train_ddp_single_node.slurm`）：

```bash
cd /public/home/acd7koea4a/work
# 先确认并改好 --partition（sinfo / sacctmgr show assoc user=$USER）
sbatch slurm/train_ddp_single_node.slurm
```

等价命令（脚本内部实际执行）：

```bash
torchrun --nnodes=1 --nproc_per_node=8 train.py \
    --hdf5_root /public/share/acd7koea4a/hdf5 \
    --epochs 100 --batch_size 2 --accum_steps 2 --val_fraction 0.2 --val_interval 2 \
    --num_workers 4 \
    --no_cbam --hr_aux_mode stage1 \
    --norm_type group --ema_decay 0.999 \
    --warmup_ratio 0.03 \
    --early_stop_patience 20 \
    --run_dir runs/exp_prod_ddp_1node
```

多节点扩展：`slurm/train_ddp_multi_node.slurm`（同一套参数，先跑通单节点再扩）。
单卡回退（无多卡资源时）：`slurm/train_gpu.slurm`（同一套参数，`--batch_size 1 --accum_steps 4`）。

**建议先做的两步冒烟**（单个 shard，16 个样本，验证管线而非精度）：

1. `slurm/train_smoke_test.slurm`：单卡，验证正式架构开关（`--no_cbam --hr_aux_mode stage1
   --norm_type group --ema_decay 0.999 --warmup_ratio`）在真实 GPU 上跑通、日志/checkpoint
   字段齐全。
2. `slurm/train_smoke_test_ddp.slurm`：2 卡 DDP，验证分布式路径（进程组/`DistributedSampler`/
   梯度同步/验证指标跨 rank 聚合/`best.pt` 落盘）本身没问题，再放心扩到 8 卡。

若通过后想单独验证 `--compile`：先在 `train_smoke_test_ddp.slurm` 上追加 `--compile` 单独跑
一次对比（不要和其它新变量一起引入），确认无异常再加进正式命令。

### 8.4 推理时如何取到这次训练的权重

正式 checkpoint 架构变了（`use_cbam=False, hr_aux_mode=stage1, norm_type=group`），`infer.py`
用 `--auto_model_cfg` 会自动识别，不需要手动传架构参数；若训练开了 EMA，推荐加 `--use_ema`
使用 `model_ema`（部署权重）而非训练用的在线权重：

```bash
python infer.py --ckpt runs/exp_prod_ddp_1node/checkpoints/best.pt \
    --auto_model_cfg --use_ema \
    --hdf5_root /public/share/acd7koea4a/hdf5 --seasons DJF \
    --out_dir infer_out_prod --output_mode per_sample --output_format nc \
    --output_space physical --amp_bf16
```

---

## 9. 关键路径速查

| 用途 | 路径 |
| --- | --- |
| 训练入口 | `/public/home/acd7koea4a/work/train.py` |
| 数据集 | `/public/home/acd7koea4a/work/dataset.py` |
| 模型 | `/public/home/acd7koea4a/work/model.py` |
| 路径配置 | `/public/home/acd7koea4a/work/paths.py` |
| 推理入口 | `/public/home/acd7koea4a/work/infer.py` |
| 说明文档（已与代码对齐） | `/public/home/acd7koea4a/work/DOWNSCALE_README.md` |
| 冒烟 slurm（单卡） | `/public/home/acd7koea4a/work/slurm/train_smoke_test.slurm` |
| 冒烟 slurm（2 卡 DDP） | `/public/home/acd7koea4a/work/slurm/train_smoke_test_ddp.slurm` |
| **正式训练 slurm（首选，单节点多卡 DDP）** | `/public/home/acd7koea4a/work/slurm/train_ddp_single_node.slurm` |
| 正式训练 slurm（多节点扩展） | `/public/home/acd7koea4a/work/slurm/train_ddp_multi_node.slurm` |
| 正式训练 slurm（单卡回退） | `/public/home/acd7koea4a/work/slurm/train_gpu.slurm` |
| 冒烟数据 | `/public/home/acd7koea4a/work/smoke_test_data` |
| 正式 HDF5 根目录 | `/public/share/acd7koea4a/hdf5` |
| 归一化统计 | `/public/share/acd7koea4a/states/global_stats_state.json` |
