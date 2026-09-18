# PixelShuffle 气候降尺度：训练说明

训练相关说明已合并到本文。命令与注意事项见 **第 1 节**。

刻意单独保留、改脚本前先读：

- [`TRAIN_INFER_NORM.md`](TRAIN_INFER_NORM.md) — 训练/推理标准化契约（禁止二次 z-score / 二次 log1p）
- [`HR_AUX_DTYPE.md`](HR_AUX_DTYPE.md) — `hr_aux` 与 bf16 特征拼接的 dtype 约定（本次不改代码）

权威实现以 `train.py` / `model.py` / `dataset.py` / `scripts/launch_platform_*.sh` 为准。

---

## 1. 训练速查

### 1.1 命令

正式长跑只走 SCNet「模型训练」控制台，**不要** `sbatch` / `slurm/*.slurm`。裸跑 `python train.py` 会落到代码默认架构（CBAM 开、`hr_aux=all`、BatchNorm、无 EMA），不要当正式训练。

| 用途 | 控制台启动命令 |
| --- | --- |
| **正式 100 epoch（唯一入口）** | `bash /public/home/acd7koea4a/work/scripts/launch_platform_train.sh` |
| 预标准化数据单卡冒烟（1 shard） | `bash scripts/launch_platform_smoke_norm_single.sh` |
| 预标准化数据 2 卡 DDP 冒烟 | `bash scripts/launch_platform_smoke_norm_ddp.sh` |
| 全量 1 epoch 探测（读盘 / 显存 / `validate()` / 落盘） | `bash scripts/launch_platform_train_copy.sh` |
| 8 卡短跑（2 epoch） | `bash scripts/launch_platform_short_8gpu.sh` |
| A800 小集单卡/2 卡冒烟 | `scripts/launch_platform_smoke_single.sh` / `smoke_ddp.sh` |
| 正式配置单卡吞吐基准 | `scripts/launch_platform_train_single_gpu_benchmark.sh` |
| 历史纯 MAE 对齐（`hr_aux=none`，非正式） | `scripts/launch_platform_train_bw_a800_aligned_benchmark.sh` |

推荐顺序：`smoke_norm_single` → `smoke_norm_ddp` → `train_copy`（1 epoch）→ `launch_platform_train.sh`（100 epoch）。冒烟只验管线，16 样本上的 MAE 没有业务意义。

脚本内部等价命令（`NPROC_PER_NODE` 必须等于控制台「每实例加速卡数量」，默认 8）：

```bash
torchrun --nnodes="${WORLD_SIZE:-1}" --nproc_per_node="${NPROC_PER_NODE:-8}" \
    --node_rank="${RANK:-0}" --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-23456}" \
    train.py \
    --hdf5_root "$(python -c 'from paths import HDF5_ROOT; print(HDF5_ROOT)')" \
    --seasons MAM JJA SON DJF --manifests cra1p5_full \
    --epochs 100 --batch_size 1 --accum_steps 2 --val_fraction 0.2 --val_interval 2 \
    --num_workers 4 \
    --no_cbam --hr_aux_mode stage1 --norm_type group --ema_decay 0.999 \
    --warmup_ratio 0.03 --early_stop_patience 20 \
    --run_dir runs/exp_prod_ddp_platform
```

损失不传参，走 `train.py` v2 默认。2 卡时先 `export NPROC_PER_NODE=2`。

### 1.2 控制台怎么填

| 项 | 值 |
| --- | --- |
| 入口 | 人工智能服务 → 模型训练 → 创建训练任务 |
| 加速卡 | BW1000 用 **DTK PyTorch**；A800 用 **CUDA PyTorch**（两边不要混用镜像） |
| 每实例卡数 | 正式 8（可先 4）；必须与 `NPROC_PER_NODE` 一致 |
| 实例数 | 1（单节点通了再考虑多实例） |
| 启动命令 | 上表对应脚本 |

平台注入的 `WORLD_SIZE` / `RANK` 是**实例数**，不是卡数。`nproc_per_node` 才是每实例卡数。

### 1.3 注意事项（硬约束）

**环境**

- BW1000：**不要** `source env/activate.sh`（家目录 conda 是 NVIDIA 版，会落到 CPU）。用镜像自带 `python`/`torch`。
- A800：conda `pytorch_downscale` 理论上可用，先确认 `torch.cuda.is_available()`；镜像与卡必须是 CUDA，不要选 DTK。
- 日志出现 `Using device: cpu` / `CUDA is not available`：几乎一定是 Python 用错，不是代码该改成 HIP API。
- 不要在登录节点 / Notebook 上跑全图训练。依赖需预装：`h5py`、`netCDF4`、`tensorboard`。

**单卡 / 多卡启动相反**

| 场景 | 做法 |
| --- | --- |
| 单卡 | 必须 `unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT`，再 `python -u train.py` |
| 多卡 | 必须 `torchrun`，并**保留**平台注入的 `WORLD_SIZE` / `RANK` / `MASTER_*` |

`train.py` 只要看到 `RANK` 和 `WORLD_SIZE` 就会 init DDP。单卡不 unset = 「1 卡假 DDP」，末尾插值还会再申请约 15.8 GiB，64GB 卡上容易 OOM。

**数据与配置**

- 训练读 `paths.HDF5_ROOT`（优先 `/public/share/acd7koea4a/hdf5_norm_fp16`，否则家目录同名目录）。磁盘已是 z-score **fp16**。
- 正式脚本已写死 `--manifests cra1p5_full`。不要指向正在上传的半成品目录。
- 8 卡保持 `--batch_size 1`，不要提到 2。`accum_steps` 只平滑梯度，不改善 BatchNorm 统计；正式训练已用 `--norm_type group`。
- 第一版不要开 `--compile`；不要叠 FFT/Grad；不要改回 `--lambda_extreme` 或通道偏置 `var_weights`。
- 只传 `--loss_gamma 0` **不是**纯 MAE：还必须 `--no_area_weight` 并关掉全部 λ。
- `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` 在本 DCU 平台不支持，不要靠它救命。
- **禁止** `infer.py --hdf5_root .../hdf5_norm_fp16`。推理契约见 `TRAIN_INFER_NORM.md`。

### 1.4 提交前清单

- [ ] 启动命令是 `launch_platform_train.sh`，不是 `sbatch`
- [ ] `NPROC_PER_NODE` = 「每实例加速卡数量」；8 卡 `--batch_size 1`
- [ ] BW1000 不要 `source env/activate.sh`
- [ ] 冒烟 / 1 epoch 探测已通过
- [ ] 损失走 v2 默认；选模型看 `Loss/val` / `best.pt`；看降水看 `MAE_val_physical/PRE`

启动后立刻核对：`Using device: cuda:0`、`distributed` 与卡数一致、`use_cbam=False`、`hr_aux_mode=stage1`、`norm_type=group`、`[EMA] decay=0.999`、`[warmup] --warmup_ratio=0.03`。

---

## 2. 任务与数据

| 项目 | 说明 |
| --- | --- |
| 目标 | 全球约 1°（180×360）→ 约 0.1°（1801×3600）；`TAS`, `PRE`, `wind10`, `Q`, `2M_RH`, `2M_TMAX`, `2M_TMIN`, `FSDS` |
| 训练 HDF5 | `paths.HDF5_ROOT`：153 shard / 14965 样本 / 533GB；磁盘 `data/x`/`data/y` 为离线 z-score **float16**，`metadata.normalized=True`，`metadata.stats_sha256` 与当前 `STATS_FILE` 绑定 |
| 原始备份 | `/public/share/acd7koea4a/hdf5`（`HDF5_ROOT_RAW`，fp32、未 z-score） |
| 预处理 | 写入原始 HDF5 前：`PRE` 已 `log1p`；`Q` 已 ×1000（g/kg） |
| 归一化 | `global_stats_state.json` 的 `var_all`：mean / `sqrt(M2/n)`（总体标准差）。预标准化数据离线完成；未标准化数据在 Dataset 在线做。**静态与 cos(SZA) 不归一化** |
| Dataset 出口 | `x_lr`/`y_hr` **bfloat16**；`hr_aux` **float32**（见 `HR_AUX_DTYPE.md`） |

数据流：读 `data/x,y,dates` → 若 `metadata.normalized` 则跳过 z-score，否则用同一组 mean/std 在线 z-score → `x/y` 转 bf16 → `hr_aux` = 6 HR 静态 + 1 在线 `cos(SZA)_hr`（不写入 HDF5）→ 模型输出 8 通道 HR（无末端激活）。

`--manifests cra1p5_full`：只保留 `shard_{tag}_{序号}.h5` 且 tag 在白名单内。不传则纳入全部 `shard_*.h5`。文件名不合约定会警告并跳过。

离线转换与校验：

```bash
python normalize_hdf5_fp16.py --dry_run
python normalize_hdf5_fp16.py --check_complete
python normalize_hdf5_fp16.py --stamp_stats --dst_hdf5_root /public/share/acd7koea4a/hdf5_norm_fp16
```

家目录那份只读、未 stamp，不要对它 `--stamp_stats`。目录护栏与推理路径见 `TRAIN_INFER_NORM.md`。

---

## 3. 模型（`PixelShuffleDownscaleNet`）

默认：`in_ch=8`、`base_ch=256`、InitConv 单层 1×1、CBAM 为 M1（仅 GAP、`reduction=4`、空间核 5）、Stage4 `shuffle_conv` 固定 1×1。历史 `--init_type` / `--no_hr_aux` 已删除。

```
180×360 ──InitConv──► base_ch(256)
  Stage1: Res×2 + CBAM + PixelShuffle(2) + [可选 concat HR_aux → 1×1] ─► 360×720
  Stage2: … ─► 720×1440
  Stage3: … ─► 1440×2880
  Stage4: …（不注 HR_aux）─► 2880×5760
  bilinear size=(1801,3600), align_corners=True
  [hr_aux_mode=all 时 concat 原生 hr_aux] ──Head──► 8ch
```

- `x_lr`：`(B, 8, 180, 360)` bf16，仅 8 个气象变量
- `hr_aux`：`(B, 7, 1801, 3600)` fp32 = `dem_hr_norm`(1) + `latlon_sincos_hr`(4) + `land_sea_mask_hr`(1) + `cos(SZA)_hr`(1)
- `cos(SZA)`：仅依赖纬度与 DOY，clip `[0,1]`；日期来自 HDF5 `data/dates`

| CLI | 作用 |
| --- | --- |
| 默认（无额外参数） | CBAM ✓、HR-aux `all` ✓、checkpoint ✓、BatchNorm — **非正式** |
| `--no_cbam` | 去掉每阶段 CBAM |
| `--hr_aux_mode {all,stage1,none}` | 全注入 / 仅 Stage1 / 不注入 |
| `--norm_type {batch,group}` | GroupNorm 与 BatchNorm checkpoint **不兼容** |
| `--no_checkpoint` | 关 gradient checkpointing（显存升、速度加快） |
| `--compile` | `torch.compile`；未在 BW1000+DDP 验证，默认关 |

---

## 4. 正式配置

已锁定（脚本写死，不要改回代码 CLI 默认）：

```
--no_cbam --hr_aux_mode stage1 --norm_type group --ema_decay 0.999
--warmup_ratio 0.03 --val_interval 2 --val_fraction 0.2
--epochs 100 --early_stop_patience 20 --manifests cra1p5_full
--batch_size 1 --accum_steps 2
```

| 项 | 说明 |
| --- | --- |
| 优化器 | AdamW，`lr=2e-4`，`weight_decay=1e-4` |
| 调度 | 线性 warmup（`warmup_ratio × total_steps`）→ 余弦至 `min_lr_ratio=0.01` |
| 精度 | `autocast(bf16)`，不用 GradScaler。读入/权重/EMA 为 fp32 |
| Checkpoint | 默认 4 个 UpStage + 末尾 interp+Head；验证 `eval()` 时不走 checkpoint |
| DDP | 检测到 `RANK`/`WORLD_SIZE` 才启用；`DistributedSampler`；GroupNorm 时跳过 SyncBN |
| 有效全局 batch | `batch_size × accum_steps × 卡数`。8 卡 × 1 × 2 = 16 |
| `--ema_decay` | 验证临时换 EMA 权重；ckpt 另存 `model_ema`；`model` 仍是在线权重（供 `--resume`） |
| `--resume` | 可恢复优化器、调度器、早停计数；每次验证覆盖 `latest.pt` |

加卡是数据并行：每张卡都装完整模型 + 自己那张 1801×3600 图，**不减单卡显存**。跨节点走以太网时扩展效率会掉，先通单节点 8 卡。

Stage4→Head 的大插值在 ROCm 上曾按 fp32 分配（约 15.8 GiB）。代码默认 `--interp_chunk_channels 32` 分块插值，A800 上也保留，不要删。继续用 `--interp_backend interpolate`。

---

## 5. 损失函数（v2）

在 **z-score 空间** 计算：

```
L_train = L_tail(γ=0.5, z_max=3, 通道权重全 1, 面积加权)
        + 0.1 · PatchExtreme(wind10, FSDS)
        + 0.05 · WindPowerSensitivityProxy(wind10)
        + 0.02 · PhysicalConsistency
        + [默认关] SpatialExtreme / FFT / Grad
```

| 子项 | 作用 | 训练时注意 |
| --- | --- | --- |
| TailWeightedMAE | 全部 8 通道；`w = 1+γ·clamp(\|y\|,0,z_max)` × `cos(lat)` | 调 γ 会影响所有变量 |
| PatchExtreme | 局部网格 max/min/**mean**；不约束极值落点 | 确定性回归下省力解是抬高整块；冒烟看 patch 均值偏差，偏大就降 `λ_pe` |
| WPS | 仅真值 3–12 m/s 的立方敏感区代理；**不是**发电量对齐 | mask 不含额定以上；日均 10m 风速可能过密。看 `wps_mask_ratio`，过高就降 `λ_wps` |
| Phys | **先** `pred·σ+μ` 再 hinge，再除 `σ_k` | **禁止在 z 空间**写 `ReLU(−z)` 或直接比 TMIN/TAS。PRE 还原后仍是 log1p |

保底等价：全部 λ=0、`--loss_gamma 0`、`--var_weights` 全 1、`--no_area_weight` 时，`L_train` 与 `nn.L1Loss()` 数值等价。这不证明其余变量不退步。

**调权**：想加强风/光，优先 `--lambda_patch_extreme` / `--lambda_wps`（只影响指定通道），不要用 `--var_weights` 抢梯度，也不要和第一版绑在同一 run。建议对照：`0.15` / `0.08`，`run_dir=runs/exp_loss_wind_solar_boost`。

FFT/Grad/Pooling/反标准化在 **float32** 上算，可能抬显存；OOM 时先关对应 λ。损失值互相 ±3× **不能**代替梯度贡献可比（Phys 多数像素 hinge=0 但局部梯度陡；WPS 的 `dP/dv` 在爬坡段内可差约 15×）。

CLI 默认（与代码一致）：`--loss_gamma 0.5`、`--loss_z_max 3`、`--var_weights` 全 1、`--lambda_patch_extreme 0.1`、`--lambda_wps 0.05`、`--lambda_phys 0.02`、旧版三项 `0`。完整消融命令见 `train.py` 顶部 A–H。

纯 MAE 对照：

```bash
python train.py --base_ch 256 --no_cbam --hr_aux_mode stage1 --norm_type group \
  --loss_gamma 0.0 --var_weights "" --no_area_weight \
  --lambda_extreme 0.0 --lambda_patch_extreme 0.0 --lambda_wps 0.0 --lambda_phys 0.0 \
  --lambda_freq 0.0 --lambda_grad 0.0 \
  --run_dir runs/exp_loss_pure_mae
```

---

## 6. 验证口径与看结果

**模型选择必须与训练损失解耦。** 组合损失跨 run 不可比，也不能用训练目标给自己打分。

| 看什么 | 看哪个 |
| --- | --- |
| 选模型、早停、论文主表 | `Loss/val`（z-score 纯 MAE，像素等权）→ `best.pt` |
| 该 run 自己的优化目标 | `Loss/val_combined` 及各子项（不可跨 run 比） |
| 风光资源 | `MAE_val_extreme/wind10`、`FSDS`；`Skill_val/*` → `best_resource.pt` |
| 降水真实误差 | `MAE_val_physical/PRE`（`expm1` 后 mm/day）。`MAE_val/PRE` 仍是 log1p |
| 面积加权诊断 | `MAE_val_area_weighted/*`（不参与 `best.pt`） |

`Skill_val` = `1 − MAE_model / MAE_baseline`，baseline 是 LR 双线性插值，>0 优于该 baseline。`best_resource.pt` 按 wind10/FSDS skill 等权平均最大保存，与 `best.pt` 互不覆盖。

TensorBoard：`--run_dir` 下 `tb/`。多个 run 可 `tensorboard --logdir runs`。

---

## 7. 平台与 DDP 细节

- 进程组：有 GPU/DCU 用 `nccl`（DTK/RCCL 提供兼容 API），否则 `gloo`。
- 梯度累积中间 micro-batch 用 `model.no_sync()`，只在 `optimizer.step()` 前 all-reduce。
- 验证指标跨进程 `all_reduce` 后再平均；日志与 ckpt 仅 rank0 落盘。ckpt 的 `model` 已 unwrap DDP，可在不同卡数间 `--resume`。
- `--eval_only --resume <ckpt>` 也可用多卡加速补算验证。
- 路径：数据 `/public/share/acd7koea4a`；代码与产物 `/public/home/acd7koea4a/work`。容器未挂 share 时，`paths.py` 回退 `~/local_data`。家目录 `~/static`、`~/states` 是指向 share 的符号链接，未挂则不可用。
- 单节点可忽略：`NCCL WARN`（无 IB）、`Could not open /var/log/hylog/`、`No device id is provided via init_process_group`。

---

## 8. 调试与消融

**hdf5_mini**（未 z-score，在线标准化；971 样本，几分钟 1 epoch）：

```bash
python train.py --hdf5_root /public/share/acd7koea4a/hdf5_mini \
    --epochs 3 --val_interval 1 --num_workers 2 --seasons DJF \
    --run_dir runs/debug
```

权威消融命令以 `train.py` 顶部 docstring 为准。架构消融 A–D；损失消融 E–H。多个开关可叠加。

无静态场 / 无 HR-aux 的 PixelShuffle 对照在 `pixelshuffle_baseline/`（`train_pixelshuffle_baseline.py`）。非正式入口。

---

## 9. CESM 推理

测试集 HDF5 尚未就绪，本轮不必改 `infer.py` 默认路径。正确做法：先把 CESM 转成未标准化 HDF5，再走 `infer.py` 的 HDF5 分支。正式权重加 `--auto_model_cfg --use_ema`。

```bash
python prepare_hdf5_cesm.py --input_format nc \
  --cesm_root /public/share/acd7koea4a/cesm \
  --out_hdf5_root /public/share/acd7koea4a/hdf5_cesm \
  --date_start 20000101 --date_end 20000101 --overwrite

python infer.py --input_source hdf5 \
  --hdf5_root /public/share/acd7koea4a/hdf5_cesm \
  --ckpt runs/exp_prod_ddp_platform/checkpoints/best.pt \
  --auto_model_cfg --use_ema \
  --output_format nc --lon_convention neg180_180 \
  --output_space physical --amp_bf16
```

写 NetCDF 必须显式 `--lon_convention`。检查清单见 `TRAIN_INFER_NORM.md`。`input_source=cesm_nc` 仅诊断用。

---

## 10. 后续优化 / A800

第一版不要绑在一起做：`--compile`、按年切验证集、wind10 非线性预处理（需重做 HDF5）、分季节归一化、8 卡 `batch_size=2`。

A800（80GB）相对 BW1000（64GB）：

- 历史单卡 `base_ch=256 + checkpoint`、`batch_size=1` 无 OOM；当时是 `hr_aux=none` + 纯 MAE，不能直接证明现行 `stage1` + v2 损失也稳，但显存边际更大。
- 单卡约 5 小时/epoch；卡数决定能不能在合理时间跑完。
- 环境变量：DCU 的 `PYTORCH_HIP_ALLOC_CONF` 换成 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`；CUDA 上可再试 `expandable_segments:True`。
- 值得单独冒烟后再考虑：`--compile`、在 `[mem]` 余量健康时试 `batch_size=2`。`validate()` 不走 checkpoint，DCU 上从未跑到验证，A800 上必须测一次验证显存峰值。
- A800 若不是同一套控制台，不要假设也要 `unset RANK`；先小规模确认平台是否注入分布式变量。
- 分块插值修复保留；不要为了切回 BatchNorm 而丢掉已有 GroupNorm 权重（checkpoint 不兼容）。

---

## 11. 路径

| 用途 | 路径 |
| --- | --- |
| 训练 / 数据 / 模型 | `train.py` / `dataset.py` / `model.py` / `paths.py` |
| 正式长跑 | `scripts/launch_platform_train.sh` |
| 冒烟 / 1 epoch 探测 / 8 卡短跑 | `scripts/launch_platform_smoke_norm_*.sh`、`train_copy.sh`、`short_8gpu.sh` |
| 训练 HDF5 | `paths.HDF5_ROOT`（fp16 预标准化） |
| fp32 备份 | `/public/share/acd7koea4a/hdf5` |
| 归一化统计 | `/public/share/acd7koea4a/states/global_stats_state.json` |
| 冒烟数据 | `work/smoke_test_data`、`work/smoke_norm_fp16_data`（后者为脚本临时软链） |
| 标准化 / dtype 专项 | `TRAIN_INFER_NORM.md`、`HR_AUX_DTYPE.md` |
| `slurm/` | 归档，不用 |
