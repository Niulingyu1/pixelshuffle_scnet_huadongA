# 全量训练指导（基于 2026-09-12 冒烟测试）

日期：2026-09-12（损失方案于 2026-09-17 更新为 v2，见第 4 节）  
范围：SCNet「模型训练」控制台 + BW1000（海光 DCU / DTK）；A800 入口见 `scripts/launch_platform_*.sh`  
依据：单卡冒烟 `runs/smoke_cra1p5_full_0038`、2 卡 DDP 冒烟 `runs/smoke_ddp_cra1p5_full_0038`

冒烟只验证管线（前向 / 反向 / 存盘 / DDP 聚合），**16 个样本上的 MAE 没有业务意义**。

---

## 1. 测试结论（已经确认）

| 项目 | 结果 |
| --- | --- |
| 卡与接口 | BW1000（海光 DCU / DTK）。代码继续写 `torch.cuda`，日志里出现 `cuda:0` 是正常的 |
| 正式架构 | `base_ch=256`、`--no_cbam`、`--hr_aux_mode stage1`、`--norm_type group`、`--ema_decay 0.999` |
| 精度 | 读入 / 权重 / EMA 为 fp32；前向反向用 **bf16 autocast**。不要改成全 fp32 或 fp16 |
| 单卡冒烟 | 必须先 `unset RANK WORLD_SIZE ...`，再直接 `python -u train.py`。3 epoch 约 2.4 分钟，无 OOM，`best.pt` / `best_resource.pt` 正常落盘 |
| 2 卡 DDP 冒烟 | **不要 unset**，用 `torchrun`。`world_size=2`，3 epoch 约 1.1–1.5 分钟，无 OOM，验证指标跨卡聚合正常 |
| 吞吐 | 2 卡大约比单卡快一倍。加卡加快「一天能看多少张图」，**不会把单张图变小** |
| `--compile` | **未测**，全量第一版不要开 |

### 1.1 单卡成功日志要点

- 设备：`cuda:0`，`distributed=False, world_size=1`
- 数据：16 样本，`train=15, val=1`
- 全局 batch：`1 × 2 × 1 = 2`
- epoch 耗时：52.6s / 46.5s / 46.5s
- `Loss/val`：0.8194 → 0.8148 → 0.8101
- 产物：`best.pt`、`best_resource.pt`（含 `model_ema`）

### 1.2 2 卡成功日志要点

- 设备：`cuda:0`，`distributed=True, world_size=2`
- 数据：16 样本，`train=12, val=4`（`val_fraction=0.25`）
- 全局 batch：`1 × 1 × 2 = 2`
- epoch 耗时：26.4s / 20.2s / 20.1s
- `Loss/val`：0.8364 → 0.8333 → 0.8298
- 产物：仅 rank0 落盘 `best.pt`、`best_resource.pt`

---

## 2. 必须避开的坑

### 2.1 平台环境

- 「模型训练」选 **DTK 版 PyTorch 镜像**，不要选 `cuda12.x`。
- **不要** `source env/activate.sh`。家目录 conda（`pytorch_downscale`）是 NVIDIA 版，在 BW1000 上会落到 CPU。
- 不要在登录节点 / Notebook 上跑全图训练。
- 日志出现 `Using device: cpu` 或 `CUDA is not available`：几乎一定是 Python 用错了，不是代码该改成 HIP API。

### 2.2 单卡和多卡启动方式相反

| 场景 | 做法 |
| --- | --- |
| 单卡 | 必须 `unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT`，然后 `python -u train.py` |
| 多卡 | 必须用 `torchrun`，并**保留**平台注入的 `WORLD_SIZE` / `RANK` / `MASTER_*` |

平台注入的 `WORLD_SIZE` / `RANK` 表示**实例数**，不是卡数。`nproc_per_node` 才是每实例卡数。

`train.py` 只要看到 `RANK` 和 `WORLD_SIZE` 就会初始化 DDP。单卡不 unset，就会变成「1 卡假 DDP」。

### 2.3 之前的 OOM 不是「卡越多越炸」

失败日志特征：

```text
distributed=True, world_size=1
torch.OutOfMemoryError: HIP out of memory. Tried to allocate 15.82 GiB
```

原因：1 卡套了 DDP 空壳（梯度桶 / 通信缓冲），末尾把 `2880×5760×256` 插到 `1801×3600` 时还要临时申请约 **15.8 GiB fp32**。64 GB 卡上碎片化后就挂。

真正的 2 卡（每进程一张卡）已经跑通。

`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` **本平台不支持**，日志会写 `expandable_segments not supported on this platform`，不要再靠它救命。

### 2.4 加卡的边界

- 当前是 **DDP 数据并行**：每张卡都装完整模型 + 自己那张 `1801×3600` 图。加卡不减单卡显存。
- 不是卡越多越好：单节点 4～8 卡收益最大；跨节点还要走以太网（日志里 `No IB NIC`），扩展效率会掉。
- 官方 slurm 写的是 8 卡 + `batch_size=2`，**冒烟只验证过 2 卡 + `batch_size=1`**。全量不要第一步就按 8×2 上。

### 2.5 可忽略的警告

单节点可忽略：

- `NCCL WARN`：无 IB、缺 `iommu=pt`、缺 `HSA_FORCE_FINE_GRAIN_PCIE`
- `Could not open /var/log/hylog/`
- `No device id is provided via init_process_group or barrier`

---

## 3. 全量开跑前先核对数据

2026-09-12 当晚共享盘状态（**当时还没齐**）：

| 季节 | shard 数量 | 备注 |
| --- | --- | --- |
| MAM | 39 | 仍有大量 `*.raysync.uploading` |
| SON | 19 | |
| JJA | 0 | |
| DJF | 0 | |

开 100 epoch 之前必须满足：

1. 四季目录齐全：`MAM / JJA / SON / DJF`
2. 没有 `*.raysync.uploading`
3. 每个 `shard_*.h5` 能打开，且含 `data/x`、`data/y`、`data/dates`
4. 加上 `--manifests cra1p5_full`，只吃白名单 tag，避免半成品 / 其它 tag 混入
5. 先用全量路径做一次 **2～4 卡、1～2 epoch** 的短跑（验证读盘和显存），再拉到 100 epoch

数据根目录：`/public/share/acd7koea4a/hdf5`  
归一化统计：`/public/share/acd7koea4a/states/global_stats_state.json`

---

## 4. 第一版正式配置（不要同时改损失）

锁定：

```text
--no_cbam --hr_aux_mode stage1 --norm_type group --ema_decay 0.999
--warmup_ratio 0.03 --val_interval 2 --val_fraction 0.2
--epochs 100 --early_stop_patience 20 --manifests cra1p5_full
```

损失走 `train.py` **v2 默认**（与 `DOWNSCALE_README.md` 第 4 节一致），不要再手写 v1 的通道偏置 / `--lambda_extreme`：

```text
面积加权 TailWeightedMAE(γ=0.5, 通道权重全 1)
+ PatchExtreme(wind10,FSDS)×0.1
+ WindPowerSensitivityProxy×0.05
+ PhysicalConsistency×0.02
FFT / Grad / 旧版 SpatialExtreme 默认关
```

纯 MAE 对照必须显式关掉全部增量项和面积加权，见 README 第 4.4 节，不要只传 `--loss_gamma 0`。
风光加强优先调 `--lambda_patch_extreme` / `--lambda_wps`，不要和第一版绑在同一 run。

| 资源 | batch / accum | 启动 |
| --- | --- | --- |
| 单卡回退 | `1 / 4` | `unset` 分布式变量 + `python -u` |
| **推荐：1 实例 × 4 卡** | `1 / 2` | `torchrun --nproc_per_node=4` |
| 稳了再试：1 实例 × 8 卡 | 先 `1 / 2`，显存够再 `2 / 2` | `torchrun --nproc_per_node=8` |
| 多实例 / 多节点 | 先不要 | 单节点 8 卡通了再扩 |

有效全局 batch = `batch_size × accum_steps × 卡数`。

- 4 卡 × 1 × 2 = 4，和单卡 `1 × 4` 同量级，学习率先不动（`2e-4`）
- 8 卡若把 `batch_size` 提到 2，全局 batch 变成 32，需要观察 `Loss/val` 是否变抖；必要时再单独调 lr，不要和第一版绑死

---

## 5. 平台「模型训练」怎么填

控制台：

| 项 | 值 |
| --- | --- |
| 加速卡 | BW1000 |
| 每实例卡数 | 4（稳了再试 8） |
| 实例数 | 1 |
| 镜像 | 与冒烟成功相同的 DTK 镜像 |
| 启动命令 | `bash /public/home/acd7koea4a/work/scripts/launch_platform_train.sh`（2 卡时先 `export NPROC_PER_NODE=2`） |

### 5.1 推荐：1 实例 × 4 卡

```bash
cd /public/home/acd7koea4a/work
mkdir -p logs runs

export NCCL_DEBUG=WARN
export NPROC_PER_NODE=4

python - <<'PY'
import os, torch
print("python cuda:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
print("platform WORLD_SIZE(实例数)=", os.environ.get("WORLD_SIZE"), "RANK(实例)=", os.environ.get("RANK"))
print("MASTER_ADDR=", os.environ.get("MASTER_ADDR"), "MASTER_PORT=", os.environ.get("MASTER_PORT"))
if torch.cuda.device_count() < int(os.environ.get("NPROC_PER_NODE", "4")):
    raise SystemExit("可见卡数不足，请把控制台「每实例加速卡数量」改成与 NPROC_PER_NODE 一致")
for i in range(torch.cuda.device_count()):
    print(f"  [{i}]", torch.cuda.get_device_name(i))
PY

LOG=logs/full_ddp_$(date +%Y%m%d_%H%M%S).log

torchrun \
    --nnodes="${WORLD_SIZE:-1}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-23456}" \
    train.py \
    --hdf5_root /public/home/acd7koea4a/hdf5_norm_fp16 \
    --seasons MAM JJA SON DJF \
    --manifests cra1p5_full \
    --val_fraction 0.2 \
    --val_interval 2 \
    --epochs 100 \
    --batch_size 1 \
    --accum_steps 2 \
    --num_workers 4 \
    --no_cbam \
    --hr_aux_mode stage1 \
    --norm_type group \
    --ema_decay 0.999 \
    --warmup_ratio 0.03 \
    --log_interval 20 \
    --early_stop_patience 20 \
    --save_top_k 3 \
    --run_dir runs/exp_prod_ddp_4gpu \
    2>&1 | tee "${LOG}"

echo "日志: ${LOG}"
```

8 卡：把控制台「每实例加速卡数量」和 `NPROC_PER_NODE` 都改成 8，`--run_dir` 改成 `runs/exp_prod_ddp_8gpu`。仍先用 `--batch_size 1`。

### 5.2 单卡回退

```bash
cd /public/home/acd7koea4a/work
mkdir -p logs runs

unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
export NCCL_DEBUG=WARN

python - <<'PY'
import os, sys, torch
print("python:", sys.executable)
print("RANK" in os.environ, "WORLD_SIZE" in os.environ)
print("cuda:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
    raise SystemExit("分布式环境变量仍在，先 unset 再跑")
if not torch.cuda.is_available():
    raise SystemExit("未检测到加速卡")
PY

LOG=logs/full_single_$(date +%Y%m%d_%H%M%S).log

python -u train.py \
    --hdf5_root /public/home/acd7koea4a/hdf5_norm_fp16 \
    --seasons MAM JJA SON DJF \
    --manifests cra1p5_full \
    --val_fraction 0.2 \
    --val_interval 2 \
    --epochs 100 \
    --batch_size 1 \
    --accum_steps 4 \
    --num_workers 4 \
    --no_cbam \
    --hr_aux_mode stage1 \
    --norm_type group \
    --ema_decay 0.999 \
    --warmup_ratio 0.03 \
    --log_interval 20 \
    --early_stop_patience 20 \
    --save_top_k 3 \
    --run_dir runs/exp_prod_single \
    2>&1 | tee "${LOG}"

echo "日志: ${LOG}"
```

### 5.3 启动后立刻核对

```text
Using device: cuda:0  (distributed=True, world_size=4)   # 单卡应是 False / 1
Model: ... use_cbam=False  hr_aux_mode=stage1  norm_type=group ...
[EMA] 已启用，decay=0.999
[warmup] --warmup_ratio=0.03 × total_steps=N → warmup_steps=...
```

查看进度：

```bash
tail -f /public/home/acd7koea4a/work/logs/full_ddp_*.log
```

---

## 6. 看结果时用哪把尺子

| 看什么 | 看哪个 |
| --- | --- |
| 选模型、早停、论文主表 | `Loss/val` → `best.pt` |
| 风光资源 | `Skill_val/wind10`、`Skill_val/FSDS`、`MAE_val_extreme/wind10`、`MAE_val_extreme/FSDS` → `best_resource.pt` |
| 降水真实误差 | `MAE_val_physical/PRE`（mm/day），不要只看 `MAE_val/PRE`（那是 log1p） |
| 训练诊断 | `Loss/val_combined` 及各子项（不可跨 run 比大小） |

推理：

```bash
python infer.py --ckpt runs/exp_prod_ddp_4gpu/checkpoints/best.pt \
    --auto_model_cfg --use_ema \
    --hdf5_root /public/home/acd7koea4a/hdf5_norm_fp16 --seasons DJF \
    --out_dir infer_out_prod --output_mode per_sample --output_format nc \
    --lon_convention neg180_180 --output_space physical --amp_bf16
```

`--auto_model_cfg` 会从 checkpoint 识别 `use_cbam=False`、`hr_aux_mode=stage1`、`norm_type=group`。开了 EMA 时加 `--use_ema`，用 `model_ema` 而不是在线权重。

---

## 7. 第一版不要一起改的东西

- 不要开 `--compile`（未在 BW1000 + DDP 上验证）
- 不要叠 FFT / Grad，不要改回 v1 的 `--lambda_extreme` 或通道偏置 `var_weights`
- 不要一上来就把 `batch_size` 提到 2（4 卡短跑显存够再提）
- 不要跨节点、多实例，除非单节点 8 卡已经稳定
- 数据没齐不要指向正在上传的整个 `hdf5/`
- 不要 `source env/activate.sh`
- 单卡不要用 `torchrun`；多卡不要 `unset RANK`

---

## 8. 建议执行顺序

1. 等四季 HDF5 传完，确认无 `*.raysync.uploading`
2. 全量路径、2～4 卡、1～2 epoch 短跑（只验证读盘 / 显存 / 落盘）
3. 同一套命令拉到 100 epoch（第一版，v2 默认损失）
4. 第一版有数后，单独开 `--lambda_patch_extreme` / `--lambda_wps` 对照
5. 再考虑：8 卡、`batch_size=2`、`--compile`、按年切验证集

---

## 9. 相关路径

| 用途 | 路径 |
| --- | --- |
| 训练入口 | `/public/home/acd7koea4a/work/train.py` |
| 平台正式启动 | `scripts/launch_platform_train.sh` |
| 平台冒烟 | `scripts/launch_platform_smoke_single.sh`、`launch_platform_smoke_ddp.sh` |
| 平台 8 卡短跑 | `scripts/launch_platform_short_8gpu.sh` |
| 历史纯 MAE 对齐 | `scripts/launch_platform_train_bw_a800_aligned_benchmark.sh` |
| 本指南 | `/public/home/acd7koea4a/work/FULL_RUN_GUIDE.md` |
| 更完整的分析与消融 | `/public/home/acd7koea4a/work/TRAINING_ANALYSIS.md` |
| 损失定案 | `/public/home/acd7koea4a/work/loss_plan/final_decision.md` |
| 冒烟数据 | `/public/home/acd7koea4a/work/smoke_test_data` |
| 正式 HDF5 | `/public/home/acd7koea4a/hdf5_norm_fp16`（fp32 备份：`/public/share/acd7koea4a/hdf5`） |
| 单卡冒烟日志 | `logs/smoke_cra1p5_full_0038_20260912_193344.log` |
| 2 卡冒烟日志 | `logs/smoke_ddp_cra1p5_full_0038_20260912_194201.log` |
