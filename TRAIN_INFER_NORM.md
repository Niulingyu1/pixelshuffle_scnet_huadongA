# 训练 / 推理标准化契约（禁止二次标准化）

日期：2026-09-18  
目的：训练可以读预标准化 fp16；**推理必须继续读「训练域物理量 + 在线 z-score」**。两边必须共用同一份统计量、同一套变换顺序，否则会出现静默错结果（重复 z-score / 重复 log1p），很难从 loss 曲线看出来。

训练命令与平台注意事项见 [`DOWNSCALE_README.md`](DOWNSCALE_README.md) 第 1 节。本文只约束目录分工与变换顺序，不要把推理指到训练用的 `hdf5_norm_fp16`。

相关代码：`dataset.py`（`_load_norm_stats`、`stats_content_sha256`、`is_pre_normalized_shard`）、`normalize_hdf5_fp16.py`、`infer.py`（`_prep_norm_tensors`、HDF5 护栏）、`train.py`（`get_norm_stats` 反标准化）、`prepare_hdf5_cesm.py`。

---

## 1. 必须遵守的目录分工

| 目录 | 磁盘内容 | 谁读 | z-score |
| --- | --- | --- | --- |
| `hdf5/`（`HDF5_ROOT_RAW`） | fp32，PRE 已 log1p、Q 已 g/kg，**未** z-score | 备份；`infer.py` 可以（一般不用） | 在线 |
| `hdf5_norm_fp16/`（`HDF5_ROOT_NORM`） | fp16，**已经** z-score，`metadata.normalized=True` | **只给训练 Dataset** | 跳过 |
| `hdf5_mini/` | 调试用 fp32，未 z-score | 训练调试 | 在线 |
| `hdf5_cesm*` | CESM：PRE 已 log1p、Q 已 g/kg，**未** z-score | **`infer.py` 主路径** | 在线 |
| CESM NetCDF | 原始降水 mm/day 等 | `infer.py --input_source cesm_nc`（诊断用） | 先 log1p，再 z-score |

**禁止：** `infer.py --hdf5_root .../hdf5_norm_fp16`。  
代码已硬拒绝：任意 shard `metadata.normalized=True` 则 `SystemExit`，避免静默二次 z-score。

切换 `paths.HDF5_ROOT` 只影响 `train.py` 默认训练数据，**不会**改 `infer.py` 的 `--hdf5_root`。推理命令必须继续指向 `hdf5_cesm*` 或未标准化的测试 HDF5。

---

## 2. 变换顺序（两边必须一致）

```
原始降水 mm/day
  → [仅 CESM NetCDF / 写训练 HDF5 时] clip≥0 后 log1p
原始 Q kg/kg
  → [写 HDF5 时] ×1000 → g/kg
HDF5 存储单位（训练域）
  → z-score：(x - mean) / std ，mean/std 来自 global_stats_state.json 的 var_all
  → 模型输入 / 模型输出都在 z 空间
  → 反标准化：x * std + mean（PRE 此时仍是 log1p）
  → 推理 physical 输出：PRE 再 expm1；Q 可选 /1000 回到 kg/kg
```

**不要做的：**

- 对已经写入 HDF5 的训练域 `data/x` 再做一次 `log1p`（会把降水分布打烂）
- 对 `hdf5_norm_fp16` 再做一次 `(x-mean)/std`（二次标准化）
- 推理用另一份 stats 文件，或自己写 `sqrt(M2/(n-1))`

`var_all` 的标准差是 **`sqrt(M2/n)`**（总体标准差），不是样本标准差 n-1。

---

## 3. 当前代码如何保证

| 环节 | 实现 |
| --- | --- |
| 同一份 mean/std | 全部走 `dataset._load_norm_stats()` → `paths.STATS_FILE`（优先 `states/global_stats_state.json`） |
| 同一套 z-score（训练 + 离线转换） | 公式都是 `(x-mean)/std`。Dataset `__getitem__` 对未标准化 shard 内联计算；`normalize_hdf5_fp16.py` 用 `_zscore_to_fp16`。mean/std 与 `dataset._load_norm_stats` 同口径（`var_all` → `sqrt(M2/n)`） |
| 训练读预标准化数据 | `DownscaleDataset` 见 `metadata.normalized` 则跳过 z-score，只转 bf16 |
| 预标准化与 stats 强绑定 | shard `metadata.stats_sha256` = 8 变量 mean/std 的内容指纹（不是路径、不是 mtime）。Dataset 在 `pre_normalized=True` 时启动即比对；缺标记、shard 之间不一致、或和当前 `STATS_FILE` 对不上都会直接报错。已转换数据补标记：`python normalize_hdf5_fp16.py --stamp_stats --dst_hdf5_root /public/share/acd7koea4a/hdf5_norm_fp16`。家目录那份只读、未 stamp，不要对它 `--stamp_stats`；`paths.HDF5_ROOT` 优先 share 已 stamp 副本 |
| 训练反标准化（Phys/WPS/验证 MAE） | `dataset.get_norm_stats()`，与上面同一组数组 |
| 推理 z-score | `_prep_norm_tensors()` 调用 `ds._load_norm_stats()`，再 `(x_t - mean) / std`（torch 手写，公式相同） |
| 推理防二次 z-score | `is_pre_normalized_shard` 为 True 则退出 |
| 推理防二次 log1p | HDF5 分支假定 PRE 已是 log1p；只有 `cesm_nc` / `prepare_hdf5_cesm.py` 对原始降水做 log1p |

---

## 4. 推理命令检查清单

提交 `infer.py` 前确认：

1. `--hdf5_root` 是 `hdf5_cesm*`、原始 `hdf5/` 或测试集，**不是** `hdf5_norm_fp16`。
2. `--stats` 未改的话，与训练同一份 `STATS_FILE`（不要另拷一份过期 json）。
3. CESM 用 HDF5 时：`prepare_hdf5_cesm.py` 已经做完 log1p / Q×1000；`infer.py` 不要再加 `--pre_transform` 之类的二次 log1p（HDF5 路径不会对 PRE 再 log1p）。
4. 权重与架构：`--auto_model_cfg --use_ema`（正式训练是 `--no_cbam --hr_aux_mode stage1 --norm_type group`）。
5. 若误指预标准化目录，应立刻看到报错  
   `该目录是预标准化数据...会静默二次标准化`  
   而不是跑出看起来正常的 NetCDF。

正确示例（与 `DOWNSCALE_README.md` 第 9 节一致）：

```bash
python infer.py \
  --input_source hdf5 \
  --hdf5_root /public/share/acd7koea4a/hdf5_cesm \
  --ckpt runs/exp_prod_ddp_1node/checkpoints/best.pt \
  --auto_model_cfg --use_ema \
  --output_space physical
```

错误示例（必须被护栏拦住）：

```bash
python infer.py --input_source hdf5 \
  --hdf5_root /public/home/acd7koea4a/hdf5_norm_fp16   # 禁止
```

---

## 5. 仍未收成单一函数的部分（以后改脚本时）

推理的 `(x_t - mean) / std` 仍是 torch 手写；离线转换是 numpy `_zscore_to_fp16`。数值与 Dataset 在线路径相同，但不是同一个函数。`train.py` 的反标准化与 `infer._to_output_space`（含 `expm1`）也未抽成共用函数。

下一轮若要再收紧：给 numpy/torch 各提供 `zscore` / `zscore_inv`，`infer.py` 与 `train.py` 都改成调用。在此之前，**目录护栏 + 共用 `_load_norm_stats` + `stats_sha256` 运行时校验是防止训练/推理用错 stats 的硬约束，不要删。**
