# `hr_aux` 与 bf16 特征拼接：dtype 约定（待完善）

日期：2026-09-18  
状态：**本次不改代码。** `x_lr`/`y_hr` 已按预标准化 fp16 → Dataset 出口 bf16 改造；`hr_aux` 继续 fp32。本文只记录现状、风险和下一步改法，避免以后改脚本时重踩坑。

相关代码：

- `dataset.py`：`_load_hr_static()`、`_cos_sza_hr()`、`__getitem__` 里 `np.concatenate`
- `model.py`：`UpStage.forward` 的 `torch.cat`、`_run_head` / `forward` 末尾 Head 前的 `torch.cat`
- `train.py`：`hr_aux.to(device)` + `torch.autocast(..., dtype=torch.bfloat16)`

---

## 1. 两条独立数据通路

| 张量 | 磁盘 / 缓存 | Dataset 出口 | 训练时 |
| --- | --- | --- | --- |
| `x_lr` / `y_hr` | HDF5：原始 fp32 未 z-score，或预标准化 **float16** | **bfloat16** | 随 autocast 走 bf16 |
| `hr_aux` | 内存：6 通道 HR 静态 + 1 通道在线 `cos(SZA)_hr`，**不写入 HDF5** | **float32** | 原样上 GPU，仍是 fp32 |

`hr_aux` 不是气象场 z-score，而是 DEM / sin·cos(lat,lon) / 海陆掩膜 / 日平均 `cos(SZA)`。它和 `x/y` 的存储精度改造解耦。本次明确：**不把 `hr_aux` 转 fp16，也不转 bf16。**

---

## 2. Dataset 内 numpy 拼接（已经安全）

```python
# dataset.py __getitem__
sza_hr = _cos_sza_hr(date_str)                          # (1, 1801, 3600) float32
hr_aux = np.concatenate([self.hr_static, sza_hr], axis=0)  # (7, 1801, 3600) float32
return x_lr, torch.from_numpy(hr_aux.copy()), y
```

加载时两边都已 `.astype(np.float32)`：

- `_load_hr_static()`：`dem_hr_norm` / `latlon_sincos_hr` / `land_sea_mask_hr`
- `_cos_sza_hr()`：纬度用 float64 算日平均，cache 前再转 float32

`np.concatenate` 两个 fp32 → 仍是 fp32。注意：

- 纬度网格 `_LAT_HR` 是 float64。若 `sza_hr` 忘了 cast，concatenate 会升成 float64，每个样本多拷约一倍内存。
- 不要在这里改成 numpy `float16`。Dataset 出口一旦变成 fp16，会把问题推到下一节的 `torch.cat`。

约定：**concat 前两边明确 fp32，返回 `torch.float32`。**

---

## 3. 真正危险的拼接在模型里

训练循环不改 dtype，只搬设备：

```python
x_lr   = x_lr.to(device, non_blocking=True)    # bfloat16
hr_aux = hr_aux.to(device, non_blocking=True)  # float32
y_hr   = y_hr.to(device, non_blocking=True)    # bfloat16
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    pred = model(x_lr, hr_aux)
```

`init_conv(x_lr)` 之后特征 `f`/`x` 为 bf16。随后两处 `torch.cat` **没有** `.to(x.dtype)`：

```python
# model.py UpStage.forward（PixelShuffle 之后）
x = self.inject_conv(torch.cat([x, hr_aux_interp], dim=1))

# model.py Head 前（仅 hr_aux_mode='all'）
f = torch.cat([f, hr_aux_for_cat], dim=1)
```

Stage 注入用的 `hr_aux_interp` 来自 `F.interpolate(hr_aux, ...)`，保持 fp32。`'all'` 时 Head 拼的是原生 1801×3600 的 `hr_aux`，不再插值。

正式配置 `--hr_aux_mode stage1`：只在 Stage 1（360×720）拼一次；Stage 2/3 与 Head 不拼。`'all'` 会在 Stage 1–3 和 Head 各拼一次。

---

## 4. 实测：哪种组合能跑

当前 conda、PyTorch 2.5.1，CPU autocast（落地前仍建议在目标 GPU/ROCm 上复核）：

| 操作 | 结果 | 含义 |
| --- | --- | --- |
| `bf16 - fp16` | 升到 `float32` | 静默变准，偏离“损失走 bf16” |
| `torch.cat([bf16, fp32])`（含 autocast） | `float32`，不报错 | **现状** |
| `torch.cat([bf16, fp16])`（autocast 内） | `RuntimeError` | **禁止把 hr_aux 改成 fp16** |
| `torch.cat([bf16, bf16])` | `bfloat16` | 下一轮目标 |

现状路径：

1. `cat(bf16 特征, fp32 hr_aux)` → 升到 fp32（隐式，代码未写明）
2. 后面的 `inject_conv` / `Head` 是卷积，autocast 再把输入转回 bf16

能跑，但 dtype 契约靠 PyTorch 提升规则，不靠代码断言。

---

## 5. 现状是不是“最佳”

在本轮约束（`hr_aux` 不动、不强制全程 bf16）下：**保持 fp32 `hr_aux` + 隐式 `cat(bf16, fp32)` 是正确选择**，不要为它对模型做半吊子修改。

它不是精度契约上最干净的方案：

- 拼接瞬间特征图也抬到 fp32，再被卷积拉回 bf16。
- `'all'` 时 Head 上 1801×3600 的 `cat` 会把大特征图抬到 fp32，显存更紧。
- 以后若有人把 Dataset 里的 `hr_aux` 改成 fp16，这里会直接崩，而不是给出明确的类型错误。

不要做的：

- `hr_aux` 存成或返回 **fp16**
- 指望 autocast 自动处理 `bf16+fp16`

---

## 6. 下一轮建议改法（脚本待完善）

目标：进 `cat` 的两边同为 **bf16**，不再依赖隐式提升。几何辅助量用 7 位尾数足够。

优先顺序：

1. **最小改动（推荐先做）**  
   只在 `model.py` 两处 `torch.cat` 前把辅助张量转到特征 dtype：

   ```python
   aux = hr_aux_interp.to(dtype=x.dtype)
   x = self.inject_conv(torch.cat([x, aux], dim=1))
   ```

   Head 同理：`hr_aux_for_cat.to(dtype=f.dtype)`。  
   Dataset 仍可返回 fp32；H2D 仍按 fp32 传 `hr_aux`。契约写在模型入口，fp16/fp32 都不会崩。

2. **更彻底**  
   `dataset.py` 出口 `hr_aux` 也 `.to(torch.bfloat16)`。H2D 减半。仍须保留（1），防止以后又传入别的 dtype。  
   numpy 缓存与 `np.concatenate` **继续 fp32**，只在 `torch.from_numpy` 之后转 bf16。不要在 numpy 层用 float16。

3. **不要**  
   Dataset 或模型把 `hr_aux` 改成 `float16`。

落地时建议同时加断言（训练冒烟即可）：

- `__getitem__`：`x_lr`/`y_hr` 为 `bfloat16`；本轮 `hr_aux` 为 `float32`；下一轮若改出口则为 `bfloat16`
- `UpStage` / Head：`cat` 两边 dtype 相同，且不是 `(bfloat16, float16)`

GPU/ROCm 上复核第 4 节那组组合（CPU autocast 与 CUDA/ROCm 路径不完全相同）。

---

## 7. 涉及文件清单（下一轮）

| 文件 | 可能改动 |
| --- | --- |
| `model.py` | `cat` 前 `.to(dtype=x.dtype)`（最小充分条件） |
| `dataset.py` | 可选：返回前 `hr_aux.to(bfloat16)`；numpy concat 保持 fp32 |
| `train.py` | 一般不用改；冒烟时打印/断言 dtype |
| `scripts/smoke_norm_fp16_step.py` | 增加 `hr_aux` 与 `cat` 后 dtype 检查 |
| `DOWNSCALE_README.md` | 改完后同步出口 dtype |

本次预标准化 fp16 改造**不包含**上表改动。`paths.HDF5_ROOT` 切换与否也与本节无关。
