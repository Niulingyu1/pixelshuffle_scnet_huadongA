# 迁移到 A800(80GB)说明

日期：2026-09-14
背景：在 SCNet BW1000（海光 DCU / DTK，64GB/卡）上完成了插值 OOM 根因修复，但 8 卡短跑验证显示
`batch_size=1` 时显存余量只有 ~1.2 GiB（`max_reserved=62.6~62.87 GiB` / `device_total=63.98 GiB`），
最终在第 20 个 optimizer step 后 `SIGSEGV` 崩溃。判断这套模型规模（`base_ch=256` +
`num_resblocks=2` + gradient checkpointing）在 64GB 卡上安全边际不足，改迁移到 80GB 的 A800。

本文档目的：完整记录 DCU 平台上踩过的坑与已完成的修复，区分「平台无关、继续有效」和
「DCU/ROCm 特有、A800 上需要重新判断」的部分，避免迁移后重复踩坑或误用过时假设。

---

## 0. 重要：仓库里已有一份真实的 A800 单卡历史基准日志，务必先看

`train_exp01_no_all_100.log`（配合 `scripts/launch_platform_train_bw_a800_aligned_benchmark.sh`
里记录的对齐基准）是**改分布式代码之前**在 A800 单卡上跑出来的真实结果：

```
Model: base_ch=256  in_ch=8  init=1x1  use_cbam=False  hr_aux_mode=none  resblocks=2
       checkpoint=True  s4_shuf=1x1  params=17,380,360
Loss: TailWeightedMAE(γ=0.0) + FFT×0.0 + Grad×0.0   ← 纯 MAE，无通道权重/无 SpatialExtreme
batch_size=1  accum_steps=4  单卡非分布式
Epoch 001~080 done in ~17960~17997s（约 5.0 小时/epoch），全程无 OOM、无崩溃
80 epoch 后 early stop（best_val_loss=0.040031，FSDS 误差 7.19，远高于其他变量——
      正是因为这次跑的是纯 MAE、无通道权重，没有针对 FSDS/wind10 加权）
```

**这份日志证明的事实**：`base_ch=256 + checkpoint=True`（跟本次准备正式训练的规模一致）
在单张 A800（80GB）上**跑 `batch_size=1` 完全没有显存问题**，与 DCU（64GB）上"batch_size=1
都只剩 1.2GB 余量"的紧张状况明显不同——这符合"80GB 比 64GB 多 25%"的预期，而且这次跑的配置
（`hr_aux_mode=none`，无 CBAM，纯 MAE）比本次准备用的
`hr_aux_mode=stage1` + v2 CombinedLoss（面积加权 TailMAE + PatchExtreme + WPS + Phys）更省显存，所以**不能直接
当作"新配置在 A800 上也稳"的证明，但可以当作"这个数量级的显存需求在 A800 上有很大安全边际"
的强有力参考**。

**这份日志同时暴露的问题**：单卡 A800 上 `base_ch=256` **一个 epoch 要 5 小时**，如果新的
A800 环境也是单卡或少卡，100 epoch 会是几百小时的量级（单卡外推约 500 小时/20.8 天），
比 DCU 8 卡估算的 129 小时还长很多。**迁移到 A800 后，卡数直接决定能不能在合理时间内跑完**，
这是第一个要确认的事——不要假设"A800 显存更大=问题都解决了"，显存和算力/卡数是两件独立的事。

---

## 1. 已完成的代码修复（平台无关，继续保留）

这些是逻辑 bug 修复和功能增强，与硬件/后端无关，迁移到 A800 后不需要改动，直接沿用：

| 问题 | 文件 | 修复内容 |
| --- | --- | --- |
| `_build_index()` 的 `manifests` 参数从未生效，导致按季节目录无差别加载全部 shard | `dataset.py` | 补上按 `shard_{tag}_{idx}.h5` 命名过滤的实际逻辑 |
| 验证阶段 `MAE_val/PRE` 只反 z-score，未做 `expm1` 物理量纲还原 | `train.py` | 新增物理量纲 MAE 指标（区分 log1p 空间误差 vs 真实 mm/day 误差） |
| README 第 4 节损失函数默认值文档过期；`--no_hr_aux`/`--init_type` 等废弃参数示例 | `DOWNSCALE_README.md` | 同步文档与当前代码 |
| `slurm/train_gpu.slurm` 锁定旧消融基线配置 | `slurm/train_gpu.slurm` | 更新为当前正式配置 |
| BatchNorm 在小 batch/DDP 场景统计噪声大 | `model.py`/`train.py` | 新增 `--norm_type group`（GroupNorm，不依赖 batch 维统计量） |
| 验证/交付稳定性 | `train.py` | 新增 `ModelEMA`（`--ema_decay`），checkpoint 额外保存 `model_ema` |
| warmup_steps 需要手动按 total_steps 回填 | `train.py` | 新增 `--warmup_ratio`，按真实 total_steps 自动换算 |
| 断点续训只能从 top-K best checkpoint 恢复，意外中断可能丢进度 | `train.py` | 新增 `latest.pt`，每次验证都覆盖保存 |
| `torch.compile` 支持 | `train.py` | 新增 `--compile`（**DCU 上未验证，A800 上值得重新尝试**，见第 3 节） |

**正式训练的架构/训练策略决策**（与硬件无关，继续沿用，不需要重新讨论）：
`--no_cbam`、`--hr_aux_mode stage1`、`--norm_type group`、`--ema_decay 0.999`、
`--warmup_ratio 0.03`、DDP 分布式、v2 CombinedLoss 默认权重
（面积加权 TailMAE + `--lambda_patch_extreme 0.1` + `--lambda_wps 0.05` + `--lambda_phys 0.02`；
旧版 SpatialExtreme/FFT/Grad 默认关闭）。

---

## 2. DCU/ROCm 平台特有问题（迁移后需要重新判断，不能想当然搬）

### 2.1 Stage4→Head 插值的 fp32 强制升级（根因已精确定位，修复保留但可能不再必要）

**现象**：`model.py` 里 Stage4 输出（2880×5760×256）插值到目标分辨率（1801×3600）这一步，
在 ROCm/HIP 后端实测按 fp32 分配输出（**不受** `torch.autocast(dtype=bf16)` 影响），
是本该按 bf16 计算的 2 倍（`15.82 GiB` vs 理论 `7.91 GiB`，与
`2880×5760×256×4byte` 精确吻合）。8 卡真实 DDP 下叠加通信开销，首个 batch 直接 OOM。

**根因判断**：大概率是 ROCm 该算子（`upsample_bilinear2d`）缺原生 bf16 kernel，内部回退到 fp32——
这是**后端实现限制**，不是我们代码的 bug。NVIDIA CUDA 后端的 `upsample_bilinear2d` 对 bf16/fp16
支持更成熟，此问题在 A800 上**很可能不存在**。

**已实现的修复**（`model.py` 的 `_interp_chunked` / `_interp_grid_sample_chunked`，
`train.py` 的 `--interp_chunk_channels`、`--interp_backend`；`infer.py` 把
`interp_chunk_channels` 写死为 32，无对应 CLI）：按通道分块插值，
每块算完立刻转回原 dtype 再拼接，数学上与不分块结果**逐元素完全相同**（已在 CPU 上验证
`max_abs_diff=0.0`），纯粹是显存分配策略，不影响模型精度。

**迁移建议**：**保留这个修复作为默认行为**（`interp_chunk_channels` 默认 32，无害），
不需要在 A800 上关闭它——即使 A800 上这个算子表现正常，分块也只是多了几次很小的 kernel
调用，开销可忽略；但不要假设"A800 上不需要它"就把相关代码删掉，万一 A800 上某个 PyTorch
版本仍有类似问题，这是现成的安全网。

### 2.2 `grid_sample` 对比实测：在 DCU 上没有优势，不建议在 A800 上花时间重测

实测 `grid_sample`（`aten::grid_sampler_2d`）反而比 `interpolate` 多占用 ~0.8GB 显存
（60.50 GiB vs 59.72 GiB allocated），推测是反向传播需要额外保存采样网格的角点索引/权重。
**结论：继续用默认的 `--interp_backend interpolate`**，除非在 A800 上遇到新的、必须绕过的
显存问题，否则不需要再对比这条路径。

### 2.3 `batch_size=2` 在 8 卡 DCU 上 backward 阶段 OOM；`batch_size=1` 余量也只有 ~1.2GB

这不是 bug，是 `base_ch=256 + num_resblocks=2 + checkpoint=True` 这套模型规模在 64GB 卡上
確實很紧。**A800 单卡 80GB，比 64GB 多 25%**，理论上：
- `batch_size=1 + accum_steps=4` 应该有明显更健康的余量（不再是 1.2GB 这种危险边际）；
- `batch_size=2` 有机会重新变得可行，但**不要凭理论直接上**，必须重新用 `[mem]` 打点验证
  （见第 4 节 checklist）。

### 2.4 平台环境变量/模块系统差异（DCU 专属，A800 不适用）

| DCU（ROCm/HIP + RCCL） | A800（NVIDIA CUDA + NCCL） |
| --- | --- |
| `PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512` | 改用 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` |
| `expandable_segments:True` **本平台不支持**（实测确认，报 `not supported on this platform`） | NVIDIA CUDA 上通常**支持** `expandable_segments:True`，可以作为额外一层保险重新启用 |
| `module load compiler/dtk/25.04`、`app/rccl/dtk-25.04/...` | 换成对应的 NVIDIA CUDA/NCCL module（需现场查 A800 平台的 module 名称） |
| `NCCL_IB_HCA=mlx5` / `NCCL_SOCKET_IFNAME=eth` 等 RCCL 专属调优变量 | 需要重新确认 A800 平台的实际网络配置（IB/以太网），不要直接照搬 |
| `Reducer: comm-optimized memory allocator not found, using regular one` 警告 | 若 A800 平台的 NCCL/PyTorch 组合支持该分配器，此警告应消失，可作为“环境是否更优”的信号 |

### 2.5 conda 环境：`source env/activate.sh` 的判断要反过来

**DCU 上明确警告过**：`~/.conda/envs/pytorch_downscale` 是 **NVIDIA 版** PyTorch 构建
（`torch==2.5.1+cu121`），在 BW1000（ROCm/DCU）上 source 它会导致 `torch.cuda.is_available()`
落到 CPU，所以 DCU 平台严格要求"不要 source env/activate.sh，用平台自带的 DTK 版镜像"。

**在 A800（NVIDIA）上，这个判断要反过来**：`pytorch_downscale` 这个 conda 环境本来就是给
NVIDIA CUDA 平台构建的，**理论上直接能用**，`source env/activate.sh` 应该是对的做法。
迁移后第一步要验证：
```bash
source env/activate.sh
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```
确认 `torch.cuda.is_available()` 为 `True` 且能看到 A800 的卡数。如果 A800 平台的 CUDA 驱动
版本与 `cu121` 不兼容，才需要考虑换成平台自带镜像/module。

### 2.6 单卡/多卡启动方式的"相反逻辑"是 SCNet「模型训练」控制台特有的机制，不要假设 A800 平台一样

DCU 平台的坑是：控制台会注入 `RANK`/`WORLD_SIZE`（表示**实例数**，不是卡数），单卡跑必须
`unset RANK WORLD_SIZE LOCAL_RANK ...` 再用 `python -u train.py`，多卡必须**保留**这些变量
再用 `torchrun`。**这是该平台的产物注入机制，不是 PyTorch/DDP 的通用规则**。迁移到 A800 后
如果是不同的作业提交系统（无论是新的平台控制台、slurm，还是别的云平台），必须重新确认：
- 平台是否会注入 `RANK`/`WORLD_SIZE`/`MASTER_ADDR` 等分布式变量，注入的语义是什么；
- 单卡回退是否也需要 `unset`，还是该平台单卡本来就不会注入这些变量。

不要直接照抄 `FULL_RUN_GUIDE.md` 第 2.2 节的表格去 A800 上操作，先在小规模冒烟里确认一遍。

---

## 3. 迁移后建议重新验证/评估的点（不是 bug，是"值得利用 A800 更好条件重新看一眼"）

- **`--compile`**：DCU 上因为兼容性未知一直没开。A800 是 NVIDIA + 成熟的 Triton/Inductor 后端，
  `torch.compile` 的稳定性和收益都会好很多，值得在小规模冒烟里单独试一次（先不要直接合入正式命令）。
- **`interp_backend grid_sample`**：DCU 上实测没有优势，但那是 ROCm 后端的结论，NVIDIA 后端
  两者的实现路径不同，如果对显存/吞吐有极端要求可以重新对比一次，但**优先级低**，默认
  `interpolate` 已经验证是安全、正确的路径，没有强烈理由才不需要折腾这个。
- **`batch_size`/`accum_steps`**：不要直接沿用 DCU 上最后跑通的 `batch_size=1, accum_steps=4`，
  A800 显存更宽裕，应该重新用 `[mem]` 打点测一次，如果余量健康（比如 `max_reserved` 距离
  `device_total` 有 >15GB 的余量）可以考虑提到 `batch_size=2`，减少总 optimizer step 数、
  提升吞吐。
- **`norm_type`**：`GroupNorm` 是为了应对"每卡 batch 很小、BatchNorm 统计噪声大"这个问题引入的。
  如果 A800 上显存宽裕到可以用更大的 `batch_size`（比如 4 张卡 × `batch_size=4`），`BatchNorm`
  的统计量会更稳定，理论上可以重新考虑要不要切回 `BatchNorm`——但**这两套 checkpoint 不兼容**，
  如果已经用 GroupNorm 跑出了有意义的结果，不建议为了这个再折腾，除非从头训练。

---

## 4. 迁移 checklist（按顺序做，不要跳步）

1. **代码迁移**：当前代码仓库还没有任何 git commit（`git log` 显示 `No commits yet`），
   迁移前先在当前环境提交一个干净的 baseline commit，用 git 搬到 A800，不要靠复制整个目录
   （避免带过去 `core.*` 崩溃转储、`runs/`/`logs/` 里的历史产物等垃圾文件）。
2. **数据路径确认**：`paths.py` 里 `DATA_ROOT = /public/share/acd7koea4a`，这是 SCNet 的共享
   存储路径。**A800 所在的平台/集群大概率有不同的挂载路径**，需要确认：
   - 数据是否需要重新同步/上传到 A800 可访问的存储；
   - 如果路径不同，是改 `paths.py` 里的 `DATA_ROOT` 常量，还是用环境变量覆盖（视 A800
     平台约定而定）。
3. **环境验证**：`source env/activate.sh` → 确认 `torch.cuda.is_available()`、
   `torch.cuda.device_count()`、`torch.__version__`。
4. **单卡冒烟**：先用小 `base_ch`（比如 4）跑一次单卡端到端，确认管线本身没问题
   （数据加载 / 前向 / 反向 / checkpoint 保存）。
5. **正式规模单卡冒烟**：`base_ch=256`（正式配置），单卡，几个 step，确认能跑（不追求速度，
   只确认不报错）。
6. **多卡 DDP 冒烟**：2 卡起步，确认 DDP 初始化、梯度同步、（如果多卡）checkpoint 只在
   rank0 落盘等逻辑正常。
7. **`[mem]` 显存余量测试**：正式规模（`base_ch=256`）+ 全量数据 + 目标卡数，跑到
   `train.py` 里新增的 `[mem]` 打点那一行，确认 `max_reserved` 距离 `device_total` 有
   健康余量（不是 <2GB 这种危险边际）。这一步决定最终 `batch_size`/`accum_steps`。
8. **短跑验证**：`--epochs 1`（或几百个 optimizer step），确认能完整跑过 `validate()`
   一次（不只是训练几步，验证阶段的全图推理是另一个显存高峰，`model.py` 里
   `ckpt = self.use_checkpoint and self.training` 意味着 `model.eval()` 时**不走
   checkpoint 路径**，这一步在 DCU 上因为训练阶段就已经崩溃而**从未被真正验证过**，
   迁移到 A800 后是第一次有机会测到，务必确认这一步不会比训练阶段更吃显存）。
9. 确认以上都过了，才提交正式的 100 epoch 长跑。

---

## 5. 遗留的、尚未在任何平台验证过的风险点（提醒，不是阻塞项）

- **验证阶段（`validate()`）的显存峰值从未被实测过**——DCU 上所有短跑都在训练阶段就已经
  OOM/崩溃，没有一次真正跑到验证。`model.eval()` 模式下不走 gradient checkpointing
  （见第 4 节第 8 步），如果 A800 上训练阶段显存余量刚好够用，验证阶段有可能是新的显存
  高峰，需要专门确认。
- 迁移后如果 A800 上依然出现类似的显存边际问题，参考本文档第 2.1~2.3 节的排查方法
  （`[mem]` 打点定位余量、`_interp_chunked` 类似的分块思路可以推广到其他大 Tensor 操作）。
