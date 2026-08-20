# CoLIF 最终模型设计与实现合同

> **2026-08-21 更新：** `dual_source_motion_anchored`已重建为v2：逐模块复用已验证的actor-source proposal作为base，沿detached proposal读取mode×time×source trajectory corridor，并把真实source选择与null gate分离。一轮receiver0/1结果base FDE5=`3.4186`、source refined=`3.6102`，因此v2默认部署proposal，refined只保留为负结果诊断。完整合同与结果见`docs/superpowers/specs/2026-08-20-dualcmp-corridor-refiner-design.md`和`HANDOFF.md`顶部。

> **版本：v23，2026-08-13。** 本文件只描述当前可运行模型；已放弃的 source-local、future-feedback 和旧三阶段方案不再混在主设计中，完整历史见 `docs/colif_failure_postmortem.md`。模型保留两个可切换预测出口：整体指标最佳的 actor-source proposal 为默认出口；trajectory-source refinement 为可选完整分支。

## 1. 论文问题与最终主张

CMP 式中间融合把 receiver 对齐的多车 BEV 压缩为一张 fused BEV。该表示适合当前检测与 persistent query 的状态校正，但 downstream forecaster 无法显式区分同一 actor 周围的证据分别来自哪一个视角。检测主要回答 actor 当前在哪里；预测还需根据 actor、传感器相对几何、时序和候选运动选择局部证据。因此最终主张为：

> **面向当前状态估计的统一协同融合，不必然是面向运动预测的充分融合。协同感知与协同预测应共享对齐方式和 LiDAR 主干，但允许采用任务条件不同的证据聚合。**

这不是“不使用 fused BEV”，也不是“每车独立检测后再跨车匹配”。CoLIF 保留 CMP FuseBEVT 作为唯一检测/跟踪测量路径，同时让 persistent actor query 从融合前、已经 receiver 对齐的 source BEV 中提取预测上下文。每个 receiver 始终只有一套物理 actor queries。

当前由实验支持的三项方法组成是：

1. **Perception/forecast fusion decoupling**：scene-level FuseBEVT 服务 detection/tracking，actor-conditioned source fusion 服务 forecasting；两者共享一套 PointPillar 特征，不增加第二个检测器。
2. **Geometry-conditioned persistent actor forecasting**：actor query 根据 actor-to-sensor 相对位置、距离、方位和航向，从各 source BEV 局部 patch 中选择证据，再结合有序可变历史与 actor interaction 预测。
3. **Motion proposal 与可选 trajectory-state refinement**：默认输出是六模态物理 proposal；可选分支把每个 mode 的 50 个 future states 作为 query，沿 proposal 再取 source evidence，并以严格 identity-initialized residual 更新。

不再声称 future 改善 tracking。所有已测试 future-to-query、future-to-association 和 forecast missing-state propagation 均无正增益；正式配置固定 `future_prior_mode=none`，forecast 只读 persistent state，不写回 tracking。

## 2. 端到端在线架构

```text
多车 LiDAR
  │ 每个 source 点云先变换到当前 receiver 坐标
  ▼
CMP PillarVFE → Scatter → BEV Backbone → Shrink → Compressor
  │
  ├── receiver-aligned source BEV [B,L,256,48,128]
  │       │
  │       ├─ 感知/跟踪路径
  │       │    FuseBEVT → fused BEV [B,256,48,128]
  │       │      ├─ CMP cls/reg heads → 当前 dense measurements
  │       │      └─ persistent query decoder
  │       │           → current box / velocity / existence / lifecycle
  │       │
  │       └─ 预测路径
  │            active actor query/current box
  │              → 每 source 采样 actor-centered 3×3 patch
  │              → actor-relative geometry-conditioned attention
  │              → cooperative actor context [B,Q,256]
  │
  └── current query + ordered history + cooperative context + actors
           → robust motion anchors + temporal/actor interaction
           → K=6, T=50 Gaussian motion proposal
                ├─ 默认：直接输出 proposal
                └─ 可选：trajectory-source state refinement
                         → refined K×T Gaussian trajectories
```

### 2.1 关键张量

| 张量 | shape | 含义 |
| --- | --- | --- |
| flattened voxels | OpenCOOD sparse voxel batch | 所有有效 CAV 扁平输入 |
| `record_len` | `[B]` | 每个 receiver batch 的有效 CAV 数 |
| source BEV | `[B,L,256,48,128]` | 对齐到 receiver 网格、尚未 FuseBEVT 的每源特征 |
| fused BEV | `[B,256,48,128]` | CMP SwapFusionEncoder 输出 |
| dense `psm/rm` | `[B,2,48,128]` / `[B,14,48,128]` | 当前 anchor classification/regression |
| query state | `[B,128,256]` | 固定容量 persistent slots；不是 128 辆 CAV |
| current boxes | `[B,128,7]` | receiver frame 中的 `[x,y,z,h,w,l,yaw]` |
| active mask | `[B,128]` | 当前活跃物理 actors |
| online history | `[B,128,11,7]` + mask | 最近最多 11 个因果 current states |
| actor source context | `[B,128,256]` | 每 actor 独立聚合的来源证据 |
| forecast | `[B,128,6,50,5]` | `dx,dy,logσx,logσy,ρ` |
| mode score | `[B,128,6]` | 六模态 logits/probabilities |

`num_queries=128` 是槽位容量，不是 agent/CAV 数；runtime `max_active_queries=24` 控制当前 ROI 中最多维护的物理 actors。V2V4Real 当前评测未发现该限制成为本轮主要误差来源，但正式投稿前仍需报告容量压力测试。

## 3. 数据与坐标合同

### 3.1 时间轴

- 每个 `(scene, receiver)` 按 receiver 自身帧建立 chronology。
- 主实验使用 V2V4Real 同步离散帧，周期为 0.1 s。
- 数据集没有真实网络到达时间；delay/dropout/asynchrony 只能作为模拟鲁棒性实验，不能写成真实测得条件。
- state 在 scene/receiver 边界必须 reset，禁止跨 scene 延续。

### 3.2 空间对齐

CMP 的真实 V2V4Real 数据链是在 voxelization 前把每个 source 点云投影到当前 receiver frame。FuseBEVT 内部不会再做一次显式 BEV warp。因此 CoLIF 输入严格遵循：

```text
source raw point cloud
  → source_to_receiver 4×4 transform
  → receiver ROI mask
  → source-wise voxelize
  → shared PointPillar
  → regroup/FuseBEVT
```

`source_to_receiver` 同时用于 actor-source geometry，不能用固定 CAV slot 代替传感器空间关系。

### 3.3 运行时与监督隔离

运行时只允许 point clouds、poses/transforms、source mask、receiver time 和 session metadata。`ass_id`、GT box、canonical identity 和 future GT 只能进入 criterion/evaluator。dataset schema 会递归拒绝监督字段混入 runtime frame。

## 4. CMP 共享编码与双路径融合

`PointPillarColif` 直接持有一套 `PointPillarCoBEVT`：

```text
PillarVFE
→ PointPillarScatter
→ BaseBEVBackbone
→ DownsampleConv
→ NaiveCompressor
→ regroup
→ SwapFusionEncoder
```

`PointPillarCoBEVT.encode_fused_bev()` 返回有梯度的：

```text
source_bev
regrouped_source_bev
cav_mask
fused_bev
```

同一个 CoBEVT 实例的 `cls_head/reg_head` 同时产生 dense measurements。CoLIF 没有再加载一个 CMP 模型，也没有“CMP 输出检测后再接另一个 tracker model”；它复用的是一个网络内的主干、fusion 和 heads。

## 5. Persistent Query Tracking

### 5.1 输入

- 当前 fused BEV；
- 上一帧 query content；
- receiver pose 变换后的 current-state reference box；
- 当前 dense measurement proposals；
- 最近四帧 query/box tracker memory。

### 5.2 预测—校正流程

1. 用上一时刻 current box、velocity、receiver poses 和 elapsed time做 CV propagation。
2. 从当前 `psm/rm` 解码最多 48 个 sparse measurement proposals。
3. active queries 与 measurements 做带 private-null columns 的 one-to-one assignment；坏匹配不能抢走其他 query 的好 proposal。
4. 未匹配 proposal 只进入 inactive slots，形成 newborn candidates。
5. persistent decoder 以 propagated/reference box 在 fused BEV 上 cross-attend，输出 box、velocity、existence 和 query embedding。
6. existence logit显式读取同一个 CMP dense heatmap 在 query center 附近的证据，并用校准后的 product-of-experts 合并，不重复计算 foreground prior。
7. learned bounded measurement gain 融合 propagated state 与 neural measurement。

Forecast distribution 不参与以上 association、reference propagation 或 existence。该边界由 `forecast_motion_prior.mode=none` 和单元测试共同保证。

### 5.3 生命周期

- 第一次有效 cooperative measurement 创建 query state，age=1；
- query 对应物理 actor，不对应 source/CAV；
- measurement miss 时只用 CV current state coast；
- `miss_ttl=4` 后失活并可回收；
- `max_active_queries=24`；
- birth 帧不输出长时 forecast；至少两次因果观测才有速度依据并进入 forecasting。

## 6. Actor-Conditioned Source Fusion

对每个 active actor，在每张 receiver-aligned source BEV 上以当前 box center采样 `3×3` patch，覆盖半径 4 m。每个 token包含：

```text
source BEV feature
+ patch relative offset
+ actor/source relative geometry
```

几何向量为：

```text
normalized relative x/y
distance
sin/cos(actor-relative bearing)
sin/cos(source-heading − actor-heading)
```

actor query 作为 attention query，所有有效 source/patch tokens 为 key/value。无效 source 在 softmax 前 hard mask。默认：

```yaml
geometry_conditioning: true
source_index_embedding: false
```

因此 source slot 置换连同 transform 一起置换时，输出保持不变；模型学的是几何关系，不是“第 0/1 辆车”的固定语义。

## 7. Ordered Variable-History and Interaction

### 7.1 有序历史

每个历史 token 编码：

- relative xy to latest state；
- one-step displacement；
- acceleration；
- z 和 box dimensions；
- sin/cos yaw；
- normalized observation time；
- speed。

两层 temporal Transformer 在 padding mask 下编码截至当前时刻的最近最多 11 帧。历史不足时右对齐，不伪造固定 11 帧有效轨迹。该编码器在已观测历史内部使用双向 self-attention，没有 triangular causal mask；“不读取未来帧”由在线输入前缀保证，而不是由因果注意力保证。

### 7.2 Robust motion descriptor

所有有效历史点对形成速度估计，取 pairwise median 作为 history-derived velocity；相对 median 的平均偏差作为 dispersion。它比只取首末点更不易受单帧 tracking jitter 影响。

### 7.3 Actor interaction

current query、actor-source context、ordered history、motion descriptor 和 normalized current box 相加形成 actor feature。仅 forecast-eligible active actors 进入两层 self-attention，避免 newborn 无速度 actor 污染成熟 actor context。

## 8. 默认预测出口：Actor-Source Motion Proposal

六个速度比例：

```text
[0.0, 0.5, 0.75, 1.0, 1.25, 1.5]
```

给出第 `k` 模态、第 `h` 步的物理 anchor：

```text
anchor[k,h] = robust_velocity × scale[k] × (h+1) × 0.1s
```

actor-conditioned mode feature输出：

- bounded residual mean；
- per-step `log_std_x/log_std_y`；
- bounded correlation `rho`；
- mode logits；
- future latent（只用于兼容/诊断，不反馈 tracker）。

mean residual heads 为零初始化，所以未训练时从物理 anchors 开始，不会随机破坏稳定 motion prior。

默认配置：

```yaml
forecast:
  architecture: motion_anchored
  mode_interaction: actor_conditioned
  temporal_history: true
  proposal_refinement: true
  refinement_type: cross_mode
  inference_layer: proposal
```

`cross_mode` layer 仅作训练期第二层监督；full-test 已证明部署该 refined output 会变差，所以默认推理明确输出 proposal。

## 9. 可选预测出口：Trajectory-Source State Refinement

选择方式：

```text
--forecast-refinement-type trajectory_source
--forecast-inference-layer refined
```

对每个 actor/mode/future step：

1. `current center + proposal mean` 得到 receiver 坐标候选位置；
2. 在每个当前 source BEV 上用 `grid_sample` 采样该位置；
3. mode feature + learned time-state embedding 查询各来源证据；
4. invalid CAV hard mask 后聚合 source context；
5. 共享 temporal state encoder 建模 50 个 proposal states；
6. 对 proposal mean/log-std/rho/logits 预测 bounded residual。

它采样的是当前 BEV 在候选路径位置的场景证据，不是假装得到 future observation。

### 9.1 Identity initialization

所有 correction heads 均零初始化：

```text
refined_mean == proposal_mean
refined_log_std == proposal_log_std
refined_rho == proposal_rho
refined_logits == proposal_logits
```

这是硬合同。早期随机替换 proposal mode score/uncertainty 的实现使 loss 崩坏，已废弃。

### 9.2 结果定位

该分支相对 no-source model 改善全部 1/3/5 s ADE/FDE，但总体不如默认 actor-source proposal。它只在 partner-support FDE5 和 age 2--3 FDE5 上有小幅优势。因此：

- 代码、训练命令和 checkpoint 全部保留；
- 可用于高容量模型选择和困难目标分析；
- 论文不写成整体 3 s 或 5 s superiority；
- 默认主表使用 actor-source proposal。

## 10. Loss、匹配与梯度

### 10.1 Phase 1

训练项包括：

```text
fused anchor focal classification
fused anchor Smooth-L1 regression
query box/state regression
query existence
velocity
lifecycle/sequential current identity supervision
```

训练 Hungarian 使用当前 GT 建 loss 对应关系；runtime forward 不接收 GT identity。

### 10.2 Phase 2

冻结 PointPillar、FuseBEVT、dense heads、measurement generator 和 query tracker，只训练：

```text
actor_cooperative_fusion
ordered history encoder
actor interaction
motion proposal
selected auxiliary/refinement branch
```

每层预测 loss：

```text
L_forecast = L_correlated_GMM_NLL(best mode)
           + λ_mode × L_mode_CE
```

future mask 按真实有效 future step计算；全 mask actor 不贡献 forecast numerator/count。双卡先 all-reduce raw numerator 与 valid count 再归一，不能平均每卡 mean。

## 11. 为什么只有两阶段

三阶段曾用于“tracking → forecast → future反馈联合训练”，但反馈主张已被同权重消融否证。当前两阶段具有明确隔离作用：

1. Phase 1建立稳定、strict-online、GT-free 的 actor set 与 history；
2. Phase 2固定这一 actor set，只比较预测信息结构。

这样 source/no-source、history/no-history 和 trajectory refiner消融不会通过改变 tracking coverage 偷换样本。没有第三阶段，因为当前没有经实验证明需要 joint future-to-tracking fine-tuning 的模块。

## 12. 固定种子公平结果

四组均从同一 Phase-1 epoch-1 checkpoint、seed 20、显式 DataLoader generator、相同 1710 clips、1 epoch/855 optimizer steps开始。四组均为 world size 2、0 AMP overflow、0 no-signal skip、DDP parameter/buffer checksum delta 0；MOTA、coverage、actor counts 完全相同。

| 配置 | minADE 1/3/5 s | minFDE 1/3/5 s | MOTA | Coverage |
| --- | --- | --- | ---: | ---: |
| 无 actor-source fusion | 1.1189 / 2.0404 / 2.7437 | 1.5749 / 3.0895 / 4.3980 | 0.6333 | 0.7894 |
| actor-source，无 ordered history | 1.0504 / 1.9313 / 2.6535 | 1.4608 / 2.9931 / 4.3827 | 0.6333 | 0.7894 |
| **actor-source + ordered history（默认）** | **0.9997 / 1.8472 / 2.5870** | **1.3708 / 2.9177 / 4.3804** | **0.6333** | **0.7894** |
| trajectory-source refined（可选） | 1.0728 / 1.9512 / 2.6714 | 1.4703 / 2.9896 / 4.3836 | 0.6333 | 0.7894 |

默认模型相对 no-source：

- minADE 1/3/5 s 改善 10.65% / 9.47% / 5.71%；
- minFDE 1/3/5 s 改善 12.96% / 5.56% / 0.40%。

默认模型相对 no-history：

- minADE 1/3/5 s 改善 4.82% / 4.35% / 2.50%；
- minFDE 1/3 s 改善 6.16% / 2.52%；
- minFDE5 只改善 0.05%，不能写成显著长时 endpoint gain。

可选 trajectory-source 相对 no-source 仍改善全部 ADE/FDE，其中 minFDE3 改善 3.23%；但它没有超过默认分支的总体 3 s/5 s 指标。小幅困难分桶结果：

```text
partner-support FDE5: 6.4371 vs default 6.4486 m
age 2–3 FDE5:         12.4346 vs default 12.4862 m
```

## 13. CMP 对比边界

本地复现 CMP no-aggregation epoch-5：

```text
ADE5/FDE5 = 4.4001 / 7.5559 m
```

默认 CoLIF：

```text
ADE5/FDE5 = 2.5870 / 4.3804 m
```

数值分别低 41.21%/42.03%，但不能写成严格 apples-to-apples SOTA superiority：CMP tracking cache 已核实先用 GT IoU匹配过滤 detections，再携带 GT ID进入 AB3DMOT并重命名 track；CMP固定 history target population 也与 CoLIF strict-online evaluator 不同。论文只能写“各自已复现协议下数值低于 CMP reference”，同时披露 protocol difference。

## 14. 两条正式训练命令

### 14.1 默认 actor-source proposal

```bash
cd /home/qkb/Project/CoLIF
ulimit -n 65536
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 -m opencood.tools.train_colif \
  --hypes_yaml opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --phase 2 \
  --init-colif-checkpoint \
  opencood/logs/colif_fused_v7_phase1_predict_correct_2epoch_20260812/checkpoint_epoch_001.pth \
  --output-dir opencood/logs/colif_actor_source_release \
  --epochs 1 --workers 2 --eval-workers 2 --log-every 50
```

### 14.2 可选 trajectory-source refined

```bash
cd /home/qkb/Project/CoLIF
ulimit -n 65536
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 -m opencood.tools.train_colif \
  --hypes_yaml opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --phase 2 \
  --forecast-refinement-type trajectory_source \
  --forecast-inference-layer refined \
  --init-colif-checkpoint \
  opencood/logs/colif_fused_v7_phase1_predict_correct_2epoch_20260812/checkpoint_epoch_001.pth \
  --output-dir opencood/logs/colif_trajectory_source_release \
  --epochs 1 --workers 2 --eval-workers 2 --log-every 50
```

完整 Phase 1、resume、smoke、独立 eval 和 inference命令见 `docs/colif_training.md`。正式协议只用 official train 优化，固定末轮评测 official test；不建 val、不按 test 自动选择 best，repeated-test monitoring 必须在论文披露。

## 15. 代码复用与文件地图

### 15.1 直接复用 CMP

- `opencood/models/point_pillar_cobevt.py`
- `opencood/models/sub_modules/pillar_vfe.py`
- `opencood/models/sub_modules/point_pillar_scatter.py`
- `opencood/models/sub_modules/base_bev_backbone.py`
- `opencood/models/sub_modules/downsample_conv.py`
- `opencood/models/sub_modules/naive_compress.py`
- `opencood/models/fuse_modules/fuse_utils.py`
- `opencood/models/fuse_modules/swap_fusion_modules.py`
- `opencood/data_utils/v2v4real/post_processor/voxel_postprocessor.py`

### 15.2 CoLIF 主路径

- 总模型：`opencood/models/point_pillar_colif.py`
- persistent decoder：`opencood/models/colif/persistent_query_decoder.py`
- state/lifecycle：`state.py`、`lifecycle.py`、`temporal_memory.py`
- dense measurements：`measurement_proposals.py`
- actor-source fusion：`actor_cooperative_fusion.py`
- motion proposal/refiner：`motion_anchored_forecast.py`
- loss：`opencood/loss/colif_loss.py`、`colif_gmm_loss.py`
- dataset：`colif_sequence_dataset.py`、`colif_target_builder.py`
- DDP train/eval：`train_colif.py`、`eval_colif_v2v4real.py`
- GT-free runtime export：`inference_colif_v2v4real.py`
- 正式 YAML：`opencood/hypes_yaml/colif/colif_v2v4real.yaml`

### 15.3 保留但不在正式主路径

旧 source-local observation、source memory、forecast association 和 FutureAdapter 文件只用于失败追溯/历史 checkpoint兼容，不能再从这些文件反推正式模型主张。

## 16. 已验证的代码合同

当前测试覆盖：

- source BEV/fused BEV shapes 和 differentiable encode API；
- runtime schema anti-leak；
- query birth/keep/miss/death 与 scene reset；
- private-null assignment；
- dense heatmap existence calibration；
- source hard mask 与 permutation invariance；
- ordered history对不同中间运动敏感；
- forecast loss到 history/source/proposal参数的有限梯度；
- trajectory refiner identity initialization；
- valid partner能改变 refined output并收到梯度；
- masked partner对 refined output无影响；
- two-GPU numerator/count与parameter/buffer同步；
- checkpoint config hash、strict resume与独立 strict-online test。

最新完整回归应从项目根目录执行：

```bash
conda run --no-capture-output -n coop python -m pytest -q tests/colif
```

## 17. 结果与证据位置

- 默认：`opencood/logs/colif_fused_v19_geometry_source_seedfixed_full_1epoch_20260813/`
- trajectory-source：`opencood/logs/colif_fused_v21_identity_trajectory_source_seedfixed_full_restart_1epoch_20260813/`
- no source：`opencood/logs/colif_fused_v18_no_actor_source_seedfixed_full_1epoch_20260813/`
- no history：`opencood/logs/colif_fused_v22_no_temporal_history_seedfixed_full_1epoch_20260813/`
- 完整失败记录：`docs/colif_failure_postmortem.md`
- 训练说明：`docs/colif_training.md`
- 当前论文草稿：`manuscript.md`

## 18. 投稿前仍需补的证据

当前模型与双卡一轮已验证，但 T-ITS 投稿前仍应补：

1. 多 seed 统计，而不是只报告 seed 20；
2. 参数量、FLOPs、online latency、显存与通信量；
3. runtime query capacity stress test；
4. partner-support/receiver-only/shared 分桶的置信区间或多 seed稳定性；
5. 与相同 target population、相同 strict-online evaluator 下的可比基线；
6. 若把 trajectory-source 放入主方法，需证明其困难分桶收益可复现，而不是只依赖单次小差值。

在这些证据完成前，最稳健的投稿版本以 actor-source proposal 为主模型，把 trajectory-source refiner作为可选高容量分支和负结果充分披露。
