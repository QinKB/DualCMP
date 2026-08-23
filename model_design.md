# CoLIF 最终模型设计与实现合同

> **2026-08-22 更新：** source/trajectory refiner 已从模型、损失、训练入口和测试中删除。当前模型只有一条预测主线：fused BEV 用于检测与 persistent tracking，receiver-aligned per-CAV BEV 用于 agent-specific feature fusion，随后直接进入 motion-anchored multimodal decoder。旧 wrapper 中的 20 轮 proposal 权重已无损提取到当前结构，online-masked FDE5=`3.2465`。

> **2026-08-23 消融合同：** 主模型默认 `actor_cooperative_fusion.representation=per_cav`。`fused` 模式只把预测 reader 的输入替换为 `fused_bev.unsqueeze(1)` 与 receiver-frame identity transform；检测、tracking、query、history、forecast head、loss 和参数量不变。`--disable-actor-cooperative-fusion` 是 no-context 诊断，不是 fused-BEV 基线。

> 本文件只描述当前可运行模型；已放弃的 source refiner、future feedback 和旧三阶段方案仅保留在 Git 历史与 `HANDOFF.md` 中。

## 1. 论文问题与最终主张

CMP 式中间融合把 receiver 对齐的多车 BEV 压缩为一张 fused BEV。该表示适合当前检测与 persistent query 的状态校正，但 downstream forecaster 无法显式区分同一 actor 周围的证据分别来自哪一个视角。检测主要回答 actor 当前在哪里；预测还需围绕目标位置，根据 actor query 与传感器相对几何聚合局部多视角证据。因此最终主张为：

> **面向当前状态估计的统一协同融合，不必然是面向运动预测的充分融合。协同感知与协同预测应共享对齐方式和 LiDAR 主干，但允许采用任务条件不同的证据聚合。**

这不是“不使用 fused BEV”，也不是“每车独立检测后再跨车匹配”。CoLIF 保留 CMP FuseBEVT 作为唯一检测/跟踪测量路径，同时让 persistent actor query 从融合前、已经 receiver 对齐的 source BEV 中提取预测上下文。每个 receiver 始终只有一套物理 actor queries。

当前由实验支持的三项方法组成是：

1. **Perception/forecast fusion decoupling**：scene-level FuseBEVT 服务 detection/tracking，actor-conditioned source fusion 服务 forecasting；两者共享一套 PointPillar 特征，不增加第二个检测器。
2. **Geometry-conditioned persistent actor forecasting**：actor query 根据 actor-to-sensor 相对位置、距离、方位和航向，从各 source BEV 局部 patch 中选择证据，再结合有序可变历史与 actor interaction 预测。
3. **Motion-anchored multimodal prediction**：agent-specific CAV context 与有序历史、actor interaction 共同形成预测特征；六个速度尺度提供物理 anchors，网络预测有界残差和高斯分布参数。

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
           → 默认推理输出 proposal
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

### 6.1 主表示与受控基线

正式 DualCMP 使用 `per_cav`，即 receiver-aligned source BEV 与对应
actor--source geometry。协议匹配的 `fused` 基线仍执行同一个
`ActorConditionedCooperativeFusion`，但只把 fused BEV 作为一个位于 receiver
原点的 source。该选择发生在 fused BEV 已完成 detection/tracking 之后，因此
不会改变 measurements、actor queries、identities 或 histories。两种模式的模型
参数完全相同；它们只检验预测端保留 source 维是否提供增量价值。

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
  inference_layer: proposal
```

`cross_mode` layer 仅作训练期第二层监督；full-test 已证明部署该 refined output 会变差，所以默认推理明确输出 proposal。

## 9. Loss、匹配与梯度

### 9.1 Phase 1

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

### 9.2 Phase 2

冻结 PointPillar、FuseBEVT、dense heads、measurement generator 和 query tracker，只训练：

```text
actor_cooperative_fusion
ordered history encoder
actor interaction
motion proposal
cross-mode auxiliary decoder
```

每层预测 loss：

```text
L_forecast = L_correlated_GMM_NLL(best mode)
           + λ_mode × L_mode_CE
```

future mask 按真实有效 future step计算；全 mask actor 不贡献 forecast numerator/count。双卡先 all-reduce raw numerator 与 valid count 再归一，不能平均每卡 mean。

## 10. 为什么只有两阶段

三阶段曾用于“tracking → forecast → future反馈联合训练”，但反馈主张已被同权重消融否证。当前两阶段具有明确隔离作用：

1. Phase 1建立稳定、strict-online、GT-free 的 actor set 与 history；
2. Phase 2固定这一 actor set，只比较预测信息结构。

这样 source/no-source 和 history/no-history 消融不会通过改变 tracking coverage 偷换样本。没有第三阶段，因为当前没有经实验证明需要 joint future-to-tracking fine-tuning 的模块。

## 11. 固定种子公平结果

三组均从同一 Phase-1 epoch-1 checkpoint、seed 20、显式 DataLoader generator、相同 1710 clips、1 epoch/855 optimizer steps开始。三组均为 world size 2、0 AMP overflow、0 no-signal skip、DDP parameter/buffer checksum delta 0；MOTA、coverage、actor counts 完全相同。

| 配置 | minADE 1/3/5 s | minFDE 1/3/5 s | MOTA | Coverage |
| --- | --- | --- | ---: | ---: |
| 无 actor-source fusion | 1.1189 / 2.0404 / 2.7437 | 1.5749 / 3.0895 / 4.3980 | 0.6333 | 0.7894 |
| actor-source，无 ordered history | 1.0504 / 1.9313 / 2.6535 | 1.4608 / 2.9931 / 4.3827 | 0.6333 | 0.7894 |
| **actor-source + ordered history（默认）** | **0.9997 / 1.8472 / 2.5870** | **1.3708 / 2.9177 / 4.3804** | **0.6333** | **0.7894** |

默认模型相对 no-source：

- minADE 1/3/5 s 改善 10.65% / 9.47% / 5.71%；
- minFDE 1/3/5 s 改善 12.96% / 5.56% / 0.40%。

默认模型相对 no-history：

- minADE 1/3/5 s 改善 4.82% / 4.35% / 2.50%；
- minFDE 1/3 s 改善 6.16% / 2.52%；
- minFDE5 只改善 0.05%，不能写成显著长时 endpoint gain。

完整 receiver `[0,1]`、20 轮训练的当前主干在 epoch 19 达到 online-masked ADE1/3/5=`0.8515/1.4083/1.9350`、FDE1/3/5=`1.0310/2.1841/3.2465`，优于修改前 FDE5=`3.4273`。

## 12. CMP 对比边界

本地复现 CMP no-aggregation epoch-5：

```text
ADE5/FDE5 = 4.4001 / 7.5559 m
```

默认 CoLIF：

```text
ADE5/FDE5 = 2.5870 / 4.3804 m
```

数值分别低 41.21%/42.03%，但不能写成严格 apples-to-apples SOTA superiority：CMP tracking cache 已核实先用 GT IoU匹配过滤 detections，再携带 GT ID进入 AB3DMOT并重命名 track；CMP固定 history target population 也与 CoLIF strict-online evaluator 不同。论文只能写“各自已复现协议下数值低于 CMP reference”，同时披露 protocol difference。

## 13. 正式训练命令

```bash
cd /home/qkb/Project/CoLIF
ulimit -n 65536
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 -m opencood.tools.train_colif \
  --hypes_yaml opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --phase 2 \
  --init-colif-checkpoint \
  opencood/logs/colif_paper_phase1_20e_seed20/checkpoint_epoch_009.pth \
  --output-dir opencood/logs/colif_actor_source_release \
  --epochs 20 --batch-size 4 --workers 4 --eval-workers 2 \
  --log-every 50 --receiver-ids 0,1
```

完整 Phase 1、resume、smoke、独立 eval 和 inference命令见 `docs/colif_training.md`。正式协议只用 official train 优化，固定末轮评测 official test；不建 val、不按 test 自动选择 best，repeated-test monitoring 必须在论文披露。

## 14. 代码复用与文件地图

### 14.1 直接复用 CMP

- `opencood/models/point_pillar_cobevt.py`
- `opencood/models/sub_modules/pillar_vfe.py`
- `opencood/models/sub_modules/point_pillar_scatter.py`
- `opencood/models/sub_modules/base_bev_backbone.py`
- `opencood/models/sub_modules/downsample_conv.py`
- `opencood/models/sub_modules/naive_compress.py`
- `opencood/models/fuse_modules/fuse_utils.py`
- `opencood/models/fuse_modules/swap_fusion_modules.py`
- `opencood/data_utils/v2v4real/post_processor/voxel_postprocessor.py`

### 14.2 CoLIF 主路径

- 总模型：`opencood/models/point_pillar_colif.py`
- persistent decoder：`opencood/models/colif/persistent_query_decoder.py`
- state/lifecycle：`state.py`、`lifecycle.py`、`temporal_memory.py`
- dense measurements：`measurement_proposals.py`
- actor-source fusion：`actor_cooperative_fusion.py`
- motion-anchored decoder：`motion_anchored_forecast.py`
- loss：`opencood/loss/colif_loss.py`、`colif_gmm_loss.py`
- dataset：`colif_sequence_dataset.py`、`colif_target_builder.py`
- DDP train/eval：`train_colif.py`、`eval_colif_v2v4real.py`
- GT-free runtime export：`inference_colif_v2v4real.py`
- 正式 YAML：`opencood/hypes_yaml/colif/colif_v2v4real.yaml`

### 14.3 历史实现

旧 source-local observation、source memory、source refiner、forecast association 和 FutureAdapter 仅保留在 Git 历史或失败记录中，不能从历史实现反推正式模型主张。

## 15. 已验证的代码合同

当前测试覆盖：

- source BEV/fused BEV shapes 和 differentiable encode API；
- runtime schema anti-leak；
- query birth/keep/miss/death 与 scene reset；
- private-null assignment；
- dense heatmap existence calibration；
- source hard mask 与 permutation invariance；
- ordered history对不同中间运动敏感；
- forecast loss到 history/source/proposal参数的有限梯度；
- two-GPU numerator/count与parameter/buffer同步；
- checkpoint config hash、strict resume与独立 strict-online test。

最新完整回归应从项目根目录执行：

```bash
conda run --no-capture-output -n coop python -m pytest -q tests/colif
```

## 16. 结果与证据位置

- 当前 20 轮无 refiner 权重：`opencood/logs/dualcmp_core_receiver01_seed20_20e_20260821/checkpoint_epoch_019.pth`
- 当前 20 轮逐 epoch 评测：`opencood/logs/dualcmp_v2_full_receiver01_seed20_20e_20260821/metrics_test.jsonl`
- 一轮 agent-specific CAV fusion：`opencood/logs/colif_fused_v19_geometry_source_seedfixed_full_1epoch_20260813/`
- no source：`opencood/logs/colif_fused_v18_no_actor_source_seedfixed_full_1epoch_20260813/`
- no history：`opencood/logs/colif_fused_v22_no_temporal_history_seedfixed_full_1epoch_20260813/`
- 完整失败记录：`docs/colif_failure_postmortem.md`
- 训练说明：`docs/colif_training.md`
- 当前论文草稿：`manuscript.md`

## 17. 投稿前仍需补的证据

当前模型与双卡 20 轮结果已验证，但 T-ITS 投稿前仍应补：

1. 多 seed 统计，而不是只报告 seed 20；
2. 参数量、FLOPs、online latency、显存与通信量；
3. runtime query capacity stress test；
4. partner-support/receiver-only/shared 分桶的置信区间或多 seed稳定性；
5. 与相同 target population、相同 strict-online evaluator 下的可比基线。

在这些证据完成前，投稿版本以 agent-specific CAV feature fusion 与 motion-anchored proposal 为唯一主模型；source refiner 不进入方法、主表或部署代码。
