# CoLIF 最终模型设计与实现合同

> **表示消融合同：** 主模型默认 `actor_cooperative_fusion.representation=per_cav`。`fused` 模式只把预测 reader 的输入替换为 `fused_bev.unsqueeze(1)`；检测、tracking、query、history、forecast head、loss 和参数量不变。`--disable-actor-cooperative-fusion` 是 no-context 诊断，不是 fused-BEV 基线。

## 1. 论文问题与最终主张

CMP 式中间融合把 receiver 对齐的多车 BEV 压缩为一张 fused BEV。该表示适合当前检测与 persistent query 的状态校正，但 downstream forecaster 无法显式区分同一 actor 周围的证据分别来自哪一个视角。检测主要回答 actor 当前在哪里；预测还需先形成不同运动假设，再读取仍保留来源维的对象级局部证据。因此最终主张为：

> **面向当前状态估计的统一协同融合，不必然是面向运动预测的充分融合。协同感知与协同预测应共享对齐方式和 LiDAR 主干，但允许采用任务条件不同的证据聚合。**

CoLIF 使用 CMP FuseBEVT 作为唯一检测/跟踪测量路径，同时让 persistent actor query 从融合前、已经 receiver 对齐的 source BEV 中提取预测上下文。每个 receiver 始终只有一套物理 actor queries。

当前已实现并进入验证的方法由三部分组成；其中 per-CAV 相对 fused-BEV 的增量价值仍以正式表示消融为准：

1. **Perception/forecast fusion decoupling**：scene-level FuseBEVT 服务 detection/tracking，mode-conditioned source fusion 服务 forecasting；两者共享一套 PointPillar 特征，不增加第二个检测器。
2. **Actor-aligned source tokenization**：根据 tracked box 的中心、尺寸和朝向，在每张source BEV上执行同一对象坐标系的RoI采样，并将source与空间维保留到预测decoder。
3. **Anchor-aware multimodal decoding**：有序历史与actor interaction先形成六个带物理anchor语义的motion queries；每个mode读取目标actor的完整多源RoI及其他actor的逐source RoI pooled tokens，再由唯一GMM head输出预测。

Forecast 路径只读 persistent state，不写回 tracking；正式配置固定 `future_prior_mode=none`。

## 2. 端到端在线架构

```text
多车 LiDAR
  │ 每个 source 点云先变换到当前 receiver 坐标
  ▼
CMP PillarVFE → Scatter → BEV Backbone → Shrink → Compressor
  │
  ├── receiver-aligned source BEV [B,L,256,48,176]
  │       │
  │       ├─ 感知/跟踪路径
  │       │    FuseBEVT → fused BEV [B,256,48,176]
  │       │      ├─ CMP cls/reg heads → 当前 dense measurements
  │       │      └─ persistent query decoder
  │       │           → current box / velocity / existence / lifecycle
  │       │
  │       └─ 预测路径
  │            active actor box
  │              → 每 source 采样 actor-yaw aligned 5×5 RoI
  │              → target local tokens [B,Q,L,25,256]
  │              → neighbour actor-source RoI max pooling [B,Q,L,256]
  │
  └── current query + ordered history + actor interaction
           → K=6 anchor-aware motion queries [B,Q,K,256]
           → target local + neighbour object cross-attention
           → 唯一 T=50 anchored Gaussian prediction
```

### 2.1 关键张量

| 张量 | shape | 含义 |
| --- | --- | --- |
| flattened voxels | OpenCOOD sparse voxel batch | 所有有效 CAV 扁平输入 |
| `record_len` | `[B]` | 每个 receiver batch 的有效 CAV 数 |
| source BEV | `[B,L,256,48,176]` | 对齐到 receiver 网格、尚未 FuseBEVT 的每源特征 |
| fused BEV | `[B,256,48,176]` | CMP SwapFusionEncoder 输出 |
| dense `psm/rm` | `[B,2,48,176]` / `[B,14,48,176]` | 当前 anchor classification/regression |
| query state | `[B,128,256]` | 固定容量 persistent slots；不是 128 辆 CAV |
| current boxes | `[B,128,7]` | receiver frame 中的 `[x,y,z,h,w,l,yaw]` |
| active mask | `[B,128]` | 当前活跃物理 actors |
| online history | `[B,128,11,7]` + mask | 最近最多 11 个因果 current states |
| actor source tokens | `[B,128,L,25,256]` | 保留来源和actor-local空间维的RoI证据 |
| mode-source attention | `[B,128,6,L,25]` | 每个motion mode对来源/空间证据的分配 |
| actor-source attention | `[B,128,6,128,L]` | 每个motion mode对其他actor逐source对象token的分配 |
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

`source_to_receiver`记录数据对齐的真实变换，但当前预测tokenizer不再重复编码
actor--sensor geometry；source slot也不进入learned embedding。

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

## 6. Mode-Conditioned Source Fusion

### 6.1 主表示与受控基线

正式 DualCMP 使用 `per_cav` receiver-aligned source BEV。协议匹配的 `fused`
基线仍执行同一个`ActorAlignedSourceTokenizer`和mode decoder，但只把fused BEV
作为一个source。该选择发生在 fused BEV 已完成 detection/tracking 之后，因此
不会改变 measurements、actor queries、identities 或 histories。两种模式的模型
参数完全相同；它们只检验预测端保留 source 维是否提供增量价值。

对每个 active actor，将规则`5×5`网格定义在actor box的局部坐标系中。局部x/y
分别按box length/width扩大2倍，并使用actor yaw旋转到receiver BEV后，从每张
source BEV双线性采样。位置输入同时包含按LiDAR范围归一化的actor-local metric
offset和receiver-frame absolute location。位置编码进入attention key和value，
使返回的context保留被选择证据的前后/左右及绝对位置。

有序历史、motion descriptor、current box和actor interaction先形成actor state。
每个mode query再显式编码robust velocity、speed、dispersion、anchor scale、5 s
anchor endpoint和actor heading。对目标actor，所有`L×25`局部tokens不做空间或
source压缩；对其他actor，只在每个source内部对RoI content执行标准max pooling，
再加该actor中心的receiver-frame位置编码。目标细粒度tokens与邻居actor-source
tokens共同作为key/value。mask只包含inactive actor、CAV padding和越界grid；没有
point occupancy、support、availability或learned router。source slot没有固定
embedding，因此source维置换时预测不变。

```text
target memory    = LayerNorm(RoI content) + RoI position
neighbour memory = LayerNorm(MaxPool(RoI content)) + actor-center position
mode context     = CrossAttention(mode query, target + neighbour memory)
```

```yaml
roi_grid_size: 5
roi_scale: 2.0
source_attention_heads: 8
source_attention_ffn: 1024
```

旧flat attention、uniform mean、relative geometry、availability router、
trajectory corridor和source residual refiner均不在当前执行路径；对应CLI会显式拒绝。

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

current query、ordered history、motion descriptor 和 normalized current box 相加形成 actor feature。仅 forecast-eligible active actors 进入两层 self-attention，避免 newborn 无速度 actor 污染成熟 actor context。source evidence在mode形成后进入，不再被提前压为actor-level共享向量。

## 8. 唯一预测出口：Anchor-Aware Source Decoder

六个速度比例：

```text
[0.0, 0.5, 0.75, 1.0, 1.25, 1.5]
```

给出第 `k` 模态、第 `h` 步的物理 anchor：

```text
anchor[k,h] = robust_velocity × scale[k] × (h+1) × 0.1s
```

每个anchor-aware mode query先执行一次pre-norm multi-head cross-attention：

```text
mode = mode + CrossAttention(mode, source RoI keys/values)
mode = mode + FFN(LayerNorm(mode))
```

唯一GMM head输出：

- bounded residual mean；
- per-step `log_std_x/log_std_y`；
- bounded correlation `rho`；
- mode logits；

mean residual heads 为零初始化，所以未训练时从物理 anchors 开始，不会随机破坏稳定 motion prior。

默认配置：

```yaml
forecast:
  architecture: motion_anchored
  trajectory_parameterization: anchored
  mode_interaction: anchor_aware
  temporal_history: true
  source_attention_heads: 8
  source_attention_ffn: 1024
```

`trajectory_parameterization=direct`是motion-anchor设计对照：同一六模态唯一head
直接回归receiver-frame displacement，不添加物理anchors，且部署proposal不经过
12 m anchor-residual `tanh`上限。history、actor interaction、source tokens、
uncertainty和mode scoring均保持不变。

当前没有cross-mode auxiliary、proposal/refined双出口或训练期隐藏第二层。

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
actor_source_tokenizer
ordered history encoder
actor interaction
anchor-aware mode/source decoder
unique anchored GMM head
```

每层预测 loss：

```text
L_forecast = L_correlated_GMM_NLL(best mode)
           + λ_mode × L_mode_CE
```

future mask 按真实有效 future step计算；全 mask actor 不贡献 forecast numerator/count。双卡先 all-reduce raw numerator 与 valid count 再归一，不能平均每卡 mean。

## 10. 两阶段训练隔离

当前训练分为两个职责清晰的阶段：

1. Phase 1建立稳定、strict-online、GT-free 的 actor set 与 history；
2. Phase 2固定这一 actor set，只比较预测信息结构。

这一隔离保证 source/no-source 和 history/no-history 消融不会通过改变 tracking coverage 偷换样本。

## 11. 固定种子公平结果

固定 seed-20 的旧no-source/no-history结果只验证被替换flat reader的历史路径。
当前mode-conditioned reader必须以`docs/colif_training.md`的一轮fused/per-CAV
结构闸门重新验证，不能继承旧数值作为新网络证据。

旧flat reader的receiver `[0,1]`、20轮per-CAV checkpoint已通过strict-online
评测，但它不能证明当前mode-conditioned reader优于协议匹配的fused-BEV表示。
当前核心增量价值仍以`HANDOFF.md`中的一轮fused/per-CAV结构闸门为准。

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

## 13. 训练与评测入口

可执行命令唯一由 `docs/colif_training.md` 维护。本文件只保留模型合同，不复制可能随配置、seed、资源或表示消融变化的命令。正式协议只用 official train 优化，固定末轮评测 official test；不建 val、不按 test 自动选择 best，repeated-test monitoring 必须在论文披露。

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
- actor-source tokenizer：`actor_source_tokenizer.py`
- motion-anchored decoder：`motion_anchored_forecast.py`
- loss：`opencood/loss/colif_loss.py`、`colif_gmm_loss.py`
- dataset：`colif_sequence_dataset.py`、`colif_target_builder.py`
- DDP train/eval：`train_colif.py`、`eval_colif_v2v4real.py`
- GT-free runtime export：`inference_colif_v2v4real.py`
- 正式 YAML：`opencood/hypes_yaml/colif/colif_v2v4real.yaml`

## 15. 已验证的代码合同

当前测试覆盖：

- source BEV/fused BEV shapes 和 differentiable encode API；
- runtime schema anti-leak；
- query birth/keep/miss/death 与 scene reset；
- private-null assignment；
- dense heatmap existence calibration；
- actor-yaw aligned RoI grid、CAV/boundary hard mask与source permutation invariance；
- target细粒度RoI保留、neighbour content pooling、position同时进入key/value；
- neighbour actor partner-source intervention及masked-token不变性；
- direct trajectory与anchored模式共享checkpoint参数schema，且不受anchor residual上限；
- ordered history对不同中间运动敏感；
- forecast loss到 history/tokenizer/source-attention/unique-head参数的有限梯度；
- two-GPU numerator/count与parameter/buffer同步；
- checkpoint config hash、strict resume与独立 strict-online test。

最新完整回归应从项目根目录执行：

```bash
conda run --no-capture-output -n coop python -m pytest -q tests/colif
```

## 16. 结果与证据位置

- 当前 Phase-1 初始化：`opencood/logs/colif_paper_phase1_20e_seed20/checkpoint_epoch_009.pth`
- 旧flat reader 20轮per-CAV权重：`opencood/logs/dualcmp_core_receiver01_seed20_20e_20260821/checkpoint_epoch_019.pth`
- 旧flat reader逐epoch评测：`opencood/logs/dualcmp_v2_full_receiver01_seed20_20e_20260821/metrics_test.jsonl`
- 旧flat reader fused seed-20：`opencood/logs/dualcmp_fused_receiver01_seed20_20e_20260823/`
- 注意力原始记录与复现元数据：`figures/dualcmp_attention_records.npz`、`figures/dualcmp_attention_summary.json`
- 训练说明：`docs/colif_training.md`
- 当前论文草稿：`manuscript.md`

旧phase、profile、single-receiver、no-source/no-history与source-refiner运行目录
已经清理。当前smoke仅验证执行路径，不能作为性能证据；未来实验不得从smoke或
旧flat reader checkpoint恢复。

## 17. 投稿前仍需补的证据

当前模型只完成单元测试、独立代码审查和双卡短训；T-ITS投稿前仍应补：

1. 多 seed 统计，而不是只报告 seed 20；
2. 参数量、FLOPs、online latency、显存与通信量；
3. runtime query capacity stress test；
4. partner-support/receiver-only/shared 分桶的置信区间或多 seed稳定性；
5. 与相同 target population、相同 strict-online evaluator 下的可比基线。

在这些证据完成前，不能把当前候选结构写成已验证优于fused-BEV的最终模型。
