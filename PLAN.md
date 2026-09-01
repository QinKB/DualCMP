结论：第一步应采用“CoLIF 数据契约 + JDT3D 三帧 clip 训练方式 + PnPNet 生命周期逻辑 + Sparse4D tensor state”，但不应整套复制任何外部 tracker。当前只给规划，没有修改代码。

## 1. 别人的 dataloader/tracker 如何组织

| 方法 | 一个 dataset item | 时间输入 | batch_size 含义 | Tracker state |
|---|---|---|---|---|
| 当前 CMP | 一个 receiver/ego 的一个 timestamp | 单帧多 CAV | 每 GPU 的独立单帧样本数，当前检测配置为 4 | 没有模型内 state；后续离线 AB3DMOT |
| CoLIF dataset | 一个 scene/receiver 的连续 clip | `warmup + unroll` 帧，每帧多 CAV | 每 GPU 独立 clip 数 | clip 内按时间递推 |
| JDT3D | 一个 3-frame LiDAR clip | 三帧一起由 item 返回 | 默认 B=1 clip/GPU | 模型内部按 `t=0→2` 更新 state |
| UniAD | 一个显式 clip | Stage 1 为 5 帧，Stage 2 为 3 帧 | B=1 clip/GPU | 训练 clip 内递推，推理跨帧 |
| Sparse4D v3 | 一个 timestamp | 单帧 | B=6 frames/GPU | state 跨 DataLoader iteration 缓存 |
| CoopTrack | 一个 timestamp | 单帧 | B=8，每条 batch lane 是连续 sequence | 每条 lane 跨 iteration 保存 state |
| S2-Track | 一个 3-frame、6-camera clip | 3 帧 | 论文未明确报告 batch | query 在 clip 内递推 |
| PnPNet | video clip | 逐帧在线预测 | 论文报告 batch 8 | detection→association→miss recovery→history |

关键证据：

- JDT3D 明确使用三帧 item，并支持把 `T×B` 合成 backbone superbatch 后重新拆回时间维：[config](/home/qkb/Project/Reference/Code/JDT3D/projects/configs/tracking/jdt3d_f3.py:86)、[feature extraction](/home/qkb/Project/Reference/Code/JDT3D/projects/tracking_plugin/models/trackers/JDT3D.py:395)。
- Sparse4D 是单帧 item 加跨 iteration state cache：[InstanceBank](/home/qkb/Project/Reference/Code/Sparse4D/projects/mmdet3d_plugin/models/instance_bank.py:83)。
- CoLIF 已有 receiver-centric clip、source mask、capture/arrival time 和 runtime/target 隔离：[sequence dataset](/home/qkb/Project/CoLIF/opencood/data_utils/v2v4real/datasets/colif_sequence_dataset.py:631)。
- 当前 CMP item 仍然只对应一个 timestamp：[CMP dataset](/home/qkb/Project/DualCMP/opencood/data_utils/v2v4real/datasets/intermediate_fusion_dataset_multi_ego.py:628)。

## 2. 我建议的样本和 batch 设计

### 一个样本是什么

正式定义：

```text
一个 item
= 一个 scene
+ 一个 receiver
+ 从同一 receiver timeline 取得的 T 个连续 timestamp
```

Tracker 首版：

```text
T = 3
warmup = 0
supervised frames = 3
stride = 1
```

每一帧内部包含：

```text
receiver frame t
├── receiver LiDAR
├── partner CAV LiDAR
├── source pose / receiver pose
├── source→receiver transform
├── capture time / arrival time / dt
├── source ID / source mask
└── 当前帧训练 target（与 runtime 输入隔离）
```

不能把下面几个维度混为一谈：

```text
B = 独立 clip 数
T = 时间帧数
L = 每帧可用 CAV/source 数
A = actor state capacity
```

逻辑 shape 是 `[B,T,L,...]`，但稀疏 voxel 的物理 collate 更适合：

```python
runtime_frames: list[T]
runtime_frames[t]:
    processed_lidar     # 当前时间所有有效 source 的扁平 voxel
    source_to_batch     # [S]
    source_slot         # [S]
    source_mask         # [B,L]
    record_len          # [B]
```

Tracker 在模型内部：

```text
for t in range(T):
    state_t = tracker.step(frame_t, state_{t-1})
```

### batch_size 是什么

`batch_size` 统一定义为“每 GPU 独立 clip 数”，不是帧数，也不是 CAV 数。

```text
effective global clips
= per_gpu_batch
× world_size
× grad_accum_steps
```

规划：

1. 第一真实 smoke：`B=1, T=3`。
2. 接口从开始就支持 `B>1`。
3. 第二验收：`B=2, T=3`。
4. 正式 batch 不照搬 PnPNet 的 8、Sparse4D 的 6 或 CoopTrack 的 8；等真实 DualCMP forward/backward 显存、吞吐和 CAV 数量测量后确定。
5. 如果 `B=2` 显存不足，保持 B=1，用 gradient accumulation；不为追求 batch 改变 tracker 语义。

首版不采用 Sparse4D/CoopTrack 那种跨 shuffled DataLoader iteration 保存训练 state 的模式。它要求每条 batch lane、worker、rank 和 sampler 始终保持 scene 连续，第一版风险过高。长 stream 训练放到第二阶段。

## 3. 数据层移植谁

### 可以直接从 CoLIF 提取并做命名/导入适配

这些是当前最成熟的 V2V4Real 基础：

- [colif_sequence_index.py](/home/qkb/Project/CoLIF/opencood/data_utils/v2v4real/datasets/colif_sequence_index.py:35)
- [colif_schema.py](/home/qkb/Project/CoLIF/opencood/data_utils/v2v4real/datasets/colif_schema.py:1)
- [colif_dataloader.py](/home/qkb/Project/CoLIF/opencood/data_utils/colif_dataloader.py:27)

保留能力：

- receiver timeline；
- source availability；
- capture/arrival/delay；
- source→receiver transformation；
- runtime 禁止包含 GT/future；
- DDP `DistributedSampler`；
- rank-aware worker seed；
- batch_size=per-rank clips。

### 从 CoLIF 复制为起点，但必须重构

- [colif_sequence_dataset.py](/home/qkb/Project/CoLIF/opencood/data_utils/v2v4real/datasets/colif_sequence_dataset.py:83)
- [colif_target_builder.py](/home/qkb/Project/CoLIF/opencood/data_utils/v2v4real/datasets/colif_target_builder.py:12)

需要删除：

- heavy UniAD tracker 假设；
- Phase-2 AB3DMOT cache 特例；
- forecast-only cache 路径；
- 旧 CoLIF 命名和已废弃 future recurrence；
- tracker 阶段无用的 50 帧 future 加载。

需要加入明确的 `stage`：

```text
detection:
  T=1
  labels=current detection only

tracking:
  T=3
  labels=current boxes/ids/velocity/existence only
  不读取 future 50 frames

forecast:
  history warmup + current
  才构建 future_xy/future_mask
```

### 不复制

- 当前 CMP 单帧 `IntermediateFusionDatasetMultiEgo`：缺少 clip/time/source slot 契约，augmentation 也不是 clip-consistent。
- CMP MTR dataset：依赖 AB3DMOT trajectory 和 fused-BEV cache，计算图已断。
- JDT3D dataset 源码：nuScenes/MMDetection3D 依赖重，且许可证不适合直接拷入。
- UniAD/CoopTrack dataset：相机、多任务、query 文件和序列 sampler 依赖过重。
- Sparse4D dataset：单帧跨 iteration cache，不适合作为第一版。

## 4. Tracker 采用谁的设计

不直接复制任何完整 tracker，而是组合已经验证的职责边界。

### PnPNet：生命周期主逻辑

采用它的三分支思路：

```text
matched existing actor
unmatched old actor
unmatched new detection
```

但不复制 LSTM 和 dense SOT voxel search。[PnPNet tracker](/home/qkb/Project/Reference/Thesis/Markdown/2020-CVPR-PnPNet.md:72)

### JDT3D：proposal 与 persistent actor 的监督规则

采用：

- detector proposal 初始化 newborn；
- 旧 actor 通过 state 继承 ID；
- train/inference 使用相同 score/quality propagation；
- 只让剩余 proposal 匹配尚未覆盖的 GT；
- 3-frame clip 作为起点。

[JDT3D method](/home/qkb/Project/Reference/Thesis/Markdown/2025-Unresolved-JDT3D.md:75)

不采用它的全局 proposal-query/track-query decoder，因为论文已经观察到 temporal confusion 和 detection AP 明显下降。

### Sparse4D：状态容器语义

采用：

- 固定容量 `[B,A,*]` tensor state；
- active/valid mask；
- timestamp/dt validity；
- ego-motion 与 velocity propagation；
- confidence decay；
- scene reset；
- cache detach boundary。

不采用：

- 900 learned anchors；
- 600 temporal queries；
- global temporal decoder；
- 不使用 tracking loss 的做法；
- 未来帧融合。

### 首版 Tracker API

```python
tracker.step(
    proposals,
    fused_bev,
    prev_state,
    receiver_pose,
    timestamp,
) -> {
    "state": next_state,
    "tracks": current_tracks,
    "assignment": measurement_assignment,
    "aux": training_aux,
}
```

`ActorState` 至少包含：

```text
runtime_id
box [7]
velocity [2]
score
quality
age
miss_count
active_mask
valid_mask
last_timestamp
short tracker feature
causal box history
```

训练时另有独立的 supervision state，用于保存 slot 对应的 GT track ID；它不能混进 runtime payload，也不能决定 runtime keep/delete。

## 5. 应监督什么

### 每帧 detection

沿用 CoBEVT：

```text
classification
box regression
```

### 旧 actor

若其训练绑定的 GT ID 在当前帧存在：

```text
alive / existence = 1
current box
yaw
velocity
propagation/update residual
可选 state quality
```

若 GT ID 当前不存在：

```text
alive / existence = 0
miss
到期 delete
```

注意：GT 只产生 loss target，runtime state 是否继续存在仍由 score、quality、dt 和 miss policy 决定。

### Newborn proposals

只对没有被 existing actor 覆盖的 GT 与 proposals 做 assignment：

```text
proposal classification
proposal box regression
newborn / duplicate classification
```

Hungarian cost 首版只使用：

```text
class cost
BEV center/box cost
可选 IoU cost
```

禁止使用 future trajectory 参与 tracker matching。

### 暂不加入

- ReID loss；
- cross-agent association graph；
- prediction-to-tracking feedback；
- S2-Track probabilistic attention；
- temporal denoising；
- JDT3D track sampling；
- SOT recovery。

这些只有基础 tracker 出现明确失败后才作为消融加入。

## 6. 是否需要本地预处理

首版不需要：

- detection cache；
- AB3DMOT tracking cache；
- fused/source BEV cache；
- trajectory prediction cache；
- track-sampling database。

首版只需要轻量索引：

```text
scene
receiver
ordered timestamps
source availability
PCD/YAML path
source/receiver pose
capture/arrival time
actor IDs
```

这个索引先在 dataset 初始化时从原始 YAML/PCD 构建，并使用 YAML LRU cache。若实测启动过慢，再把它持久化为版本化 metadata manifest；manifest 只存路径和元数据，不存模型 feature。

Tracker targets 可以按需从原始标注构建：

- current boxes/IDs：当前 YAML；
- velocity label：相邻 GT 的有限差分，只放 target；
- existence/miss：同一 GT ID 在相邻帧是否存在；
- future 50 帧：只有 forecast stage 才读取。

JDT3D 的 track database 只在后续决定启用 temporally consistent track sampling 时才生成。

## 7. 下一步实际改动顺序

### 第一次修改：纯数据契约

预计文件：

```text
opencood/data_utils/v2v4real/datasets/dualcmp_sequence_index.py
opencood/data_utils/v2v4real/datasets/dualcmp_schema.py
opencood/data_utils/v2v4real/datasets/dualcmp_sequence_dataset.py
opencood/data_utils/v2v4real/datasets/dualcmp_target_builder.py
opencood/data_utils/dualcmp_dataloader.py
```

并注册 dataset，加入数据测试。此步不写 tracker。

验收：

- 真实 receiver 连续三帧；
- source 0 始终是 receiver；
- source arrival 不晚于 receiver time；
- runtime 无 GT/future；
- `B=2` collate 的 source mask 与扁平 source 完全对应；
- clip 不跨 scene；
- tracker stage 不读取 future PCD/YAML。

### 第二次修改：ActorState 和纯几何传播

预计文件：

```text
opencood/models/dualcmp/state.py
opencood/models/dualcmp/geometry.py
```

验收：

- scene reset；
- ego-motion transform；
- velocity×dt propagation；
- invalid dt；
- batch 中两条 clip state 完全隔离；
- detach 行为明确。

### 第三次修改：轻量 tracker

预计文件：

```text
opencood/models/dualcmp/tracker.py
opencood/models/dualcmp/roi_sampler.py
opencood/loss/dualcmp_tracker_loss.py
```

验收：

- matched actor 保持 ID；
- 高质量 unmatched proposal 出生；
- unmatched old actor miss 后删除；
- duplicate proposal 不产生第二个 ID；
- 一真实三帧 clip 完成 `forward → L_det + L_track → backward → optimizer step`；
- actor-local RoI 梯度能到 fused BEV；
- detection AP/newborn recall/FP/FN/IDS 和 proposal-vs-track score 分布可报告。

第一批改动以双 GPU 训练和冻结 48 帧 evaluator 为唯一功能验收。
中间模块不创建 reviewer；只在阶段端到端验收通过后做一次独立审查。

## 8. 完整 DualCMP 实施主线

`PLAN.md` 不再只停留在 dataloader/tracker。最终边界以
`model_design.md` 为准：

```text
fused BEV -> detection + lightweight actor-state tracker + causal history
source BEVs -> actor-aligned source tokens
history + MTR intention queries -> source cross-attention -> multimodal forecast
final -> joint perception + tracking + forecasting fine-tuning
```

### Phase A：因果数据与跟踪闭环（进行中）

已完成：

- `(scene, receiver)` 的 T=3 真实 clip 和 B>1/DDP 数据契约；
- runtime/target 隔离、evaluator-native label 和 receiver-1 -> CAV0 评估坐标；
- tensor `ActorState`、AB3DMOT GIoU matching/lifecycle、DMSTrack Kalman 更新；
- fused-BEV actor RoI、Sparse4D-style 同 slot temporal feature recurrence；
- `forward -> loss -> backward -> optimizer -> checkpoint -> online inference -> metric` 路径已接通。

未完成：

- 用当前最新代码重新进行双 GPU、40 epoch、44 clips 过拟合；
- 新 checkpoint 在冻结 2x24 帧上同时通过 tracking metric 和
  birth/keep/delete/duplicate 门槛；
- 失败时只依据真实 evaluator 失败项修复，不改冻结门槛。

### Phase B：Source-preserved forecasting（待 Phase A 通过）

直接复用：

- CMP/OpenCOOD `PointPillarCoBEVT` 产生的 `source_bev` 和 `fused_bev`；
- 当前 `ActorAlignedSourceTokenizer` 的 actor-centered grid sampling；
- CMP 的 V2V4Real MTR adapter/decoder/loss/metric；
- MTR official intention query、GMM regression/NLL 和 trajectory NMS。

需要适配：

- tracker 的 runtime actor box/history 取代 AB3DMOT 预处理 trajectory 作为预测输入；
- 在每个 source BEV 的同一 actor 位置取样，保留 `[B,A,L,C]`
  source 维；
- MTR motion intention query 形成后才 cross-attend source tokens，不提前
  mean/max/gate 融合；
- prediction 不反向写回 tracker。

功能验收：

- 真实 forecast item 含长因果 history 和 future target，runtime 仍无 GT/future；
- 完成 `online tracking -> actor history/source tokens -> MTR loss -> backward -> checkpoint -> inference`；
- source-token 和 intention cross-attention 梯度到达 per-CAV encoder feature；
- 预先冻结的小子集过拟合 minADE/minFDE/MR 达标，不以 loss 下降代替。

### Phase C：联合微调（待 Phase B 通过）

按 UniAD/UniV2X/V2XPnP 的 staged initialization 思路，不重造 loss：

```text
L = lambda_det * CoBEVT detection loss
  + lambda_track * existing DualCMP tracking loss
  + lambda_pred * MTR forecasting loss
```

功能验收：

- 双 GPU 走通真实 joint `forward/backward/optimizer/checkpoint/inference`；
- prediction loss 梯度到达 source representation，detection/tracking 不回归；
- 固定协议下同时报告 detection、tracking 和 forecasting metric。

### Phase D：实验与最终验收

仅保留与核心主张直接相关的对照：

```text
forecast representation: fused / source / dual
tracker: heavy baseline / lightweight actor-state tracker
training: staged-only / joint fine-tuning
```

最终完成标准：真实数据、双 GPU、新 checkpoint、权威 evaluator、固定协议；
然后才创建一个 fresh reviewer 做最终独立审查和 Git checkpoint。

## 9. 效率约束

- 同一阶段只保留一个 writer，复用原 agent，不重复建任务。
- 中间语法、shape 和单元测试只作 preflight，每个阶段只以一次真实功能验收收口。
- runner 只负责长时双 GPU 训练/评估；失败日志回到同一 writer 修复。
- 不因中间 checkpoint、oracle、loss 下降或局部测试宣称模型完成。
