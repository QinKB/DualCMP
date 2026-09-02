# DualCMP Model Design

## 1. Goal

设计 DualCMP，使其成为一个面向协同轨迹预测的统一 perception–tracking–forecasting 模型。

核心问题不是单独提高检测或跟踪性能，而是解决：

> Cooperative perception and motion forecasting do not necessarily require the same fused representation.

对于检测和跟踪，需要一个稳定、统一的 receiver-centric representation 来确定：

- 当前有哪些 actor；
- actor 在哪里；
- actor identity 是否连续；
- actor 的历史状态是什么。

但对于轨迹预测，过早把不同 CAV 的 feature 完全融合成单一 fused BEV，可能丢失不同观测来源中与未来运动相关的 feature。

因此 DualCMP 的总体设计应采用 **task-asymmetric dual representation**：

- **Fused BEV**：负责 cooperative detection、lightweight tracking 和 causal actor history；
- **Source-preserved BEVs**：保留各 CAV 的 feature，并在 forecasting 阶段按 actor 和 motion intention 进行 cross-attend/aggregate。

Tracker 是两种 representation 之间的桥梁，而不是一个复杂的独立 MOT 系统。

---

## 2. Overall Architecture

总体模型设计如下：

```text
Multi-CAV LiDAR
      │
      ▼
Shared PointPillar / BEV Encoder
      │
      ▼
Receiver-aligned per-CAV BEVs
      │
      ├───────────────────────────────┐
      │                               │
      ▼                               │
Cooperative BEV Fusion                │
      │                               │
      ▼                               │
Fused BEV                             │
      │                               │
      ├── Cooperative Detection       │
      │                               │
      └── Lightweight Actor Tracker   │
                  │                   │
                  ▼                   │
        Actor State / Identity        │
        Current Box / History         │
                  │                   │
                  └── spatial anchor ─┘
                             │
                             ▼
                 Actor-aligned Source Features
                             │
                             ▼
                   Temporal History Encoder
                             │
                             ▼
                    MTR Actor Interaction
                             │
                             ▼
                   Endpoint Intention Queries
                             │
                             ▼
          Intention-conditioned Source Cross-Attention
                             │
                             ▼
                    Multi-modal Trajectories

模型需要保持这个明确的任务边界：

```text
Fused BEV
→ detection
→ actor state
→ identity
→ causal history

Source-preserved BEVs
→ current actor feature
→ intention-conditioned forecasting
```

不要在早期把 source BEVs 再融合掉，也不要让复杂 tracker feature 成为 forecasting 的唯一输入。

---

## 3. Cooperative Perception Backbone

### What to do

继续采用当前 DualCMP/CMP 的 PointPillar + CoBEVT cooperative perception backbone。

每个 CAV 的 LiDAR 首先转换到 receiver coordinate frame，通过 shared PointPillar encoder 得到 per-CAV BEV。

需要同时保留：

```text
source_bev:
[B, L, C, H, W]

fused_bev:
[B, C, H, W]
```

其中：

- `source_bev` 后续给 forecasting 使用；
- `fused_bev` 给 detection 和 tracker 使用。

不要为了新模型重新设计一个 cooperative perception backbone。

### Reference

主要参考：

- CMP
  `tasl-lab/CMP`
- OpenCOOD / CoBEVT
- 当前 DualCMP `PointPillarCoBEVT`

CMP 提供 cooperative LiDAR backbone 和 V2V4Real perception training baseline。

### Why

这一部分不是论文需要重新创新的地方。

需要研究的是：

> 同一个 cooperative feature encoder 产生的不同 representation 应该如何被 tracking 和 forecasting 使用。

---

## 4. Forecast-Oriented Lightweight Actor Tracker

### Motivation

当前 DualCMP 的 persistent-query tracker 较重，包含全局 Transformer decoder、temporal query memory、measurement update 等多个模块。

但 DualCMP 中 tracker 的真正职责不是学习一个非常强的高维 motion representation。

Tracker 只需要稳定维护：

```text
who
where
alive / disappeared
causal history
```

Forecasting 所需要的丰富当前 feature 可以直接从 source-preserved BEVs 获取。

因此需要把 tracker 重构成一个：

> **Forecast-Oriented Lightweight Actor Tracker**

它将 dense fused BEV 转换成 sparse causal actor states，并为 source feature sampling 提供稳定的 actor spatial anchor。

不要把目标设成在 V2V4Real 上重新做一个 tracking SOTA。

---

### Tracker Design

Tracker 需要包含三个基本过程：

```text
Actor Initialization
        ↓
Actor Propagation
        ↓
Actor Update / Matching
```

#### Actor Initialization

使用 fused BEV 上已有 cooperative detection head 产生的 detection proposals 初始化 newborn actors。

不要继续依赖大量随机 learned queries 在整张 BEV 上寻找新目标。

参考：

- S2-Track：query initialization / propagation / matching 的总体分解；
- JDT3D：LiDAR detection proposal + track query；
- 当前 DualCMP `dense_measurement.py`。

#### Actor Propagation

已有 actor 根据：

- previous box；
- velocity；
- receiver ego-motion；
- frame time difference；

传播到当前 frame 得到 reference state。

可以保留简单 kinematic propagation。

不要引入 trajectory prediction feedback，也不要让 forecasting result 写回 tracker。

参考：

- JDT3D；
- Sparse4D temporal instance propagation；
- 当前 DualCMP `query_geometry.py`。

#### Actor-local Feature Update

这是新 tracker 与当前 heavy tracker 最重要的区别。

不要让每个 actor query cross-attend 整张 fused BEV。

根据 propagated/detected actor box，只读取 actor 附近的 fused BEV feature：

```text
Fused BEV
    +
Actor Reference Box
    ↓
Actor-local RoI Feature
    ↓
Lightweight Query Update
    ↓
Updated Actor State
```

可以使用简单的 actor-aligned RoI sampling + lightweight cross-attention / feature update。

具体层数、hidden dimension、RoI size 等不要提前写死，应通过实现和消融确定。

参考：

- JDT3D：track query propagation；
- Sparse4D v3：sparse temporal instance representation；
- 当前 `ActorAlignedSourceTokenizer` 中的 actor-oriented grid sampling。

#### Matching and Lifecycle

已有 actor 保持 identity，新检测主要负责 newborn actor。

Matching 可以继续使用简单、可靠的 spatial assignment / Hungarian mechanism。

Lifecycle 负责：

- birth；
- keep；
- miss；
- delete；
- runtime ID。

不要增加复杂 ReID network、graph association network 或 source-to-source association network，除非实验明确证明必要。

参考：

- MOTR / UniAD persistent query identity；
- JDT3D runtime tracker；
- 当前 DualCMP `lifecycle.py`。

---

## 5. Actor State Memory

Tracker 输出的核心不是一个复杂 prediction feature，而是 sparse actor state。

Actor state 至少需要表达：

```text
actor identity
current box
velocity
active state
age / miss state
short tracker feature
causal box history
receiver pose / timestamp
```

其中 tracker internal feature 只用于下一帧 tracking update。

Forecasting 主要读取：

```text
current actor box
causal actor history
```

不要默认把 tracker latent feature直接作为 motion prediction feature。

### Why

这样可以避免重新建立：

```text
LiDAR
→ fused representation
→ tracker latent
→ predictor
```

这种单一 information bottleneck。

DualCMP 应该保持：

```text
fused BEV
→ stable actor state

source BEVs
→ rich current forecasting feature
```

### Reference

主要参考：

- Sparse4D v3 InstanceBank；
- JDT3D persistent track instances；
- UniAD track query state。

---

## 6. Dual Representation for Forecasting

这是 DualCMP 的核心。

### Fused Representation

Fused BEV 的正式职责：

```text
cooperative detection
actor localization
identity continuity
history maintenance
```

它建立统一的 receiver-centric actor registry。

### Source-preserved Representation

每个 source BEV 始终单独保留：

```text
[B, L, C, H, W]
```

根据 tracker 输出的 actor box，在每个 source BEV 的相同 actor location 上获取 actor-aligned feature：

```text
Actor box
      │
      ├── Source 0 BEV → actor feature
      ├── Source 1 BEV → actor feature
      └── ...
```

输出仍然显式保留 source dimension。

不要提前：

- average；
- max pool；
- fixed weighted fusion；
- source gate。

当前 `ActorAlignedSourceTokenizer` 的整体方向可以保留。

### Why

不同 CAV 对同一个 actor 的当前观测存在视角、遮挡和可见性差异。

这些差异对于当前 detection 可能可以被 fused representation 消除，但对未来轨迹预测可能仍然有用。

因此 source-preserved feature 应该一直保留到 forecasting query 根据具体 motion intention cross-attend/aggregate。

---

## 7. Temporal Actor Representation

Prediction 仍然需要 actor history。

维护最多约 1 s 的 causal actor history，与 CMP 当前 V2V4Real protocol 保持可比较，例如现有的 11-frame observation history。

History 应由一个独立 temporal encoder 编码：

```text
historical actor boxes/states
        ↓
Temporal Encoder
        ↓
actor temporal feature
```

直接复用 CMP/MTR 的 actor polyline encoder 编码 11 帧几何历史。当前
DualCMP 尚未接入该 history encoder，不得把 tracker 的单 token recurrent
feature 当作已完成的 motion-history representation。

不要要求 tracker 自己通过一个很大的 recurrent feature memory承担全部 motion history。

也不要因为 prediction 使用长历史，就要求整个 perception/tracking network 做同长度 BPTT。

需要区分：

```text
long causal runtime history
≠
long differentiable tracker unroll
```

参考：

- CMP：历史轨迹输入；
- MTR：actor motion history；
- Sparse4D：temporal state 与梯度传播可以解耦。

---

## 8. Intention-Conditioned Cooperative Forecasting

Motion decoder 继续参考 MTR。

不要重新设计一套 trajectory decoder。

### Base Forecasting

使用：

- actor temporal feature；
- surrounding actor interaction；
- fixed/data-clustered endpoint intentions；
- multi-layer trajectory refinement；
- GMM trajectory prediction；
- final trajectory NMS。

参考：

- MTR official implementation；
- CMP 对 MTR 的 V2V4Real adaptation；
- CoLIF 已验证的 `IntentionQueryForecastHead` 移植候选。

### DualCMP Modification

DualCMP 真正需要修改的是：

```text
MTR intention query
       ↓
cross-attend actor-aligned source features
       ↓
updated motion query
       ↓
trajectory prediction
```

每个 motion intention 可以对不同 CAV、不同 actor-local feature 产生不同 attention。

因此 source aggregation 不应该发生在 prediction 之前，而应发生在 motion intention 已经形成以后。

正式模型优先采用：

```text
history representation
+
source-preserved current feature
```

进行预测。

当前 `fused + source` forecast path 可以继续作为 ablation，而不是默认必须保留在正式模型中。

---

## 9. Dataset / Training Protocol

继续使用当前 receiver-centric causal sequence dataset，不重新构造新的 V2V4Real 数据格式。

需要支持不同 training stages 使用不同 sequence requirements。

### Detection Stage

单帧 cooperative detection training。

目标是获得稳定的 CoBEVT initialization。

正式实验最好自己训练，而不是只依赖 CMP 下载的 checkpoint。

CMP checkpoint 可以用于开发和结果对照。

### Tracking Stage

使用 short consecutive clips 训练 lightweight tracker。

参考 JDT3D / S2-Track 的短 sequence tracking training。

默认可以从约 3-frame clips 开始，但不要把具体 clip length 写成不可修改的 architecture constraint。

Tracking frames 应正常接受 tracking supervision，不再默认采用当前长期无监督 burn-in。

Phase-specific dataloader 不应为了 tracking 额外加载几十帧 future trajectory。

### Forecast Stage

Prediction 使用 causal actor history + current source BEVs，预测未来 trajectory。

长 history 主要来自 actor-state buffer，不要求 perception/tracker 对全部历史帧保留完整 autograd graph。

### Joint Fine-tuning

最终模型需要增加 joint perception–tracking–forecasting fine-tuning：

```text
L_detection
+
L_tracking
+
L_forecasting
```

Prediction loss 应能够更新 source representation，并在合理路径上影响 upstream cooperative representation。

不要把最终模型保持成：

```text
frozen perception
+
frozen tracker
+
train prediction only
```

---

## 10. Training Strategy Reference

训练策略参考以下端到端模型，而不是照搬 CMP 的完全模块化训练：

### UniAD

参考：

```text
perception/tracking pretraining
        ↓
load checkpoint
        ↓
joint track + motion + downstream task training
```

重点学习它的 staged initialization + final joint optimization。

### UniV2X

参考：

```text
subsystem pretraining
        ↓
cooperative perception training
        ↓
joint cooperative track / motion / planning optimization
```

重点学习 cooperative representation 先稳定，再联合优化 downstream tasks。

### V2XPnP

重点参考：

```text
detection pretraining
        ↓
prediction warm-up
        ↓
unfreeze
        ↓
joint perception + prediction fine-tuning
```

它对当前 DualCMP 最有直接参考意义。

### CMP

CMP 主要参考：

```text
V2V4Real cooperative LiDAR backbone
CoBEVT
MTR/V2V4Real prediction protocol
```

不要继承其：

```text
detector
→ AB3DMOT
→ independent predictor
```

这种完全 decoupled optimization pipeline。

---

## 11. Loss Design

不要为了新模型增加大量新 loss。

总体只需要三类：

```text
Detection Loss
Tracking Loss
Forecasting Loss
```

### Detection

继续使用 CoBEVT / PointPillar 原有 classification + box regression objective。

### Tracking

监督：

```text
actor existence
actor box
actor motion / velocity
persistent identity matching
```

Persistent actors 保持上一帧 identity，新 actor 再做 matching。

参考：

- MOTR；
- UniAD；
- JDT3D；
- 当前 DualCMP `ColifLoss`。

### Forecasting

继续使用 MTR 的 intention classification + GMM trajectory regression / NLL 形式。

不要设计与主问题无关的新 trajectory loss。

最终 joint stage：

```text
L = λ_det L_det
  + λ_track L_track
  + λ_pred L_pred
```

具体权重通过训练稳定性和实验确定，不要在第一版设计文档中写死。

### Frozen full-data schedule

正式训练使用 epoch，而不是把小样本 overfit 的 optimizer steps 外推到
全数据：

```text
Phase A: 20 epochs, JDT3D-style T=3 tracking one-cycle cosine
Phase B: 30 epochs, official V2V4Real CMP/MTR milestone schedule
Phase C: 20 epochs, UniAD/UniV2X-style warmup + cosine joint fine-tuning
```

Phase A 直接对应 JDT3D 的 detection-pretrained T=3 tracking stage；Phase B
直接对应同数据、同 MTR head 的 CMP 配置；Phase C 对应从 staged checkpoint
开始的 end-to-end joint stage。Phase C 的 first ten history frames remain
causal state-building frames without dense autograd; only the current frame
retains the joint graph. The full commands and measured per-GPU resource
settings are maintained in `README.md`.

---

## 12. Code Refactor Direction

不要直接删除现在的模型。

保留当前 heavy persistent-query tracker 作为 baseline。

建议新增：

```text
actor_state_tracker.py
actor_roi_sampler.py
```

并重构：

```text
point_pillar_colif.py
state.py
colif_sequence_dataset.py
colif_loss.py
train_colif.py
intention_query_forecast.py
actor_source_tokenizer.py
```

旧的：

```text
persistent_query_decoder.py
temporal_memory.py
measurement_state_update.py
forecast_motion_prior.py
future_adapter.py
```

暂时保留，便于 baseline / ablation 和回退。

实现时通过 config 选择：

```text
heavy_query_tracker
actor_state_tracker
```

不要直接破坏当前可运行 baseline。

---

## 13. Main Research Logic

实现过程中始终围绕下面这条逻辑判断模块是否应该存在：

```text
Why dual representation?
        ↓
Detection/tracking and forecasting need different features
        ↓
What should fused BEV do?
        ↓
Build stable receiver-centric actor states
        ↓
Why lightweight tracker?
        ↓
Forecasting does not need a heavy recurrent tracking representation;
tracker mainly provides identity, geometry and causal history
        ↓
What should source BEVs do?
        ↓
Preserve per-CAV current actor features
        ↓
How are source features used?
        ↓
Motion intentions cross-attend/aggregate them during forecasting
```

如果某个新增模块不能明显服务于这条逻辑，不要默认加入。

特别避免为了“增加创新”加入：

```text
extra gates
reliability networks
complex association graphs
prediction-to-tracking feedback
additional trajectory anchors
multiple handcrafted losses
```

除非实验明确表明它们解决了真实问题。

---

## 14. Reference Map

实现时优先查看以下代码和论文：

| Part | Main Reference |
|---|---|
| Cooperative LiDAR backbone | CMP / OpenCOOD / CoBEVT |
| Tracker framework decomposition | S2-Track |
| LiDAR query tracking | JDT3D |
| Sparse temporal actor state | Sparse4D v3 |
| Persistent identity / matching | MOTR / UniAD |
| Cooperative tracking comparison | CoopTrack |
| Motion decoder | MTR |
| V2V4Real prediction protocol | CMP |
| Staged end-to-end training | UniAD / UniV2X / V2XPnP |
| Actor-aligned source feature | existing DualCMP implementation |

Do not copy a complete tracker from one repository.

The tracker should be derived from the requirements of DualCMP:

```text
fused BEV
→ sparse actor state

actor state
→ source spatial anchor

source BEVs
→ forecasting feature
```

---

## 15. Expected Final Model

### Current implementation boundary

As of `CP-20260901-03`, the real online path implements detector-proposal
initialization, receiver-frame propagation/KF update, runtime identity and
lifecycle.  It does not yet implement the complete model in this document:

- `ActorState` is initialized with only one history frame, not the required
  11-frame causal forecasting history;
- `PointPillarFax` returns `source_bev`, but no tokenizer, MTR head, prediction
  loss or online forecast consumer is connected;
- the tracker recurrent feature is tracking-internal and must remain excluded
  from the forecasting input;
- the original frozen tracker gate remains failed; the user explicitly chose
  to proceed using the source-backed tracking baseline comparison, without
  rewriting that historical result.

因此当前状态是“lightweight tracker 可用于接入”，不是“DualCMP 模型已完成”。

最终目标不是：

```text
CoBEVT
+
a new generic tracker
+
MTR
```

而是一个完整、连贯的模型：

> **DualCMP maintains a unified receiver-centric actor state from the fused cooperative representation, while preserving per-source BEV features for motion forecasting. A lightweight actor-state tracker maintains identity and causal geometry history without forcing forecasting through a heavy tracking representation. Motion intention queries then cross-attend and aggregate actor-aligned source features to produce multimodal future trajectories.**

实现和实验都应围绕这一模型主张展开。
