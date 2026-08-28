# DualCMP 模型合同

## 1. Goal

DualCMP研究同一组协同LiDAR特征在感知与运动预测中的不同表示需求：

> 一张fused BEV适合提供稳定的协同共识，但单独使用它可能丢失未来推理所需的视角差异；因此预测端同时保留fused consensus和source-preserved evidence，并由每个运动意图分别读取两种表示。

正式模型只包含两个方法要点：

1. **Dual BEV Representation**：fused BEV为感知、跟踪和预测提供共识表示；receiver-aligned per-CAV BEVs为预测保留来源证据。
2. **Intention-Conditioned Cooperative Forecasting**：复用MTR的64个数据聚类endpoint intention queries，使每个query分别读取consensus与source evidence。

history encoder、actor interaction、GMM、layered decoding和NMS属于标准预测scaffold，不作为额外贡献。Forecast只读persistent state，不写回tracking。

## 2. Architecture

```text
多车 LiDAR
  → source_to_receiver 变换
  → shared PointPillar encoder
  → receiver-aligned per-CAV BEVs [B,L,256,48,176]
       ├─ FuseBEVT → fused BEV [B,256,48,176]
       │    ├─ dense detection measurements
       │    ├─ persistent query tracking
       │    └─ fused consensus actor RoI [B,Q,1,25,256]
       └─ source-preserved actor RoIs [B,Q,L,25,256]

persistent query + learned temporal history
  → actor features [B,Q,256]
  → MTR object decoder interaction
  → 64 fixed endpoint intention queries
       ├─ cross-attend fused consensus memory
       └─ cross-attend per-CAV source memory
  → shared residual FFN
  → layerwise 64-mode GMM trajectories
  → predicted endpoint updates next-layer query center
  → endpoint NMS (2.5 m)
  → 6 public trajectories [B,Q,6,50,5]
```

### 2.1 Dual representation

对actor `i`，同一个`ActorAlignedSourceTokenizer`分别处理：

```text
consensus_i = RoI(fused_bev, box_i)       [1,25,256]
source_i    = RoI(per_cav_bevs, box_i)    [L,25,256]
```

两路共享tokenizer及decoder attention参数。`dual`、`fused`和`source`三种表示因此具有相同的总参数与trainable参数；变化只来自可读memory：

| representation | consensus memory | source memory | 用途 |
| --- | --- | --- | --- |
| `dual` | fused actor RoI | per-CAV actor RoIs | 正式模型 |
| `fused` | fused actor RoI | 无 | fused-only消融 |
| `source` | 无 | per-CAV actor RoIs | source-only消融 |

检测、measurement、tracking、actor identity和history在三种设置中完全相同。

### 2.2 Actor-aligned tokenization

RoI网格在tracked box局部坐标系内定义，随真实box yaw旋转到receiver BEV并执行bilinear sampling。供intention attention使用的patch position随后转换到与64个MTR intentions一致的legacy local frame；tokenizer输出content、position和hard mask。source slot没有ID embedding，因此交换CAV顺序不会改变预测，只会交换对应的attention权重。

当前正式输入不包含point support、availability router或手工可靠度。相对source geometry/support只有在受控实验证明独立增益后才允许进入主模型，不能作为未经验证的附加模块。

### 2.3 Learned temporal history

最近最多11个因果observations经右对齐mask进入两层Transformer encoder。每个history token只包含标准状态表示：

```text
relative xyz to the latest observation
normalized box dimensions
sin/cos yaw
relative observation time
```

relative position scale由LiDAR range自动得到，box-size scale由YAML显式记录为`[3,3,8]`；二者仅用于数值归一化，不定义运动模式。

模型不再计算pairwise-median velocity、dispersion、手工motion descriptor或速度尺度。至少两次有效观测后才输出长时预测；这是可观测性边界，不是trajectory anchor。

### 2.4 MTR intention queries

正式模型直接读取：

```text
preprocessed_data/v2v4real/v2v4real_cluster_64_center_dict.pkl
```

`TYPE_VEHICLE`包含64个二维endpoint points。它们只作为MTR的static intention query与训练期positive-query assignment，不与constant-velocity trajectory相加。

每个decoder layer遵循MTR合同：

```text
static intention query + current dynamic endpoint
  → MTR object cross-attention
  → fused consensus cross-attention
  → per-CAV source cross-attention
  → variance-preserving residual sum + shared post-norm FFN
  → 64 GMM trajectories and mode logits
  → predicted final endpoint becomes next dynamic query center
```

consensus/source两路从同一个MTR base query并行读取，使用同一组attention/FFN参数：

```text
e = (Attn(q, consensus) + Attn(q, source)) / sqrt(number_of_active_paths)
q' = LayerNorm(q + Dropout(e))
q_out = LayerNorm(q' + Dropout(FFN(q')))
```

没有concatenation MLP、learned gate、source refiner、trajectory refiner或第二套预测出口。`query_dim=256`到MTR `d_model=512`只通过一层共享attention memory projection完成；它定义attention feature width，不负责融合两种表示。

### 2.5 Outputs

| tensor | shape | semantics |
| --- | --- | --- |
| fixed intentions | `[B,Q,64,2]` | 固定endpoint query locations |
| layer scores | `[B,Q,6,64]` | 六层raw-64 logits |
| layer trajectories | `[B,Q,6,64,50,5]` | 六层raw-64 GMM |
| public scores | `[B,Q,6]` | NMS选择的mode probabilities |
| public trajectories | `[B,Q,6,50,5]` | receiver-frame `dx,dy,logσx,logσy,ρ` |
| consensus attention | `[B,Q,6,25]` | selected modes读取fused RoI |
| source attention | `[B,Q,6,L,25]` | selected modes读取per-CAV RoIs |

训练始终保留全部64个queries；64→6只发生在public inference output。

## 3. Perception and tracking contract

DualCMP只持有一套`PointPillarCoBEVT`。fused BEV同时产生dense cls/reg measurements并供persistent query decoder读取。在线状态维护：

- 128个query slots是容量，不是CAV数；runtime最多保留24个active actors；
- active query与dense proposals执行带private-null columns的一对一匹配；
- learned bounded measurement gain融合propagated state与当前measurement；
- miss时使用current-state CV coast，`miss_ttl=4`后回收；
- forecast distribution不参与association、existence或lifecycle。

以上边界由`forecast_motion_prior.mode=none`和测试共同约束，保证表示消融不会改变actor population。

## 4. Coordinates and runtime isolation

- per-CAV point clouds在voxelization前已投影到当前receiver frame；FuseBEVT内不重复warp。
- tracking state使用receiver-frame radians。
- 本地CMP/MTR的V2V4Real intentions/checkpoint由degree-valued heading作为trigonometric angle生成；该legacy兼容只存在于MTR local-coordinate边界，public output使用同一角度逆变换。
- runtime forward只允许point clouds、poses/transforms、CAV mask、receiver time与session metadata。
- `ass_id`、GT boxes、canonical identity和future GT只进入criterion/evaluator，不能进入model forward或recurrent state。

## 5. Loss

Phase 2对每个decoder layer使用相同的MTR监督：

```text
positive_query = argmin_k ||fixed_intention_k - GT_endpoint||
L_layer = correlated_GMM_NLL(trajectory_positive_query)
        + cross_entropy(raw_64_logits, positive_query)
```

六层loss取平均。`loss_intention_points`始终为固定64点，不使用网络预测终点重新定义分类标签。future mask按真实有效步计算；DDP先all-reduce raw numerator/count再归一。

MTR dense future prediction保留为标准object-context auxiliary objective。检测、existence、box和velocity losses只属于Phase 1 frozen tracker。

## 6. Training and checkpoint contract

1. **Phase 1**：从CoBEVT checkpoint训练fused detection/persistent tracking，forecast disabled。
2. **Phase 2**：从固定Phase-1 checkpoint恢复tracker，同时通过`--init-mtr-checkpoint`严格加载`motion_transformer.motion_decoder.*`；冻结perception/tracking，只训练history encoder、shared tokenizer、MTR decoder与dual attention。

旧motion-anchor、flat reader、source-refiner及旧`PersistentQueryMTRAdapter` checkpoint与当前结构不兼容，禁止静默加载或resume。`--resume`只允许完全相同config hash的当前结构checkpoint。

唯一可执行训练/评测命令由`docs/colif_training.md`维护。

## 7. Validation

当前组件/集成合同必须覆盖：

- fixed 64 intentions、six-layer raw-64 supervision与NMS-6 output；
- 每层dynamic endpoint确实来自上一层预测终点；
- consensus与source任一分支干预都会改变trajectory；
- 任一分支fully masked与`None`完全等价且attention为零；
- source permutation prediction invariant；
- loss梯度到达history、tokenizer、shared dual attention和所有MTR motion heads；
- dual/fused/source参数schema相同；
- Phase-1 + MTR双checkpoint初始化严格且无unexpected keys；
- 单卡真实batch forward/loss/backward/optimizer及双卡DDP同步smoke；
- 完整strict-online receiver `[0,1]`评测。

旧fused-vs-source或motion-anchor训练结果不能证明当前`dual`模型有效。正式结构闸门必须重新比较：

```text
Fused only
Source only
Fused + Source (DualCMP)
```

只有`dual > source > fused`或至少`dual`稳定优于两个单表示对照时，互补表示主张才得到支持。

## 8. Code map

- model integration：`opencood/models/point_pillar_colif.py`
- fused/source tokenizer：`opencood/models/colif/actor_source_tokenizer.py`
- intention decoder：`opencood/models/colif/intention_query_forecast.py`
- reused MTR decoder：`MTR/mtr/models_v2v4real/motion_decoder/mtr_decoder.py`
- persistent tracking：`persistent_query_decoder.py`、`state.py`、`lifecycle.py`
- loss：`opencood/loss/colif_loss.py`、`colif_gmm_loss.py`
- train/eval：`opencood/tools/train_colif.py`、`eval_colif_v2v4real.py`
- formal config：`opencood/hypes_yaml/colif/colif_v2v4real.yaml`

## 9. Open problems

- 尚未获得当前dual结构的正式训练指标；旧checkpoint只能作为历史证据。
- source geometry与point support尚未证明独立收益，不进入主模型。
- 64-point文件在本地可验证为CMP同名文件的字节相同副本，但仓库缺少生成脚本、train-split清单和k-means随机种子；正式数据provenance仍需补齐。
- 投稿前仍需多seed、参数/FLOPs/latency/显存、query capacity与matched-population基线。
