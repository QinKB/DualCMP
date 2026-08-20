# DualCMP Corridor Refiner v2 Design

## 1. 目标

重建 `dual_source_motion_anchored`，使模型满足以下不可妥协的合同：

1. fused BEV 负责唯一 receiver-centric actor state 与可靠的 base forecast；
2. source BEV 只提供沿候选未来走廊的增量证据，不得绕过 source evidence 形成第二个无条件 decoder；
3. source 分支关闭、无有效 source、gate 选择 null 或 residual 初始为零时，final distribution 必须逐元素等于 base distribution；
4. source final loss 不回传 base，但也不能通过多层 loss 平均、全局梯度裁剪或随机数消耗削弱 base 的训练；
5. source correction 必须具有时间连续性，不能出现 1 s 大跳变而 5 s 修正反而更小的非物理轨迹；
6. forecast 仍是 prediction-only，不进入 tracker propagation、association、identity 或 lifecycle。

## 2. 已验证的 v1 失败机制

### 2.1 Base 梯度被减半

v1 将 base 与 detached final 两层 GMM loss 做算术平均。由于 final 对 base 完全 detach，base 实际只收到 `0.5 L_base`。最小 autograd 复现得到 base gradient ratio 精确为 `0.5`；真实首批日志对应 A0 gradient norm `85.85`、A1 `42.19`。

### 2.2 Source residual 存在 base-query 旁路

v1 residual MLP 输入为 `[source_query, source_context]`，其中 `source_query` 直接由 base GMM 生成。对 A2 checkpoint 保留相同 router weights、清零 source context 后，1/3/5 s endpoint correction 仍保留约 60%/45%/51%。因此 source branch 可以不依赖 source BEV，退化为 detached base 上的第二个 decoder。

### 2.3 Source reader 没有读取预测走廊

v1 只在 actor 当前中心采样一个 `3×3` patch，然后压成每个 mode 一个 token并一次性输出 50 步 residual。各 mode 虽然 query 不同，但读取的是相同当前位置 memory；未来时刻轴在 source read 前已经消失。

### 2.4 Null route 无约束塌缩

v1 把固定零 logit 的 null 与无界 source relevance logits 放入同一个 softmax。source context 又除以总 source mass，所有 source logits 同时增大只会打开 residual gate，不改变 source mixture。forecast loss 因而可以无代价地把 source mass 推到 1。A2 的 300 帧诊断为平均 source mass 99.04%、null 0.96%；A3 证明 availability bias 不是根因。

### 2.5 Position-wise residual 缺少运动约束

v1 独立预测 50 个二维位置偏移。A2 的 1 s correction P90 为 1.47 m，而 5 s 为 0.37 m，说明 correction 没有位移连续性或速度积分语义。

## 3. 总体架构

```text
receiver-aligned multi-CAV LiDAR
        │
        ├── FuseBEVT ──> fused BEV ──> persistent actor state/query
        │                                │
        │                                └── mature motion proposal core
        │                                         │
        │                              physics/cross-mode proposals
        │                                         │
        ├── fused BEV corridor sampling ──────────┤
        │                                         ▼
        │                                  fused-base GMM
        │                                         │ detach
        └── source BEV corridor sampling ─────────┤
                                                  ▼
                                   source selection + null gate
                                                  │
                                   integrated velocity residual
                                                  ▼
                                         final source-refined GMM
```

完整模型分三层监督：

1. `motion_proposal`：成熟 motion-anchored proposal core；
2. `fused_base`：沿 proposal 读取 fused corridor 后的 base；
3. `source_final`：沿 detached fused-base corridor 读取 source evidence 后的 final。

source-disabled 消融只暴露前两层；full 模型暴露三层。source final 对前两层全部 detach。

## 4. 成熟 Motion Proposal Core

### 4.1 复用边界

v2 首先构造一个 `MotionAnchoredForecastHead` 子模块，并且在任何 v2 随机模块之前构造。这样在相同 seed、相同 `PointPillarColif` 构造顺序下，其参数初始化与 A0 的成熟 head 一致。

该子模块继续使用：

- robust pairwise-median velocity；
- ordered temporal history encoder；
- motion descriptor 与 box encoding；
- actor interaction；
- six physics motion anchors；
- proposal 与 cross-mode deep supervision。

v2 不复制或重新解释上述逻辑。它消费 proposal core 已公开的 proposal GMM，后续 corridor modules 作为独立 refinement。

### 4.2 Fused BEV 职责

proposal core 的 `query_feat` 已由 fused BEV tracker 产生。除此之外，v2 沿 proposal future states 再读取 fused BEV，使 base forecast 能访问 query 压缩后仍保留的空间证据。source BEV 在 fused-base 形成前不参与。

## 5. Trajectory Corridor Sampling

### 5.1 坐标合同

`PointPillarCoBEVT.encode_fused_bev` 已验证：每个 source 点云在 voxelization 前投到 receiver frame，`fused_bev` 与 `regrouped_source_bev` 均使用同一 receiver-frame metric grid。v2 不执行第二次 warp。

每个相对 proposal mean `mu[b,q,k,t,2]` 转成绝对 receiver position：

```text
p_abs[b,q,k,t] = actor_box[b,q,:2] + mu[b,q,k,t]
```

根据相邻 proposal points 计算 tangent；速度近零时回退到 actor current yaw。每个 state 使用五个固定 metric offsets：

```text
center, +1.5 m normal, -1.5 m normal,
+1.5 m tangent, -1.5 m tangent
```

这些 offsets 随 trajectory tangent 旋转后映射到 BEV normalized grid。采样覆盖全部 50 个 future steps，不提前压缩 time axis。

### 5.2 张量形状

设 `S=5` corridor samples：

```text
fused corridor:  [B,Q,K,T,S,C]
source corridor: [B,Q,K,T,L,S,C]
spatial valid:   [B,Q,K,T,S]
source valid:    [B,Q,K,T,L,S]
```

`grid_sample(..., align_corners=False, padding_mode="zeros")` 与现有 BEV 合同一致。越界 grid、无效 CAV 与显式 source mask 均在 softmax 前 hard-mask。

### 5.3 模块边界

新增 `opencood/models/colif/trajectory_corridor.py`：

- `TrajectoryCorridorSampler`：只负责 metric proposal 到 grid、tangent/normal offsets、fused/source sampling 与 validity；
- `FusedCorridorRefiner`：只负责 fused tokens 到 fused-base GMM residual；
- `SourceCorridorRefiner`：只负责 per-source tokens、source selection、null gate 与 source final residual。

`dual_source_forecast.py` 只编排 proposal core、两级 corridor refiner、输出与 mask，不继续扩成单个千行模块。

## 6. Fused Corridor Base Refiner

### 6.1 State tokens

proposal GMM 的每个 `(mode,time)` state 由以下信息形成 query：

```text
[proposal mean(2), log_std(2), rho(1), normalized time(1)]
    -> point MLP -> D
```

fused corridor 的五个 samples 经 `fused_projection -> GELU -> LayerNorm`，加入 offset embedding。state query 对五个 samples 做 pre-LN cross-attention，随后使用 temporal Transformer 沿 `T` 建模。

### 6.2 Base correction

fused base 不直接独立输出 50 个 position offsets，而是输出 bounded velocity correction。速度采用向量范数约束，而不是分别截断两个坐标分量：

```text
delta_v_fused = unit(raw_delta_v) * 4.0 * tanh(||raw_delta_v|| / 4.0)  [m/s]
delta_mu_fused[t] = cumsum(delta_v_fused * 0.1)[t]
delta_mu_fused = 6.0 * tanh(delta_mu_fused / 6.0) [m]
```

log-std、raw-rho 与 mode-logit updates 保持有界。所有 fused correction heads 零初始化，所以 v2 初始化时 fused-base 精确等于 mature proposal。

## 7. Source Corridor Refiner

### 7.1 Source tokens

source sampling 使用 detached fused-base mean 作为 corridor center。每个 token保留 `[K,T,L,S]`，并加入：

- corridor offset embedding；
- source origin relative geometry；
- `delta_t_s` age embedding；
- actor/source availability logit，仅作为 relative source quality；
- normalized future time embedding。

不使用 fixed source-slot embedding，联合置换 source BEV、mask、geometry、age 后输出保持不变。

### 7.2 Source selection 与 null gate 分离

source relevance 只在真实 sources 内 softmax：

```text
w_source[b,q,k,t,l] = softmax_l(relevance + beta * log_sigmoid(availability))
```

聚合 source token：

```text
z_source[b,q,k,t] = sum_l w_source * z[b,q,k,t,l]
```

独立 gate 决定是否采用 source correction：

```text
g[b,q,k] = sigmoid(gate_logit)
alpha_null = 1 - g
alpha_source[b,q,k,t,l] = g * w_source[b,q,k,t,l]
```

gate 为 mode-level，source selection 为 mode/time-level。无有效 source 时强制 `g=0`、`alpha_null=1`、source weights=0。

gate 最后一层 weight 零初始化，bias 固定初始化为 `logit(0.1)=-2.1972246`，所以初始 source mass 为 0.1，而不是 v1 的约 2/3。gate 训练使用 forecast loss与显式 Bernoulli prior共同约束；availability BCE 不被错误解释成 gate target。

### 7.3 禁止 base-query 旁路

source residual value network不接收 `source_query`、base latent或base trajectory embedding。base state只允许用于：

1. 决定 corridor sampling positions；
2. 形成 attention query；
3. 计算 gate 所需的 detached uncertainty。

source/fused evidence contrast、base uncertainty、geometry与age只允许进入 source selection 和 gate。residual value只消费经过 source selection 的 source corridor value tokens，不能消费 fused token、base query或base trajectory embedding。

source value path采用零保持结构：BEV value projection无bias；time conditioning使用乘法FiLM而非加法position embedding；temporal attention与FFN使用无bias线性层和无affine LayerNorm；最终velocity/log-std/rho/logit heads无bias。由此 `source_context=0` 必然推出全部 source corrections 精确为零，不依赖训练后bias是否漂移。

### 7.4 连续运动 residual

source temporal encoder输出 bounded velocity correction：

```text
delta_v_source = unit(raw_delta_v) * 3.0 * tanh(||raw_delta_v|| / 3.0) [m/s]
delta_mu_source[t] = cumsum(delta_v_source * 0.1)[t]
delta_mu_source = 4.0 * tanh(delta_mu_source / 4.0) [m]
final_mean = base_mean.detach() + g * delta_mu_source
```

因此第一帧 correction 的二维范数不超过 0.3 m，并由速度积分自然保持位置连续。log-std、raw-rho和mode logits分别更新，均乘同一 mode gate。

## 8. Loss 与梯度合同

### 8.1 显式 layer weights

`ColifLoss` 不再对任意 forecast layers 盲目求均值。head 输出：

```text
forecast_loss_layer_names
forecast_loss_layer_weights
```

默认权重：

```text
motion proposal layers: 0.5, 0.5
fused base:             1.0
source final:           1.0  (仅full模型)
```

source-disabled 模型完全不输出 detached source-final loss layer。full 模型的 proposal/base gradients 与 source-disabled 模型一致；source-final loss只训练 source corridor modules。

### 8.2 Source regularizers

仅对匹配且 forecast-valid actors、按 detached base mode probability 加权：

- `source_gate_prior`：`KL(Bernoulli(g) || Bernoulli(0.1))`，默认权重 0.02；
- `source_velocity_smoothness`：相邻 `delta_v_source` 的 Smooth-L1，默认权重 0.05；
- `source_residual_magnitude`：`||g * delta_mu_source|| / 4.0`，默认权重 0.01。

原 matched actor×CAV availability BCE 保留，默认权重 0.1；它只训练 availability quality。

### 8.3 梯度裁剪隔离

head 提供三个 parameter groups：

```text
motion_core
fused_base
source_refiner
```

trainer 在 `GradScaler.unscale_` 后分别计算并裁剪三组 norm，默认上限均为 1000。source gradients不再通过一个全模型 norm缩放 motion/fused base gradients。日志记录每组 pre-clip norm。

### 8.4 RNG 隔离

source-disabled 与 full 模型始终构造相同模块。为保持跨帧 base dropout RNG一致，source-disabled 模型仍执行 source stochastic path，但在输出前强制 gate=0且冻结 source parameters。base core始终在全部 v2随机模块之前构造，保证相同 seed 下 base初始化一致。

## 9. 输出与诊断

保留 legacy forecast keys，并新增：

```text
motion_proposal_pred_scores/trajs
base_pred_scores/trajs
source_gate                    [B,Q,K]
source_selection_weights       [B,Q,K,T,L]
source_router_weights          [B,Q,K,T,L+1]
source_velocity_delta          [B,Q,K,T,2]
source_mean_delta              [B,Q,K,T,2]
fused_mean_delta               [B,Q,K,T,2]
```

训练日志至少记录：

- motion/fused/source gradient norms；
- gate mean、P50、P90；
- null mean；
- source selection entropy；
- 1/3/5 s residual displacement mean与P90；
- receiver-only、partner-only、shared matched counts。

official evaluator继续只用公开 `pred_scores/pred_trajs` 计算原指标；诊断字段不改变样本、matching或metric protocol。

## 10. 配置与 checkpoint 合同

v2 仍通过 `forecast.architecture=dual_source_motion_anchored` opt-in，但强制：

```text
dual_source_version: 2
forecast_motion_prior.mode: none
forecast_motion_prior.phase2_mode: none
```

默认关键参数：

```text
corridor_lateral_offset_m: 1.5
corridor_longitudinal_offset_m: 1.5
source_gate_prior: 0.1
source_gate_prior_weight: 0.02
source_velocity_limit_mps: 3.0
source_residual_limit_m: 4.0
source_velocity_smoothness_weight: 0.05
source_residual_magnitude_weight: 0.01
source_dropout: 0.1
```

失败的 v1 A1/A2/A3 checkpoints只保留作诊断，禁止resume到v2。v2只从 Phase-1 tracker checkpoint初始化。A0 legacy `motion_anchored` checkpoints与行为不变。

## 11. 测试合同

### 11.1 单元测试

必须先以失败测试固定以下行为：

1. source-disabled layer集合不包含 detached final，base gradient不再为单层的0.5；
2. full 与 source-disabled 的 motion/fused base gradients在相同输入和RNG下逐元素一致；
3. 初始 motion proposal=fused base=source final；
4. all-source masked、forced-null 与 gate=0 时 final逐元素等于base；
5. 令source context为零且保持任意base query、gate和router weights时，全部source correction严格为零；
6. source joint permutation保持输出不变；
7. 不同mode/time可以选择不同source；
8. corridor sampling位置随proposal trajectory变化，而非固定actor中心；
9. source correction第一步不超过`source_velocity_limit_mps * frame_period_s`；
10. integrated position correction等于velocity correction累计积分并保持有限；
11. final-only loss不回传motion/fused base；base-only loss不回传source；
12. group-wise clipping不让source norm改变base clipping scale；
13. invalid actor、空source与越界corridor均无NaN/Inf；
14. dual-source forecast仍拒绝任何tracking prior mode。

### 11.2 真实数据 smoke

按以下顺序执行：

1. 单卡一批 forward、loss、backward、optimizer；
2. 双卡一批 unused trainable=0、parameter/buffer checksum delta=0；
3. 双卡连续5批，AMP overflow=0，各gradient group有限；
4. source-disabled 与 full 在相同seed下首批 motion/fused base输出和梯度一致；
5. 128帧严格在线评测，gate/null/residual诊断有限且不塌缩。

## 12. 一轮验收门槛

工程测试通过后，使用与历史一轮结果一致的初始化、seed、receiver IDs、batch、训练样本和metric protocol训练一轮。v2只有同时满足以下条件才进入多seed或长训：

1. source-disabled fused-base 不因loss合同而低于自身 motion proposal；
2. full source-final 的 minADE/minFDE 在1/3/5 s均不劣于同checkpoint fused-base超过0.01 m；
3. `minFDE@5s`直接对照历史一轮 `3.427`，不得用不同样本口径替代；
4. source gate mean位于 `[0.02,0.60]`，null未塌缩；
5. 1 s source correction P90小于0.6 m，且5 s correction允许大于1 s而不出现反向时间尺度；
6. partner-only与shared切片至少一个获得正向5 s minADE增益，且另一个退化不超过0.02 m；
7. tracking、actor population和coverage在同协议下保持一致。

若第2、4或5项失败，停止长训并判定source refiner仍不合格；不得用增加epoch掩盖结构失败。

## 13. 非目标

- 不让forecast写回tracking；
- 不引入camera或HD map；
- 不使用future labels构造runtime input或gate target；
- 不通过切换metric protocol、receiver population或diagnostic layer制造性能提升；
- 不把参数量增加本身作为论文贡献；
- 不修改A0 legacy路径或历史checkpoint语义。
