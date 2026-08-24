# CoLIF 项目失败复盘：从 CMP 复现到当前架构否证

> 状态：截至 2026-08-12 的完整记录。当前项目没有完成，代码修改已暂停；本文记录的是已经发生并可追溯的设计、实现、训练和论证失败，而不是对未来方案的成功声明。

> 归档说明（2026-08-24）：文中旧 `opencood/logs/...` 路径是历史证据标识，对应的大体积 checkpoint、smoke/profile 日志和失败运行已在当前模型收敛后清理。结论与数值保留在本文和 Git 历史中，但这些路径不再是可恢复资产，也不得作为当前实验初始化来源。

## 1. 本文的证据规则

本文把结论分成三类：

- **已证实**：由当前代码、配置、checkpoint、完整 test 日志或同权重控制实验直接支持。
- **证据性推断**：多个独立诊断一致支持，但尚未通过单变量实验完全隔离。
- **未证实**：曾经提出过但没有被数据或消融支持，不能写进论文结论。

本项目的证据优先级为：

1. 完整 official test 与同 checkpoint 开关对照；
2. 真实数据的前向、反向、DDP 和诊断实验；
3. 当前代码调用链；
4. 设计文档、论文草稿和历史讨论。

如果历史文档与当前代码或日志冲突，以代码和日志为准。

## 2. 一页结论

项目至今经历了两条结构不同的路线：

1. **source-local/no-fuse 路线**：各 CAV 独立检测，再进行跨来源观测关联和 persistent query 更新。
2. **fused-BEV 路线**：多车点云按 CMP 语义变换到 receiver 坐标，经同一个 CoBEVT/FuseBEVT 融合，再由单一 persistent query bank 跟踪和预测。

第一条路线被放弃，不是因为代码不能训练，而是因为它偏离了最终确认的 CMP fused-BEV 研究设定，并把“跨来源 handover/association”写得过重。第二条路线先后经历了 v1--v11。

截至当前，事实是：

- **运行链已经打通**：双卡 DDP、AMP、checkpoint/resume、strict-online test、GT-free inference 均有真实验证。
- **跟踪从失败提升到可用**：MOTA 从 fused v1 的 `0.1934` 提升到 v7 的 `0.6251` 左右，主要来自 CMP dense detection evidence、measurement proposals 和 predict-correct state update，而不是 future feedback。
- **直接复用 CMP MTR 到 query history 失败**：多个版本的 5 s minFDE 仍为 `10.78--13.23 m`；恢复更多 CMP 模块和更“精确”的 loss 反而没有解决问题。
- **运动锚头把 minFDE 降到约 4.3--4.4 m**：但训练前三步已经有 `4.3250 m`，完整一轮为 `4.4161 m`；这表明成绩主要来自六个手工速度尺度的 best-of-six 覆盖，而不是学到的 residual 或闭环反馈。
- **future feedback 的主张被同权重对照直接否证**：v11 同一 checkpoint，反馈开启时 `minFDE5=4.3946`、`MOTA=0.6320`；关闭时 `minFDE5=4.3403`、`MOTA=0.6419`，coverage、recall 和 IDS 也全部改善。因此当前代码不能声称“未来预测改善跟踪”。
- **“第一帧即可做有效 5 秒预测”不成立**：单个位置不能识别速度。当前模型已改为第一帧创建 actor state，至少两次因果观测后才输出运动预测；age 2--3 的 5 s minFDE 仍约 `12.5 m`。

当前模型最关键的架构缺陷不是门控数值，而是**未来分布没有进入真正决定跟踪的空间路径**：

```text
上一帧 future cue
       ↓
只扰动 query content
       ↓
query decoder

而真正的：
propagated box → Hungarian measurement assignment → observed/miss/TTL
```

在当前实现中，association 已经在 query decoder 之前由传播框和 dense proposals 完成；缺测时 measurement gain 为 0，future 也不能移动 state box；生命周期只看 measurement assignment 和 TTL。因此 future cue 既不能改变匹配，也不能在漏检时提供空间传播，反而会在有新 fused-BEV 观测时扰动已较可靠的 query content。

## 3. 项目时间线与每次失败

### 3.1 2026-08-07：先复现 CMP，暴露基线和协议问题

项目最初从本地 CMP 复现开始，而不是直接实现 CoLIF。

已完成的基线：

| 模型/权重 | test 记录/目标 | 5 s minADE | 5 s minFDE | 说明 |
| --- | ---: | ---: | ---: | --- |
| CMP official aggregation checkpoint | 2,858 / 29,500 | 3.9257 | 7.0533 | 官方感知、tracking cache、aggregation |
| 自训 CMP no-aggregation epoch 5 | 2,858 / 29,500 | 4.4001 | 7.5559 | 当前目标中使用的数值阈值 |
| 自训 CMP no-aggregation epoch 28 | 同上 | 4.9348 | 9.4701 | 后期退化 |
| 自训 CMP no-aggregation epoch 30 | 同上 | 5.2554 | 10.3258 | 后期进一步退化 |

这一步发现的失败不是 CoLIF 的模型失败，而是 CMP 原协议的可比性风险：

- 检测先与 GT 以 IoU 匹配，未匹配 detection 被删除；
- GT object ID 被写入 tracking cache，AB3DMOT track 最终又按该 ID 重命名；
- MTR dataset 通过 GT/pred key 交集保留轨迹，跨 CAV aggregation 也依赖精确 ID。

因此 CMP 的 prediction cache 不是完全无 GT 的在线链。CoLIF 后续 evaluator 使用 `strict_online=true` 和 `runtime_uses_ground_truth=false`，科学上更严格，但也意味着 CMP 与 CoLIF 的数值不是完美同协议比较。

本阶段还暴露了以下工程问题：

- MTR dataset 的 `collate_batch` 对 batch size 大于 1 会静默丢弃样本；已改为显式拒绝，并通过梯度累积获得等效全局 batch。
- 原 DDP test 没有跨进程正确聚合指标；完整 CMP test 因此使用单卡独立运行。
- train/test 含一个完全重复的 294 帧场景；沿用作者 split 时无法把结果解释为严格无泄漏泛化。
- 只保存最近三个 checkpoint 导致 epoch 1/2 在请求评测前被清理；后续以独立快照避免竞态。

本阶段的重要教训是：**训练更久不等于更好**。自训 CMP 在 epoch 5 的 5 秒指标优于 epoch 28--30，而项目协议又明确不使用 test 自动选模；所以“再跑更多 epoch”不能作为架构失败的回答。

### 3.2 2026-08-09：第一条 CoLIF 路线——source-local/no-fuse

#### 当时的设计

```text
每车点云
  → SharedLocalPillarTrunk
  → 每车 local detection/observation queries
  → pose/time alignment
  → 跨来源 association + persistent query bank
  → source memory / lifecycle
  → 6-mode forecast
  → future likelihood 参与下一帧 association/state update
```

这条路线没有使用 FuseBEVT。实现包括 local observation encoder、source-aware query bank、soft/Hungarian association、birth grouping、source memory、GMM forecast 和 FutureAdapter。

#### 为什么这条路线最终失败

1. **研究设定错位**：用户最终确认主模型应尽量复用 CMP，并采用 fused BEV。source-local 路线实际上重做了 cooperative association/tracker，而不是修改 CMP 的 fused pipeline。
2. **主张与融合设定冲突**：一旦采用 receiver-centric fused BEV，模型内部不再存在 ego/partner 两套 actor observation 供 future 去消除跨车匹配歧义。因此 source handover 不能继续作为 fused 主模型的核心机制。
3. **数据支撑不足**：V2V4Real 没有真实网络 acquisition timestamp、显式 occlusion label 或真实 packet delay。`asynchronous/capture-time/occlusion handover` 只能做模拟扰动或离线分桶，不能当数据集天然事实。
4. **标注合同有缺陷**：agent0 常见 `ass_id=-1`，partner 对同一 actor 才有有效 `ass_id`。旧 `_actor_key` 会把同一物理目标拆成 `source_local` 和 `canonical` 两个 actor。
5. **source 点支持稀疏**：早期 248 个 source-frame 抽样中，311 个量程内 label 只有 76 个满足至少 3 点，约 24.4%。这使每车独立 detection/association 的训练监督比 fused 目标更困难。

#### 这条路线为什么一度看起来很好

早期 Phase 1 日志曾报告极低 FDE：epoch 14 的 `minFDE5=0.8022 m`，epoch 7 为 `1.2170 m`。但相应 coverage 只有 `11.1%` 和 `11.0%`。模型只对极容易的一小部分目标输出预测，条件式 minFDE 因而产生假象。

当完整 Phase 5 把 coverage 提升到 `59.17%` 后：

- `minADE5=5.4771 m`；
- `minFDE5=8.1841 m`；
- unmatched-active rate=`96.73%`。

这揭示了项目早期的评价错误：**没有把 minFDE 与 forecast coverage、target denominator 和 tracking recall 绑定报告**。旧 no-fuse 的 3,986 帧、8,992 target actor-frames，与后续 fused receiver0 的 1,993 帧、约 32,408 target actor-frames 也不同，不能直接称为严格更好。

#### 可保留的工程资产

虽然主路线被放弃，这一阶段留下了可复用的 sequence index、runtime/target 分离、GT anti-leak、tensor-only recurrent state、DDP clip unroll、checkpoint/resume 和 strict-online evaluator 框架。

### 3.3 2026-08-10：切换为 fused-BEV persistent query

最终主链改为：

```text
receiver-aligned 多车点云
  → CMP PointPillar/CoBEVT
  → FuseBEVT fused feature
  → 单一 persistent query decoder
  → current box/existence/velocity/lifecycle
  → multimodal forecast
  → bounded future cue 写入下一帧 query content
```

这里有两个早期误判：

- 曾认为仓库 `point_pillar_cobevt_motr.py` 可以直接复用；实际仓库缺顶层 `MOTR/` 和对应 YAML，原文件直接 import 失败。后续只能固定外部依赖或移植最小组件，不能声称“CMP 已有现成可跑的 MOTR”。
- 曾把论文训练描述成三阶段：tracker、forecast、joint feedback fine-tuning。真实实验表明第三阶段不是必要的论文贡献，且旧 Phase 3 的解冻策略造成结构性崩塌。当前训练器已经只接受 Phase 1/2。

### 3.4 fused v1：query tracker 稀疏，三阶段闭环崩塌

#### 结果

- Phase 1 训练 20 epoch，末轮 MOTA=`0.1934`、recall=`0.3487`。
- Phase 2 的 5 s minFDE 依次为：`11.1361, 10.0142, 10.0918, 12.0699, 12.9909, 13.8127, 12.5830, 11.7916, 11.2964, 11.8827 m`。
- Phase 3 修复 DDP 后的 epoch 0/1：
  - FDE=`14.1183 / 13.5075 m`；
  - coverage=`15.85% / 5.05%`；
  - MOTA=`-0.0165 / 0.0447`。

#### 已证实根因

1. query existence 没有连续读取同一个 CoBEVT 实例已经预训练好的 dense detection heatmap，导致 query births/active population 极稀疏。
2. Phase 3 解冻 tracker、改变学习率和闭环输入后，tracking/coverage 发生结构性崩塌；这不是“还没训练完”。
3. actor-local 与 receiver-frame 坐标合同一度混用，预测 target 与输出旋转不一致。

#### 工程故障

- 条件 MTR 分支整体转 SyncBatchNorm 后，不同 rank collective 次数不同，forward 卡住。
- 一个 DDP backward 前连续调用 wrapper 多次导致 reduction 语义错误，最终改为模型内部完成整个 causal clip unroll。
- Phase 3 首次命令失败曾只显示 `ChildFailedError`，真正错误需要从 rank 日志定位；之后修复 checkpoint/phase 合同与 DDP reduction。

### 3.5 fused v2：接入 dense evidence 后 tracker 改善，MTR 仍失败

#### 修复

- query existence 在 decoded center 采样同一 CoBEVT 的 `cls_head(fused_bev)`；没有引入第二个 CMP 模型。
- 修正 actor-local forecast target/output 旋转。
- 调整 birth/keep threshold、active cap 和 identity release。
- 收敛为两阶段：Phase 1 tracker；Phase 2 冻结 tracker、训练 forecast/adapter。

#### 结果

- Phase 1 最好 MOTA=`0.4366`（epoch 16），末轮用于 Phase 2 的 checkpoint 只有 `0.4033`。
- Phase 2 epoch 0--4 的 5 s minFDE：
  `12.9098, 12.6569, 11.9540, 12.6820, 11.7361 m`。
- 最好 epoch 4 的 age>=4 FDE 仍为 `11.5874 m`。

这证明 tracking coverage 改善并没有自动解决 prediction。继续训练五轮后性能仍大幅落后 CMP，不支持“只是没收敛”的解释。

### 3.6 fused v3：恢复 11×22 MTR 输入合同，仍只有有限改善

#### 发现的合同错误

此前只把当前 `query_feat/current_box` 直接交给 MTR decoder，实际上 CMP checkpoint 是在 `MTREncoder` 编码的 11 帧、22 属性 trajectory/context 上训练的。还存在：

- 历史 box 跨 receiver frame 未正确处理；
- 训练从空状态只看 4 帧，而 test 长序列持续递归；
- loss 曾在 NMS 后的 6 模态上计算，而不是原始 64 intentions；
- box dimension 次序和正模态选择不一致；
- batch 中仅一个 actor 时，forecast BatchNorm running statistics 不可靠。

#### 修复后结果

完整双卡一轮：

- `minADE5=5.3127 m`；
- `minFDE5=10.7844 m`；
- coverage=`74.84%`；
- MOTA=`0.4581`；
- age 1 FDE=`22.91 m`；age 2--3 FDE=`18.49 m`。

相比 v2 最好 FDE 改善约 `0.95 m`，说明输入合同确实是一个根因；但修复完整合同后仍远落后 CMP，说明它不是唯一根因。

#### AMP 隐蔽断图

在同一个 autocast context 内，7 帧 `no_grad` burn-in 会缓存 FutureAdapter 的无梯度 FP16 权重，后续有梯度帧复用该缓存，造成 cue MLP/gate 参数 `grad=None`。修复为 FP32 adapter island 后，真实双卡 recurrent BPTT 能看到非零梯度。这个修复只证明反馈路径“可反传”，不证明它有用。

### 3.7 fused v4--v7：真正的 tracking 瓶颈来自 measurement correction

一系列单变量诊断表明，persistent query tracking 的主要改善来自当前帧 fused detection measurement，而不是 future：

- dense measurement proposal/correction 前：MOTA=`0.0271`、recall=`0.7295`、IDS=`3574`；
- correction 后：MOTA=`0.4746`、recall=`0.8062`、IDS=`1039`。

v5 Phase 1 epoch 0 达到 MOTA=`0.6133`、recall=`0.8393`，但 active query 过多，生命周期/cap 不稳定，epoch 1 退到 MOTA=`0.5063`。

v7 predict-correct tracker：

- epoch 0：MOTA=`0.6225`、recall=`0.8310`、IDS=`664`；
- epoch 1：MOTA=`0.6251`、recall=`0.7926`、IDS=`701`。

这是当前可用 tracker 的来源。它依靠同一 CoBEVT 的 dense heatmap/proposals、propagated box 与 measurement update；不能把这部分提升归因给未来预测。

v7 的 query-MTR recurrent Phase 2 完整一轮仍为：

- `minFDE5=11.9681 m`；
- coverage=`81.02%`；
- MOTA=`0.6394`；
- age 1 / age 2--3 FDE=`29.63 / 28.08 m`。

更好的 tracker 提升了 coverage，但原预测头仍没有恢复到 CMP 水平。

### 3.8 fused v8--v9：越严格复用 CMP MTR，结果反而更差

v8 恢复 CMP 多层 GMM loss、64 intention 训练合同和更多 decoder 细节，完整一轮结果：

- `minADE5=7.0475 m`；
- `minFDE5=13.2302 m`；
- coverage=`79.63%`；
- MOTA=`0.6252`。

附加控制：

- exact legacy contract FDE=`14.4895 m`；
- fixed measurement gain=1 FDE=`14.6259 m`；
- motion-line diagnostic FDE=`13.0687 m`。

训练日志中 GMM/gradient 曾非常大：首 batch GMM=`2755.33`、gradient norm=`597442`。这不是简单的 AMP NaN；完整训练数值有限，但说明 checkpoint 所在表示分布与 online query history 严重错配。

GT-history oracle 进一步定位了问题：完整 11 帧 GT history 下，原 CMP MTR 的 age 11 FDE 可到约 `4.717 m`；而 persistent query history 下同一类 decoder 长期在 `10--15 m`。因此结论是：

> CMP MTR 的权重不是一个可直接迁移到 persistent-query online history 的通用预测器。它依赖 AB3DMOT/oracle-like cache 的轨迹平滑、身份和固定历史分布。

v9 去除了第一帧 actor 对 MTR context 的污染并增加 maturity/heading 处理，但没有恢复 MTR。此时继续修补输入算子已经不能解释主要差距。

### 3.9 fused v10：多尺度运动锚成功降低 minFDE，但未证明学习有效

诊断发现，Phase 1 online history 本身包含足够的运动信息：

- stationary FDE 约 `15.16 m`；
- last-two online constant velocity 约 `10.02 m`；
- robust history median constant velocity 约 `8.00 m`；
- 六个速度尺度 `[0, 0.5, 0.75, 1.0, 1.25, 1.5]` 的 best-of-six FDE 约 `4.67 m`。

因此新增 motion-anchored head：稳健历史速度提供 6 个基础轨迹，神经网络只学 residual、mode score、uncertainty 和 actor interaction。

结果：

| run | 训练量 | coverage | 5 s minADE | 5 s minFDE | MOTA |
| --- | ---: | ---: | ---: | ---: | ---: |
| v10 DDP smoke checkpoint | 3 optimizer steps | 78.76% | 2.4248 | **4.3250** | 0.6293 |
| v10 full epoch | 855 updates | 78.63% | 2.8861 | **4.4161** | 0.6266 |

完整一轮反而没有优于三步 smoke。mode loss 的 19 个记录均值为 `1.7827`，非常接近均匀六模态的 `ln(6)=1.7918`。因此当前低 minFDE 的可靠解释是：

- 六个运动锚覆盖了真实速度误差；
- minFDE 只取六条中的最好一条；
- 学到的 mode ranking、概率校准和 residual 改善尚未被证明。

这不是人为修改 evaluator，也不是 GT 泄漏；作为 K=6 的 minFDE 候选集合是合法的。但它只能证明**多模态集合覆盖**，不能证明模型能选出正确模式，更不能证明 future feedback 改进 tracking。

第一次 full run 还因 `workers=4`、文件描述符上限 1024，在 batch 1 报 `Too many open files`。改为 `workers=2` 后完成。这是数据加载资源故障，不是模型根因。

### 3.10 fused v11：同权重开关实验否证 future feedback

v11 从 Phase 1 tracker 训练 motion-anchored head 和 bounded FutureAdapter。完整双卡一轮完成 855 updates，DDP 参数/缓冲同步检查为 0，运行时不使用 GT。

最关键实验使用**同一个 checkpoint、同一套 test、只开关 feedback**：

| 指标 | feedback ON | feedback OFF | 哪个更好 |
| --- | ---: | ---: | --- |
| 5 s minFDE | 4.3946 | **4.3403** | OFF |
| 5 s minADE | 2.8717 | **2.8519** | OFF |
| coverage | 78.88% | **79.94%** | OFF |
| MOTA | 0.6320 | **0.6419** | OFF |
| recall | 79.49% | **80.53%** | OFF |
| IDS | 685 | **670** | OFF |

mode loss 均值仍为 `1.7826`，接近均匀模式。

这项对照直接覆盖以下说法：

- 不能说 current future cue 改善 tracking；
- 不能把 v10/v11 相对 CMP 的低 minFDE 归因于 feedback；
- 不能说 gate 学会了“何时相信未来”；
- 不能用“再训练更多 epoch 可能变好”替代已经出现的同权重反事实证据。

## 4. 为什么之前的模型这么差：根因矩阵

| 根因 | 证据 | 影响阶段 | 状态 |
| --- | --- | --- | --- |
| 只看条件式 minFDE，忽略 coverage | no-fuse FDE 0.8--1.2 m 时 coverage 约 11%；coverage 59% 后 FDE 8.18 m | source-local | 已证实并修正评价口径 |
| source-local 路线偏离最终 fused 设定 | 旧主链无 FuseBEVT，重做跨来源检测/association | source-local | 路线已放弃 |
| canonical identity 构造不一致 | agent0 `ass_id=-1` 与 partner canonical key 被拆开 | source-local/data | 已审计；旧结果受影响 |
| query 未利用预训练 dense existence evidence | fused v1 active/recall 极低；接入后 tracker 显著改善 | fused v1 | 已证实并修复 |
| Phase 3 解冻造成 tracker/coverage collapse | coverage 15.85%→5.05%，MOTA 近 0 | fused v1 | 已废除第三阶段 |
| actor/receiver 坐标和历史合同错误 | 修复 11×22、旋转、历史后 FDE 改善约 0.95 m | v1--v3 | 已证实并修复 |
| 训练短 clip 与 test 长递归分布不一致 | v2 训练只见 age<=4，test 大量 mature actor | v2 | 后续用 11 帧 lifecycle 修正 |
| CMP MTR 权重与 query history 表示分布错配 | query history FDE 10--15 m；GT 11 帧 oracle age11 约 4.72 m | v2--v9 | 已证实，直接复用路线失败 |
| 第一帧无法识别运动 | age1 FDE 22--32 m；单位置在理论上不确定速度 | v2--v9 | 已禁用 age1 forecast |
| tracker 缺少显式 measurement correction | correction 前/后 MOTA 0.027→0.475 | v4 | 已证实并修复 |
| FutureAdapter AMP burn-in 断图 | gate/cue 参数 grad=None；FP32 island 后非零 | v3 | 已修复，但只解决可训练性 |
| future 只改 content，不改 association/space/lifecycle | 当前调用链直接证明；OFF 对照全面更好 | v10--v11 | **核心未解决** |
| 六模式排序几乎未学习 | mode loss 均值约 1.783≈ln6；3步优于/接近一轮 | v10--v11 | **核心未解决** |
| 阶段 2 冻结 tracker | tracking loss只能训练 forecast/adapter去扰动固定 decoder，无法共同学习关联策略 | v2--v11 | **架构限制，未解决** |

## 5. 当前 future 为什么不能改善 tracking

当前调用链是：

```text
state.ref_box + state.velocity
    → transform_and_propagate_boxes
    → dense CMP proposals
    → distance/Hungarian _assign_measurements
    → decoder_reference + measurement_observed_mask

state.future_cue
    → FutureAdapter
    → query_prior content
    → query decoder

最后：
measurement_observed_mask + existence score
    → QueryLifecycle observed/miss/TTL
```

由代码可直接得到五个结论。

### 5.1 future 不参与匹配

`_assign_measurements()` 只使用 `propagated_box`、dense proposal box 和欧氏距离/Hungarian。它在 query decoder 之前完成。FutureAdapter 生成的 `query_prior` 不进入 matching cost、gate 或多假设 association。

### 5.2 future 不改变空间 reference

`_prepare_prior()` 中 reference box 只由上一帧 `ref_box + velocity + receiver pose/time` 传播。FutureAdapter 的明确定义是只修改 query content，不移动 reference。

### 5.3 漏检时 future 不能桥接位置

`MeasurementStateUpdate` 的 gain 会乘 `observed_mask`。没有 measurement 时 gain=0，state box 等于普通 propagated box。即使 future cue 改变 decoder box，缺测时也不会写入 state geometry。

### 5.4 future 不改变生命周期证据

`QueryLifecycle` 的 observed 由 `previous_active & measurement_observed_mask & score` 决定，survival 由 observed 或固定 miss TTL 决定。Future cue 没有显式改变 observed/miss/TTL 的空间证据。

### 5.5 当前 future cue 甚至不是显式未来分布摘要

motion-anchored head 中：

```python
mode_tokens = future_token(mode_feature)
cue = sum_k softmax(logit_k) * mode_tokens[k]
```

该 cue 与 forecast head 共享 mode feature，但并没有显式编码预测均值、协方差或短时 posterior。把它称为完整 `future belief` 过强；更准确的名称是 `forecast-head latent cue`。

所以，当前架构对“future 改进 tracking”没有充分理论路径。它只是在新 fused-BEV measurement 已经可用时给 query content 加一个小扰动；同权重 OFF 更好与这一结构一致。

## 6. 文献真正支持什么，不支持什么

以下文献说明“未来可以帮助跟踪”并非错误命题，但它们都有当前 CoLIF 缺失的显式机制：

- [PF-Track (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Pang_Standing_Between_Past_and_Future_Spatio-Temporal_Modeling_for_Multi-Camera_3D_CVPR_2023_paper.html) 用预测轨迹在遮挡/漏检期间空间传播 track query，并支持 re-association。future 真正改变了 tracking 的空间状态。
- [ForeSight (ICCV 2025)](https://arxiv.org/abs/2508.07089) 维护多假设 forecast memory，让 forecast 进入后续 detection 的空间推理，而不只是无条件叠加一个 latent。
- [StreamMOTP (ACCV 2024)](https://openaccess.thecvf.com/content/ACCV2024/html/Zhuang_StreamMOTP_Streaming_and_Unified_Framework_for_Joint_3D_Multi-Object_Tracking_ACCV_2024_paper.html) 共享 object/track context 并有显式 association matrix；其结果说明联合建模可能带来有限 tracking 增益，但不能证明任意 future latent 都会改善 tracking。

HPNet 使用历史 prediction 维持 forecasting consistency，主要是 prediction→prediction，不是 future→tracking 的直接证据。

因此，future 改善 tracking 至少需要满足：

1. 短时预测分布进入空间 gating/association，或直接传播缺测 actor；
2. 保留多模态与 uncertainty，避免把长期 5 秒平均未来当成唯一位置；
3. 新 measurement 能对 prediction prior 做校正；
4. 用同权重 on/off、遮挡/漏检分桶和 IDS/MOTA 证明因果增益。

当前 CoLIF 只部分满足第 3 条，未满足第 1 条，因此现有负结果不是意外。

## 7. 已经被推翻或必须降级的论文主张

以下表述目前不能使用：

- “从第一次协同观测起即可做有效长时预测。”
- “future belief 能消除 fused-BEV 中跨 CAV 竞争观测的歧义。”
- “future feedback 改善 tracking、association 或 lifecycle。”
- “当前低 minFDE 主要来自 learned recurrent future reasoning。”
- “完整复用 CMP MTR 即可适配 persistent query history。”
- “V2V4Real 提供真实异步、真实 capture time 或显式遮挡 handover。”
- “三阶段训练是模型成立所必需的。”

当前最多可以诚实表述为：

- persistent actor state 在第一次证据时创建；
- 至少两次因果观测后输出多模态 motion-anchored forecast；
- 同一 CoBEVT 的 fused measurement 持续校正 persistent state；
- 六个速度尺度使 best-of-six minFDE 获得较强覆盖；
- bounded forecast latent recurrence 已实现、可反传，但当前实验显示它有害，尚不能成为贡献。

## 8. 当前成绩应如何解释

### 已成立

- strict-online runtime 不读取 GT/`ass_id`/future target。
- 双卡完整一轮、同步、保存、恢复和完整 test 链可运行。
- fused tracker 已从不可用提升到 MOTA 约 0.62--0.64。
- motion-anchored K=6 在当前 evaluator 上达到 5 s minFDE 约 4.3--4.4 m，数值低于已复现 CMP no-aggregation epoch 5 的 7.5559 m。
- age1 预测已经停止，不再报告不可辨识的伪“即时运动预测”。

### 没有成立

- CoLIF 的核心 future→tracking 机制没有正增益。
- 学习部分没有被证明优于未训练/少量训练的 motion anchors。
- top-1、NLL、mode calibration 和 safety-relevant mode selection 尚未证明；只看 minFDE 会掩盖均匀模式概率。
- 与 CMP 仍有 protocol 差异：CMP cache 带 GT-ID oracle，CoLIF strict online；不能只用一个 FDE 数字宣称全面优于 CMP。
- age2--3 的 5 秒预测仍差，v11 feedback OFF 为 `12.5007 m`。

## 9. 完整实验账本

| 阶段 | 关键日志/结果文件 | 结论 |
| --- | --- | --- |
| CMP official full test | `/home/qkb/Project/data/V2V4Real/derived/CMP/runs/mtr_full_20260807/eval/log_eval_20260807-120255.txt` | official aggregation FDE5=7.0533 |
| CMP 自训 no-agg epoch5 | `/home/qkb/Project/data/V2V4Real/derived/CMP/runs/mtr_noagg_epoch5_test_20260807/eval/log_eval_20260807-145046.txt` | FDE5=7.5559；当前数值门槛 |
| 旧 source-local Phase1 | `opencood/logs/colif_phase1/metrics_test.jsonl` | 极低 FDE 来自约 11% coverage |
| 旧 source-local full | `opencood/logs/colif_phase5_full_verified_seed20/metrics_test.jsonl` | coverage 59.17%，FDE5=8.1841 |
| fused v1 tracker | `opencood/logs/colif_fused_phase1/metrics_test.jsonl` | 末轮 MOTA=0.1934 |
| fused v1 Phase2 | `opencood/logs/colif_fused_phase2/metrics_test.jsonl` | FDE 长期约 10--14 m |
| fused v1 Phase3 | `opencood/logs/colif_fused_phase3_fixed_20260811/metrics_test.jsonl` | tracking/coverage collapse |
| fused v2 tracker | `opencood/logs/colif_fused_v2_phase1/metrics_test.jsonl` | 最好 MOTA=0.4366 |
| fused v2 forecast | `opencood/logs/colif_fused_v2_phase2/metrics_test.jsonl` | 最好 FDE5=11.7361 |
| fused v3 MTR contract | `opencood/logs/colif_fused_v3_phase2_epoch1_final_20260812/metrics_test.jsonl` | FDE5=10.7844 |
| fused v7 tracker | `opencood/logs/colif_fused_v7_phase1_predict_correct_2epoch_20260812/metrics_test.jsonl` | MOTA≈0.625 |
| fused v7 query MTR | `opencood/logs/colif_fused_v7_phase2_query_mtr_recurrent_b1w2_2epoch_20260812/metrics_test.jsonl` | FDE5=11.9681 |
| fused v8 exact CMP loss | `opencood/logs/colif_fused_v8_phase2_cmp_multilayer_1epoch_20260812/metrics_test.jsonl` | FDE5=13.2302 |
| motion anchor diagnostic | `opencood/logs/colif_fused_v8_phase2_cmp_multilayer_1epoch_20260812/diagnostic_multiscale_motion_anchors_full.json` | 未训练 best-of-six≈4.67 m |
| v10 3-step smoke test | `opencood/logs/colif_fused_v10_motion_anchored_ddp_smoke_20260812/metrics_test_full_strict_online.json` | FDE5=4.3250 |
| v10 full epoch | `opencood/logs/colif_fused_v10_motion_anchored_full_w2_1epoch_20260812/metrics_test.jsonl` | FDE5=4.4161 |
| v11 feedback ON | `opencood/logs/colif_fused_v11_motion_feedback_full_1epoch_20260812/metrics_test.jsonl` | FDE5=4.3946，MOTA=0.6320 |
| v11 same checkpoint OFF | `opencood/logs/colif_fused_v11_motion_feedback_full_1epoch_20260812/metrics_test_feedback_disabled.jsonl` | minFDE、MOTA、coverage、recall、IDS 优于 ON；fragments 与 FP 略差，不能概括为全部指标更好 |

## 10. 工程失败与修复账本

这些问题影响了可信度或训练效率，但不能被冒充为模型性能根因：

| 故障 | 真实表现 | 处理 |
| --- | --- | --- |
| CMP MOTR 依赖缺失 | `ModuleNotFoundError: MOTR` | 固定/移植最小依赖；不再声称原型可直接跑 |
| DDP 多次 wrapper forward | reduction/graph 错误 | 一个模型 forward 内完成 causal clip |
| 条件 SyncBN | ranks collective 次数不同，可能卡死 | 条件 MTR 保持普通 BN，边界同步 buffer |
| 无 target rank 伪分母 | 分布式 loss 统计失真 | all-reduce raw numerator/count，空 rank 贡献 0/0 |
| AMP burn-in 权重缓存 | FutureAdapter 部分参数 grad=None | adapter 使用 FP32 island |
| checkpoint 跨阶段缺键误判 | BN counter/未训练模块加载混乱 | 只加载源阶段实际训练模块并严格校验 |
| 冻结 GRU eval backward | cuDNN backward 报错 | 参数冻结但 recurrent BPTT 时保持 train mode |
| workers=4 文件描述符耗尽 | full run batch1 `Too many open files` | workers=2 完成；未掩盖模型问题 |
| 旧 Phase3 `ChildFailedError` | torchrun 只显示子进程失败 | 追踪 rank 日志修复真实错误，不跳过 DDP 检查 |
| train/test 目标合同变化 | 不同阶段 target denominator 差异巨大 | 当前统一 receiver0 strict-online evaluator，并在表中保留协议说明 |

## 11. 当前停止点

项目现在不应继续用以下方式“修指标”：

- 延长同一失败架构的 epoch；
- 用 test 选择 best checkpoint；
- 降低 target coverage、提高筛选阈值或只报 matched easy actors；
- 使用 GT identity、GT current state 或 GT history进入 runtime；
- 仅调整 gate 大小、dropout 或 loss 权重后宣称 future 有效；
- 用 best-of-six minFDE 代替 top-1/calibration/feedback 因果实验。

下一次继续代码前，必须先重新回答并验证：

1. future distribution 具体进入哪一个 tracking 决策：spatial gating、association cost、missing-state propagation，还是 lifecycle survival？
2. 为什么使用短时 distribution，而不是把 5 秒多模态 latent 无条件写回 query？
3. prediction prior 与 fused-BEV measurement 如何做 uncertainty-aware correction？
4. 哪一个同权重实验能证明 future 对 IDS/MOTA/miss recovery 有正增益？
5. 如何在不使用 GT runtime 和不降低 coverage 的前提下证明收益？

在这些问题没有形成新的、可证伪架构前，当前 `recurrent_bptt` 不应作为论文主模型默认结论。现有代码与日志应保留为否证证据，而不是继续通过小补丁掩盖架构问题。

## 12. 最终责任归因

项目之前表现差，不是一个单一 bug，也不能归咎于“训练没跑够”。责任链是：

1. **最初问题定义摇摆**：source-local handover 与 fused-BEV persistent query 是两种不同论文路线，早期没有及时固定。
2. **过早实现完整系统**：在没有先证明 tracking state、MTR 输入分布和 future→tracking 因果路径之前，就加入 source memory、association、GMM、lifecycle、feedback 和多阶段训练，导致根因混在一起。
3. **错误相信模块接口等于语义兼容**：MTR tensor shape/权重能加载，不等于 persistent query history 与 CMP tracking cache 分布一致。
4. **评价口径不完整**：早期低 FDE 没有和 coverage、target denominator、age 分桶一起解释。
5. **把可反传误当成有用**：FutureAdapter 有梯度、gate 非零、DDP 同步，只能证明代码链存在；同权重关闭反馈更好，证明这条链当前没有产生所主张的效果。
6. **当前核心机制缺少空间因果路径**：future 没有进入 association、缺测传播或 survival，是论文主张失败的根本架构原因。

因此，当前最诚实的项目结论不是“CoLIF 已经成功”，而是：

> strict-online fused persistent tracking 和多模态 motion-anchor forecasting 已经建立了可运行基座；CMP MTR 直接迁移失败，现有 latent feedback 被对照实验否证。若论文仍要主张 future improves tracking，必须重构 future 到空间关联/缺测传播的决策路径，并重新做同权重、同 coverage、GT-free 的因果验证。

## 13. 2026-08-12 架构再审计：HPNet、缺测语义与理论路径

### 13.1 HPNet 不能证明 future improves tracking

本地正式论文原文显示，HPNet 的 Historical Prediction Attention 在已知 agent、time、mode 对应关系下，让当前 prediction embedding 关注同一 agent、同一 mode 的历史 prediction embeddings。它的目标是改善**当前预测的准确性、稳定性和有效历史窗口**；输入中的 agent identity/track 已经给定，论文不执行 detection、runtime association 或 lifecycle，也不报告 MOTA/IDS。

因此 HPNet 支持的是：

```text
historical prediction embeddings → current prediction
```

而不是：

```text
future prediction → detection/association/tracking
```

当前 CoLIF 也没有忠实实现 HPNet 式 prediction recurrence：它没有对齐相邻时刻预测的重叠时间段，也没有保持 per-mode historical prediction attention，而是先把所有 mode token 按概率加权成一个 cue。该操作会提前折叠多模态信息。

真正更接近 future→perception/tracking 的参考是 ForeSight、PF-Track 一类显式空间传播方法。StreamMOTP 同时展示了另一条较弱路径：prediction loss 通过共享特征和联合训练间接改善 association representation。当前 CoLIF 的 Phase 2 冻结整个 tracker，因此这种 joint-representation 路径也被切断。

### 13.2 `measurement_observed_mask=False` 的真实含义

当前代码中的“缺测”只表示 query 没有得到一个最终分配的 dense proposal。一个 proposal 必须依次通过：

1. CoBEVT dense heatmap score threshold；
2. 3×3 local-maximum selection；
3. 2 m center suppression；
4. pre-top-k / 最多 48 proposals；
5. active queries 与 proposals 的纯中心距离 Hungarian；
6. 4 m match gate；
7. one-to-one proposal ownership。

所以缺测可能由下列情况产生：

- actor 确实离开联合 LiDAR 范围；
- actor 仍在范围内但点很少、被遮挡或 detector 漏检；
- proposal score 低于 0.05；
- peak 被相邻 peak 抑制；
- proposal 数量上限截断；
- 恒速传播框偏离超过 4 m；
- 拥挤场景中 proposal 被另一个 query 分走；
- Hungarian 在没有显式 null/dummy 的情况下先强制全局匹配，再于事后 gate 拒绝。

它不是 visibility label，也不是 point-support label。当前 evaluator 没有把缺测按上述原因分桶，因此还不知道 V2V4Real 中每类事件的实际占比。

### 13.3 “只在缺测时使用 future”同样是错误修补

原因有三项：

1. 缺测标志是在 proposal selection 和 Hungarian **完成以后**产生，future 此时已经错过了影响本帧 association 的机会。
2. 缺测时 `MeasurementStateUpdate` 把 gain 乘为 0，state box 仍是普通 `box+velocity` 传播；future-influenced decoder box 不会写回空间状态。
3. 缺测混合了 out-of-range、detector miss、association failure 和 proposal competition，不能用同一策略处理。

正确原则不是“有测量完全不用 future、无测量才用 future”，而是标准 predictive-filtering 结构：future 在测量到来前构成 prior；强且一致的 measurement 应主导 posterior，弱、歧义或缺失 measurement 时 prior 权重自然上升。

### 13.4 future 改善 tracking 的理论条件

令 `X_t` 为 actor state，`Z_t` 为当前 fused-BEV measurement，则理论上应实现：

```text
p(X_t | Z_1:t) ∝ p(Z_t | X_t) · p(X_t | Z_1:t-1)
```

forecast 对 tracking 有用的部分不是任意 5 秒 latent，而是上一时刻预测分布在当前时刻的短期边缘分布。它必须在 observation assignment 前形成多模态、带 covariance 的 spatial prior，例如：

```text
C_ij = -log Σ_k π_i^k N(z_j ; μ_i,k(t), Σ_i,k(t)+R_j)
       + appearance/query compatibility
```

并包含显式 null/newborn 项。若无 measurement，posterior 才退化为 prediction prior，同时 existence 应根据“预计可见但漏检”与“本来就在感知范围外”采用不同更新。

这只说明 future **可能**改善 tracking，不保证一定改善。V2V4Real 主协议为同步 10 Hz，邻帧只有 0.1 s，当前 4 m gate 下恒速模型已经可能足够；5 秒意图分布对下一帧 association 的边际价值可能很小。是否存在足量 hard miss/ambiguity 事件必须先做事件 census，不能从理论直接假定正收益。

### 13.5 当前代码的新增架构判定

- runtime matching 是 `no_grad`、纯中心距离、无 future/appearance/uncertainty/null 的 Hungarian；future 无法学习改变匹配决策。
- training criterion 对已有 slot 通过 GT actor ID 延续监督，而 runtime 通过 proposal Hungarian 维持身份；训练与推理的 association mechanism 不同。
- missed query 的 decoded box 和 dense existence evidence都不能重置 observed 状态，因为 lifecycle 硬要求 `measurement_observed_mask=True`。
- FutureAdapter 对所有 forecast-valid query 生效，主要扰动已有强 measurement 的 query；而真正缺测 query 的几何状态又不接受该输出。
- `future_uncertainty` 只是 mode entropy，没有使用预测的 `log_std/rho`；`future_cue` 也不是由预测均值/协方差显式编码。因此“future belief”命名缺乏实现支撑。
- Phase 2 冻结 CoBEVT、query decoder、temporal memory 和 measurement update；prediction loss不能通过共享参数改善 tracking representation。
- hard Hungarian、threshold 和 TTL 使 tracking 分支对 future 的关键决策不可微。即使 gate 只有约 0.0045，小 content 扰动也可能跨过 existence threshold并造成离散生命周期分叉。

最终判定：**当前不是“future 理论上不能改善 tracking”，而是当前 CoLIF 没有实现使它能够改善 tracking 的理论路径。模型架构确实有问题；仅把 feedback 限制到缺测帧仍是错误方向。**

## 14. 2026-08-12 显式空间先验的实现与最终否证

在上述审计后，代码实现并真实验证了六种同权重模式：`none`、
`association`、`propagation_detached`、`propagation_bptt`、`detached`、
`recurrent_bptt`。新实现包含短时五步多模态均值/方差、receiver-pose
变换、CV fallback、pre-gated Hungarian private null、measurement correction
和跨帧 BPTT。它通过单元测试、真实 11 帧单卡 forward/backward、双卡同步
smoke，以及一轮 1710 clips/855 updates 的完整训练；因此结论不是代码没跑通。

评测同时修复了一个可复现性问题：test loader 原先仍随机
`shuffle_points`，导致相同 checkpoint 独立运行并非完全相同输入。现在
train 保留随机 shuffle，test 保留原始点序；`none` 重复完整评测的全部
指标逐位一致。

在 v13 epoch-0 checkpoint、official test 1993 帧、strict-online、无 GT
runtime、确定性点云顺序下：

| 模式 | MOTA | IDS | recall | coverage | minFDE 5 s |
|---|---:|---:|---:|---:|---:|
| `none` | 0.631619 | 599 | 0.794234 | 0.788663 | 4.399249 |
| 单状态 propagation | 0.639193 | 607 | 0.794051 | 0.788231 | 4.340486 |
| 双状态 gap imputation | 0.628718 | 648 | 0.793532 | 0.788015 | 4.371122 |

单状态版本虽降低 FP/FDE，但增加 IDS、略降 recall/coverage；进一步把
association filter state 与 imputed output state 分开后，MOTA、IDS、recall、
coverage 仍全面差于 `none`。更早的 forecast-association 与联合传播消融也
同样有负增益。按预先规定的强验收条件，future-improves-tracking 主张被
拒绝，不能通过挑单项 FDE 或调整 test threshold 包装为成功。

最终代码决策：默认 `future_prior_mode=none`；所有 future→tracking 分支
只作为可复现负消融保留。当前可成立的是 strict-online fused persistent
query tracking 与 causal variable-history motion forecasting，不能成立的是
“预测反过来改善 tracking”。

## 15. 2026-08-13 从“预测反馈”转向“任务条件协同融合”

### 15.1 新架构问题如何产生

future→tracking 被完整消融否证后，继续围绕 gate、TTL 或 missing frame 修补已没有研究依据。重新检查当前最强代码发现：

- current detection/tracking 只需要一张 FuseBEVT 的 receiver-centric feature；
- motion forecast 也只读该张压缩后的统一 feature 与 query history；
- 每个 actor 在不同 CAV 上的局部证据在 FuseBEVT 后不再显式可分；
- 早期预测实现把 persistent query 或 robust median velocity 当成完整时序表示；
- additive `actor+mode` 再接共享线性 score 时，actor 项对所有 mode 是相同常数，softmax 后抵消。

这使研究问题从“future 能否帮助 tracking”转为：**为当前状态估计学习的统一协同融合，是否足以服务运动预测；还是应保留 dense fused measurement，同时让每个 actor 从 aligned source features 中选择预测相关证据。**

对应代码加入 geometry-conditioned actor-source patch attention、有序 variable-history encoder、actor-conditioned mode interaction；tracking 分支保持冻结，保证预测消融不改变 actor set。

### 15.2 公平性漏洞及修复

第一轮 source/no-source 对照存在隐藏随机性漏洞：PyTorch DataLoader iterator 的 base seed 来自全局 torch RNG，而不同模型结构在构造参数时消耗的 RNG 数量不同。因此即使 CLI seed 相同，worker 内 point shuffle 仍可能不同；旧的毫米级差异不能采信。

修复包括：

1. DataLoader 使用独立 `torch.Generator(seed + rank)`；
2. worker 用 `torch.initial_seed() % 2**32` 同步 Python/NumPy RNG；
3. geometry/source-slot 两个分支始终实例化相同参数结构，未启用分支只冻结；
4. 固定 seed smoke 验证 source ON/OFF 的 active query、birth、box、aux box 与 tracking loss逐 batch完全一致。

修复后四组完整实验均使用同一 Phase-1 checkpoint、1710 clips、855 updates、seed 20、official test 1993 frames。四组 MOTA=`0.633329`、coverage=`0.789404`、actor_count_5s=`25583` 完全相同；0 AMP overflow、0 no-signal skip、双卡 parameter/buffer checksum delta 0。

### 15.3 最终固定种子结果

| 预测结构 | ADE 1/3/5 s | FDE 1/3/5 s | 结论 |
|---|---|---|---|
| 无 actor-source | 1.1189 / 2.0404 / 2.7437 | 1.5749 / 3.0895 / 4.3980 | 公平预测基线 |
| actor-source，无 ordered history | 1.0504 / 1.9313 / 2.6535 | 1.4608 / 2.9931 / 4.3827 | source 分支有效，history未用 |
| actor-source + ordered history | **0.9997 / 1.8472 / 2.5870** | **1.3708 / 2.9177 / 4.3804** | 总体最佳，默认分支 |
| trajectory-source refined | 1.0728 / 1.9512 / 2.6714 | 1.4703 / 2.9896 / 4.3836 | 总体次优，困难目标可选分支 |

默认分支相对无 source 的主要收益在短中期和整段 ADE：ADE1/3/5 改善 10.65%/9.47%/5.71%，FDE1/3 改善 12.96%/5.56%；FDE5 仅改善 0.40%。ordered history 相对无 history 的 ADE1/3/5 改善 4.82%/4.35%/2.50%，FDE5 只改善 0.05%。因此不能把这些模块写成显著降低五秒终点误差；可支持的是更好的短中期与全轨迹建模。

### 15.4 trajectory-source 的失败、修复与保留边界

第一版 trajectory-source refiner 沿候选轨迹采样 per-source BEV，并用 mode×time state encoder 输出第二层 GMM。真实训练到 batch 100 时发现 GMM 明显恶化。代码审计确认：mean residual 虽为零初始化，但 mode score随机替换 proposal score，log-std/rho也独立替换 proposal统计量；它不是稳定 residual refinement。

修复后所有统计量都以 proposal 为基准，只学习有界增量；所有增量 heads 零初始化。新增测试保证：

- 初始化时 refined trajectory/logits 与 proposal逐元素相等；
- 有效 partner source 能改变 refined output并得到梯度；
- hard-masked partner无任何影响。

修正后完成双卡一轮与 official test。它相对无 source 仍改善全部 ADE/FDE，但总体不如简单 actor-source。它在 partner-only FDE5（6.4371 vs 6.4486）及 age2--3 FDE5（12.4346 vs 12.4862）略好。按用户决定，代码和训练入口都保留，以便选择更完整的 mode/time/source研究分支；论文必须同时报告其总体负结果，不能只挑困难分桶包装 superiority。

### 15.5 当前最终判定

- 默认整体最佳：actor-source + ordered history + proposal deployment。
- 可选完整分支：trajectory-source refined；增加工作量和困难目标分析，不是总体最优。
- 已证实的论文主张：检测/跟踪的 dense fused measurement 与预测的 actor-conditioned source evidence可以解耦，且 source/history分支在冻结 tracker的公平训练中改善预测。
- 未证实：来源信息显著改善 FDE5；trajectory-state refinement总体优于简单分支；future 改善 tracking；首次观测即可产生可靠五秒预测。
- 与 CMP no-aggregation `ADE5/FDE5=4.4001/7.5559` 相比，默认 CoLIF 数字更低，但 CMP tracking cache有 GT-ID oracle且目标协议不同，不能把该数字写成严格 apples-to-apples 全面超越。
