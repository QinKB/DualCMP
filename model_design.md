# CoJTP：基于假设条件证据路由的端到端协同联合轨迹预测

> 文档状态：设计冻结候选稿，尚未进入实现。
> 目标代码基线：`/home/qkb/Project/CMP copy`。
> 主要真实数据集：V2V4Real。
> 本文档描述完整研究方案，不把现有 checkpoint 的推理结果当作新模型证据。

## 1. 一句话研究主张

现有协同模型通常先把各观察车特征融合成一个面向检测的 BEV，再把该统一表示送给预测器。这样虽然扩大了可见范围，却在预测之前抹去了“哪辆观察车、哪个历史时刻提供了什么证据”；而检测认为等价的融合结果，对不同联合未来假设未必等价。CoJTP 使用融合 BEV 形成因果、连续的在线车辆状态，同时保留融合前的来源—时间车辆证据，直到每个联合未来假设内部再进行软路由，使不同场景分量能够从同一组协同历史观测形成不同的证据解释，并由场景联合似然端到端训练。

更通俗地说：

- 持续对象查询回答“场景中有谁、过去 1 秒怎么运动”；
- 来源—时间证据库保留“哪辆观察车在何时看到了什么局部 LiDAR 证据”；
- 假设条件证据路由回答“对当前这一种完整未来假设，哪些历史协同证据更有用”；
- 联合场景模式回答“接下来可能出现哪几种完整交通局面”；
- 每一种局面同时包含所有车辆的未来，不能再为每辆车各自挑一个最有利的预测结果。

对应的实际交通问题是：自车驶入交叉口或汇入区时，自己可能只看到前车尾部，协作车辆却能看到该车侧面姿态及另一辆冲突车辆。普通早期融合可能足以给出两个当前检测框，但“车辆 A 先行、B 让行”和“B 先行、A 让行”需要对历史协同证据作不同解释。CoJTP 不是等待更多通信，也不是在预测后修补碰撞；它在当前已有的协同特征内保留来源和时间，在生成两种完整场景假设时分别取证并联合精修所有车辆轨迹。

整篇论文只保留一个系统主张：

> **对于存在跨视角遮挡和多车交互不确定性的协同交通场景，预测所需证据不应在联合未来假设形成前被压成一个假设无关表示；把来源—时间证据的最终聚合放入场景假设内部，可以在保持在线检测与身份误差真实存在的前提下，生成概率更可信、车辆间更一致的联合未来。**

暂定英文题目：

> **CoJTP: Hypothesis-Conditioned Evidence Routing for End-to-End Cooperative Joint Trajectory Prediction**

暂定中文题目：

> **基于假设条件证据路由的端到端协同联合轨迹预测**

题目中的 “Joint” 同时强调两件事：感知—跟踪—预测联合优化，以及多交通参与者未来的场景级联合建模。最终投稿时应避免让标题产生“首次联合轨迹预测”的误解，因为单车观测条件下的联合预测已有大量工作。

---

## 2. 研究边界与不能声称的内容

### 2.1 数据边界

V2V4Real 在当前本地版本中提供两辆协作车辆的 LiDAR、位姿和车辆标注，但不提供可直接用于本研究的高清地图、交通灯状态、行人标注、路侧单元观测或真实无线逐包日志。因此：

- 模型输入只使用历史多车 LiDAR 和车辆位姿；
- 预测对象只冻结为车辆；
- 不引入不存在的地图、交通灯或行人输入；
- 不声称解决真实路侧基础设施协同、行人过街或真实通信延迟问题；
- “车路协同”只能作为面向 V2X 的上位研究方向，V2V4Real 实验直接验证的是车车协同实例。

如需在论文标题或摘要中直接使用 infrastructure/vehicle–infrastructure，必须增加包含真实路侧观测的数据集，不能由 V2V4Real 结果外推。

### 2.2 新颖性边界

本文不主张以下单点首次出现：

- BEV 特征融合；
- 对象查询或轨迹查询；
- 端到端检测、跟踪和预测；
- 多智能体联合轨迹预测；
- Transformer、可变形注意力或匈牙利匹配；
- 感知不确定性向预测传播。

持续查询、沿轨迹读取历史 BEV、模式条件注意力和场景级联合混合分布分别已有 ViP3D/CoopTrack、PnPNet/MASAR、DeTra/QINET 和 QCNeXt 等强先例。可审查的创新定位因此严格收缩为：

> 在协同 LiDAR 输入条件下，用融合 BEV 和持续查询建立在线车辆集合，但不让检测导向的早期融合成为预测器唯一证据。模型保留 observer×time provenance-resolved 的融合前车辆证据，并把其最终聚合推迟到联合世界假设内部：每个场景分量对同一组来源分离历史证据进行条件路由，再共同生成所有车辆未来。完整评测必须证明该精确耦合优于融合后历史证据、无模式条件的来源路由和等参数联合解码器。

本文研究的是在线检测与身份不确定性下的 cooperative sensor-to-scene forecasting，不声称首次提出协同端到端预测或模式条件注意力。若来源—时间证据置乱无影响，或 HCER 不优于无模式条件路由，则核心模型主张直接失败，不能退回“完整系统组合”包装为创新。

---

## 3. CMP 基线的已验证问题

### 3.1 现有信息链

CMP 在 V2V4Real 上的实际链路为：

```text
multi-agent LiDAR
    -> PointPillar / CoBEVT
    -> per-frame detections and fused BEV
    -> AB3DMOT
    -> deterministic 11-frame tracks
    -> MTR marginal predictions
    -> optional prediction aggregation
```

其中 AB3DMOT、缓存文件和轨迹预处理切断了未来预测误差到感知网络的梯度。当前 `no-aggregation` MTR 在训练时会在 motion aggregator 之前返回；它训练的是基于预处理轨迹的 MTR，并不是完整端到端 CMP。

### 3.2 对交通预测的直接影响

#### 问题 A：硬轨迹接口丢失预测所需证据

检测框和 AB3DMOT 轨迹只保留位置、尺寸、速度等压缩状态。远距离稀疏点、遮挡后的弱证据、车辆朝向变化以及协作车辆补充的局部几何信息不能由未来任务决定是否保留。上游漏检、定位偏差和身份错误会直接成为预测器的输入错误。

#### 问题 B：历史状态形成与未来任务分离

现有预测器接收已经完成的轨迹，无法通过未来损失纠正“过去 1 秒应该如何估计”。因此即使未来预测发现当前速度或转向趋势不合理，也不能反向修正历史查询或 BEV 特征。

#### 问题 C：边缘最优不等于场景合理

常见输出可写成：

\[
q_{\mathrm{marg}}(\mathbf Y\mid X)
=
\prod_{i=1}^{N}
\sum_{k=1}^{K}
\pi_{i,k}
q_{i,k}(\mathbf Y_i\mid X).
\]

它允许不同车辆独立选择模式。对两辆交互车辆，模型可能分别预测出合理的候选轨迹，但独立选择后组成“两车同时抢行”等不合理场景。边缘 minADE/minFDE 仍可能很好，因为评测会分别为每辆车选择不同的最优模式。

#### 问题 D：检测导向的早期融合形成预测信息瓶颈

CoBEVT 把不同观察车的特征压缩成一张统一 BEV，这对扩大检测范围有效，但融合后不再显式保留“哪辆车在何时提供了哪部分局部几何”。在交叉口、汇入和遮挡场景中，自车可能只看到目标尾部，协作车却看到其侧面朝向或相邻冲突车辆。两组观测可以支持相同的当前检测框，却对转向、直行或让行等未来假设提供不同证据。若所有未来模式只读取同一个检测导向的融合表示，未来任务无法在不同联合交通假设下重新组织这些互补观测。

CoJTP 的核心不是取消空间融合：融合 BEV 仍用于可靠发现和持续跟踪车辆；模型另外保留融合前的来源—时间车辆证据，并让证据路由与联合未来假设在同一解码层中共同演化。该信息瓶颈是否真实存在必须由 fused evidence、来源平均、无模式条件路由和完整模型的受控实验验证，不能仅凭架构描述预设成立。

### 3.3 CoLIF 失败记录对本设计的硬约束

`/home/qkb/Project/CoLIF/docs/colif_failure_postmortem.md` 已经对 fused-BEV persistent-query 路线进行了多轮真实训练和同权重否证。与本设计直接相关的结论是：

1. fused persistent tracker 可以运行并达到约 0.62--0.64 MOTA，但这主要来自 CoBEVT dense measurement 和 predict-correct update；
2. 把 CMP MTR 直接接到 online query history 后，5 s minFDE 长期处于约 10--15 m，严格恢复 11×22 MTR 输入合同也不能消除表示分布错配；
3. 六个手工速度尺度能够把 best-of-six minFDE 降至约 4.3--4.7 m，但 mode loss 接近均匀分布，不能证明学习到了模式排序或 residual；
4. future latent 写回 query content 的反馈在同一 checkpoint 下使 minFDE、MOTA、coverage、recall 和 IDS 变差，因此已被直接否证；
5. 低条件式 minFDE 必须与 coverage、target denominator、top-1/NLL 和 actor age 同时报告。

因此 CoJTP 明确作出以下改变：

- 不把“persistent query”本身当作新贡献或成功依据；
- 不直接复用 CMP MTR checkpoint，不假设 tensor shape 一致等于表示语义兼容；
- 不让预测 latent 反馈 tracking，不主张 future improves tracking；
- 不只把 query latent 或预测框序列送入预测器，而是保留与在线框对齐的多帧、分来源融合前 BEV 证据；
- 预测头从当前训练分布重新训练，并把未训练运动锚、constant velocity、CoLIF query-MTR 作为必须超过的诊断基线；
- 第一次观测只建立车辆状态；没有至少两次有效因果观测时，不声称模型已辨识速度，输出需单独标记为 short-history prediction。

---

## 4. 任务定义

### 4.1 输入

设接收车辆为 (e)，可用协作车辆集合为 \(\mathcal A\)。历史窗口包含当前帧在内共 \(T_p=11\) 帧，采样间隔固定为 0.1 s：

\[
\mathcal X=
\left\{
X_{a}^{t-T_p+1:t},
P_{a}^{t-T_p+1:t}
\right\}_{a\in\{e\}\cup\mathcal A},
\]

其中 \(X_a^s\) 是车辆 \(a\) 在时刻 \(s\) 的原始 LiDAR 点云，\(P_a^s\) 是其位姿。所有输入只包含预测时刻及其之前的数据。

### 4.2 输出

模型首先输出当前时刻的车辆集合：

\[
\hat{\mathcal B}_t=
\{(\hat c_i,\hat b_i,\hat q_i)\}_{i=1}^{\hat N},
\]

其中 \(\hat c_i\) 为存在置信度，\(\hat b_i\) 为当前 3D 框，\(\hat q_i\) 为携带身份和历史的对象查询。

随后输出 \(K=6\) 个场景级未来：

\[
\hat{\mathcal Y}
=
\left\{
\pi_k,
\{\hat{\mathbf Y}_{i}^{(k)}\}_{i=1}^{\hat N}
\right\}_{k=1}^{K},
\]

其中 \(\hat{\mathbf Y}_{i}^{(k)}\in\mathbb R^{T_f\times 3}\)，\(T_f=50\)，三个输出维度为未来位置 \((x,y)\) 和航向角 \(\psi\)。尺寸默认沿用当前查询框，不预测没有监督价值的未来尺寸变化。

### 4.3 预测对象

主实验只预测当前时刻存在、能够与冻结 GT 集合匹配且具有至少一个可评测未来点的车辆。未来进入场景但当前尚不存在的车辆不属于标准 motion forecasting 输出，应在研究边界中明确说明。

---

## 5. 总体架构

```text
Multi-agent LiDAR history (11 frames)
                 |
                 v
 Shared PointPillar -> source-tagged BEVs
                 |
       +---------+----------------+
       |                          |
       v                          |
 Per-frame CoBEVT                 |
       |                          |
 Persistent Track Queries         |
       |                          |
 online boxes + state history ----+--> source x time actor crops
                                           |
                                  provenance evidence tokens
                                           |
 Actor x World tokens <--------------------+
       |
       v
 Recurrent HCER Scene Layers
 - world-conditioned source/time routing
 - within-world multi-actor interaction
 - actor/world token refinement
       |
       v
 K scene probabilities + N x K future trajectories
```

图中两条线不是两个独立预测头：融合分支先产生在线车辆框和身份，这些框随即决定应从分来源 BEV 的哪里采样；来源证据再与状态历史共同进入唯一的 HCER 场景解码器。整个链路处于同一次前向和反向中，不存在独立检测结果再送入另一个脚本、离线 AB3DMOT、离线轨迹缓存、旧 MTR 权重直连或预测后聚合器。

---

## 6. 模块一：逐帧协同 BEV 编码

### 6.1 共享 LiDAR 编码器

为与 CMP/V2V4Real 已验证的数据语义保持一致，先把每个来源点云投影到该时刻接收车坐标系，但仍保留来源槽位，再使用权重共享的 PointPillar 编码器：

\[
\widetilde X_{a\rightarrow e}^{s\mid s}
=T(P_a^s,P_e^s)X_a^s,
\qquad
F_{a\rightarrow e}^{s\mid s}
=E_{\theta}(\widetilde X_{a\rightarrow e}^{s\mid s}),
\qquad
F_{a\rightarrow e}^{s\mid s}\in\mathbb R^{C\times H\times W}.
\]

这里的特征是 **source-tagged、receiver-aligned**，不是 observer-native BEV。编码器沿用 CMP 已验证的 PillarVFE、Scatter、BEV backbone 和下采样配置。这样保持输入范围、分辨率和官方前端语义可比；observer-native voxelization + feature warp 作为更高风险备选，不进入第一版冻结模型。

### 6.2 位姿对齐与空间融合

来源特征已经在接收车网格中对齐，按来源维重组后通过 CoBEVT 融合：

\[
\bar F_e^{s\mid s}
=
G_{\phi}
\left(
\{F_{a\rightarrow e}^{s\mid s}\}_{a\in\{e\}\cup\mathcal A},
M_{\mathrm{source}}^s
\right).
\]

这里的通信和融合机制沿用 CMP 的可比设置，不把压缩、时延或传输策略作为本文贡献。上标 \(s\mid s\) 表示特征处于时刻 \(s\) 的接收车坐标系。前端同时返回仍有梯度的分来源特征 \(F_{a\rightarrow e}^{s\mid s}\) 和融合特征 \(\bar F_e^{s\mid s}\)：前者供 HCER 使用，后者供检测和持续查询使用。所有历史时刻分别编码；后续查询传播显式补偿接收车自运动。

外部接口冻结为显式 padded 形式，不能把来源顺序隐含在扁平 batch 中：

```text
source_bev:         [B, A_max, C, H, W]
fused_bev:          [B, C, H, W]
source_mask:        [B, A_max]
source_role:        [B, A_max]
source_to_receiver: [B, A_max, 4, 4]
```

当前 CoLIF 内部实际产生的是 `[sum(record_len), C, H, W]`，适配层必须使用 collate 给出的 `source_to_batch/source_slot` 显式填入上述张量，并以合成多 batch、缺失协作车和 collaborator-dropout 测试验证映射；禁止假设扁平位置永远等于固定 CAV 身份。

### 6.3 必要实现约束

- 不允许在最终训练路径中保存后再读取 `.npy` 融合特征；
- 不允许对发送端或融合特征调用 `detach()`；
- 不能直接使用 CMP `forward()` 中 `clone().detach()` 的 `source_feature/fused_feature` 输出，需采用精简且无断图的 encoder 接口；
- 删除官方模型 forward 从未使用的 `graph_sage_net/post_graph_*` 大参数头，只按允许前缀加载 PointPillar、backbone、CoBEVT 和检测头；
- 位姿变换必须在训练和推理中使用同一坐标定义；
- 缺失协作车辆使用显式 mask，而不是伪造空点或零位姿；
- 每个历史时刻完成在线查询更新和局部 evidence crop 后立即压缩为 token，避免一次性保存所有 CAV 的 11 帧高分辨率 BEV。

---

## 7. 模块二：持续协同对象查询

### 7.1 为什么需要对象查询

融合 BEV 描述的是空间证据，不天然等同于车辆身份和历史。模型不能从单张当前 BEV“凭空”获得过去 1 秒运动。持续对象查询负责把逐帧融合证据组织成有身份的车辆状态序列。

每个查询状态定义为：

\[
Q_i^s=
(q_i^s,r_i^s,b_i^s,o_i^s,m_{i,\mathrm{hist}}^s),
\]

其中：

- \(q_i^s\)：车辆语义和运动隐状态；
- \(r_i^s\)：BEV 参考点；
- \(b_i^s\)：当前 3D 框和速度状态；
- \(o_i^s\)：存在置信度；
- \(m_{i,\mathrm{hist}}^s\)：由模型运行时状态产生的历史查询有效 mask。

### 7.2 查询传播

已有查询根据上一时刻状态预测当前参考位置：

\[
(q_i^{s-},r_i^{s-})
=
\mathcal P\!\left(
q_i^{s-1},
T(P_e^{s-1},P_e^s)b_i^{s-1\mid s-1},
\Delta t
\right).
\]

其中 \(T(P_e^{s-1},P_e^s)\) 先把上一帧车辆框转换到当前接收车坐标系。\(\mathcal P\) 使用轻量 GRU/MLP 预测参考点位移和隐状态，不把常速度模型当作最终预测器。它的作用仅是给当前帧证据检索提供合理初值。

### 7.3 协同证据校正

传播后的查询在当前融合 BEV 的参考点附近执行稀疏可变形交叉注意力：

\[
c_i^s
=
\operatorname{DefAttn}
(q_i^{s-},\bar F_e^{s\mid s},r_i^{s-}),
\]

\[
q_i^s
=
\operatorname{GRU}(q_i^{s-},c_i^s).
\]

检测头从 \(q_i^s\) 输出存在性、3D 框、航向和速度。若自车看不到车辆但协作车辆在融合 BEV 中提供了有效点云，查询可以被协同证据校正；若当前帧短时没有证据，查询可依靠历史状态保留有限帧，但不无限维持。

### 7.4 新车辆发现与消失查询

- 每一帧保留一组 discovery queries，由 CMP 现有 anchor-based `psm/rm` 头解码 top anchor proposals，再在 proposal 中心从融合 BEV 采样特征以初始化参考点；
- active track queries 与 discovery queries 在同一个 set decoder 中联合自注意力和 BEV 校正，而不是各自独立输出两套车辆；
- 已有 track queries 优先继承其 GT 身份匹配；
- 训练匹配先保留仍存在 GT 的 inherited query，再让 discovery queries 只匹配新出现 GT；重复覆盖已有 GT 的 discovery query 被监督为 no-object；
- 已有 track query 不与 anchor proposals 做 decoder 之前的硬 Hungarian measurement assignment，而是按自身传播参考点直接从当前融合 BEV 更新；
- 推理时 query slot 自身维持匿名身份；高置信 discovery query 只有在不与 active-query 输出形成明显重复框时才出生，随后依据存在置信度和连续消失帧数管理生命周期；
- 查询生命周期属于模型推理规则，不再调用 AB3DMOT 或外部卡尔曼滤波器。

这一约束专门避免 CoLIF 中“训练以 GT ID 延续、推理却在 decoder 前用纯距离 Hungarian 维持身份”的机制错配。匈牙利匹配仍可用于训练监督分配，但不成为在线观测到已有查询的运行时数据关联器。birth 去重阈值属于固定生命周期规则，必须在 val 冻结并报告其敏感性；它不能偷偷承担完整帧间身份匹配功能。

Anchor proposal head 不是只在初始化时借用的无监督旁路：它在训练每一帧沿用 CMP 的 anchor 分类与框回归监督，在推理中只负责产生新车候选；已有 track query 的校正仍来自其自身参考点附近的 BEV 证据。这样既不虚构 CenterPoint 式 heatmap，也避免在解冻 backbone 后由于 top-k 不可导而让 proposal head 漂移。若后续改用中心热图头，必须作为新模块单独实现和验证，不能再称为复用现有 CoBEVT 检测头。

### 7.5 固定查询容量与空场景

第一版固定 \(Q_{\max}=64\) 个 persistent track slots 和每帧 \(Q_{\mathrm{disc}}=64\) 个临时 anchor discovery candidates；数值只能依据 train/val 的 actor-count/overflow 统计在实现前冻结，不能查看 test 后修改。Anchor proposals 按 `psm` 置信度和固定 NMS 规则排序，birth 只填入空闲 track slots；若候选或 active tracks 超限，使用确定性的 runtime 排序截断，并把 GT/query truncation 计入 coverage 报告。

当没有 active actor 时，HCER 仍以固定形状执行：mask-aware pooling 返回可学习 `null_scene`，每个世界概率仍归一化，SceneBlock 通过一个仅用于数值路径的 `null_actor` 执行。`null_actor` 不进入输出、匹配或未来损失。这样避免全 mask softmax/NaN，并保证所有 DDP rank 调用相同参数。无匹配未来场景按第 8.7/9.2 节返回 graph-connected zero，不计入 future scene denominator。

### 7.6 显式状态历史

最终历史不是只有一个当前查询向量。对每个有效车辆保存：

\[
\mathcal H_i=
\{q_i^s,b_i^{s\mid s},o_i^s,m_{i,\mathrm{hist}}^s\}_{s=t-T_p+1}^{t}.
\]

状态历史本身只提供运动几何，不能单独代表原始协同传感证据。它首先形成每帧的 actor reference：

\[
R_i=\{(b_i^{s\mid s},o_i^s,m_{i,\mathrm{hist}}^s)\}_{s=t-T_p+1}^{t}.
\]

该 reference 全部来自在线预测查询，不读取 GT 历史，也不依赖离线轨迹文件。为避免坐标含混，运行时首先保留查询在各自时刻接收车坐标系中的框 \(b_i^{s\mid s}\)，随后仅利用已知自车位姿将其转换到当前接收车坐标系：

\[
b_i^{s\mid t}=T(P_e^s,P_e^t)b_i^{s\mid s}.
\]

### 7.7 显式状态编码与来源—时间证据库

CoLIF 已证明，直接把 online query latent 或框历史接入旧 MTR 会产生严重表示错配。CoJTP 首先用完全显式的因果状态历史形成基础 actor token：

\[
\Delta\xi_i^s=
\left[
R(-\psi_i^t)(p_i^{s\mid t}-p_i^{t\mid t}),
\operatorname{wrap}(\psi_i^{s\mid t}-\psi_i^{t\mid t}),
R(-\psi_i^t)v_i^{s\mid t}
\right],
\]

\[
g_i^s=E_{\mathrm{state}}
(\Delta\xi_i^s,o_i^s,m_{i,\mathrm{hist}}^s,t-s),
\qquad
h_i=\operatorname{TemporalEncoder}(\{g_i^s\}_{s=t-T_p+1}^{t}).
\]

与此同时，不先把各观察车证据平均成一个 actor feature。对每个当时已经在线存在的查询 \(i\)、观察车来源 \(a\) 和历史时刻 \(s\)，使用预测框在 source-tagged、receiver-aligned 的融合前 BEV 上做定向连续采样：

\[
e_{i,a,s}
=\operatorname{DiffActorCrop}
(F_{a\rightarrow e}^{s\mid s},b_i^{s\mid s}).
\]

`DiffActorCrop` 由预测框构造 `affine_grid`，再用双线性 `grid_sample` 和小型池化器立即压缩为 token；不得 rounding、转 NumPy 或 `detach`。证据 token 保留来源和时间元数据：

\[
\chi_{i,a,s}=
\left[
e_{i,a,s},
\phi(\operatorname{role}_a),
\phi(t-s),
\phi(T_{a,s\rightarrow e,t}),
\rho_{i,a,s}
\right].
\]

其中来源编码只使用 ego/collaborator 角色和相对几何，不使用固定车辆或场景 ID，避免记忆序列身份。\(\rho_{i,a,s}\) 只包含能从当前输入计算的距离、方位、有效范围、局部点/占据支持等质量特征，不使用 GT 可见性、GT 点数标签或未来信息。路由 mask \(M_{i,a,s}\) 只表示来源帧是否可用、该查询在时刻 \(s\) 是否已经因果出生、采样区域是否落在有效网格以及 padding；禁止从当前身份向出生前补写历史证据。所有来源均无效时使用明确的 null-evidence token，避免全 \(-\infty\) softmax。

这样得到的 \(\mathcal E_i=\{\chi_{i,a,s},M_{i,a,s}\}\) 在进入场景解码器前不沿 observer 或 time 维求和。在 CMP/V2V4Real 的 receiver-specific projected-point 实现中，接收端本来就能得到各来源对应的 CoBEVT 前特征；本设计只把这些特征保留到当帧查询更新完成并裁剪 token，不增加第二轮请求或新的 payload。这里不能泛化成所有 V2X 部署都“零额外通信”：该实现需要按 receiver 投影/编码，多接收车系统是否重算、发送何种 source-native payload 属于未验证范围。论文只报告相对当前 CMP payload 没有新增空中消息，高分辨率 source BEV 随后即可释放。

沿历史 BEV 采样并非本文首次提出，持续查询也不是核心创新。本模块的可审查点是把 **observer×time provenance-resolved evidence** 保留到联合未来假设形成之后。若去除来源 ID、打乱时间或先做来源平均而结果基本不变，则该证据接口及后续 HCER 主张 No-Go。

---

## 8. 模块三：假设条件协同证据路由与场景解码

### 8.1 设计动机

如果给每辆车独立生成 6 条轨迹和独立概率，最终场景实际上具有 \(6^N\) 种任意组合，但模型没有学习哪些跨车辆组合能够共同发生。现有场景联合解码通常又只读取一个已经统一融合的历史表示，不同世界模式无法访问不同的来源—时间证据。

CoJTP 使用 \(K=6\) 个共享世界查询表示完整交通局面，并提出 Hypothesis-Conditioned Cooperative Evidence Routing（HCER）。HCER 不是在解码前一次性做 attention，而是在每一层先形成粗联合未来，再用该未来假设读取 observer×time evidence，然后进行同一世界内的多车交互和轨迹精修。证据解释与联合未来由此逐层共同演化。

### 8.2 对象—世界查询构造

设 \(z_k\in\mathbb R^D\) 为第 \(k\) 个世界查询，\(h_i\) 为显式在线状态历史编码，构造基础 actor–world token：

\[
u_{i,k}^{0}
=
\operatorname{LN}
\left(
W_hh_i+W_zz_k+W_bb_i^t
\right).
\]

这里 \(i\) 决定车辆身份，\(k\) 决定场景模式。二者不通过复制最终轨迹拼接，而是在解码前形成统一 token。

对象—世界张量和共享场景概率与 QCNeXt 类场景联合预测已有紧密先例，本文不把该因子化本身称为创新。它在本模型中的作用是给 HCER 提供完整交通场景假设，而不是给每辆车独立设置无关模式。

### 8.3 轨迹参数化的联合假设状态

第 0 层先从基础 token 产生无需额外监督的粗未来，后续层使用上一层精修结果：

\[
\widetilde{\mathbf Y}_{i,k}^{0}
=H_{\mathrm{prop}}(u_{i,k}^{0}),
\qquad
\widetilde{\mathbf S}_{k}^{\ell}
=\operatorname{MaskedPool}_{i}
\left(
E_{\mathrm{scene}}(\widetilde{\mathbf Y}_{i,k}^{\ell});
M_{\mathrm{actor}}
\right).
\]

路由查询必须同时包含 actor token、该车粗未来和全场景粗假设：

\[
r_{i,k}^{\ell}
=E_r
\left[
u_{i,k}^{\ell},
E_y(\widetilde{\mathbf Y}_{i,k}^{\ell}),
\widetilde{\mathbf S}_{k}^{\ell}
\right].
\]

因此路由查询不只含任意 mode 编号，还显式包含整场轨迹参数。这里安全地称 \(\widetilde{\mathbf Y}\) 为 **trajectory-parameterized routing state**；在没有额外监督和干预证据前，不声称中间层轨迹已经具有可辨识的真实未来语义。粗轨迹不另加 proposal loss，最终场景 NLL 是其唯一未来监督路径；必须通过固定 actor/mode/evidence 后交换或扰动 \(\widetilde{\mathbf Y}_k\) 的反事实实验，证明它确实改变路由和最终场景分布。

### 8.4 假设条件协同证据路由

为同时表达“同一世界对某个来源的共同解释”和“不同车辆所需局部证据不同”，路由 logit 分解为场景级来源项与对象级相关项：

\[
\ell_{i,k,a,s}^{\ell}
=
\underbrace{
f_{\mathrm{world}}
(\widetilde{\mathbf S}_{k}^{\ell},z_k,\eta_{a,s})
}_{\text{该世界对来源--时间的一致偏好}}
+
\underbrace{
\frac{Q(r_{i,k}^{\ell})^{\top}K(\chi_{i,a,s})}{\sqrt d}
}_{\text{该车辆的局部证据相关性}},
\]

\[
\alpha_{i,k,a,s}^{\ell}
=\operatorname{MaskedSoftmax}_{a,s}
(\ell_{i,k,a,s}^{\ell};M_{i,a,s}),
\qquad
c_{i,k}^{\ell}
=\sum_{a,s}\alpha_{i,k,a,s}^{\ell}V(\chi_{i,a,s}).
\]

\(\eta_{a,s}\) 只包含来源角色、整帧可用性、相对位姿、时间和帧级质量；车辆局部点数、ROI 占据率等 actor-specific 信息只放在 \(\rho_{i,a,s}\) 中。\(f_{\mathrm{world}}\) 只称 **scene-conditioned shared source–time log-prior/bias**：它给同一世界内所有车辆共享一个来源偏置，但随后仍由对象项共同决定每辆车的 softmax，因而不保证所有车辆选择同一来源，也不是来源可靠性后验。普通早期融合、来源均匀平均以及所有世界共享同一路由均是该模型的受限对照。

\(\alpha\) 只称 routing weights，不能解释为可辨识的来源可靠性概率或可见性后验；不同权重可能产生相同输出，因而模型有效性必须通过来源/时间干预和预测指标证明，不能靠 attention 热图证明。

### 8.5 路由—交互—精修的层内闭环

证据更新后，对每个世界模式 \(k\)，所有车辆 token 在同一层内执行集合自注意力：

\[
U_k^{\ell+1}
=
\operatorname{SceneBlock}
\left(
\{\operatorname{LN}(u_{i,k}^{\ell}+W_cc_{i,k}^{\ell})\}_{i=1}^{N},
M_{\mathrm{actor}}
\right),
\]

其中 \(U_k^{\ell}=\{u_{i,k}^{\ell}\}_{i=1}^{N}\)，\(M_{\mathrm{actor}}\) 是由固定查询容量和 train/inference 相同的 runtime lifecycle 产生的 candidate/padding mask，不使用 GT 匹配、未来有效性或评测置信阈值。低置信但仍在 candidate bank 的 query 继续参与 raw future forward；最终置信阈值只影响输出与评测。离散 birth/death 决策主要由感知监督训练，不能声称 future NLL 对其成员选择可导。SceneBlock 由同一世界内的车辆间自注意力和前馈网络组成；下一层再由更新后的整场粗未来重新路由证据。这样交互结果可以改变下一层取证，而来源证据也可以改变后续交互，不是把一个 source attention 模块简单接在联合解码器前面。

每层在交互后预测轨迹残差：

\[
\widetilde{\mathbf Y}_{i,k}^{\ell+1}
=\widetilde{\mathbf Y}_{i,k}^{\ell}
+H_{\Delta}^{\ell}(u_{i,k}^{\ell+1}).
\]

位置残差按上式累加，航向更新固定为 \(\psi^{\ell+1}=\operatorname{wrap}(\psi^{\ell}+\Delta\psi^{\ell})\)，路由中的航向使用 \((\sin\psi,\cos\psi)\) 编码，避免 \(+\pi/-\pi\) 不连续。下一层路由显式读取这一更新后的整场轨迹。最终均值轨迹固定为 \(\mu_{i,k}=\widetilde{\mathbf Y}_{i,k}^{L}\)，防止“粗假设只进入一个可被忽略的旁路”。

### 8.6 轨迹和概率输出

每个对象—世界 token 的最终均值来自上述逐层精修轨迹，不确定性头输出位置尺度和航向集中度：

\[
(\hat\beta_{i,k,h},\hat\kappa_{i,k,h})
=
H_{\mathrm{unc}}(u_{i,k}^{L}),
\quad h=1,\ldots,T_f.
\]

对位置采用二维独立 Laplace 密度，对航向采用 von Mises 密度。为避免连续密度通过尺度坍缩获得无界训练目标，参数固定为：

\[
\beta_{i,k,h,d}=\beta_{\min}+\operatorname{softplus}(\hat\beta_{i,k,h,d}),
\qquad
\kappa_{i,k,h}=\kappa_{\max}\operatorname{sigmoid}(\hat\kappa_{i,k,h}),
\]

其中 \(\beta_{\min}>0\)、\(\kappa_{\max}<\infty\) 只在 train/val 冻结。单步完整对数密度为：

\[
\log q_{i,k,h}
=-
\sum_{d\in\{x,y\}}
\left[
\log(2\beta_d)+\frac{|y_d-\mu_d|}{\beta_d}
\right]
+\kappa\cos\!\left(\operatorname{wrap}(\psi-\mu_\psi)\right)
-\log(2\pi I_0(\kappa)),
\]

实现必须使用数值稳定的 \(\log I_0\)。世界概率由模式 token 和同一个 runtime candidate/padding mask 的 mask-aware pooling 产生：

\[
\pi_k
=
\operatorname{softmax}_k
\left[
H_{\mathrm{score}}
\left(z_k,\operatorname{Pool}_i u_{i,k}^{L}\right)
\right].
\]

这里一个 \(\pi_k\) 对应整个场景，而不是第 \(i\) 辆车自己的第 \(k\) 条轨迹概率。

### 8.7 概率边界

设 \(\mathcal A_t(\mathcal X)\) 为模型根据历史输入形成的固定容量候选库，\(\mathcal J_t=\operatorname{dom}(\sigma_t)\subseteq\mathcal A_t\) 为 criterion 在当前帧匹配到 GT 且至少有一个有效未来坐标的监督槽位。模型对 \(\mathcal J_t\) 的 observed future 使用如下有限混合联合分布：

\[
q(\mathbf Y_{\mathcal J_t,\Omega}
\mid\mathcal X,\mathcal A_t)
=
\sum_{k=1}^{K}
\pi_k
\prod_{i\in\mathcal J_t}
\prod_{h:m_{i,\mathrm{future}}^h=1}
q_{i,k,h}(y_{i,h}\mid\mathcal X).
\]

给定世界模式后，各车辆未来坐标的显式密度因子化，但每个因子的参数已经通过模式内对象注意力依赖其他车辆。因此它是有效的条件联合混合模型，但不应宣称能够表示任意联合分布。

未来有效 mask 来自标注可评测性。对缺失坐标只计算已观测坐标的模型边缘项；在没有证明缺失机制外生或可忽略之前，论文只称其为 masked/observed-coordinate likelihood，不称为完整物理未来分布上的严格 proper score。

该分布严格限定为“给定模型候选库后，当前匹配监督槽位的未来已观测坐标条件场景密度”。它不是所有 active queries 的变基数未来密度，也不是车辆存在性与未来轨迹的统一集合密度；当前存在性仍由感知—查询监督学习。若 \(\mathcal J_t=\varnothing\)，该 scene 跳过 future loss 且不进入 future batch 分母；query 容量不足造成的 unmatched GT 必须进入 truncation/coverage 统计。世界模式数固定为 \(K=6\) 只是默认设置，必须按场景 actor 数报告并消融 \(K=3,6,12\)。

### 8.8 模型主张的理论边界

设全部来源—时间证据为 \(E\)，传统早期融合只保留 \(Z=g(E)\)，未来场景为 \(Y\)。数据处理不等式给出：

\[
I(Y;Z)\le I(Y;E).
\]

这只说明不可逆融合 **可能** 丢失与未来相关的信息，不证明 CoBEVT 一定丢失，也不证明 HCER 必然更优，因为 \(Z\) 可能已经是充分统计量。HCER 的实验假设正是检验：在来源互补、遮挡或来源冲突的真实交通样本中，保留 \(E\) 是否比只保留 \(Z\) 提供更好的场景 NLL 与 top-1 联合预测。

所有世界共享路由是 HCER 的受限特例：

\[
\alpha_{i,k,a,s}=\alpha_{i,a,s},\quad\forall k.
\]

因此模型的关键不是“多一个 attention”，而是粗联合未来是否真正改变模型读取的协同证据，并在干预下改变整个场景分布。若完整函数族的额外自由度不能带来稳定泛化收益，或只增加参数量，则理论包含关系不能替代实验证据，核心主张仍判失败。

---

## 9. 监督、匹配与梯度路径

### 9.1 跨帧查询监督

第一历史帧使用预测集合与 GT 集合之间的匈牙利匹配。匹配成本只包含车辆类别、3D 中心、尺寸和航向。后续帧中：

- 仍然存在的 GT ID 继承其查询身份；
- 新出现 GT 与空闲 discovery queries 匹配；
- 消失 GT 仅监督存在性，不把其未来位置伪造为零。

GT ID 仅进入 criterion 的监督对应关系，不进入 model forward、query reference、runtime lifecycle 或未来解码器。查询的传播、激活、出生和死亡只能由历史模型状态及当前传感输入决定；criterion 得到的 GT 对应关系不得重置 query、reference 或 box。必须增加 runtime schema anti-leak 测试：对完全相同的 LiDAR 输入，任意置换 GT ID、替换未来坐标或未来有效 mask，纯 `model.forward()` 的所有 raw outputs 必须完全不变。

感知—查询监督为：

\[
\mathcal L_{\mathrm{perception}}
=
\sum_{s=t-T_p+1}^{t}
\left(
\lambda_d\mathcal L_{\mathrm{dense}}^s
+
\lambda_c\mathcal L_{\mathrm{cls}}^s
+\lambda_b\mathcal L_{\mathrm{box}}^s
+\lambda_v\mathcal L_{\mathrm{vel}}^s
\right).
\]

其中 \(\mathcal L_{\mathrm{dense}}\) 监督 discovery heatmap/box，后三项监督持续查询。它们是建立新车候选、车辆身份和历史状态所必需的中间监督，不把这些标准检测/跟踪损失包装成创新。

### 9.2 场景级未来监督

`model.forward()` 始终对由历史输入形成的固定容量 query candidates 输出 raw futures；它不知道哪些 query 最终能匹配 GT，也不知道哪些未来步有标注。criterion 在任何评测置信阈值之前对全部候选做当前帧匹配 \(\sigma_t\)，再选择监督对象，并用独立的未来目标 mask \(m_{i,\mathrm{future}}^h\) 组成联合未来目标。这样模型不能通过压低置信度、减少预测覆盖率来逃避 future loss。未来损失直接使用场景混合负对数似然：

\[
\mathcal L_{\mathrm{future}}
=
-\log
\sum_{k=1}^{K}
\pi_k
\exp\left[
\sum_{i=1}^{N}
\sum_{h:m_{i,\mathrm{future}}^h=1}
\log q_{i,k,h}(y_{i,h})
\right].
\]

同一个模式 \(k\) 必须解释所有车辆，不允许训练或评测时按车辆分别选择赢家模式。实现使用 `logsumexp`，避免概率下溢。模式指数内部不得按 actor 数或有效未来点数做除法“平衡难度”，否则不再是上式的正规化场景密度；训练只在 batch 的 scene 维求平均，不同 actor 数和未来有效长度的分层统计在评测端完成。

若某 scene 的 \(\mathcal J_t=\varnothing\)，只保留感知—查询监督，future loss 返回 graph-connected zero 但不增加 future scene count；若候选容量不足导致 GT 未匹配，不能把它从 denominator 静默消失，必须计入 detection/forecast coverage 与 query truncation。

### 9.3 总训练目标

\[
\mathcal L
=
\mathcal L_{\mathrm{perception}}
+\lambda_f\mathcal L_{\mathrm{future}}.
\]

不增加重建、通信、碰撞、蒸馏、对比学习或额外模式分类损失。这里不是用损失相加“伪造端到端”：\(\mathcal L_{\mathrm{perception}}\) 保证运行时必须使用的 discovery 与 track queries 能形成正确历史，\(\mathcal L_{\mathrm{future}}\) 则必须沿同一实际推理主链反向到历史证据和 LiDAR 前端。碰撞率首先作为联合建模是否成功的外部评价，而不是用碰撞惩罚强行制造提升。

### 9.4 端到端成立条件

“端到端”不由损失数量决定，而由最终训练图和梯度路径决定。最终模型必须满足：

```text
scene future NLL
  -> recurrent HCER scene layers
  -> source x time evidence tokens
  -> each live pre-fusion source BEV
  -> shared PointPillar encoder
  -> raw LiDAR from ego and collaborator

and, through the shared online actor states during each actor's causal lifetime:
scene future NLL
  -> recurrent HCER scene layers
  -> persistent query updates from birth to the prediction time
  -> per-frame CoBEVT fused BEV
  -> shared PointPillar encoder
  -> raw LiDAR
```

论文必须报告一次真实 batch 梯度审计：未来损失单独反向时，轨迹头、粗联合未来、HCER router、ego 与 collaborator 的 source-BEV 张量、查询更新器、CoBEVT 和 PointPillar 中均存在有限非零梯度。共享编码器参数有梯度不等于协作来源被使用，必须分别检查每个来源特征张量。匈牙利匹配只确定监督对应关系，不应被描述为可微算子；匹配之后的连续模型参数仍由未来损失更新。

---

## 10. 数据协议

### 10.1 场景级样本，而不是单目标缓存

当前 MTR dataset 以单目标和预处理轨迹为中心，不能直接支持场景联合预测。新数据入口以接收端—时间窗为一个样本，返回：

- 11 帧内所有有效 CAV 的原始点云；
- 每帧 CAV 位姿、可用 mask 和接收端 ID；
- 每帧 canonical 车辆 GT 集合及稳定 ID；
- 当前帧所有可预测车辆的 50 帧未来和有效 mask。

模型输入结构只包含历史点云、历史位姿和历史可用性；GT ID、当前匹配、未来轨迹及 \(m_{\mathrm{future}}\) 只能放在 criterion/evaluator target 结构中。`query_active_mask`、`history_observation_mask` 和 `future_target_mask` 必须使用三个不同字段，禁止复用一个 `mask` 导致未来标注控制前向 actor 集合。

所有未来 GT 坐标必须一次性转换到预测时刻 \(t\) 的接收车坐标系 \(E_t\)。不能沿用“每个未来时刻各自在当时 receiver frame”的轨迹；后者会把自车运动混入目标运动，也无法正确计算多车碰撞和场景联合指标。

主协议固定 **receiver=CAV0**，每个 CAV0 时间窗构成一个样本；CAV1 只作为 collaborator。这样监督、预测坐标、target denominator 和 CMP 的 CAV0 `Car` 轨迹口径完全一致，也避开本地 val 中 CAV1 比 CAV0 多出的 18 帧尾段。将 CAV1 设为 receiver 只可作为另行定义的扩展协议，不混入主训练/主表。

主监督冻结为 CAV0 local object key 和 `Car` 类别；CAV1 的 `ass_id` 只用于标注审计，不进入模型 runtime，也不把无法可靠映射的 CAV1 独有标注强行并入联合集合。Forecast denominator 是当前帧接收端 ROI 内的全部 CAV0 `Car`，只按当前帧几何和类别确定，不依赖未来长度、预测置信度或点云支持；不能复用 CoLIF 的 `current ∩ point-supported evidence` 筛选。未来不完整只影响 \(m_{\mathrm{future}}\) 和完整 5 s 子集归属，不把该车从当前 target denominator 删除。

只读 train/val 审计已发现跨观察者标注并非无噪声共享 ID：train 有 628 个稳定映射、486 个未解决 local track、41 个映射冲突和 24 个几何冲突；val 分别为 50、36、0 和 14。因此不能直接复用 CoLIF 的跨车合并 target。实现前仍需全量审计 CAV0 ID 连续性、同帧重复框、类别和变换误差。

### 10.2 时间定义

- 历史采样间隔固定为 0.1 s；
- 历史长度为 11 帧，覆盖当前帧及之前 1 s；
- 未来长度为 50 帧，覆盖未来 5 s；
- 现有数据代码中的 0.01 s 时间编码不能继续使用；
- 所有时间戳、速度和轨迹时间位置需在单元测试中对应 10 Hz。

### 10.3 切分与模型选择

当前本地 V2V4Real 为 32 个 train 场景、3 个 val 场景和 9 个 test 场景。原始 train/test 含一个逐文件完全重复的场景。论文同时保留两套协议：

1. **Author protocol**：沿用公开 train/test，仅用于与已发布 CMP 数字和官方 checkpoint 对照，并明确披露重复场景；
2. **Strict protocol**：从 train 删除与 test 完全重复的场景，使用独立 val 选择 checkpoint，冻结设置后只运行一次 test。

核心结论以 strict protocol 为准。不能在 test 上比较多个 epoch、调整阈值或选择模型。

由于 val 只有 3 个场景，所有核心门禁至少运行多个预先冻结随机种子，报告 scene-level 均值、方差和逐场景结果；不能依赖单 seed、单场景的微小均值差异决定 HCER 是否成立。

### 10.4 联合场景可用性的 train/val 预审

在不读取 test 的前提下，已对 CAV0 原始车辆 ID 做过一次只读粗统计：

- train：7,105 帧中，4,776 帧同时具有至少两辆连续存在完整 50 帧未来的车辆；
- val：748 帧中为 598 帧；
- train 中上述联合帧里，2,885 帧的任意车辆对在未来 5 s 内中心距离曾小于 8 m；val 中为 456 帧。

这些结果说明 V2V4Real 并非几乎全是单目标预测，联合场景方向有继续审计的必要。但该统计尚未应用 canonical-ID 冲突处理、点云支持规则、接收端 ROI 和重复标注清洗，不能作为论文最终数据。阶段 0 必须用正式 target builder 重做同样统计，并报告每个 clip 的有效车辆数、交互距离和标注冲突率；正式统计若显示联合有效样本大幅减少，则场景联合主张 No-Go。

---

## 11. 现有 checkpoint 的合法用途

| checkpoint | 可以做什么 | 不能证明什么 |
|---|---|---|
| 官方 V2V4Real CoBEVT | author protocol 下初始化同构 PointPillar/backbone/CoBEVT、接口调试和复现作者感知基线 | 训练数据含 strict-test 重复场景，不能支撑 strict 无泄漏主结果，也不能证明查询、HCER 或联合预测有效 |
| 官方完整 CMP | 在 author protocol 下复现作者基线 | 不能作为 CoJTP 权重；不能与 strict protocol 新模型直接公平比较 |
| 官方 no-aggregation MTR | 检查 MTR 数据接口或作为历史轨迹基线 | 不能初始化没有同构参数语义的世界查询解码器 |
| 本地自训 no-aggregation epoch 5 | 调试当前 MTR 复现链路 | 由 test 选择，不能作为论文最终 checkpoint |
| 仓库 `point_pillar_cobevt_motr.py` | 参考未完成的 CoBEVT+MOTR 接口尝试 | 没有未来头、完整配置和训练证据，不能视作已有可用模型 |
| CoLIF v7/v10/v11 | 作为 strict-online tracker、query-MTR、motion-anchor 和 feedback 反例/诊断基线 | 不能把 v10/v11 的低 best-of-six minFDE 解释为 learned forecast；不能继续沿用被否证的 latent feedback |

使用旧参数初始化时必须记录：来源数据、任务、输入语义、加载键、未加载键和是否在最终阶段解冻。仅“能够 strict load”不等于对新任务有效。strict protocol 最终前端必须在删除重复场景的 strict-train 上重新训练，或使用能证明未见 test 的权重；官方/CoLIF 权重只能服务 author protocol、结构调试和诊断。

离线缓存融合 BEV 可以用于第 1–3 阶段的结构调试和消融，但任何依赖缓存或 `detach` 的实验都必须标为 frozen-perception，不得称为最终端到端结果。

---

## 12. 分阶段实现与 Go/No-Go 门禁

模型完整设计保持不变，但实现按可验证阶段推进。

### 阶段 0：冻结数据与基线

工作：

- 修复时间间隔定义；
- 建立 scene-clip loader；
- 冻结 `Car` 类 CAV0 canonical GT，把全部未来统一到预测时刻 receiver frame，并在已知位姿跳变处切断窗口；
- 审计 canonical ID、坐标变换、重复框、有效未来和 query 截断率；
- 在 strict protocol 下重新训练必要 CMP 基线；
- 建立精简无断图 CoBEVT core，并在 strict-train 重训/验证感知前端；
- 冻结 val 选择规则和 test 评测脚本。

Go 条件：随机抽样和全量统计均未发现跨 split 复用、时间错位、未来泄漏或同一物理车辆被重复计数。

### 阶段 1：只验证持续查询跟踪

工作：

- 使用阶段 0 在 strict-train 重新训练并冻结的 CoBEVT 前端；官方 checkpoint 只允许 author-protocol 对照和结构调试，不得沿用到 strict 主模型；
- 训练 query discovery、query propagation 和当前框头；
- 不加入联合未来解码器。
- 复用 CoLIF 已验证的 strict-online 序列索引、因果运行时、anti-leak 检查和评测口径，但不复用 decoder 前 Hungarian measurement assignment、旧 query-MTR 或 future feedback；
- 同时保留 CoBEVT+AB3DMOT 和 CoLIF v7 作为已知强弱点不同的诊断对照。

Go 条件：在相同 strict-online 协议和相同 target denominator 下，持续查询的检测/跟踪综合指标至少不低于 CoLIF v7，并与 CoBEVT+AB3DMOT 的 track recall、false positive、ID switch、MOTA/HOTA 和历史速度误差逐项比较。不能只凭一个 MOTA 数字通过；若查询跟踪明显降低预测目标覆盖率，则先停止，不用预测模块掩盖跟踪失败。

### 阶段 2：验证来源分离的历史证据确实被预测使用

工作：

- 从头训练独立的每车 6 模态预测头，不直连 CMP MTR 权重；
- 在相同在线查询、相同预测头和相同监督下，对比 current-BEV-only、在线框历史、query latent、融合 BEV actor crop、分来源均匀平均和模式无关 source routing；
- 与 AB3DMOT+MTR、CoLIF query-MTR、constant velocity 和未训练六速度锚比较；
- 仍使用每车边缘模式，仅作为中间基线。

Go 条件：在相同预测覆盖率和 target denominator 下，来源分离证据不仅降低 oracle minADE/minFDE，还必须相对融合 crop、在线框历史和未训练运动锚改善 top-1 ADE/FDE 或 NLL，并在 ego-weak/collaborator-strong 或短时遮挡子集上呈现一致收益。移除 source ID、置乱时间/actor evidence 应造成可测下降。若只得到 best-of-six 改善，或来源/时间干预没有影响，则 provenance 接口 No-Go。

### 阶段 3：加入联合场景模式

工作：

- 用对象—世界查询解码器替换独立预测头；
- 加入“粗联合未来 -> HCER 取证 -> 同世界交互 -> 精修”的循环层；
- 使用场景混合 NLL，不对粗轨迹或 attention 增加辅助监督；
- 不加入显式碰撞惩罚。

Go 条件：冻结同一 tracker、actor set、状态历史和 scene decoder，完成以下 \(2\times2\) 对照：A=`fused evidence + mode-independent read`，B=`fused evidence + world-conditioned read`，C=`source×time evidence + mode-independent read`，D=`source×time evidence + full HCER`。四项必须匹配参数量、FLOPs 和输入 token 预算，例如给 fused baseline 提供等量复制/空容量，C 保留同规模 router 但删除 \(k\) 条件。只有 D 在 scene NLL 和 top-1/joint miss 等指标上同时优于 A、B、C，且优势集中于遮挡、来源冲突或 ego-weak 子集，HCER 主张才成立。另需证明轨迹参数化 routing state 优于只有 mode embedding，共享来源 bias 优于纯 actor 独立路由。边缘 minADE/minFDE 不得出现不可接受退化。

### 阶段 4：全链端到端微调

工作：

- 先解冻 CoBEVT，再解冻 PointPillar；
- 使用较小 backbone 学习率和完整查询/解码器学习率；
- 真实 raw-LiDAR scene clips 前向、未来 NLL 和反向；
- 验证无 `.detach`、缓存和外部 tracker。

Go 条件：未来损失对所有声明模块产生有限非零梯度；端到端微调相对 frozen-perception 版本在 val 上改善主要预测指标或预先定义的困难子集。若没有改善，只能报告为 differentiable pipeline，不能声称联合优化带来效果。

### 阶段 5：论文级完整实验

只有阶段 0–4 全部通过，才运行冻结 test 和扩展实验。任何阶段失败都回到该阶段修正，不通过增加新模块或额外损失绕过。

---

## 13. 实验设计

### 13.1 基线

至少包含：

1. Ego-only detection + AB3DMOT + MTR；
2. CoBEVT + AB3DMOT + MTR（无预测聚合）；
3. 完整 CMP prediction aggregation；
4. V2XPnP 风格的多帧 BEV anchor-based 端到端预测适配；
5. CoLIF v7 strict-online tracker + query-MTR，以及同一权重下 future-feedback OFF 的诊断结果；
6. CoLIF 未训练六速度锚/constant-velocity 诊断基线；
7. CoJTP source-resolved evidence + independent per-agent modes；
8. 同一 joint decoder 下的上述 A/B/C 三个受限模型；
9. CoJTP source×time evidence + iterative HCER scene modes（完整模型）。

其中必须包含两个命名且忠实的近邻适配：`QCNeXt joint decoder + fused history`，以及 `QCNeXt joint decoder + source×time tokens but no trajectory-conditioned/world-prior routing`。还需在同一数据协议下适配 DeTra/QINET 风格的模式—时间读取，以及 V2X-Graph/V2X-RECT 风格的 mode-independent 多视角历史融合；不能只引用论文表中跨数据集数字。

所有需要公平比较的模型必须在相同 strict train/val/test、相同历史和未来长度、相同 canonical GT 与感知范围下重新训练。不能把官方 checkpoint 的 author-protocol 数字与 strict-protocol 新模型数字放在同一列直接排名。

### 13.2 指标

#### 感知与跟踪

- 3D detection AP；
- AMOTA/AMOTP 或 HOTA；
- ID switch、false negative、track recall；
- 历史位置和速度误差。

#### 端到端预测

- EPA 或等价 perception-aware forecasting AP；
- minADE、minFDE、Miss Rate；
- top-1 ADE、top-1 FDE、轨迹 NLL 和模式置信度校准；
- 预测覆盖率、实际参与预测的 target denominator 和 tracking recall；
- 按 actor age/有效历史长度分层，首次观测目标单独报告且不用于证明速度辨识；
- 按 1 s、3 s、5 s 分别报告。

#### 联合场景

- minSADE / minSFDE，先对同一场景模式中的所有车辆求平均，再选模式；
- scene top-1 ADE/FDE、scene NLL 和 energy score；
- 对预先定义且能从 GT 唯一计算的交通事件概率报告校准；不把每帧 oracle argmin 的匿名世界索引伪装成类别标签；
- scene miss rate；
- 预测车辆有向框的时空碰撞率；
- 同一模式下的最小车辆间距分布。

不能用边缘 oracle 模式重新组合联合输出后再计算联合指标。

主联合表在所有参与车辆均具有完整 5 s 未来的场景子集上报告；部分未来样本可以进入 masked observed-coordinate NLL，但须单独列出，不能与完整场景指标混为一个分母。

#### 系统代价

- 参数量、FLOPs、峰值显存；
- 单帧更新和 11 帧 clip 训练/推理延时；
- 在与 CMP 相同通信特征设定下报告通信量，不把它作为新贡献。

### 13.3 预先冻结的困难子集

阈值只允许在 train/val 上确定：

- **ego-weak / collaborator-strong**：GT 框内自车 LiDAR 点数低、协作车点数明显更高；
- **long range**：当前目标距接收端 30–50 m、50 m 以上；
- **short occlusion**：当前或最近若干帧自车点数接近零，但前后存在有效观测；
- **high interaction**：根据历史相对状态计算 TTC 较小或未来 GT 最小距离较小；
- **new/short history**：当前存在但历史有效帧数不足 11。

这些子集用于回答“协同证据和持续查询在什么交通条件下真正有用”，而不是只报告总体均值。

### 13.4 协同收益的公平验证

不能在只见过完整协同输入的 checkpoint 上推理时突然把协作车辆清零，再将下降解释为协同贡献。采用两种合法对照：

1. 独立训练一个 ego-only 模型，保持 backbone、数据量和训练轮数匹配；
2. 完整模型训练时进行显式 collaborator dropout，使有协作和无协作都属于训练分布，再用同一 checkpoint 做条件对照。

Collaborator dropout 必须按整段 clip 删除协作来源，并同步更新 flattened voxels、`record_len`、source mask、CoBEVT communication mask 和 HCER evidence mask；不能在融合后简单把协作特征置零。后者只能称为分布外压力测试，不能作为主要因果证据。

### 13.5 HCER 的因果干预实验

主消融先固定完全相同的在线 boxes、query candidates 和 scene decoder，只更换 evidence 表示/路由，避免把检测变化误认为 HCER 收益。除 A/B/C/D 外，必须包含：

- 去掉来源角色或先平均 source tokens；
- 保持 mask 数量不变，置乱同一 batch 内的 actor、observer 或时间 token；
- 只扰动一个来源的位姿、时间或局部 crop，观察 scene probability 与整场轨迹如何变化；
- 强制所有世界共享同一组路由；
- 用固定 mode embedding 替代显式粗联合未来；
- 删除场景级来源项，只保留每车独立相关项；
- 分别移除高权重和低权重来源，检查输出变化是否与声称的证据使用相符。

这些实验使用 scene NLL、top-1 scene ADE/FDE、joint miss 和来源扰动下的性能退化作为证据。Routing weight 可视化只能辅助展示，不能替代干预结果。

---

## 14. 消融与贡献—证据映射

| 论文主张 | 必要对照 | 主要证据 | 失败判据 |
|---|---|---|---|
| 持续查询比 AB3DMOT 硬接口更适合协同预测 | AB3DMOT vs query tracker；同一 CoBEVT | tracking、history error、EPA、困难子集 | 查询跟踪更差且预测无改善 |
| 保留 provenance 能突破检测导向融合的预测瓶颈 | fused crop、source average、mode-independent source routing；同一 tracker/head | 1/3/5 s top-1 与 oracle ADE/FDE、NLL、来源互补分层、source/time干预 | source-resolved 不优于 fused，或去 source ID/时间置乱无影响 |
| 粗联合未来条件的 HCER 让不同场景解释读取不同协同证据 | A/B/C/D 2×2；mode embedding vs coarse future；无场景级来源项 | scene NLL、top-1 scene error、joint miss、responsibility、来源扰动响应 | D 不同时优于 A/B/C，去粗未来/去条件无下降，或 attention 变化但输出不变 |
| 共享场景模式产生更一致的多车未来 | 同一 provenance evidence 下 independent modes vs joint world modes | minSADE/minSFDE、top-1 scene error、scene NLL、energy score、scene miss、collision rate | 只改善 oracle 指标、模式责任/轨迹塌缩或靠碰撞 loss 才改善 |
| 未来任务能够反向改善协同表示 | frozen perception vs full fine-tuning | 梯度审计、EPA、hard subsets | 梯度断开或解冻后无稳定收益 |
| 收益来自协同信息而非训练分布差异 | matched ego-only、collaborator dropout | ego-weak/collaborator-strong 分层 | 不公平清零实验才有收益 |

删除任一核心模块，都应对应删除一条论文主张。不要为了增加消融数量加入没有明确假设的模块。

---

## 15. 工程可实现性

### 15.1 可复用部分

- CMP 的 V2V4Real 点云读取、体素化和位姿变换；
- PointPillar、CoBEVT 结构；官方 V2V4Real checkpoint 仅用于 author protocol、接口调试或同构前端 warm-start；
- CoLIF 已验证的无 `detach` `encode_fused_bev()` 思路、strict-online 序列运行时和 anti-leak 工具；
- 现有 MTR 指标实现中的 ADE/FDE/MR 计算；
- 仓库未完成的 CoBEVT+MOTR 文件只能参考接口，不直接视为可运行实现；
- AB3DMOT 和现有 MTR 保留为基线，不进入新模型推理链。

### 15.2 必须新建或重写的部分

- scene-clip V2V4Real dataset/collate；
- 跨帧 canonical GT 和未来集合构造；
- 持续协同对象查询更新器；
- 无断图且同时返回 `source_bev/fused_bev/source_mask` 的精简 CoBEVT core；
- 来源—时间 evidence token bank；
- 粗联合未来条件的 HCER 循环场景解码器；
- 场景混合 NLL；
- 联合场景评测和困难子集审计；
- 完整 raw-LiDAR 训练入口。

这不是对 CMP 做一个最小模块替换，而是保留已验证的协同 BEV 前端，重构其后的对象状态形成、未来分布和训练协议。

当前官方 CoBEVT 实例约 184.8M 参数，其中约 173.7M 来自 forward 未使用的 `graph_sage_net/post_graph_*`，实际有效前端和检测头约 11.1M。新模型只按明确前缀加载有效模块，不能把未调用参数带入 checkpoint、optimizer 或 DDP 后再把参数量虚报为模型能力。

### 15.3 显存策略

11 帧、两车、全分辨率 BEV 的端到端反向可能超过单卡显存。优先策略为：

- 按历史时刻顺序编码，不一次性堆叠所有高分辨率 BEV；
- 对 PointPillar/CoBEVT 使用 AMP；
- 对每帧 BEV encoder/fusion 使用非重入 activation checkpointing，保留随机状态并冻结 BatchNorm running statistics，避免重算两次更新；
- batch size 从每卡 1 开始，使用梯度累积；
- 每帧在线更新后立即把 source actor crop 压缩为 token；查询和低维 token 保留完整图，高分辨率 source/fused BEV 由 checkpoint 重算；
- 一个 DDP forward 必须完成整段 clip；所有 rank 固定执行 HCER，不能按有效车辆数条件跳过参数；
- 先做 11 帧真实样本、future-only scene NLL 的 forward/backward，检查峰值显存和每个来源梯度，再决定训练配置。

CoLIF 的 11 帧日志实际采用 7 帧 `no_grad` burn-in + 4 帧 BPTT，不能作为 11 帧全链反向已通过的证据。未跑上述真实门禁前，不能声称可以完整训练。

---

## 16. 主要风险与退出条件

### 风险 1：系统组合被认为缺乏方法创新

缓解：论文不把 query、沿轨迹 BEV 采样、模式 attention、actor×world tensor 或 joint prediction 单独称为新颖。核心精确锁定为“在粗联合未来形成后，仍保留 observer×time provenance，并在每层通过场景级来源项和对象级相关项共同路由证据，再精修同一联合未来”。必须用 A/B/C/D 和干预实验逐项证明；若只相当于给 mode-query attention 增加 source 轴，则删除模型创新主张。

### 风险 2：V2V4Real 交互场景不足

缓解：先在 train/val 统计每个 scene clip 的目标数、TTC、未来最小距离和联合有效未来比例。如果大多数样本只有一个目标，联合预测主张在该数据集上不可充分验证，需要增加 OPV2V 或其他顺序 V2X 数据集；不能在单目标样本上用“joint”包装边缘预测。

### 风险 3：标注身份和跨 CAV 一致性不足

缓解：固定 canonical GT，不直接合并两个 CAV 的 raw IDs；在 train/val 审计身份连续性、重复框和坐标变换。无法构造稳定场景级 GT 时，项目 No-Go，而不是通过宽松去重阈值隐藏错误。

### 风险 4：查询跟踪不如成熟 AB3DMOT

缓解：阶段 1 单独门禁。若查询跟踪和未来预测均不优于 AB3DMOT，不进入联合模式阶段。

### 风险 5：重复 CoLIF 的“可运行但不可预测”失败

缓解：不把 query latent、框历史或旧 MTR 当作充分历史表示；阶段 2 必须在完全相同的在线目标集合上证明来源分离 evidence 超过融合 evidence、在线框历史、CoLIF query-MTR、constant velocity 和未训练六速度锚，并同时审查 top-1、NLL、覆盖率和 actor age。若只能提高 oracle minFDE，或 source/time 置乱不影响结果，则删除该预测接口，不进入联合解码。

### 风险 6：HCER 退化为装饰性 attention

缓解：粗联合未来必须显式进入路由查询；来源与时间维在路由前不得压缩；报告不同世界的路由差异、responsibility 和来源扰动响应。若所有模式共享等价路由、移除粗未来条件无下降、或高/低权重来源移除的影响无差异，则 HCER 主张 No-Go，attention 可视化不能作为补救。

### 风险 7：联合模式只减少碰撞但损害预测

缓解：不加入碰撞训练损失；同时看联合指标和边缘指标。若 joint head 不能在无显式碰撞惩罚下学习到更合理场景，则核心假设不成立。

### 风险 8：最终训练并未真正端到端

以下任一情况出现都禁止使用“end-to-end”主张：

- 输入为缓存 BEV 或缓存轨迹；
- 特征被 `detach()`；
- AB3DMOT 仍在推理主链；
- 未来损失不能到达融合和 LiDAR backbone；
- 只有检测 loss 更新感知，未来 loss 未参与；
- 新模型只加载旧 checkpoint 推理，没有在对应训练分布上训练。

---

## 17. 论文叙事与章节结构

### 17.1 Introduction 的逻辑

1. 在交叉口、汇入和遮挡交通中，不同协作车辆会看到同一目标或冲突车辆的不同局部证据；
2. 现有协同模型通常在预测之前把 observer 和 time 维压成一个检测导向表示，所有未来模式只能共享这一假设无关的信息瓶颈；
3. 同时，硬跟踪接口会传播漏检和身份误差，逐车边缘模式又可能组合出相互冲突的通行结果；
4. 因而本文研究在线检测与身份不确定性下，从多车历史 LiDAR 直接估计场景级联合未来的问题；
5. CoJTP 用融合 BEV/持续查询形成车辆历史，却把来源—时间证据保留到粗联合未来出现之后，并通过 HCER 让取证、交互和场景精修逐层共同演化；
6. 论文贡献落在“假设无关早期融合瓶颈”的系统主张、精确的 hypothesis-conditioned provenance routing 机制，以及固定检测分母下的因果干预与联合场景评测闭环。

### 17.2 Method 章节

1. Problem Formulation；
2. Source-Preserving Cooperative BEV Encoding；
3. Persistent Cooperative Track Queries；
4. Provenance-Resolved Actor Evidence Bank；
5. Hypothesis-Conditioned Cooperative Evidence Routing；
6. Scene-Level Likelihood and End-to-End Training。

### 17.3 Experiments 章节

1. Dataset integrity and protocols；
2. Implementation details；
3. Comparison with cooperative and end-to-end baselines；
4. Perception-aware and joint-scene evaluation；
5. Hard-condition analysis；
6. Ablations；
7. Efficiency and failure cases。

---

## 18. 架构图设计说明

论文主架构图使用从左到右的单一信息流，不画与主张无关的图标：

1. 左侧：上下两辆 CAV 及其 11 帧点云序列；
2. 中左：共享 PointPillar 后保留两行不同颜色的 `Ego Source BEV` 与 `Collaborator Source BEV`，二者再汇入逐帧 CoBEVT；
3. 中部上方：CoBEVT fused BEV、三到四个连续时间步车辆方框和同色 track-query 连线，明确融合分支负责发现和在线历史；
4. 中部下方：在线框向同一时刻两张 source BEV 发出定向采样线，形成按 observer×time 排列的小方块证据库；
5. 中右：先画 6 个粗联合未来，再画一个重复两层的 HCER 方框，内部固定顺序为 `Hypothesis-Conditioned Routing -> Within-World Interaction -> Scene Refinement`；
6. 右侧：6 个完整交通场景，每个场景同时显示所有车辆轨迹和一个 scene probability；
7. 底部：从 Scene-Level NLL 分别返回两张 source BEV、CoBEVT 和 PointPillar 的反向箭头；
8. 用灰色删除线标注 “No AB3DMOT / No offline track cache / No prediction feedback / No extra V2X request”。

全图只使用方框、箭头、BEV 网格、点云、小车和轨迹线，不使用无线塔、时钟、数据库、云端等无关图标。

---

## 19. 证据账本

| 文献 | 已核实用途 | 不能据此声称 |
|---|---|---|
| CMP, RA-L 2025 / arXiv:2403.17916 | 协同 LiDAR、CoBEVT、AB3DMOT、MTR、预测聚合；论文承认非完全端到端 | 不能把“协同预测”本身称为首次 |
| PnPNet, CVPR 2020 | tracking-in-the-loop；沿历史轨迹从每帧 BEV 双线性采样外观证据并与运动编码结合 | 不能把轨迹条件历史 BEV 取证或 future loss 进入跟踪称为首次 |
| ViP3D, CVPR 2023 | 3D agent queries 连接检测、跟踪和预测 | 不能把持续对象查询称为首次 |
| V2XPnP, ICCV 2025 | 多智能体多帧时空融合和端到端 anchor 轨迹输出 | 不能把“协同端到端预测”称为首次 |
| Towards Collaborative Joint Perception and Prediction, arXiv 2026 | 协同联合感知—预测问题及端到端原型 | 不能把 collaborative joint P&P 或 sensor-to-forecast formulation 称为首次 |
| ForeSight, ICCV 2025 | streaming joint detection and forecasting | 不能把 streaming joint detection/forecasting 称为首次 |
| Joint Metrics Matter, ICCV 2023 | 边缘指标可能掩盖不一致多车未来；JADE/JFDE/碰撞率 | 不能把 joint metrics 称为本文发明 |
| FJMP / Scene Transformer / MotionLM | 多智能体联合未来已有成熟研究 | 不能把 scene-level joint prediction 称为首次 |
| QCNeXt, 2023 | actor×scene-mode tokens、模式内多车交互、共享场景概率和 mixture-of-products likelihood | 对象—世界张量与联合似然必须明确归属，不能称为 CoJTP 新解码器 |
| DeTra, ECCV 2024 / QINET, Sensors 2024 | future/mode query 读取传感器或历史时刻证据 | 不能把 mode-conditioned feature attention 本身称为首次 |
| TraF-Align, CVPR 2025 / CoST, ICCV 2025 | 协同 source×time 特征与对象轨迹对齐/统一 attention | 不能把 observer×time 路由本身称为首次 |
| V2X-RECT, 2025 / V2XPnP, ICCV 2025 | 多源轨迹特征或多帧特征先融合、再由预测模式读取 | HCER 差异只能落在“联合未来假设形成后才聚合 provenance evidence” |
| MASAR, 2026 | 过去轨迹假设沿 temporal BEV 取证并以外观兼容性精修 | 不能把 hypothesis-guided temporal evidence 称为首次 |
| V2X-Graph, NeurIPS 2024 | 各观察视角历史轨迹独立编码并保留来源，通过 link-guided 端到端融合 | 不能把 provenance preservation 或多视角历史融合称为首次；差异只能是来源在联合世界假设之后才聚合 |
| Co-MTP, ICRA 2025 / CoPAD, IROS 2025 | 多来源历史节点、外部未来节点或多源轨迹的图/模式融合 | 不能把 prediction-side collaboration、协同历史与未来联合融合称为首次 |
| Collaborative Trajectory Prediction via Late Fusion, 2026 | 在预测阶段而非感知阶段融合协作者输出 | 不能把 prediction-side late collaboration 称为首次；HCER 融合的是场景假设条件下的传感证据而非最终预测 |
| Towards Motion Forecasting with Real-World Perception Inputs, ICRA 2024 | 真实感知输入、漏检和跟踪误差造成预测差距，简单微调不一定解决 | 不能未经实验直接声称 CoJTP 已消除域差距 |
| Propagating State Uncertainty Through Trajectory Forecasting, ICRA 2022 | 上游状态不确定性传播已有工作 | 不把 uncertainty propagation 作为本文核心新颖性 |
| CoopTrack, ICCV 2025 / DMSTrack, ICRA 2024 | 协同端到端或可微跟踪近邻 | 不把 cooperative differentiable tracking 单独称为首次 |
| CoLIF failure postmortem（本地真实训练记录） | persistent-query tracker 可运行；query-MTR 表示错配；未训练运动锚可产生低 oracle FDE；future feedback 被同权重实验否证 | 不能把内部失败当作外部文献或公开基线结论；只能用于约束本项目设计并在后续正式消融中重新验证 |

正式写论文前，所有 “first/novel/SOTA” 措辞必须重新经过最新文献检索和逐项审计。

---

## 20. 当前决策

当前只冻结以下设计决策：

1. 放弃 deadline/progressive communication、网格 cardinality field 和观察者接力身份主张；
2. 保留 CMP 的多车 LiDAR 与 CoBEVT 前端作为可比基础；
3. 用持续对象查询替换检测—AB3DMOT—缓存轨迹接口；
4. 显式构建 11 帧在线状态历史，并从每个 observer 的融合前 BEV 提取来源—时间证据；在 HCER 前不对 observer/time 维求和；
5. 禁止 future latent 写回 tracking；未来预测只消费因果历史，不声称未来任务改善在线身份维持；
6. 每个场景分量先形成显式粗联合未来，再由 HCER 的场景级来源项与对象级相关项路由证据，并逐层精修所有车辆未来；
7. actor×world 联合密度沿用已有场景模式因子化思想，不作为单独创新；
8. 最终模型必须从 raw LiDAR 训练，未来损失能分别到达 ego/collaborator source BEV、融合、查询和 PointPillar；
9. strict protocol 最终模型不得以见过重复 test 场景的官方/CoLIF 权重作为已训练前端；
10. 所有旧 checkpoint 只按第 11 节限定使用；
11. 实现前必须关闭理论和工程 P0；论文主张只有通过 A/B/C/D、source/time 干预和联合指标后才成立。

本文档不包含已完成实验或预期提升数值。任何性能结论只能来自后续按冻结协议实际运行的实验。
