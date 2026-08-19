# HANDOFF

## 2026-08-19 修改模型代码前的Git恢复点

- 当前所有已跟踪代码、配置和文档将以`dualcmp-pre-code-change-20260819`标签保存，作为下一轮模型代码修改前的恢复点。
- 本恢复点中的三处代码/config差异仅修改注释和docstring措辞，不改变张量计算、模型参数、forward、loss或推理行为。
- `.serena/`、训练日志、未跟踪图片、审计草稿、`paper.md`和`third_party/`不纳入代码checkpoint，均保留在工作目录中且不会删除。
- 恢复整个已跟踪项目可切换或读取该标签；仅恢复模型代码可对指定路径执行`git restore --source=dualcmp-pre-code-change-20260819 -- <paths>`。

## 2026-08-19 Introduction段首段尾与来源对照重写

- `manuscript.md` 的 Introduction 已重组为四个正文段：任务 stakes/V2X opportunity、intermediate/trajectory/prediction-level fusion stages、target-conditioned source relevance limitation、DualCMP design/evidence；贡献列表同步保留 online states、shared LiDAR encoder、source-specific spatial evidence、motion-anchor motivation 和 fixed-tracking protocol。
- 按用户要求没有因为本地语料词频低而删除必要概念；正文保留 `source-specific spatial evidence`，并用 `aligned CAV feature maps` 指明其实现承载。P2 保留 prediction-level fusion，P4 保留 persistent history 长度/状态噪声到 motion anchors 的因果说明。
- 对段首用词做了精确词频和段落功能核对：`cooperative viewpoint` 在本地 Markdown 中为0次，`multiple viewpoints`为18次、`different viewpoints`为10次；正文已改为 `different viewpoints ... observing CAVs`，只替换不自然的名词组合，没有删去 target-dependent viewpoint relevance 的科学内容。
- `To implement this interface` 在本地语料中为0次，且 `interface` 在该位置容易产生软件API语感；正文已改为 `One limitation ...` → `To address this limitation, DualCMP ...`，P1 也改用 `When forecasting begins`，避免在引言段首和段尾堆叠 interface/this/gap 这类抽象指代。
- 引言正文和 roadmap 已进一步移除 `this/these` 的代词回指：使用 `the inputs`、`CAV observations`、`the fused BEV representation`、`the limitation` 和 `the remainder of the paper`，确保每个段首/段尾的技术对象可直接定位。
- 新增 `docs/dualcmp_introduction_rewrite_ledger.md`，逐句记录 29 个正文/贡献/roadmap 句子的本地来源、模仿的句法功能、段首段尾职责和四段之间的直接衔接。IS-FUSION 与 QD-DETR 已从本地 Markdown 核验并纳入来源卡；它们仅作为 scene/instance 粒度和 query-dependent relevance 的结构参照，不被写成 cooperative motion prediction 的直接证据。
- PaperMeld 只读验证通过：232 条 catalog records、2567 个本地图像链接、23 个 code links。`git diff --check` 通过；Introduction audit 脚本报告正文区域 861 词，但因 Markdown 图片、caption、贡献列表和 roadmap 将其计为10段，并且不识别 `\cite{}` BibTeX keys，报告0个编号引用是已知诊断误报。独立检查确认 Introduction 的11个引用键均存在于当前 working references，来源表29句经引用/标点归一化后全部对应正文。
- 本次任务修改了 `manuscript.md` 和新增上述来源审计文件；模型代码、配置、日志、Abstract、Related Work、Method 和文献库目录未修改。工作树仍保留本任务开始前的既有用户改动。

## 2026-08-18 DualCMP Introduction两类pipeline与Fig. 1 v3

- Introduction当前为“五段正文＋贡献列表”：stakes、current-state estimation、motion forecasting、existing interfaces、DualCMP proposal/evidence。五段为93/114/150/133/207词；审计脚本统计978词。
- 用户提供的三块结构已分别成段：current-state段解释fusion价值与separate-stream代价；forecasting段解释target-dependent CAV use和fused-BEV failure mode；interface段用early feature fusion与later trajectory/prediction fusion收束trade-off。proposal段继续解释persistent query和motion anchor。
- 已删除新增P5（persistent box/velocity/existence、ordered history、motion-anchored decoder、runtime无需precomputed tracks及验证的独立段），并逐字恢复此前159词第四段。没有把被删句子拆散后塞回其他段落。
- 按用户要求完成句级语料复核：对viewpoint作用、三类cooperative interface、Dual representation/persistent reference、persistent state/multimodal decoder四个论证单元各读取约8个正文句，共32个source cards。最后一组曾用于被删除P5，现只作为拒绝记录保留，不能声称8篇论文共同支撑P5中的每个独立意思。
- 纠正此前“8条句子=8篇论文”的错误：`docs/dualcmp_introduction_audit.md`第7.2节现为P2 viewpoint/context、P3三类interface、P3 single-state reference三个论证单元分别列出8篇不同论文；同一论文在每组中只计一次。
- `manuscript.md`已单独纳入Git版本管理：`83993f7`为上一可恢复版本，当前提交为`056d2a4`（`docs: restructure DualCMP fusion argument`）。相关提交均只包含`manuscript.md`；模型代码、配置、日志和其他脏文件未进入提交。
- 当前P1--P5句长序列为`14/17/19/23/20`、`13/15/23/10/21/15/17`、`7/21/19/28/13/29/21/12`、`17/22/30/25/39`、`26/21/10/20/25/21/14/23/8/20/19`。禁用`yet`和`Taken together`；所有新增句在写入前均检索本地语料或核对当前代码。
- 第二段不再以7词的`Consider...`开头，改为18词主题句后在段中给出场景。删除`This variation matters because`等解释性口语，影响关系统一使用`may/can`。第三段改用`While...`让步主题句，Introduction正文不再点名V2VNet、CMP、V2XPnP、PP-VIC、CoPAD或Co-MTP，只保留方法族及引用；具体模型比较留在Related Work。
- Fig. 1在正文只引用一次：`As shown in Fig. 1`后直接对照aligned maps先于agent-state formation、target reference后于state estimation的接口时序。删除`Figure 1 compares...`、逐panel讲解及proposal段中的`Figure 1(c) summarizes...`；panel说明由caption承担。
- Proposal与evaluation合并为一个方法--证据段，按`aggregates → updates → integrates → generate → We evaluate → To isolate → Results show`推进。贡献列表改为三个具体交付物：persistent-query LiDAR framework、agent-specific CAV feature aggregation、motion-anchored decoder及fixed-tracking evidence；不再把泛化`strategy`或单独短验证段列为贡献。
- Abstract已在用户指出核心转折无意义后从头重写。新版本不再以“aligned CAV feature maps are not used”作为问题，也删除`variable-length histories`、runtime annotations与precomputed tracking cache；论证链改为：两类pipeline分别改善scene understanding与historical motion representation → motion prediction还需要随agent/viewpoint变化的spatial context → DualCMP用同一LiDAR encoding形成fused BEV并执行agent-specific aggregation → decoder与V2V4Real结果。当前约155词、9句。
- 新增`docs/dualcmp_abstract_audit.md`。实际阅读全文摘要：CMP、V2XPnP、V2X-Graph、ViTraj、CoPAD、Co-MTP、HPNet、QCNet及T-ITS Social-Pose/TrajPRed/Efficient Baselines。可稳定提取的173篇`ALL_MARKDOWN`摘要词数P25/median/P75为161/183/208；20篇`TITS_MARKDOWN`为161/184/206，句数中位数均为8。审计文档记录各论文的sentence-role、可复用功能与不宜模仿之处。
- PaperMeld与正式论文页面复核了trajectory-fusion路线：V2X-Seq的PP-VIC先以CBMOT融合vehicle/infrastructure trajectories再运行TNT/HiVT；V2X-Graph学习across-view trajectory associations并融合motion/interaction features；CoPAD采用Hungarian matching和Kalman filtering；Co-MTP在heterogeneous graph中融合vehicle/infrastructure histories。CMP与2026 late-fusion还包含prediction aggregation，但按用户要求不作为Fig. 1的独立类别，只在Related Work中承认。
- Abstract和Introduction现并列两种代表性输入组织：`intermediate fusion`与`trajectory fusion`。共同的可核实接口差异是：前者向下游提供fused BEV features，后者提供cooperative trajectory features；两者均未在agent states形成后用aligned CAV feature maps构造agent-specific forecasting context。该表述不推导任何一种fusion必然造成精度下降。
- 用本地229篇Markdown做术语复核：`feature fusion`386次/76篇、`intermediate fusion`235/52、`trajectory fusion`23/7、`historical trajectories`102/36、`cross-view`84/19、`data association`133/17、`agent-centric`71/34、`agent-specific`21/9、`agent queries`97/15、`motion history`45/9、`interaction features`80/18。摘要与引言已改用这些术语，移除核心叙事中的`actor-level`、`per-CAV`和`dual-granularity cooperation`。
- Fig. 1最终构图为三行横向pipeline：`(a) Intermediate feature fusion`、`(b) Trajectory association`、`(c) DualCMP`。文件为`figures/dualcmp_introduction_fig1_v3.png`（1774×887 RGB PNG）；图中实际FuseBEVT输出统一标为`Fused BEV`，不再标为`Shared BEV`。生成规格见`figures/dualcmp_introduction_fig1_v3_prompt.md`；v1/v2均为已替换草稿。
- 论文题目相应更新为 **DualCMP: Task-Adaptive Cooperative Motion Prediction with Fused and Source-Specific BEV Representations**。Method中的FuseBEVT输出及符号统一为`fused BEV`/`F_fuse`，`shared`仅保留在weight-shared encoder等确有共享参数含义的位置。
- 新增working references/citation ledger：`yu2023v2xseq`、`wu2025copad`、`zhang2025comtp`。图中不画prediction-level aggregation，但Related Work明确记录CMP和2026 late-fusion的forecast aggregation，避免把两行Figure比较误写成完整领域taxonomy。

## 2026-08-17 DualCMP Introduction深度审读与重写（历史；以2026-08-18条目为准）

- 初次Fig. 1专项审读查看了CMP、V2XPnP、V2X-Graph、HPNet、V2VNet以及T-ITS H2C、Social-Pose、TrajPRed，并统计正文引用与caption长度；该阶段形成的v1构图后来被2026-08-18的两类pipeline审计替换。
- 该阶段先删除了空泛的`Fig. 1 illustrates this distinction`并建立panel-specific引用；当前引用与caption以顶部2026-08-18条目及`manuscript.md`为准。
- `figures/dualcmp_introduction_fig1_v1.png`是已替换的初始草稿；当前稿为`figures/dualcmp_introduction_fig1_v3.png`。
- 使用PaperMeld逐段阅读全文Introduction：CMP、V2XPnP、V2X-Graph、ViTraj、UniMM-V2X、HPNet、QCNet，以及T-ITS TrajPRed/MapFE4MP；另用Cross-Stitch、MTAN、DSN复核shared/specific representation叙事。完整段落职责、连接词、相邻工作边界、拒稿检查和最终claim map写入`docs/dualcmp_introduction_audit.md`。
- Introduction主线改为：motion-prediction stakes → intermediate-fusion与trajectory-fusion两条代表性pipeline → aligned CAV feature maps在agent-centric prediction中的接口差异 → DualCMP的task-adaptive feature fusion → fixed-tracker验证 → 3条贡献。相邻工作改为方法族总结，不在Introduction展开模型结构。
- Dual设计理由现限定为：fused BEV更新single receiver-side persistent agent state；agent query形成后再访问aligned CAV feature maps。论文不声称intermediate fusion必然损害预测，也不把不同视角清晰度本身直接等同于性能下降。
- 追加阅读全文T-ITS H2C/Social-Pose/SGANv2及HiVT/V2VNet。论断强度按原文惯例分层：代码接口写成可核实事实；可能影响使用`may/can`或研究问题；性能变化只引用受控实验。摘要与Introduction已移除`all/always/inevitably/necessarily/guarantee/once fusion then...`式推导；可见Introduction约648词，Abstract约153词。
- Introduction结果段按用户要求删除1/3/5秒和5秒弱增益的具体数字，只保留固定tracker、agent sets和coverage下aligned CAV feature maps与ordered motion history改善预测的结论。
- 新增并核验`ruan2024v2xgraph`、`you2025vitraj`、`madjid2026latefusion`三条working references/citation-ledger记录。Introduction使用10个BibTeX keys，全部存在ledger；审计脚本不识别BibTeX keys而报告0个编号引用，这是已知false positive。
- Figure 1的早期隐藏占位已经由`figures/dualcmp_introduction_fig1_v3.png`替换并写入Introduction。

## 2026-08-17 DualCMP Method章节按本地论文与代码重写

- 使用PaperMeld定位并阅读全文中的Method章节：CMP、V2XPnP、V2X-Graph、ViP3D、HPNet、HiVT、QCNet和MTR；其主小节通常按`problem/overview → representation/encoder → fusion/interaction → decoder → objective`组织，主小节数量中位数约为4。
- `manuscript.md`现将Problem Formulation独立为Section III，并把Method设为Section IV的7个小节：Method Overview、Shared and Source-Specific BEV Encoding、Persistent Query-Based Agent Tracking、Agent-Conditioned Source Feature Aggregation、Motion Context Encoding、Motion-Anchored Trajectory Decoding、Training Objectives。标题采用相关论文常见的“对象＋操作”形式；原Multimodal Motion Prediction因同时包含history/interaction encoding与trajectory decoding而拆成两个小节。optional trajectory-source refiner降为decoder小节中的短段落，strict-online行为并入问题定义和训练/推理边界；框架图及其正文引用按用户要求暂时留空。
- 新Method逐项对照当前`PointPillarColif`、actor cooperative fusion、motion-anchored forecast、sequence dataset和`ColifLoss`。明确默认forecast不反馈tracking、至少两次有效观测后预测、Phase 2冻结current-state路径，并把旧文档中错误的“按平均位移选正模态”修正为代码实际使用的最终有效目标点/预测endpoint匹配。
- 保留10个承担定义或核心机制的公式；删除张量尺寸、patch大小、TTL、模态数、预测时长、DDP归一化等Implementation Details。Method共6个小节、公式编号1--10连续、display/aligned环境成对、无tab或Git whitespace错误；引用键`wang2025cmp`和`gu2023vip3d`均已在citation ledger中登记。
- 对八篇本地Method的连接词统计显示：`where`82次，`Here`5次，`Yet`0次，`However`8次；当前Method使用`where`5次、`Following`2次、`Given`1次，不使用`Here/There/Yet/However/Moreover/Furthermore/Consequently`。本次只修改论文文档，没有修改或重新测试模型代码。
- 追加审计PointPillars、CoBEVT、V2X-ViT、Where2comm、DiscoNet、V2VNet、V2XPnP与CMP的点云符号和framework overview。Problem Formulation现采用“Consider N agents”结构，定义CAV j的LiDAR L_t^j、pose xi_t^j、receiver r及代码实际使用的T_{j->r}^t=(xi_t^r)^{-1}xi_t^j；删除与“可用CAV集合”重复的runtime mask，mask只在padded BEV fusion处定义。Problem Formulation与Method Overview已拆为独立章节；Overview沿aligned CAV feature maps、fused BEV、persistent agent queries、agent-specific aggregation和forecast的实际数据流介绍。
- 对比CMP、V2XPnP、V2X-Graph、HiVT、HPNet和ViP3D实际框架图后，用内置ImageGen生成新的DualCMP方法图构图初稿`figures/dualcmp_framework_imagegen_v1.png`（1846×852 RGB）。图以白底、横向流水线、低饱和分支配色表示aligned CAV features、fused-BEV tracking、agent-specific aggregation、motion context和decoder；删除旧图TACF/SFQT/GAF/CHIE/MAMD缩写，不画loss、训练phase、optional refiner或forecast-to-tracking feedback。该PNG尚未写入manuscript，也不是最终可编辑矢量稿。

## 2026-08-15 epoch 9 corrected fixed-horizon independent evaluation

- Independently evaluated `opencood/logs/colif_actor_source_release/checkpoint_epoch_009.pth` on the complete official test split with two RTX 3090 processes, `receiver_ids=[0,1]`, and `forecast_metric_protocol=cmp_fixed`. Strict checkpoint loading succeeded with all 542 model-state keys; the run completed 3,986 receiver frames without traceback.
- Corrected 10 Hz metrics use ADE windows `[:10]/[:30]/[:50]`, FDE indices `9/29/49`, and one common complete-5-s population of 32,833 matched actor-frames. Results: minADE `0.908741/1.496983/2.125372 m`, minFDE `1.092558/2.184541/3.350971 m`, MR@2m `0.133281/0.275546/0.334359` at 1/3/5 s.
- Tracking/lifecycle results: MOTA `0.650443`, MOTP center `0.726322 m`, recall `0.810801`, precision `0.848606`, IDS `1,037`, fragments `693`, forecast coverage `0.804925`.
- Result: `opencood/logs/colif_actor_source_release/metrics_test_cmp_fixed_epoch009.json`; log: `opencood/logs/colif_actor_source_release/eval_test_cmp_fixed_epoch009.log`.
- The checkpoint embeds Phase 2, epoch 9, seed 20, 50 future frames, both receivers, and Git commit `9cafb8c`; its saved dirty status includes the model files. Current model-code differences relative to that commit are documentation-string wording only, while the metric evaluator is the intentional later change. Thus the checkpoint is directly loadable for current evaluation, but corrected fixed-horizon numbers must not be mixed with the older masked-variable-horizon rows without a protocol label.

## 2026-08-14 DualCMP 题目与论文主张收紧

- 论文题目当时暂定为 **DualCMP: Task-Adaptive Cooperative Motion Prediction with Shared and Source-Specific BEV Representations**；2026-08-18已按实际FuseBEVT输出更正为`Fused and Source-Specific`。
- 当前主张不再把 feature aggregation 直接定义为已证实的“information-loss bottleneck”，而是提出可由现有固定 tracker 消融回答的问题：fused BEV features与aligned CAV feature maps是否为预测提供互补信息。persistent agent queries只作为系统接口，不作为首创贡献。
- Introduction 已明确 V2VNet、UniMM-V2X、ViP3D、CoopTrack 的相邻工作边界，限定 V2V4Real 为 dual-vehicle 证据，并如实报告 5 s minFDE 仅改善 0.4%；当前证据支持 short/intermediate-horizon conditional forecasting，不支持长期意图、跟踪改善或多源扩展性主张。

## 2026-08-14 CMP multi-ego receiver 接口与恢复点

- 已创建当前 CoLIF 主仓库恢复点：`5e3477a checkpoint: current CoLIF implementation`。训练日志/checkpoint（约194 GB）和`.serena/`索引没有提交；MOTR 外部依赖的两处本地兼容修复位于其独立仓库提交`cbdde0c fix: support batched CoLIF queries`，未推送。
- 已逐段复核 CMP `IntermediateFusionDatasetMultiEgo`：它对每个scene的每个`ego_cav_id`建立独立记录，其他CAV根据该ego坐标变换点云；CoLIF 的`ColifSequenceDataset`已同构地为`(scene_id, receiver_id)`构建clip，`_source_ids_for_receiver`保证当前receiver位于source slot 0，`SceneReceiverDistributedSampler`和runtime state使用`(scene, receiver)`边界。因此不向`PointPillarColif`增加receiver-ID embedding、第二个FuseBEVT或第二套query bank。
- 新增`opencood/tools/colif/receiver_protocol.py`：三个入口共享`--receiver-ids`（例如`0`或`0,1`），在dataset构建前覆盖`fusion.args.receiver_ids`；训练日志、strict-online metrics JSON与GT-free runtime artifact都会记录最终列表。默认YAML仍是`[0]`，而`[0,1]`必须从Phase 1重新训练并与同协议评测，`--resume`继续通过既有config hash拒绝跨协议恢复。
- 回归：新增receiver协议/slot-0测试；完整`tests/colif`为`121 passed`（19个已知warning），相关入口`compileall`通过，`git diff --check`通过。真实V2V4Real单卡Phase-1一批`--receiver-ids 0,1 --smoke-batches 1 --no-eval`完成：dataset records=`3420`（先前receiver-0 protocol的1710的两倍）、`receiver_ids=[0,1]`写入log、forward/backward/optimizer成功、未用可训练参数0、runtime GT guard保持启用。该随机初始化单批仅验证通路，不是性能实验。

## 2026-08-13 最终固定种子消融、双预测分支与当前主张（当前最新）

- 当前论文主张已从被否证的“future improves tracking”切换为：**面向 current detection/tracking 的单一 dense fused BEV 不必然是 motion forecasting 的充分融合；CoLIF 保留 CMP FuseBEVT 测量路径，同时让 persistent actor query 从 receiver-aligned per-source BEV 中选择预测相关证据。** forecast不再反馈 tracking，主模式固定为`future_prior_mode=none`。
- 最终在线结构：receiver-aligned multi-CAV LiDAR → CMP PointPillar source BEV → (a) FuseBEVT current measurement/query tracking；(b) actor-relative geometry-conditioned source patch attention → ordered variable history → actor interaction → six-mode motion proposal。第一次观测只建立 state，至少两次因果观测后输出 K=6、T=50 forecast。
- 保留两个可切换预测出口。默认`cross_mode + inference_layer=proposal`用于整体最佳结果；可选`trajectory_source + inference_layer=refined`使用 mode×time state tokens，沿候选轨迹在各 source BEV重新采样。按用户决定，两者均保留，不删除复杂分支。
- 已修复公平性漏洞：旧结构消融的 DataLoader iterator seed受模型参数初始化消耗的全局torch RNG影响。现在训练loader使用独立`torch.Generator(seed+rank)`，worker同步Python/NumPy seed；geometry/slot分支始终实例化相同参数、未用分支冻结。固定seed smoke中source ON/OFF的active queries、birth、current box与tracking loss逐batch一致。
- 四组公平完整实验共同使用`colif_fused_v7_phase1_predict_correct_2epoch_20260812/checkpoint_epoch_001.pth`、seed20、official train 1710 clips、双卡855 updates、固定1 epoch、official test 1993 frames。四组均0 AMP overflow、0 no-signal skip、parameter/buffer checksum delta0；MOTA=0.633329、coverage=0.789404、actor_count_5s=25583完全一致。

| 结构 | ADE 1/3/5 s | FDE 1/3/5 s | 结果目录 |
|---|---|---|---|
| 无 actor-source | 1.1189 / 2.0404 / 2.7437 | 1.5749 / 3.0895 / 4.3980 | `colif_fused_v18_no_actor_source_seedfixed_full_1epoch_20260813` |
| actor-source、无 ordered history | 1.0504 / 1.9313 / 2.6535 | 1.4608 / 2.9931 / 4.3827 | `colif_fused_v22_no_temporal_history_seedfixed_full_1epoch_20260813` |
| **默认 actor-source + history** | **0.9997 / 1.8472 / 2.5870** | **1.3708 / 2.9177 / 4.3804** | `colif_fused_v19_geometry_source_seedfixed_full_1epoch_20260813` |
| trajectory-source refined | 1.0728 / 1.9512 / 2.6714 | 1.4703 / 2.9896 / 4.3836 | `colif_fused_v21_identity_trajectory_source_seedfixed_full_restart_1epoch_20260813` |

- 默认分支相对无 source：ADE1/3/5改善10.65%/9.47%/5.71%，FDE1/3/5改善12.96%/5.56%/0.40%。ordered history相对无 history：ADE1/3/5改善4.82%/4.35%/2.50%，FDE1/3改善6.16%/2.52%，FDE5仅0.05%。因此主张短中期和整段轨迹改善，不声称显著降低5秒终点。
- trajectory-source总体不如默认分支，但partner-only FDE5=6.4371（默认6.4486）、age2--3 FDE5=12.4346（默认12.4862）略好。它可用于困难目标分析/增加方法完整性；必须同时报告总体负结果，不能只挑分桶。
- trajectory-source第一版在真实训练batch100暴露非identity初始化：随机mode score、独立uncertainty替换proposal。已改为mean/logstd/rho/logits全部对proposal做零初始化有界residual；新增初始化等价、有效partner梯度、masked partner不变测试。完整`tests/colif`为110 passed。
- CMP no-aggregation reference为ADE5/FDE5=4.4001/7.5559；默认CoLIF为2.5870/4.3804，数值低41.21%/42.03%。但CMP cache已核实有GT-ID oracle且目标数/协议不同，只能写“各自已复现协议下数字更低”，不能写严格apples-to-apples全面超越。
- 当前训练与选择命令已重写到`docs/colif_training.md`；`model_design.md`已清除混杂的旧蓝图并成为唯一现行实现合同；完整失败与修复账本位于`docs/colif_failure_postmortem.md`第15节。
- `manuscript.md`已按当前代码和消融从头改写Introduction、Related Work、Method与当前实验边界；主叙事为task-adaptive dual-path fusion，明确trajectory-source总体不如默认分支、future不反馈tracking、birth帧不预测。`tits-academic-writing` audit已运行；脚本因正文使用BibTeX keys而不统计编号citation，ledger已逐项绑定本地论文/章节。
- 最终回归：`tests/colif`为110 passed、19 warnings；相关模型/数据/loss/train/eval/inference通过compileall；默认与trajectory-source两套YAML覆盖均可CPU构建，参数量分别195,675,723和196,283,730；`git diff --check`通过。
- 未完成风险：容量上限（runtime active queries=24、target/collate max actors=128）尚未做压力测试；当前one-epoch/单seed消融证明方向，不等于多seed统计显著性；与CMP的target population/evaluator协议仍不严格等价。

## 2026-08-12 项目完整失败复盘与当时停止点（历史）

- 已新增独立复盘：`docs/colif_failure_postmortem.md`。它按 CMP 复现 → source-local/no-fuse → fused v1--v11 的顺序，记录设计变更、完整 test 数字、工程故障、已证实根因、未证实推断和被实验推翻的论文主张。后续不能只引用本文件较早阶段的“已实现/已验收”描述而忽略该复盘。
- 架构再审计已追加到复盘第13节。HPNet 已由本地正式论文核实为 `historical prediction → current prediction`，假设 agent track/identity 已知，不做 detection/association/MOTA/IDS；它不能为 `future → tracking` 背书。当前 CoLIF 也没有 HPNet 式 per-mode、overlap-aligned historical prediction attention。
- `measurement_observed_mask=False` 不是 actor 离开视野：它只是 dense peak 经过 score/local-max/suppression/top-k 后，没有被纯距离 Hungarian 在4 m gate内一对一分给该 query。它可能表示 out-of-range、低点数、detector miss、NMS抑制、proposal竞争、传播误差或association failure；当前 evaluator尚未按原因分桶。
- “只在缺测时开启 future”被判定为错误修补：缺测在 matching 之后才产生，而且缺测时 measurement gain=0、future-influenced decoder box不写回空间状态。正确理论路径应让上一帧 forecast 在本帧 measurement 之前形成短期多模态 spatial prior，并由 measurement likelihood做连续校正。
- 当前最重要的同权重反事实：`opencood/logs/colif_fused_v11_motion_feedback_full_1epoch_20260812/checkpoint_epoch_000.pth` 在 feedback ON 时 5 s minFDE/MOTA/coverage/IDS 为 `4.3946/0.6320/0.7888/685`；关闭 feedback 后为 `4.3403/0.6419/0.7994/670`。关闭反馈后这些主要目标指标均更好，当前模型不能声称 future cue 改善 tracking；但 fragments 为 `368`（OFF）对 `365`（ON）、FP 为 `4682` 对 `4647`，不能表述成所有指标都更好。
- 当前约 4.3--4.4 m 的 minFDE 主要由六个速度尺度 motion anchors 的 best-of-six 覆盖贡献：未训练诊断约 4.67 m，3 optimizer-step checkpoint 为 4.3250 m，完整一轮为 4.4161 m；mode loss 均值约 1.783，接近 `ln(6)`。学习到的 mode ranking、calibration、residual 和 recurrent gain 尚未成立。
- 当前 future cue 只更新 query content；measurement Hungarian 已在 decoder 前由 propagated box 与 CMP dense proposals 完成，缺测时 measurement gain 为 0，lifecycle 又只看 assigned measurement/score/TTL。future 没有进入 spatial association、missing-state propagation 或 survival decision，这是论文主张失败的核心架构原因。
- 第一帧只有位置，无法识别速度。当前代码已改为第一帧创建 persistent state、至少两次因果观测后才 forecast；旧“first evidence immediate useful 5-second forecast”主张已失效。
- 用户已要求暂停继续修改模型代码并先完成架构复盘。当前不得以延长训练、test 选模、降低 coverage、使用 GT、修改 evaluator 或只调 gate/loss 的方式人为改善指标。若继续，必须先提出 future 进入空间 gating/association 或缺测传播的可证伪新路径，再做同权重 on/off 验证。

## 2026-08-12 fused CoLIF v3 完整输入输出审计、修复与双卡一轮验收（历史记录）

- 已按 Data → receiver-frame preprocessing → CMP CoBEVT/FuseBEVT → persistent query → lifecycle/history → CMP MTREncoder/MTRDecoder → GMM loss → DDP train → strict-online test → GT-free inference 完整复核。逐模块张量合同、坐标约定、监督隔离和实际 shape 见 `docs/colif_model_io_audit.md`。
- 已核实并修正上一版预测输入的结构错配：原代码只给 MTR decoder 当前 query/current box，未运行 checkpoint 对应的 11×22 MTREncoder；历史框跨 receiver frame 未变换；训练只从空状态看4帧而test长序列递归；loss在NMS后的6模态而非原始64 intention上计算；box尺寸次序与正模态选择也不符合CMP。当前实现严格加载MTREncoder 166 + decoder 714，共880个state tensors，missing/unexpected均为0。
- 当前 Phase 2 每个样本使用11个连续runtime frames，其中前7帧只做causal burn-in、后4帧计算loss；tracker temporal GRU仍只读与Phase 1一致的最后4槽，MTR读取完整11槽。所有history boxes先变换到当前receiver frame。MTR属性严格为22维，输出训练64模态、公开NMS后6模态×50帧。
- 每卡batch=1时，只有一个age=1 actor会使预训练MTR的BatchNorm统计不可靠；现在Phase 2冻结36个BN running statistics，但保留affine梯度。修复前同一首batch GMM/grad norm为115.5/1714，修复后为5.678/21.654。无future target的rank现在贡献raw zero count，不再制造伪分母。
- 当前诊断配置明确为`phase2_feedback_mode=none`：Phase 2只训练正确输入合同下的forecast predictor，冻结CoBEVT/query tracker，避免把预测输入修复与future-feedback闭环混在一起。本轮结果不能用于证明future cue改善tracking；bounded FutureAdapter代码保留但未参与本结果。
- 双卡完整一轮已完成：目录`opencood/logs/colif_fused_v3_phase2_epoch1_final_20260812`，1710个global clips、855个optimizer updates、0 AMP overflow、0 no-signal skip；parameter checksum和显式同步后的buffer checksum跨卡最大差异均为0，checkpoint已保存。Phase 1冻结的400个tracker tensors与初始化checkpoint逐项完全相同；新checkpoint含1294个model state tensors，浮点tensor无NaN/Inf。
- 同一训练进程随后自动完成双卡整场景分片的1993帧official test：`strict_online=true`、`runtime_uses_ground_truth=false`；5 s minADE/minFDE=`5.3127206163/10.7843668489 m`，MR@3.6m=`0.7866859027`，tracking recall/precision=`0.7465033897/0.7590436268`，MOTA=`0.4581017529`，IDS=`1684`，forecast coverage=`0.7483727674`。
- 相比旧 fused Phase-2 已完成epoch中最好的5 s minFDE 11.7361256393 m，新结果降低0.9517587904 m（8.11%）；5 s minADE由5.9129101300降至5.3127206163 m（10.15%）。但它仍落后旧source-local/no-fuse与CMP no-aggregation的不同协议结果，不能写成已经达到基线或论文性能。
- 独立GT-free inference已从新checkpoint连续输出2帧到`runtime_test_2frames.pth`；输出仅含scene/frame元数据、匿名runtime ID、box/existence/age/miss、6×50 forecast、source availability/timing，不含GT、`ass_id`或future target，所有浮点输出有限。
- 继续检查完整future-feedback分支时发现并修复了一个真实架构/数值边界错误：同一外层AMP context先以`no_grad`运行7帧burn-in时，会缓存FutureAdapter Linear层的无梯度FP16权重；后4帧虽开启梯度仍复用该缓存，表现为首batch有64个kept feedback query-frames且gate非零，但cue MLP/gate共8个参数`grad=None`，10步后这些权重仍完全不变。`FutureAdapter.forward`现固定为FP32小岛，避免混合`no_grad`/grad时的autocast weight-cache断图。
- 修复后最终代码双卡recurrent-BPTT 2-batch smoke位于`opencood/logs/colif_fused_v3_phase2_feedback_ampfixed_2batch_20260812`：首batch64个kept query-frames，max gate/delta=`0.0044966/0.0179863`，未覆盖可训练参数0；第二步后cue projection、gate末层及原本因零初始化首步梯度为0的gate首层均发生有限非零更新；0 AMP overflow，parameter/buffer checksum delta均为0。默认YAML已恢复`phase2_feedback_mode=none`，因此该smoke不改变上一条完整一轮无反馈指标，也不能当闭环性能结果。
- 新增针对“AMP内先no_grad burn-in、再grad suffix”的回归测试。最终回归：`conda run --no-capture-output -n coop python -m pytest -q tests/colif`为38 passed、1个仓库既有Matplotlib弃用warning；相关data/model/loss/train/eval/inference模块`compileall`成功；`git diff --check`成功。
- 剩余研究风险：age=1与age=2--3 actor的5 s minFDE仍为22.91/18.49 m；“第一证据即可forecast”的执行能力成立，但早期forecast是否有用尚未成立。当前仓库仍有大量本任务历史日志和未跟踪CoLIF文件，未提交；未做破坏性Git操作。

## 2026-08-12 fused CoLIF v2 两阶段性能诊断（覆盖“继续训练即可收敛”的假设）

- 截至 2026-08-12 07:53（UTC+8），`opencood/logs/colif_fused_v2_phase2/metrics_test.jsonl` 已完成的 epoch 0--4 均为同一 strict-online official test（1993 帧、约 23.5k 个 matched forecast actor-frames、`runtime_uses_ground_truth=false`）。5 s minFDE 依次为 12.9098 / 12.6569 / 11.9540 / 12.6820 / **11.7361** m；虽有波动改善，仍不能用“再训练几轮就会自然接近基线”解释当前约 4.18 m 的差距。训练已进入 epoch 5，本文不把未完成 epoch 当结果。
- 当前两阶段主模型明显落后于已复现的 CMP no-aggregation epoch 5（5 s minFDE 7.5559 m），也落后于旧 source-local/no-fuse CoLIF 已有 full epoch（8.1841 m）。后两个结果与当前 strict-online evaluator 的 target eligibility、identity/aggregation 协议不完全相同，因此只用于确认量级差距，不能作为严格单变量消融。
- Phase 1 tracking 本身较弱且波动：20 个 official-test epoch 中最好 MOTA 为 0.4366（epoch 16），实际初始化 Phase 2 的末轮 epoch 19 为 0.4033；Phase 2 epoch 0--4 的 tracking recall 约 0.724--0.728、IDS 为 1653--1779。它会压低 detection-aware forecast coverage/质量，但不能单独解释预测头退化：当前最好 epoch 4 中 age>=4 的 query 仍有 11.5874 m 的 5 s minFDE。
- 已核实的首要结构问题是 MTR 输入合同错配。`PointPillarColif._forward_single` 只向 `PersistentQueryMTRAdapter` 传当前 `query_feat/ref_box/active_mask`；adapter 把当前 query embedding 直接当作 MTR 的 center/object context，并只从当前框构造相对位置。CMP 原 MTR checkpoint 的 decoder 权重则是在 `MTREncoder` 对 11 帧、22 属性轨迹及 mask 编码出的 context feature 上训练。当前仅加载 `motion_transformer.motion_decoder.*` 的 714 个 tensor；shape/name 严格匹配不代表输入表示语义匹配。Phase 2 首 batch 的 GMM component=2409.508、gradient norm=32531.09 是该分布跳变的直接旁证。
- 当前 forecast head 也没有显式消费已经存在的 `history_box/history_mask/velocity/age`；4 帧非重叠训练 clip 每个 sample 都从空 runtime state 开始，而 official test 在整段 scene 中持续递归，形成训练只见 query age<=4、测试大量 query age>4 的状态分布偏移。这是代码已确认的第二项风险，但它对 FDE 的独立贡献尚未通过实验定量。
- 因此不应通过恢复第三阶段或单纯延长当前 Phase 2 解决。下一版必须先实现 age/mask-aware 的 variable-history actor encoder，把 query/box/velocity history 映射为 MTR-compatible center/object features，同时保持 age=1 可预测；先在 feedback=off 下证明 forecast head 本身有效，再开启 bounded future recurrence。
- 最小根因消融应依次为：同 checkpoint 的 feedback on/off strict-online 对照；GT-current-state/正确 identity 的 forecast-head oracle（隔离 tracking 错误）；当前 4 帧 reset clip 对比带 causal burn-in 的 8--12 帧 clip（隔离 query-age 分布偏移）。完成这些前，不得把 future feedback、短历史或 tracking 任一项单独写成已证实主因。

## 2026-08-11 fused CoLIF v2 根因修复与两阶段重新验收（历史记录）

- 旧 Phase 1/2/3 性能差不是用“尚未训练完”解释。代码与真实日志审计确认三项结构问题：query existence 没有连续使用同一 CoBEVT 实例已有的 fused detection heatmap；MTR actor-local 轨迹/上下文曾被当作 receiver-frame；旧 Phase 3 解冻 tracker 并重新升高学习率后 tracking/coverage 发生结构性崩塌。旧三阶段 checkpoint 只保留作故障证据，不能继续训练或作为当前模型结果。
- 当前仍然只有一个 `PointPillarCoBEVT` 实例 `cobevt`。query existence 现在在 decoded query center 连续采样该实例的 `cls_head(fused_bev)`，与 query logit相加；没有第二个模型、NMS、检测框后处理或 AB3DMOT。
- MTR 合同已修正为 center-actor heading frame：actor context位置和 forecast target先旋入 actor-local frame，GMM mean/covariance在公开输出时精确旋回 receiver frame；loss 使用 `pred_trajs_local`，evaluator 使用 receiver-frame `pred_trajs`。旧预测 checkpoint 在错误坐标下训练，不能仅靠新推理代码修复，必须重训 Phase 2。
- 生命周期配置改为 birth/keep=`0.30/0.20`、最多24 active queries；criterion identity在对应 runtime slot死亡时释放，避免训练时用GT identity维持一个推理时已被重置的slot。Phase 1使用同一模型的 dense evidence辅助 query objectness。
- 训练协议已收敛为两阶段：Phase 1训练 fused query tracker、forecast关闭；Phase 2冻结 CoBEVT/query tracker，只训练CMP MTR与bounded FutureAdapter，同时允许下一帧tracking loss穿过冻结计算图回到上一帧future cue。冻结的CoBEVT/query decoder保持eval；无BN/dropout的冻结temporal GRU保持train mode，仅为满足cuDNN recurrent backward，参数仍不更新。
- 跨阶段加载只导入 Phase 1实际训练的400个tracker state tensors，MTR decoder从本地CMP checkpoint严格加载714个state tensors，Phase 1未训练的forecast/FutureAdapter保持当前初始化。加载器校验允许PyTorch不报告BN `num_batches_tracked`的兼容行为，但任何非forecast/future缺键或unexpected key仍立即报错。
- 修复过程中真实双卡依次抓出并修复了三项错误：跨阶段missing-key校验误判BN计数；Phase 2不计算冻结aux detection loss却仍在weighted sum读取其键；冻结GRU eval forward后执行cuDNN backward报错。均已加入或扩展回归测试，未通过跳过检查或降低标准隐藏。
- Phase 1修复版真实双卡10-batch smoke：`opencood/logs/colif_fused_v2_phase1_repair_smoke_20260811/checkpoint_epoch_000.pth`；20 global micro-batches、10 optimizer steps、0 overflow、首反向unused trainable=0、双卡checksum delta=0。
- Phase 2修复版真实双卡10-batch smoke：`opencood/logs/colif_fused_v2_phase2_smoke_fixed4_20260811/checkpoint_epoch_000.pth`；20 global micro-batches、10 optimizer steps、0 AMP overflow、0 no-signal skip、45,804,760个trainable参数首反向全部覆盖、双卡checksum delta=0。与Phase 1逐tensor比较，400个冻结tracker tensors变化数为0；与MTR初始化比较，714个decoder tensors中427个已实际更新。
- Phase 2 strict resume已从epoch0/global step10恢复，在双卡epoch1继续到global step12，0 overflow、checksum delta=0；输出位于`opencood/logs/colif_fused_v2_phase2_resume_smoke_20260811`。
- 为检查早期大幅波动的GMM NLL，又从同一epoch0/global step10断点覆盖50个不同真实batch，连续更新到global step60：100 global micro-batches、0 AMP overflow、0 no-signal skip、双卡checksum delta=0，mean clip loss=71.5166；记录和checkpoint位于`opencood/logs/colif_fused_v2_phase2_stability50_20260811`。这证明有限性和训练稳定通路，不证明已收敛。
- 真实两帧因果梯度隔离测试只使用`t+1` tracking loss，不使用forecast loss：上一帧latent projection梯度L1=0.09235、FutureAdapter=45.7853、MTR decoder=0.24667，均有限非零；冻结query decoder参数梯度张量数为0。证明future cue确实进入下一帧tracking recurrence。
- 双卡strict-online截断test已严格加载当前checkpoint并完成128帧，`runtime_uses_ground_truth=false`、`world_size=2`；输出为`opencood/logs/colif_fused_v2_phase2_smoke_fixed4_20260811/metrics_test_64x2.json`。其中tracking recall=0.7641、precision=0.8654、MOTA=0.5711、coverage=0.7665、minFDE5=12.058；该权重只更新10步且test被截断，指标仅证明通路，严禁作为论文结果或模型选择依据。
- 最终回归：`tests/colif` 34/34 passed；相关data/model/loss/train/eval模块`compileall`通过。正式命令见`docs/colif_training.md`；默认每个完整epoch自动official test，不建val、不自动选择best。
- 尚未完成：修正后Phase 1/2收敛训练、完整official test、no-feedback/detached与公平CMP基线/消融。因此当前可以确认代码通过既定工程合同和双卡通路验收，不能确认论文性能已经成立。

## 2026-08-11 Phase 3 正式启动 DDP reduction 故障与修复

- 用户从 `colif_fused_phase2/checkpoint_epoch_009.pth` 初始化双卡 Phase 3 时，在第一个 batch 成功、后续 iteration 启动前收到 `ChildFailedError`。保存的 JSONL 只显示首 batch 成功；使用 `TORCH_DISTRIBUTED_DEBUG=DETAIL` 和相同 seed 重现后，真实根因是 DDP `Expected to have finished reduction`，rank 0 未触发 `forecast_head.latent_projection.{weight,bias}` 的梯度 hook。
- 根因是 Phase 3 的 `future_cue` 属于跨帧 recurrent 输出：actor 若只在 clip 最后一帧出现，latent projection 会生成并通过最终 `state` 返回，DDP 因此把参数判断为已使用，但本 clip 已无下一帧 loss 消费该 cue，backward 不会触发对应 hook。它不是 checkpoint 损坏、数据 worker、非有限 loss 或永久 dead parameter。
- `opencood/models/point_pillar_colif.py` 的 `training_anchor` 现在额外加入终帧 `future_cue` 与 `future_uncertainty` 的零值 autograd 路径。它不改变 loss、推理输出或较早帧 cue 的真实梯度，只让合法的终帧 recurrent graph 完成 DDP reduction；完全未执行的条件分支仍由 `find_unused_parameters=True` 识别。
- 修复后双卡真实数据连续 30 batches（workers=0）完成 30 optimizer steps：60 global micro-batches、0 overflow、0 no-signal skip、双卡 parameter checksum delta=0；日志与 checkpoint 位于 `opencood/logs/colif_fused_phase3_ddp_terminal_anchor_20260811`。
- 使用与正式命令相同的 workers=4 又完成双卡 10 batches：20 global micro-batches、10 optimizer steps、0 overflow、checksum delta=0；位于 `opencood/logs/colif_fused_phase3_ddp_workers4_anchor_20260811`。原错误在第 3 iteration 前稳定复现，两个修复后运行均已越过该点并正常退出。
- 回归验证：`tests/colif` 28 passed；相关模型/训练/loss `compileall` 通过；`git diff --check` 通过。首次直接调用环境里的 `pytest` 因其可执行文件未包含项目根目录而 collection 失败，改为该环境的 `python -m pytest` 后通过；这不是模型代码失败。
- 原 `opencood/logs/colif_fused_phase3/train_log.jsonl` 含失败启动的部分记录且没有正式 checkpoint。正式重启建议换一个干净 output directory，避免把失败与完整训练日志混在一起。

## 2026-08-10 fused-BEV CoLIF 已实现与验收（覆盖下方旧路线记录）

- 当前运行主链已切换为：receiver-aligned multi-CAV LiDAR → CMP CoBEVT/FuseBEVT → 单一 persistent 3D query bank → current box/velocity/existence → CMP MTR 6-mode×50-step forecast → bounded latent-only recurrence。旧 source-local observation/association文件仍在仓库，但不在`PointPillarColif`主路径。
- MOTR官方依赖固定于`third_party/MOTR`，CUDA deformable-attention已编译并完成GPU forward/backward；CMP MTR decoder严格加载714个参数；CoBEVT官方checkpoint严格加载259个参数且真实batch数值等价。
- 数据已恢复CMP真实几何语义：source点云先投到receiver frame再voxelize。canonical identity以agent0 local key为基准，partner稳定`ass_id`映射并向前回填；该映射仅进入target/evaluator，runtime递归GT guard已通过。
- Phase 1双卡完整一轮位于`opencood/logs/colif_fused_phase1_full_fp32_20260810`：1764 global clips、882 updates、0 skipped、checksum delta 0；自动完整official test共1993帧。该一轮tracking尚未收敛，不能作为论文性能。
- 用户随后明确“batch不需跑完，只需足够验证模型正确”。Phase 2连续10-batch双卡checkpoint位于`opencood/logs/colif_fused_phase2_ddp_throughput10_packedcounts/checkpoint_epoch_000.pth`：0 overflow、checksum delta 0；Phase 3最终2-batch checkpoint位于`opencood/logs/colif_fused_phase3_ddp_finalsmoke_20260810/checkpoint_epoch_000.pth`：FP32、0 skipped、首反向未覆盖参数0、checksum delta 0。
- Phase 3严格resume已通过：`opencood/logs/colif_fused_phase3_resume_verified_20260810`从epoch0/global step2恢复，在epoch1更新到global step3，optimizer/scheduler/scaler/参数组均连续。
- Phase 2全局无forecast target保护已通过强制空actor双卡测试：`global_step=0`、LR不变、`no_signal_skipped_steps=1`、checksum delta 0；空批次仍完成DDP backward，但不推进Adam/GradScaler/scheduler。
- 闭环梯度已用真实两帧直接验证：loss只取`t+1` current boxes，不含forecast loss；梯度仍到达`t`的MTR latent projection（norm 3.8415）、FutureAdapter cue projection（2.7875）和gate（0.1437），均有限非零。
- 双卡strict-online test子集结果位于`opencood/logs/colif_fused_phase3_ddp_verified3_20260810/metrics_test_50frames_per_rank.json`：100帧、`runtime_uses_ground_truth=false`、minADE 1/3/5s=2.139/5.464/7.907m、minFDE=3.756/10.752/15.502m、coverage=0.476、MOTA=0.227。仅用于证明指标路径，checkpoint只训练数步，禁止作为论文结果。
- 最终checkpoint的20帧双卡回归位于`opencood/logs/colif_fused_phase3_ddp_finalsmoke_20260810/metrics_test_agebins_10frames_per_rank.json`，同时验证age=1、age=2–3、age>=4的5秒minADE/minFDE/count可跨卡汇总；截断test结果同样不得作为论文结果。
- 最终checkpoint的GT-free inference已输出2个连续runtime records到`opencood/logs/colif_fused_phase3_ddp_finalsmoke_20260810/runtime_test_2frames.pth`；记录只含runtime ID、box、score、age/miss、forecast、source/time metadata，递归检查无GT/`ass_id`/target/label字段。
- 已修复两项仅在真实双卡暴露的问题：条件MTR分支不能整体转换为SyncBatchNorm，否则不同rank collective次数不一致并卡在forward；一个DDP backward前必须由模型内部完成整个causal clip unroll，不能连续调用wrapper四次。当前使用普通BN、clip-boundary buffer广播、单次DDP clip forward与条件unused-parameter发现。
- Phase 1/3因真实长链梯度在FP16下溢出而使用FP32；Phase 2冻结tracker后使用AMP。loss counts按每帧一个packed all-reduce同步；无forecast target的rank贡献0 numerator/denominator，不再插入伪计数。
- 训练协议仍为官方train优化、每个完整epoch后official test、不建val、不根据test自动选best。短smoke使用`--no-eval`；正式训练默认自动test。完整命令见`docs/colif_training.md`。
- 尚未完成：收敛训练、完整Phase 2/3 official-test、CMP/no-feedback/fixed-history与gate/dropout消融、论文级统计。`manuscript.md`仍包含部分旧source-aware叙述，必须在投稿写作前按当前fused-BEV方法重写。
- 最终静态与单元复核：`tests/colif`为28 passed；相关模型、数据、loss、训练、inference/evaluator均通过compileall；`git diff --check`通过。当前工作树保留用户原有与本任务新增的未提交文件，没有执行commit或清理日志。

## 2026-08-10 CoLIF 主路线切换为 CMP fused-BEV persistent forecast queries

- 用户最终确认放弃当前 source-local observation/跨车 association 主路线，主模型固定为：多车 LiDAR 按 CMP 语义对齐并生成 fused BEV，单一 receiver-centric persistent actor query 同时承担当前状态、生命周期和即时多模态预测，预测 latent 仅以有界方式参与下一帧 query-content recurrence。完整实施合同已重写到 `model_design.md`，它覆盖本文件后续旧章节中的 source-local 设计结论。
- 已用 Serena 激活并复核当前项目，真实 CMP 路径为 `IntermediateFusionDatasetMultiEgo` 在 voxelize 前把 source 点云投到 receiver 坐标，随后 `PointPillarCoBEVT` 执行 PillarVFE、Scatter、BEV Backbone、regroup 和 `SwapFusionEncoder`。现有 model forward 不执行第二次显式 BEV warp；新 sequence loader 必须恢复这一真实语义。
- 最大复用边界已固定：完整复用 CMP CoBEVT 主干、FuseBEVT、anchor auxiliary loss和官方 checkpoint；复用仓库 `point_pillar_cobevt_motr.py` 的 sequential query 结构；通过薄 adapter 复用本地 CMP MTR decoder、GMM loss和NMS；继续复用现有 `train_colif.py` 的 clip/DDP/AMP/checkpoint/evaluator框架。
- 当前 CMP MOTR prototype 不是现成可运行依赖：仓库缺少顶层 `MOTR/`、对应 YAML 与可验证 checkpoint，直接 import 会报 `ModuleNotFoundError: MOTR`。实施 M0 必须从官方 MOTR 固定 commit恢复依赖并验证许可证、编译和最小 forward；只能改 3D box/fused-feature 接口，不能重新自造完整 Transformer tracker。
- 当前 `PointPillarColif`、source-local dataset、`ForecastAssociation`、source-aware query bank和旧 `FutureAdapter` 与新设计结构性不兼容。现有 Phase 1/Phase 5 checkpoints、30 tests、双卡一轮与日志仅验证已放弃旧路线，不证明新 fused-BEV CoLIF 可运行，也不能用于新模型 resume。
- 新 FutureAdapter 的硬边界：按 mode probability聚合 future latent，经 `g<=g_max` 和 update-norm bound只更新下一帧 query content；空间 reference只由 receiver pose与当前 box/velocity传播；forecast不进入 identity matching，不用多模态轨迹均值移动 box。
- 数据监督另有已核实问题：agent0 常见 `ass_id=-1`，partner 对同一 actor 才带有效 `ass_id`，当前 `_actor_key` 会把二者拆开。新 target builder以 agent0 local key为 canonical、partner有效 `ass_id`映射并对后续映射向前回填；该映射只进 criterion/evaluator，严禁进入 runtime。
- 训练协议仍为官方 train训练、每个 epoch后在官方 test监控、不构造val、不按test自动选择best。论文三阶段为 tracking warm-up、anytime forecast、bounded recurrent joint fine-tuning；工程验收按 M0–M5 逐门槛推进，只有新模型完成双卡真实一轮、strict resume和official-test后才能声称“完整可训练、双卡可运行”。
- 本次只重写设计文档并更新交接记录，没有修改模型/数据/训练代码，没有运行新模型训练或把旧结果冒充为新路线结果。

## 2026-08-10 CoLIF 论文 Introduction 与 Methodology 草稿

- 论文草稿位于 `manuscript.md`。当前包含英文 Introduction、strict-online problem formulation、完整 CoLIF methodology、24 个公式、learning objectives、working references、Introduction citation ledger 和投稿前主张边界；尚未写 Abstract、Related Work、Experiments、Discussion 与 Conclusion 正文。
- Introduction 已使用本机 `tits-academic-writing` skill 从头重写，而非沿用早期版本。实际核对的 T-ITS 引言包括 2024 *Efficient Baselines*、2024 *TrajPRed*、2025 *Social-Pose* 和 2026 *I2XTraj*；采用“交通运行后果 → V2V 任务条件 → 现有范式 → 一个机制缺口 → 设计原则 → 验证问题 → 三项贡献”的论证链。当前约 1,062 词，详细论文目录不堆在 Introduction，留给 Related Work。
- 论文核心主张不再是错误的通用 `hard track-to-forecast interface`，也不声称 CMP 存在 partner→ego track handover。当前主张是：现有代表性协同预测方法主要把 future distribution 作为当前 receiver time 的输出；CoLIF 维护 source-aware receiver-centric persistent actor belief，并让上一时刻完整多模态高斯 future belief 参与下一时刻异步跨来源 association、state prior 和 recurrent query update。
- Introduction 明确认可 V2XPnP 可在无独立 finalized tracker 的情况下端到端预测；ViP3D、StreamMOTP、ForeSight、CoopTrack、SparseCoop 已分别覆盖 persistent query、joint tracking/forecasting、forecast feedback 与 sparse cooperative instances。不能将这些单项算子写成 CoLIF 首创；详细区别应放入 Related Work。
- Methodology 已逐项对照当前代码：source-local PointPillar proposals、center `grid_sample`、pose/time metadata、FutureAdapter 全模式/协方差变换、full-mixture motion likelihood、训练 soft assignment/部署 per-source Hungarian、per-source FIFO memory、GRU query update、cross-source birth grouping、observed/coasting/expiry lifecycle、actor Transformer、6-mode correlated GMM、training-only target matching 与 truncated BPTT。文稿明确离散 NMS、Hungarian、query allocation/lifecycle control 不可微。
- 当前方法只声称 forecast feedback 作用于 cooperative association、state propagation 和 recurrent query content，不声称上一时刻 forecast 直接改变本帧 source-local detection proposals。
- Introduction 使用 7 个临时 BibTeX keys，均在 citation ledger 中绑定到真实论文、章节和本地路径/稳定 URL。`tits-academic-writing` audit 已执行；其数字引用计数为 0 是因为脚本只识别 `[n]` 而正文使用 `\cite{...}`，7 个 key 已另行集合核对无缺失；`git diff --check` 通过。
- 当前不得加入优于基线的 headline result。现有一轮 Phase 5 仅证明训练/evaluator 闭环，远未收敛；coverage、minADE、minFDE、MR 可按已实现 evaluator 报告，onset、continuity、source-transition、communication 和 robustness 指标在真正实现并运行前不能写成已完成实验。
- Introduction Fig. 1 已按 `tits-academic-writing` 的 visual-argument 规范设计并用内置 ImageGen 生成，最终修正版位于 `figures/fig1_colif_introduction.png`（1855×848）。图采用三面板“变化的协同证据 → representative open-loop terminal forecast → CoLIF closed-loop actor belief”，不展示未经验证的性能/安全增益；完整视觉 claim map、生成提示和建议 caption 位于 `figures/fig1_colif_introduction_prompt.md`。该 PNG 是论文构图初稿，投稿前应按批准构图重绘为矢量图。

## 2026-08-10 CoLIF 全模型双卡完整一轮与 official-test 闭环

- 用户最终指定 CMP-compatible protocol：官方 V2V4Real `train` 全量训练，每个完整 epoch 后直接评测官方 `test`；不构造 val、不按 test 指标自动选择 best。配置已移除 disjoint manifest，`validate_dir/test_dir` 都指向官方 test；CLI 的 eval split 只允许 `test`。已明确记录 repeated-test monitoring 与已知 294-frame train/test 重复场景风险。
- Phase 1 增加 `local_targets_only` 快路径：仍完整构造 point-supported source boxes 与 local anchor labels，但跳过 Phase 1 loss 不使用的 50-frame global targets；test 与 Phase 2–5 强制恢复完整 future target。YAML 做 worker-local LRU cache，非重叠 2-frame clips 覆盖每个 receiver frame 一次。
- 正式 Phase 1 双卡一轮完成：官方 train 7,084 clips、每 rank 1,771 batches、全局 3,542 micro-batches、1,768 有效 updates、3 次 AMP overflow 安全跳过，mean clip loss 0.656338。checkpoint：`opencood/logs/colif_phase1_full_verified_seed20/checkpoint_epoch_000.pth`。其 0.2 阈值 test 无 active query，指标诚实记为 null；50 帧 anchor score 抽样 max=0.05253、`>=0.02` 为 355 个。
- 新增显式 `--observation-score-threshold`，同时控制 local selection 与 bank evidence threshold并写入 checkpoint。standalone eval/inference 从 checkpoint config 恢复非参数 runtime 语义。早期完整模块验证固定为 0.02，不是 test-only override。
- 修复随机 forecast head 的百万级初始 GMM NLL：trajectory mean/rho 初始化为 0，log-std 初始化为 `log(5 m)`，mode scores 初始等概率；跨 phase 初始化只加载 source phase 实际训练/继承的模块，未训练模块保留当前稳定初始化。10-batch 自然 Phase 5 preflight 中 GMM 从约 `1.05e7` 降到 8.57，真实 births/active queries/forecast loss 均参与。
- 正式全模块 Phase 5 双卡一轮完成：10,992,784 参数全部 trainable，官方 train 7,084 clips、全局 3,542 micro-batches、1,765 有效 updates、6 次 AMP overflow 安全跳过、9,813 birth observations、平均末帧 active queries 16.5186、mean clip loss 3.83901。自然 hard batches 已记录非零 association/state/GMM/mode loss，不依赖 forced-birth。
- Phase 5 checkpoint：`opencood/logs/colif_phase5_full_verified_seed20/checkpoint_epoch_000.pth`，131,642,063 bytes。CPU strict load missing/unexpected 均为 0，全部浮点 state finite；optimizer、scheduler、AMP scaler(scale=16)、phase/config hash/threshold/provenance 均存在。
- 自动 strict-online official test 完成 3,986 frames，world size 2，整条 `(scene,receiver)` 序列不跨 rank，raw sums/counts all-reduce。结果：coverage 0.591748（5,321/8,992 actor-frames），minADE 1/3/5s = 2.7658/4.5977/5.4771 m，minFDE = 4.1071/7.1636/8.1841 m，MR@2.0m = 0.5307/0.5952/0.5221，MR@3.6m = 0.3155/0.4131/0.3648。unmatched-active rate 0.9673，说明只训练一轮远未收敛，不能当论文性能。
- 独立两卡 evaluator 用 Phase 5 checkpoint 再跑 4 frames/rank 成功，checkpoint runtime config、sequence sharding、进度日志和 all-reduce 路径均通过。当前完整 CoLIF tests 为 30 passed（最终全套复核待本节写入后再执行）。
- 最终复核：30/30 tests passed、相关模块 `compileall`、`git diff --check` 均成功。Phase 5 checkpoint strict-resume 双卡 1 batch 成功：从 epoch 0/global step 1765 恢复到 epoch 1/global step 1766，AMP scale 16、scheduler/optimizer 连续，association/state/GMM/mode 均为非零有限值；config hash 不一致时训练器现在会拒绝 resume。
- 已发现 Phase 2–5 主要吞吐瓶颈是在线构造 point-supported local labels 与 50-step receiver future target，而非 DDP；正式多 epoch 实验宜新增 supervision-only cache，但必须保持 runtime 输入无 GT、坐标/时间/证据规则和 cache provenance 一致。

## 2026-08-09 CoLIF 完整模型与双卡闭环

- 在 Phase 1 基础上新增 `GlobalForecastState`、`PoseTimeAligner`、`PersistentQueryBank`、`SourceMemoryUpdater`、`ActorContextEncoder`、`MTRStyleGMMForecastHead` 和 `FutureAdapter`。统一接口为 `model(runtime_frame, prev_state) -> outputs, next_state, aux`，model 不接收 targets。
- persistent bank 当前实现训练时 soft observation→query assignment、部署时每个 source 内 one-to-one Hungarian decoding、forecast motion prior、hard anonymous birth/runtime ID、cross-source near-center birth suppression、coasting/miss TTL/death、per-query/per-source length-6 memory。完整输出为 state/existence/lifecycle 加 6-mode×50-step `[x,y,log_std_x,log_std_y,rho]` forecast。
- receiver forecast targets 已从 supervision-only YAML 构造，当前 actor box 固定在 receiver frame，future 是相对当前中心位移；runtime clip 仍不含任何 label/future。loss 已包含 local cls/reg、query state、existence、canonical association/birth、masked GMM NLL 和 mode CE；unknown `ass_id=-1` association ignore。
- 训练器支持 Phase 1–5 freeze/unfreeze、同阶段 strict resume、跨阶段 strict weight init、truncated BPTT、AMP、梯度裁剪、分组 LR、DistributedSampler、DDP 和 SyncBatchNorm。循环必须调用 DDP wrapper 的 `forward`，不能绕过到 `module.forward_frame`。
- 双卡根因修复：recurrent clip 多次 forward 后一次 backward 时，DDP 默认每帧广播 BN buffers 会触发 autograd version error；采用 SyncBatchNorm 并设置 `broadcast_buffers=False` 后通过。
- 最严格真实 smoke：`torchrun --nproc_per_node=2 ... --phase 5 --smoke-force-birth`，全部 `10,926,989/10,926,989` 参数可训练，world_size=2，loss `0.23044865`、gradient norm `1.5253581`，成功保存 checkpoint。forced-birth 同时覆盖 birth、memory、lifecycle、forecast 与 FutureAdapter，不只是 local loss。
- GT-free inference 新增 `inference_colif_v2v4real.py`，用 Phase 5 双卡 checkpoint 在 test split 完成 2 个 chronological runtime records，输出不含 GT；加入部署 Hungarian 后再次完成同样的两帧回归。
- 当前 23 个 CoLIF tests 通过；真实 CoBEVT checkpoint trunk warm-start 匹配加载 142 个 tensors，21 个 compressor/fusion 等不兼容项被明确跳过。
- 运行说明见 `docs/colif_training.md`。尚未做完整数据 census、strict CMP adapter、正式 evaluator/长训练/论文指标；不能把 smoke checkpoint 当作收敛模型。

## 2026-08-09 CoLIF Phase 1 开始实施

- 新增 `colif_sequence_index.py`：receiver 自身帧定义时间轴，source missing/100 ms delay 通过 causal lookup 和 mask 表示，不再使用 legacy 最短-CAV时间轴。
- 新增 `build_split_manifest.py` 并生成 `opencood/hypes_yaml/colif/splits/v2v4real_disjoint.json`：实际得到 train/val/test 31/3/9 scenes；重复 `testoutput_CAV_data_2022-03-15-09-54-40_0` 保留 test、从 train 排除。
- 新增 runtime schema anti-leak 检查、source evidence audit、`ColifSequenceDataset` 和 `CanonicalTargetBuilder`。训练 runtime 与 targets 分开；只有 source PCD 中达到最小点数的 box 是 local positive，unknown identity 只忽略 association、不丢 detection positive。
- 初步均匀 smoke（每 source 4 帧，共 248 source-frames）中，311 个量程内 label 只有 76 个达到 3 点，约 24.4%；99 个 canonical actor-frame 均只在一个 source label set 出现。此结果是小样本预警，不是完整 Gate A 结论，训练前仍需全量/分层 census。
- 新增 `PointPillarColif` Phase 1、`SharedLocalPillarTrunk` 和 `LocalObservationEncoder`。真实 V2V4Real forward 输出 local BEV `(1,256,48,128)`、cls `(1,2,48,128)`、reg `(1,14,48,128)`；没有调用 FuseBEVT，selected observation feature 的梯度可回到 local BEV/trunk。
- 新增 `ColifLoss` local detection、`train_colif.py`、分组 optimizer、CoBEVT trunk partial loader、完整 checkpoint/resume。真实两帧 GPU smoke：首次 step loss `0.019075`、gradient norm `0.078293`；resume 后 epoch/global_step 从 `0/1` 到 `1/2`，config hash 一致。
- 几何复核后修正 source→receiver 组合为仓库矩阵约定 `inv(receiver_lidar_pose) @ source_lidar_pose`；禁止使用反向组合。
- 验证：`PYTHONPATH=. conda run -n coop pytest -q tests/colif` 为 17 passed；py_compile、`git diff --check` 通过。仅有仓库既有 Matplotlib `get_cmap` deprecation warning。
- 未完成：strict-online CMP、完整 evidence census、persistent bank/association/source memory/lifecycle、forecast/FutureAdapter/BPTT/evaluator。当前模型只可称 Phase 1 local observation path。

## 2026-08-09 CoLIF 设计审计

- 用户已明确当前只更新设计文档，不修改模型代码、不运行训练；目标是形成达到 IEEE T-ITS 工作量与证据标准的 CoLIF 方案。完整设计位于 `model_design.md`。
- 已用 Serena 激活 `/home/qkb/Project/CoLIF`，审计正式 V2V4Real 路径的 Data → Preprocessing → CoBEVT → detection → AB3DMOT → MTR → loss/train/inference/metrics，并保存项目 memories。
- 已核实正式 tracking cache 存在 GT-ID oracle：检测先以 IoU 匹配 GT，未匹配 detection 被删除，匹配 GT ID 进入 AB3DMOT，最终 track 又按该 ID 重命名；MTR dataset 和 multi-ego aggregator 继续用精确 ID 取交集/分组。现有结果不能当作无标签在线 pipeline。
- 已核实 MTR 只接受完整 11 帧 tracking history，并要求完整 50 帧 future GT；no-aggregation MTR 实际不消费缓存 CoBEVT BEV。正式代码中不存在 persistent forecast query 或 future-belief recurrence。
- 在复核用户早期对话与旧附件后，最终主方案不是 `FuseBEVT → 单一 MOTR query bank`，而是：各 CAV 共享 local encoder 独立产生当前 observation queries，经 pose/time alignment 进入每个 `(scene/session, receiver)` 唯一的 source-aware persistent forecast query bank；不先生成多套 local tracks，也不做 trajectory stitching。
- CoLIF 的方法闭环是：query 从协同点云 evidence 首次支持时出生，以可变历史立即预测；多模态 future belief 同时作为下一帧 query prior 和跨 CAV association 的运动兼容项；source-aware memory 保留 observation ownership/新鲜度/置信度。FuseBEVT 方案保留为强基线或 receiver scene-context 消融。
- 训练 GT ID 只进入 target matching/loss，运行时使用匿名 ID；先建立 strict-online CMP 与 anti-leak tests，再做 local observations、global lifecycle、anytime forecast、full-mixture feedback 和 raw-LiDAR joint path。
- 已完成外部创新边界与 T-ITS 论文尺度核对：ViP3D、ForeSight、CoopTrack、SparseCoop 等只限制“首次”措辞，不构成删除既有算子的理由。CoLIF 保留 local observation query、source memory、persistent bank、lifecycle、multimodal decoder 和 future feedback 完整结构；T-ITS 主贡献定位为这些模块围绕 receiver-centric cooperative forecast lifecycle 的统一机制、严格无 GT 协议与充分基线/消融/鲁棒性证据，不要求每个 attention、gate、GMM 或 matcher 都是新算子。
- 关键 go/no-go 风险：source-local 标注/可见性语义尚需真实点云统计；自然 partner-supported/handover/miss 事件数量尚未 census；`Qmax`、miss TTL、clip length 和 loss 权重需数据确定；train/test 有一个已知 294 帧重复场景，模型选择需使用 scenario-disjoint validation，并优先评估第二数据集泛化。
- 已把 `model_design.md` 扩展为基于当前 CMP 代码的实施蓝图：保留 legacy CMP 文件，新增 `ColifSequenceDataset`、`PointPillarColif`、独立 `ColifLoss/train_colif/inference/eval`；列出 factory 注册、runtime/target schema、逐模块 shape/API、optimizer/checkpoint、strict-online baseline、阶段提交和测试门槛。
- 新核实：`opencood/models/point_pillar_cobevt_motr.py` 依赖 `sys.path.append("./MOTR")` 后的外部 `MOTR.models.*`，但当前仓库没有顶层 `MOTR/` 目录且没有对应 submodule；该实验文件当前不能直接导入运行。CoLIF 不引入这一缺失依赖，只参考接口并独立实现 tensor state/matcher/query update。
- 实施入口结论：model/loss factory 已支持动态 import，无需修改；V2V4Real dataset factory 是固定白名单，需注册 `ColifSequenceDataset`；原 `train.py` 不支持 recurrent clip、BPTT、分组 LR 和完整 checkpoint，因此新增 `train_colif.py`，不改变 baseline trainer。
- 最终实施合同进一步明确：receiver 自身 frame keys 定义主时间轴（source 缺帧只 mask）；strict exporter 直接调用底层 prediction-only `VoxelPostprocessor.post_process`，不调用 GT wrapper/旧 tracker runner；association 使用带 null/birth 的双向约束匹配；criterion supervision 绑定 `(slot,generation,GT-id)` 且不回流 model；训练 soft state 与部署 hard state分离；Stage A 数据证据未通过时硬停止。

## 当前任务

- 用户目标：在 conda 的 `coop` 环境中，使用 `/home/qkb/Project/data/V2V4Real` 与固定官方 CoBEVT 感知 checkpoint，从训练开始复现 CMP 的轨迹预测部分（no-aggregation MTR）。
- 验收标准：依赖和原生算子可导入；官方 CoBEVT 生成的 train/test 融合特征、检测与跟踪结果可用；no-aggregation MTR 从随机初始化完成 30 epoch 训练；对可用 checkpoint 做完整单卡 minADE/minFDE/MR 评测。用户已将 CMP 预测聚合器训练排除在本轮范围之外。
- 当前状态（2026-08-07 18:51）：no-aggregation MTR 双卡 30 epoch 训练已完成；epoch 5、28、29、30 均已在同一 test cache 上完成独立单卡评测。按用户关注的 5 秒 minFDE，epoch 5 独立快照为已测试 checkpoint 中最优（7.5559）；末轮 epoch 30 为 10.3258。聚合器尚未启动，符合“只训练轨迹预测”的当前范围。

## 背景与动机

- 原仓库配置把 V2V4Real 指向 `/data1/Datasets/V2V4Real`，缓存也包含 `/data3` 的旧绝对路径，不能直接用于本机数据目录。
- 目标数据位于 `/home/qkb/Project/data/V2V4Real`。实际检查到 train 有 14,210 个 YAML/PCD 帧对，test 有 3,986 个 YAML/PCD 帧对，test 分为 9 个场景。
- `docs/prepare_dataset_checkpoints.md` 指定了 V2V4Real 感知权重和 MTR 权重的官方 Google Drive 目录；仓库本身没有所需 checkpoint 或完整的 CoBEVT 融合特征。

## 已完成

### Git 与参考实现审计

- 修改前：`main` 分支，提交 `de0f839`，CMP 工作区干净。
- 已创建隔离任务分支：`codex/cmp-v2v4real-repro`。
- 已检查 `/home/qkb/Project/Reference/CMP`：与当前 CMP 同提交；其中没有可复用 checkpoint 或融合特征。该参考目录有一个用户未跟踪的论文 PDF，未作任何修改。
- 项目起初没有 `HANDOFF.md`；本文件由本任务创建。

### 环境审计与修复

- 实际环境：`coop`，Python 3.10.20，PyTorch 2.1.1+cu121，torchvision 0.16.1+cu121，torch-geometric 2.6.1，spconv 2.3.6，Open3D 0.19.0；机器有两张 RTX 3090。
- 补齐 CMP 直接导入所缺的 `tensorboardX==2.6.4`、`timm==0.9.16`、`Pympler==1.1`，并安装 `gdown==5.2.0` 以取得官方权重。
- 在当前 Python 3.10 / CUDA 12.4 工具链下重建 MTR 的 KNN 与 attention CUDA 扩展（架构 8.6），并重建 OpenCOOD `box_overlaps` 扩展；实际导入和 GPU 前向均已成功。
- `pip check` 仅留下环境中原有且与 CMP 无调用关系的冲突：`openxlab 0.1.3` 要求 `rich~=13.4.2`，当前为 `rich 15.0.0`。未擅自修改共享环境中的无关包。
- CMP 文档声称的 Python 3.7.11 / torch 1.13.1 与实际可用环境不同；当前运行结果确认这套 Python 3.10 / torch 2.1.1 组合可用。

### 权重与本地路径

- 按文档的官方 Drive 清单下载了以下文件（均被 Git 忽略）：
  - `pretrained/v2v4real/point_pillar_cobevt_multiego_256x/net_epoch75.pth`，739,231,572 bytes，Drive file id `1AkGmL4yWbUfnUviPDq8Wfrmm6z7K6cxn`。
  - `MTR/output/v2v4real_multiego_cobevt_c256/ckpt/best_model.pth`，2,540,101,497 bytes，Drive file id `1KEKbNa0tu3CTb35415lxDTKPVuRissnI`。
  - 同时下载了文档中提供的 no-aggregation 对照权重，未用于本次主评测。
- 四个 MTR V2V4Real 配置与四个 OpenCOOD V2V4Real 配置已改为本机数据根目录，并把 MTR cache 设为 `/home/qkb/Project/data/V2V4Real/derived/CMP/cache`，避免读写旧机器路径。

### 真实数据、数据流与最小验证

- 原始 YAML 包含 `gps`、4x4 `lidar_pose` 及车辆的角度、中心、尺寸、位置和类型字段；GT 与跟踪 pickle 的样本时间轴为 147 帧，使用末列有效 mask。
- 已用官方下载的 CoBEVT 配置构建实际 OpenCOOD test loader：共 3,986 个记录；单样本 `origin_lidar` 为 `(25731, 5)`，batch voxel 特征为 `(5530, 32, 4)`。
- 数据流：原始 V2V4Real YAML/PCD → OpenCOOD CoBEVT 感知网络 → 每帧融合特征 `.npy` → V2V4Real MTR dataset 的历史轨迹、future mask 与融合特征 → MTR 编码器/解码器 → minADE、minFDE、MR。
- 通过真实单样本 smoke cache 验证：`obj_trajs=(1,1,11,22)`、`obj_trajs_mask=(1,1,11)`、future 为 `(1,50,2)`，融合特征为 `(1,256,48,128)`，均无 NaN/Inf。
- 完整感知阶段于本次任务中实际完成：3,986 / 3,986 个特征文件，目录约 24 GB。首、中、末三个样本均为 `float32` 的 `(1,256,48,128)` 且所有值有限。
- 官方 c256 MTR checkpoint 已实际加载：250,669,208 个模型参数、1,059 个磁盘键、缺失键 0、意外键 0。
- 单 batch 的真实端到端 smoke 已完成（只含 1 样本，不代表完整数据集性能）：minADE 1/3/5s = 0.2515 / 0.3355 / 1.0657；minFDE 1/3/5s = 0.2361 / 1.0137 / 4.0619；MR@3.6 = 1。日志：`/home/qkb/Project/data/V2V4Real/derived/CMP/runs/mtr_smoke_20260807/eval/log_eval_20260807-115323.txt`。

### 当前完整评测

- 实际命令：

  `PYTHONPATH=/home/qkb/Project/CMP:/home/qkb/Project/CMP/MTR CUDA_VISIBLE_DEVICES=0 conda run -n coop python MTR/tools/test_multiego.py --cfg_file MTR/tools/cfgs/v2v4real/v2v4real_multiego_cobevt_c256.yaml --batch_size 1 --workers 0 --launcher none --fix_random_seed --ckpt MTR/output/v2v4real_multiego_cobevt_c256/ckpt/best_model.pth --output_dir /home/qkb/Project/data/V2V4Real/derived/CMP/runs/mtr_full_20260807`

- 数据验证阶段已于 2026-08-07 12:06:17 结束，生成 2,858 个可评估记录，分布为 2 CAV：2,604，1 CAV：254。
- 当前为单 GPU 执行：上游自定义评测没有跨进程指标汇总，且脚本中的分布式路径硬编码为 8 GPU；为避免重复/遗漏样本和错误指标，本次没有强行改为双卡。
- 完整运行的退出码为 0。GPU 推理进度为 2,858 / 2,858，耗时约 4 分 20 秒；日志的指标聚合包含 29,500 个预测目标、Filtered=0：minADE 1/3/5s = 1.1375 / 2.6684 / 3.9257，minFDE 1/3/5s = 2.0376 / 4.8467 / 7.0533，MR@3.6/3.0/2.0/1.8 = 0.3299 / 0.3393 / 0.3628 / 0.3673。
- 完整日志：`/home/qkb/Project/data/V2V4Real/derived/CMP/runs/mtr_full_20260807/eval/log_eval_20260807-120255.txt`。

### 训练复现审计（已完成：轨迹预测阶段）

- 实际训练入口为 `MTR/tools/train_multiego.py`：训练循环执行 `loss.backward()` 和 `optimizer.step()`，并保存带 optimizer state 的 checkpoint。
- 配置表明作者采用两阶段初始化：`v2v4real_multiego_cobevt_c256_no_agg.yaml` 的 `PRETRAINED_MOTION_TRANSFORMER` 为空，应从随机初始化训练；聚合版 `v2v4real_multiego_cobevt_c256.yaml` 会加载 no-aggregation MTR，并对 MTR 使用 1/100 学习率、对 aggregation 模块使用完整学习率。第二阶段将指向本机第一阶段的自训 checkpoint，而不是官方 no-aggregation 权重。
- 用户于 2026-08-07 明确选择固定官方 CoBEVT、只训练轨迹预测；官方 `net_epoch75.pth` 位于 `pretrained/v2v4real/point_pillar_cobevt_multiego_256x/`，大小 739,231,572 bytes。该模型生成的 17,902 个融合特征、64 个 train tracking pickle、18 个 test tracking pickle，以及对应 MTR cache 已实际核验存在。
- 已完成 11 帧真实 train 感知 smoke：官方 epoch 75 CoBEVT checkpoint 可加载并前向。为防止 smoke 污染后续训练，本任务修正了 `--max_frames`：它不再生成不完整的 tracking cache。该 smoke 未写 detection/tracking cache，留下的少量 train 融合特征可由完整预处理安全复用。
- train/test 目录有一个同名场景 `testoutput_CAV_data_2022-03-15-09-54-40_0`；比对结果为 294 个 YAML、294 个 PCD 和目录差异均为 0，即原始数据集本身的重复场景。融合特征的同名文件因此语义一致；为复现作者的原始切分，本任务不擅自移除该重复，但将其标记为数据泄露风险。
- 双卡训练可使用 `DistributedSampler` 和 DDP；但 `eval_one_epoch_custom` 没有跨进程聚合。`train_multiego.py` 的 `--not_eval_with_train` 原先仍会在训练后调用该分布式评测，现已修正为完整跳过它。双卡训练只生成 checkpoint，最终指标统一用单卡完整评测产生。
- 完整 train 预处理已以退出码 0 完成：17,902 个唯一融合特征（约 105 GB），检测 cache 为 25,789,678 bytes，64 个 tracking pickle 与 64 个 GT pickle 名称完全一致；多出的 `tracking/train/log.txt` 不是轨迹数据。MTR train cache 含 3,684 个记录，分布为 1 CAV：2,586，2 CAV：1,098。
- 发现并修正了上游 V2V4Real batching 根因：`collate_batch` 只使用 `batch_list[0]`，所以 batch_size>1 原先会静默丢弃其余场景。现在显式拒绝 batch_size>1；真实 4-item loader 已验证报出该错误。为在两张卡上保持作者 8 卡×每卡 1 场景的等效全局 batch=8，训练入口新增 `--grad_accum_steps`，每卡 1 个场景累积 4 步后更新一次。
- 训练信号验证：no-aggregation 模型的真实 batch 为 2 个 CAV、每 CAV `(1,256,48,128)` 融合特征，`obj_trajs=(2,1,11,22)`；loss=63.3177 且有限，456 个参数梯度张量均有限，完成 AdamW step。4 个真实 micro-batch 的累积 smoke 完成 1 次更新，参数绝对变化和为 0.53487045。
- 正式 no-aggregation MTR 训练已启动：`CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop torchrun --standalone --nproc_per_node=2 MTR/tools/train_multiego.py --cfg_file MTR/tools/cfgs/v2v4real/v2v4real_multiego_cobevt_c256_no_agg.yaml --batch_size 2 --workers 2 --launcher pytorch --fix_random_seed 42 --extra_tag trajectory_only_seed42 --not_eval_with_train --grad_accum_steps 4 --max_ckpt_save_num 3 --ckpt_save_interval 1 --logger_iter_interval 50`（另设项目内 `PYTHONPATH`）。日志确认从随机初始化开始，`PRETRAINED_MOTION_TRANSFORMER` 为空，加载 3,684 条 train cache 和 2,858 条 test cache。每 rank 有 1,842 个 micro-batch，累积 4 步后为 461 次 optimizer 更新/epoch，等效全局 batch=8。
- 第 1/30 epoch 于 14:32:46 完成，双卡显存约 1.98/1.85 GB、稳定吞吐约 0.55 秒/更新，训练 loss 均为有限值。`MTR/output/v2v4real_multiego_cobevt_c256_no_agg/trajectory_only_seed42/ckpt/checkpoint_epoch_1.pth` 为 699,853,318 bytes；在 CPU 上加载得到 epoch=1、iter=461、880 个模型张量及 optimizer state。第 2 个 epoch 已于 14:32:47 开始。
- 训练以 `--max_ckpt_save_num 3` 仅保留最近 3 个 checkpoint；因此 checkpoint epoch 1 在 epoch 4 保存后被自动清理。用户在其被清理后请求 epoch 1 FDE，无法再诚实评测该具体权重。随后尝试 epoch 2 测试时，epoch 5 已于 14:49:37 保存并清理 epoch 2；测试于 14:49:47 启动，实际报 `FileNotFoundError`，没有进入推理或产生指标。
- 为避免与训练清理并发，已将 epoch 5（epoch=5、iter=2305）checkpoint 复制为独立快照后单卡完整评测，配置为 `v2v4real_multiego_cobevt_c256_no_agg.yaml`，checkpoint 880 个键严格加载、missing/unexpected 均为 0。2026-08-07 14:50:49--14:57:36 对 2,858 条 test record、29,500 个预测目标完成评测（Filtered=0）：minADE 1/3/5s = 1.3007 / 3.0554 / 4.4001；minFDE 1/3/5s = 2.4689 / 5.4988 / 7.5559；MR@3.6/3.0/2.0/1.8 = 0.3327 / 0.3447 / 0.3775 / 0.3834。日志为 `/home/qkb/Project/data/V2V4Real/derived/CMP/runs/mtr_noagg_epoch5_test_20260807/eval/log_eval_20260807-145046.txt`。这是训练中途的 no-aggregation 基线，不是最终 CMP 聚合结果。
- 正式 no-aggregation MTR 训练在 2026-08-07 16:36:16 正常结束：日志最后记录为 `epoch 29/30, acc_iter=13830, cur_iter=460/461`，随后输出 `End training`。因正式命令带有 `--not_eval_with_train`，训练入口明确跳过了最终评测；这避免了该脚本不正确的分布式指标汇总，最终指标均由下述独立单卡 test 产生。
- 训练自动保留的 epoch 28/29/30 checkpoint 与 epoch 5 独立快照均已严格加载（880 个模型键、missing/unexpected 均为 0），并在相同的 2,858 条 test record、29,500 个预测目标上完成评测（Filtered=0）。其中 epoch 28、29 分别在 GPU 0、GPU 1 上独立执行，epoch 30 在 GPU 0 上独立执行；每次评测均为单进程，未混合各 GPU 的样本或指标。

  | checkpoint | iter | minADE 1/3/5s | minFDE 1/3/5s | MR@3.6/3.0/2.0/1.8 |
  | --- | ---: | --- | --- | --- |
  | epoch 5 snapshot | 2,305 | 1.3007 / 3.0554 / 4.4001 | 2.4689 / 5.4988 / **7.5559** | 0.3327 / 0.3447 / 0.3775 / 0.3834 |
  | epoch 28 | 12,908 | 1.1806 / 3.1383 / 4.9348 | 2.1798 / 5.9895 / 9.4701 | 0.6317 / 0.6729 / 0.7456 / 0.7580 |
  | epoch 29 | 13,369 | 1.1875 / 3.1478 / 5.0544 | 2.1683 / 6.1815 / 9.9534 | 0.6140 / 0.6447 / 0.7178 / 0.7325 |
  | epoch 30 | 13,830 | 1.2384 / 3.2854 / 5.2554 | 2.2775 / 6.3733 / 10.3258 | 0.6433 / 0.6621 / 0.7300 / 0.7519 |

- 训练日志：`MTR/output/v2v4real_multiego_cobevt_c256_no_agg/trajectory_only_seed42/log_train_20260807-142831.txt`。epoch 28/29/30 的评测日志分别为 `/home/qkb/Project/data/V2V4Real/derived/CMP/runs/mtr_noagg_epoch28_test_20260807/eval/log_eval_20260807-184238.txt`、`/home/qkb/Project/data/V2V4Real/derived/CMP/runs/mtr_noagg_epoch29_test_20260807/eval/log_eval_20260807-184238.txt`、`/home/qkb/Project/data/V2V4Real/derived/CMP/runs/mtr_noagg_epoch30_test_20260807/eval/log_eval_20260807-184746.txt`。
- 在已评测的 checkpoint 中，epoch 5 是 5 秒 minFDE、5 秒 minADE 和全部报告 MR 阈值的最佳者；epoch 28 的 1 秒 minADE（1.1806）及 epoch 29 的 1 秒 minFDE（2.1683）略低，但均未改善 5 秒终点误差。故若后续只按 5 秒 minFDE 使用本阶段模型，应使用独立快照 `/home/qkb/Project/data/V2V4Real/derived/CMP/runs/mtr_noagg_epoch2_test_20260807/checkpoint_epoch_5_snapshot.pth`，而非末轮 checkpoint。
- CoBEVT 从随机初始化的真实训练 smoke（batch=4）已通过：`record_len=(4,)=[1,1,1,1]`、voxel 特征 `(23676,32,4)`，输出 `psm=(4,2,48,128)`、`rm=(4,14,48,128)`、融合特征 `(4,256,48,128)`；loss=1314.1504 有限，159 个梯度张量均有限，Adam step 后参数绝对变化和为 230.2341，峰值显存 6036.7 MB。
- CoBEVT 双卡 DDP 真实 batch smoke 已以退出码 0 完成：每 rank batch=4、全局 batch=8，rank 0/1 loss 分别为 1819.8065/1566.5241，159 个梯度张量均有限，全局峰值显存为 6749.5 MB。PyTorch 仅报告一个卷积梯度 bucket stride 的性能警告，不影响数值或更新正确性。
- 首次 CoBEVT 80 epoch 训练在 epoch 0 第 3 个混合 CAV batch 以退出码 1 终止，未生成 epoch checkpoint。根因是 `collate_batch_train` 把只供 batch=1 tracking 推理使用、长度随 in-range CAV 数变化的 `ego_transformation_matrix` 列表做 `np.array`/stack；实际混合 batch 为 `[2,2,1,2]`，因而触发不等长数组错误。模型 forward 与检测 loss 对该字段均无读取路径。
- 修复后，`collate_batch_train` 仅在 `len(batch)==1` 时保留与原先相同的 tensor `transformation_matrix_list`，多样本训练 batch 不再携带该未消费字段。用原训练的 `DistributedSampler.set_epoch(0)` 顺序在双卡真实模型上执行 3 次更新：rank 1 的第 3 个 batch 正是 `[2,2,1,2]`，两 rank 均完成前向、loss、反向和 step；159 个梯度张量均有限，峰值显存为 8722.7 MB。单样本 test collate 保持矩阵形状 `(1,1,4,4)`。
- 修复后一次 CoBEVT 重新启动已正常越过第 38 个训练 step，GPU 利用率为 100%，显存约 18.7/19.4 GB，loss 有限；它在 epoch 0 尚未生成 checkpoint 时被有意停止，因为原 `opencood/tools/train.py` 没有对 Python、NumPy 和 PyTorch 设置训练随机种子。现已添加 `--seed`：每个 DDP process 使用 `base_seed + rank`（本次为 42/43），模型会在 DDP 建立时由 rank 0 同步参数，数据增强/worker 随机流则按 rank 分离。`--help` 与 `py_compile` 均已验证该入口。
- seed=42 的后续重启正常运行至第 337 step（loss 仍有限），但审计发现验证循环将 rank 0 的半个 test sampler 均值直接写为全量验证 loss，且两个 rank 会竞争创建 run-dir/config 与写同一 `net_epoch*.pth`。该运行也在 epoch 0、无 checkpoint 时有意停止。现在只有 rank 0 创建目录、写 config/TensorBoard/checkpoint，目录通过 DDP object broadcast 传给另一 rank；每个 epoch checkpoint 写完后 barrier；验证 loss 按 `sum(loss) / count` 的 float64 all-reduce 聚合。双卡最小测试实际验证了路径 broadcast 和 loss sum/count all-reduce，两个 rank 都得到路径一致和全局均值 1.5。
- 正式 CoBEVT 运行命令：`PYTHONPATH=/home/qkb/Project/CMP CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop torchrun --standalone --nproc_per_node=2 opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_cobevt_multiego_256x_v2v4real.yaml --dataset_name v2v4real --seed 42`。输出目录为 `opencood/logs/point_pillar_cobevt_v2v4real_2026_08_07_13_44_41`，其 `config.yaml` 已实际记录 `train_params.seed: 42`。
- 正式 run 的 epoch 1 已完成：1,776 个每-rank train step 后仅由 rank 0 写出 `net_epoch1.pth`（739,231,489 bytes，2026-08-07 14:12:00）。跨两 rank、共 3,986 个 validation records 聚合的检测 validation loss 为 `2.0913016796`（2026-08-07 14:15:51）；不是 rank 0 半个 split 的局部均值。此后已进入第 2 个 epoch，实时 rank-0 的 regression/confidence loss 仍有限。
- `net_epoch1.pth` 的 checkpoint load 已在 CPU 上严格验证：state dict 含 259 个键，按当前 CoBEVT c256 配置构建的模型 `strict=True` 加载后 missing/unexpected keys 均为 0。

## 修改内容

- `.gitignore`：忽略本机解释器生成的扩展 `.so` 和 MTR 版本文件，避免将环境构建产物提交。
- `MTR/tools/cfgs/v2v4real/*.yaml`：改为本机数据及 cache 路径。
- `opencood/hypes_yaml/*v2v4real.yaml`：改为本机数据路径。
- `MTR/tools/eval_utils/eval_utils.py`：移除不参与 V2V4Real 自定义评测、却会强制导入缺失 Waymo/TensorFlow 依赖的顶层 Waymo evaluator import；数据集的原有 Waymo evaluation 路径保留其惰性导入。
- `MTR/mtr/models_v2v4real/context_encoder/mtr_encoder.py`：移除不参与上下文编码器的 `this` 与 transformers 导入，避免启动时打印 Zen of Python 和引入无关依赖。
- `opencood/tools/multi_ego_inference_v2v4real.py`：增加 `--refresh_perception_features` / `--max_frames`；修正融合特征实际 `.pkl.npy` 名称检查、缓存命中时丢失检测元数据、`None` 输出路径和跟踪缓存误读检测缓存的问题；默认行为仍可使用已有缓存。
- `MTR/tools/train_multiego.py`：使 `--not_eval_with_train` 同时跳过训练结束后的最终评测，避免 DDP 下未聚合的自定义指标被误当作完整结果。
- `MTR/mtr/datasets/v2v4real_multiego_dataset.py`：拒绝不受支持的 batch_size>1，避免静默丢弃场景。
- `opencood/data_utils/v2v4real/datasets/intermediate_fusion_dataset_multi_ego.py`：训练时不堆叠不等长且不参与模型/loss 的 tracking 变换矩阵；仍为 batch-size-one 推理保留原有 tensor 接口。
- `opencood/tools/train.py`：增加 `--seed`，在建立 dataset/model 前初始化 Python、NumPy 与 PyTorch 的每-rank 随机状态。
- `opencood/tools/train.py`：DDP 下只由 rank 0 建立 run 目录、写 TensorBoard/checkpoint，路径广播并在写 checkpoint 后同步；验证 loss 通过跨 rank 的 sum/count all-reduce 汇总，避免将半个 validation split 误报为完整指标。
- `MTR/tools/train_multiego.py` 与 `MTR/tools/train_utils/train_utils.py`：加入正确缩放 loss、只在每个累积组末尾更新参数/调度器、DDP 中间 micro-batch 不同步梯度的 `--grad_accum_steps` 支持。
- YAML 静态检查：MTR 的 4 个 V2V4Real 配置及 OpenCOOD 的 CoBEVT 1x、CoBEVT 256x、SinBEVT 配置均可由 PyYAML 解析；`point_pillar_v2vnet_multiego_v2v4real.yaml` 在当前和 `de0f839` 基线中都在 `preprocess.args.voxel_size` 存在同一处过度缩进，因而都不能解析。本次只替换了其数据路径，没有修复未使用的上游 V2VNet 配置语义。

## 未完成

- CMP 预测聚合器未启动；这是用户选择“只训练轨迹预测”后的有意范围限制，因此当前结果不能称为完整两阶段 CMP 聚合模型的结果。
- 未保存或评测 epoch 1--4、6--27；因此 epoch 5 只能称为“已评测 checkpoint 中按 5 秒 minFDE 最优”，不能称为 30 个 epoch 的全局最优。若需要无测试集选择偏差的模型选择，应另设验证划分或在下一次训练中保存/验证全部 checkpoint。
- 未对 Waymo 数据集执行测试；它不属于本次 V2V4Real 任务，并且环境没有安装 Waymo/TensorFlow 栈。

## 已知问题与风险

- 上游 V2V4Real 自定义 evaluator 的评测逻辑沿用 OPV2V 的 prediction-dict 生成路径；它已在真实 V2V4Real smoke 与完整缓存构建中执行，但未改写其算法语义。
- 原始 train/test split 含一个 294 帧的完全重复场景；若研究目标是严格隔离的泛化评测，应另行设计去重 split，而不能把本任务的作者协议指标直接解释为无泄露泛化性能。
- 项目文档中的 V2V4Real 评测 `--batch_size 8` 与该 dataset 的单场景 collate 实现不兼容；本任务的先前完整评测和后续训练都使用每进程 batch_size=1，避免遗漏数据。
- `opencood/hypes_yaml/point_pillar_v2vnet_multiego_v2v4real.yaml` 是基线已有的无效 YAML，未用于本次 CoBEVT c256 复现；如需运行 V2VNet 变体，应先单独修复并验证其 `voxel_size` 缩进。
- 全量融合特征约 24 GB、MTR checkpoint 约 2.5 GB，均为可再下载/再生成的忽略文件，不随 Git 提交。
- 单样本 smoke 指标仅用于通路验证；完整指标才是本任务的复现结果，但它们是否与论文数值完全一致仍需在相同数据版本、评测协议和权重来源下另行逐项比对。
- 两阶段训练的完整时长取决于 train record 数、batch size 和双卡吞吐；在完成单 batch 反向和 DDP 采样审计前，不应估计为已完成或直接引用官方 checkpoint 代替自训结果。
- 训练 checkpoint 当前只保留最近 3 个；若需在训练进行中评测特定 epoch，必须先复制其 checkpoint 到独立结果目录，避免与下一次 epoch 保存的自动清理发生竞态。
- 这次为了回答用户的 checkpoint 性能问题，使用 test cache 比较了 epoch 5、28、29、30；这可以证明末期 checkpoint 相对已测试 epoch 5 退化，但不应把该比较作为无偏的 test-set model-selection 协议或论文最终泛化结论。
- 固定的官方 CoBEVT checkpoint 与数据版本是本次轨迹预测复现的外部前提；因此最终结果只能支持“在官方感知输出上的轨迹预测复现”，不能声称感知模块从随机初始化已复现。

## 下一步

1. 当前“只训练轨迹预测”目标已完成；如需使用本阶段模型且以 5 秒 minFDE 为准，使用保存的 epoch 5 独立快照，不要默认使用 epoch 30。
2. 如需正式选择全局最优的 no-aggregation checkpoint，重新训练时应保存每轮 checkpoint，并在独立 validation split 上选择；test 仅用于一次最终报告。
3. 只有用户重新授权完整 CMP 两阶段复现时，才以选定的 no-aggregation checkpoint 初始化并训练预测聚合器，然后单卡完整评测。

## Git 状态

- 当前分支：`codex/cmp-v2v4real-repro`。
- 修改前基线：`main` 的 `de0f839`。
- 已提交的相关检查点：`9491046`（本机路径/缓存/推理修复）、`92997c6`（smoke cache 安全）、`3e84a44`（跳过不安全的分布式评测）、`4c5ffb0`（MTR batch 语义与梯度累积）、`889abe8`（CoBEVT 混合 CAV batch）、`d7ee7da`（CoBEVT seed）、`aff428b`（DDP run/checkpoint 单写与验证聚合）、`f86d0e1`（切换为官方 CoBEVT + 自训轨迹预测）、`667f9ba`（记录训练启动）、`44992e9`（记录 epoch 5 评测）。下载权重、缓存、特征和运行日志均在 Git 忽略路径中。
- 恢复方式（提交后）：`git revert <任务提交>`；或切回 `main` 查看原始版本。不得删除生成数据，除非用户另行授权。

## 更新时间

- 2026-08-07，本次 CMP V2V4Real 复现任务；用户已切换为官方 CoBEVT + 自训轨迹预测。no-aggregation MTR 的 30 epoch 训练及 epoch 5/28/29/30 完整单卡评测均已完成，聚合器未训练。
# CoLIF 2026-08-13 最新状态（优先于下方历史记录）

- 当前主张已从被实验否证的“future 改善 tracking”改为可检验的任务融合问题：CMP FuseBEVT 为当前检测/跟踪提供统一 receiver-centric measurement，但预测不应只使用已压缩的同一融合表示。主模型保留 FuseBEVT，并对每个 persistent actor query 在融合前 receiver-aligned source BEV 上采样局部 patch，再跨 source/patch attention 构造预测上下文。
- 新路径的实际输入输出：`regrouped_source_bev[B,L,256,48,128] + cav_mask + query[B,Q,256] + box[B,Q,7] -> actor cooperative context[B,Q,256]`；无效 CAV hard-mask，有效 partner token 有梯度的单测已通过。
- motion head 现同时编码 ordered variable history、robust pairwise-median velocity/dispersion、actor interaction、actor-conditioned mode token 和 source context。修复了旧 `actor+mode -> shared linear score` 导致 softmax 模态比例中 actor 项被抵消的结构问题。
- 预测使用两层深监督，但第二个 cross-mode refinement 只在训练中作 auxiliary head，推理默认 `inference_layer=proposal`。同一 v16 checkpoint 完整 test：refined `ADE5/FDE5=2.632675/4.401278`，proposal `2.550686/4.381799`，MOTA 均为 `0.633329`。
- v16 proposal 相比旧 v13 `ADE5/FDE5=2.789291/4.399249` 分别改善 `0.238605 m` 与 `0.017450 m`。v17 从头只训单层得到 `2.601196/4.384774`，说明第二层的训练期深监督有小幅正则作用，但不应部署它。
- same-checkpoint 因果关闭结果：关闭 actor source context 使 ADE5 从 `2.632675` 变为 `2.737838`；关闭 ordered history 使 ADE5 变为 `2.731695`。它们对 FDE 的影响只有约 `0.004 m`，因此当前只支持“改善整段轨迹拟合/ADE”，不支持夸大 endpoint 主张。
- 手工 lateral anchors 的完整 test FDE 为 `4.88/4.78/4.73 m`，比原 multi-scale CV anchor oracle `4.648545 m` 差，已拒绝写入主模型。修正坐标系后的 train-only residual prototypes 在 test 的 frozen oracle FDE 为 `4.392662 m`，证明 collinear anchors 有 coverage gap，但已训练 v16 proposal `4.381799 m` 更好，因此保留 learned residual，不注入静态 prototype。
- v18 公平 no-actor-source 对照已通过 5-step 双卡 smoke：可训练参数 `4,289,783`，unused trainable=0，AMP overflow=0，DDP parameter/buffer checksum delta=0。完整 855-step 训练与 official test 尚在执行；完成前不得把 same-checkpoint 关闭结果写成独立重训结论。
- 当前主配置仍为两阶段：Phase 1 训练 CMP fused query tracker；Phase 2 冻结 perception/tracker，训练 actor source fusion、ordered history 和 forecast head。`future_prior_mode=none`，第一帧只 birth，至少两次观测后预测。

# CoLIF 2026-08-12 历史状态

- 当前主配置：`future_prior_mode=none`。旧 latent feedback、forecast
  association、missing-state propagation 和双状态 gap imputation 均已由
  deterministic same-checkpoint full-test 消融否证，只保留实验开关。
- test 点云不再随机 `shuffle_points`；train 仍 shuffle。相同 v13
  checkpoint 的 `none` 重复完整评测全部指标逐位一致。
- v13 双卡完整一轮：1710 global clips、855 optimizer updates、AMP
  overflow=0、no-signal skip=0、parameter/buffer checksum delta=0；checkpoint
  为 `opencood/logs/colif_fused_v13_propagation_bptt_full_1epoch_20260812/checkpoint_epoch_000.pth`。
- v13 deterministic `none`：MOTA=0.631619、IDS=599、recall=0.794234、
  coverage=0.788663、minFDE5=4.399249。双状态 gap imputation：
  MOTA=0.628718、IDS=648、recall=0.793532、coverage=0.788015、
  minFDE5=4.371122；不满足强验收条件。
- 当前实现增加 `association_ref_box`，确保负消融中的 imputed output 不会
  成为下一帧 CV association origin；主模式 `none` 下它与 `ref_box` 相同。
- 完整失败链和实验账本见 `docs/colif_failure_postmortem.md` 第14节；当前
  架构入口和边界见 `model_design.md` 顶部“当前规范”。

---

# 2026-08-18 `paper.md` 全量审计归档

- 新建 `/home/qkb/Project/CoLIF/paper.md`，作为论文、模型、数据、训练、评测、失败链和学术写作审计总档；它不是投稿正文。
- 综合正文使用“当前已证实 / 证据性推断 / 未证实 / 已否证 / 历史已替换 / 待核验”状态，避免旧CoLIF、future-feedback和当前DualCMP结论混写。
- 原文附录包含当前 `model_design.md`、`docs/colif_failure_postmortem.md`、`docs/colif_training.md`、Abstract/Introduction审计、`manuscript.md` 与本 `HANDOFF.md`。
- 已通过Codex任务历史接口分页回读当前任务：17页、1,400条可见用户/助手消息、486,839字符；全部写入附录H。隐藏推理、工具调用与命令输出没有伪装成审计结论。
- 已回读任务开头引用的Codex任务：5页、294条可见消息、113,637字符，写入附录I；已回读引用的ChatGPT对话：4页、62条可见消息、320,340字符，写入附录J。
- ChatGPT历史接口单条消息上限为20,000字符；引用对话有3条达到上限，已明确标记为可能截断。当前任务没有消息达到上限。
- 最终复核时 `paper.md` 为45,765行、156,940词、2,071,721字节，SHA-256为 `6fa137a74642d1a33b512683e32bb9a33236e7cc7a396930e38cc8590c85ce38`（后续继续编辑时该hash自然变化）。
- 未修改模型代码、配置、训练日志或实验结果；当前工作树原有大量用户/历史改动均保留。
