# HANDOFF

## Current

- seed-20 fused-BEV、`w/o Relative Geometry`和`w/o Motion Anchors`均已完成独立`online_masked`评测；当前主模型仍为`per_cav + attention + geometry + anchored`。
- 论文采用最小组件消融：fused reader移除source-specific BEV，geometry-off移除relative geometry，direct regression移除motion anchors。uniform mean只保留为诊断开关，不进入论文消融。
- 两项现有checkpoint已有正式结果；但direct使用每卡batch 4，而Full和geometry-off使用每卡batch 2。若论文要把差异严格归因于motion anchors，仍需按匹配训练预算重训direct。

## Verified

- fused ADE@1/3/5s=`0.854988/1.428909/1.972203`，FDE=`1.044122/2.209837/3.278118`；per-CAV六个ADE/FDE、三档MR及三个support bucket方向均更优，样本计数完全一致。
- geometry-off使用最后完整epoch-15：ADE@1/3/5s=`0.853783/1.420999/1.956899`，FDE=`1.039269/2.199123/3.255097`；六项均略差于Full，但该结果不是20-epoch严格等训练量对照。
- direct regression使用完整epoch-19：ADE@1/3/5s=`1.270013/2.307474/3.119810`，FDE=`1.794235/3.912687/5.568135`；相对Full分别增加49.16/63.85/61.23%和74.03/79.14/71.51%。该结果表明当前direct formulation明显较差，但训练batch/updates不匹配，不能把全部差异严格归因于anchors。
- Full、fused、geometry-off和direct均覆盖3,986 frames、52,597 matched actor frames，coverage=`0.804925`，receiver-only/partner-only/shared counts=`15,847/11,902/24,848`，因此预测指标差异不来自评测对象变化。
- `--source-aggregation {attention,mean}`、`--source-geometry {on,off}`和`--forecast-trajectory-parameterization {anchored,direct}`已进入训练、checkpoint config、phase log和evaluator metrics。
- uniform mean只平均有效source/patch token并冻结attention/geometry；direct由相同proposal head直接回归位移，不使用anchors且不受12 m anchor-residual上限。
- 完整`tests/colif`为141 passed（18个既有warning）；三个变体均完成真实双卡forward/loss/backward/optimizer，unused trainable=0、AMP overflow=0、DDP parameter/buffer checksum delta=0。
- 新旧四种模式state schema一致；per-CAV和fused epoch-19 checkpoint均严格加载为0 missing/0 unexpected。
- 历史 source refiner 对 core proposal 是梯度隔离的：corridor 位置、base trajectory/logit 均 detach，source 随机过程使用独立 RNG fork，proposal/cross-mode 权重仍为 0.5/0.5；迁移后的 core checkpoint 可用于当前推理，但没有 optimizer/scheduler/scaler state。
- `opencood/logs`当前为4个目录、约2.4 GiB、3个checkpoint；fused只保留epoch-19、训练日志和正式metrics，中间epoch 10--18已清理。

## Open

- 按已验证结果更新`DualCMP_TITS/bare_jrnl_new_sample4.tex`的Ablation Study表格与文字。
- 若需要投稿级单模块归因，以每卡batch 2、grad accumulation 2重训direct 20轮；否则将现有行明确写为`Direct Regression`设计对照并披露训练预算差异。
- 用户确认不再需要恢复训练后，再清理query epoch 0--14和direct epoch 0--18中间权重；保留被正式评测的epoch-15/19、训练日志和metrics JSON。

## Risks

- 训练入口会复用既有输出目录并追加日志；所有新 run 必须先通过 `check_run_id.py --must-not-exist`，既有任务只用显式 `--resume`。
- `--disable-actor-cooperative-fusion` 是 no-context 诊断，不是 fused-BEV 基线；注意力图也不能替代 fused-vs-per-CAV 因果消融。
- geometry-off仅完成epoch 0--15；其方向一致的小幅差异只能作为relative geometry的有限支持，不能声称20-epoch等训练量因果结论。
- geometry-off使用每卡batch 2、428 updates/epoch、累计6,848 updates；direct使用每卡batch 4、214 updates/epoch、累计4,280 updates，二者与Full的训练预算并非全部一致。
- direct变体已完成20轮并在相同测试对象上显著退化；结论只能限定为当前anchored parameterization优于当前direct regression formulation，不能量化anchors单独贡献。
- 当前query/direct目录仍保留全部中间checkpoint，约29 GiB；清理前必须保留正式评测对应的epoch-15和epoch-19。
- 已删除的约 233 GiB 二进制 checkpoint 不在 Git 中且不可恢复；历史数值只能用于复盘，不能重新作为初始化资产。

## Checkpoints

| ID | Commit | Outcome | Validation | Risk / next |
| --- | --- | --- | --- | --- |
| CP-20260825-01 | `14cc0a9` | uniform mean、no-geometry和direct trajectory正式开关与命令 | 141 tests；三种双卡optimizer smoke；旧checkpoint strict load | 顺序完成三个20轮消融 |
| CP-20260824-04 | `f40c26b` | 清理旧日志、权重、缓存、草稿与被替代资产 | 6 个权重留存；strict/cross-phase load；8 plot tests | fused 完成后再清理中间 epochs |
| CP-20260824-02 | `afa5364` | 论文证据边界、实验、结论与结果图收敛 | 数值/主张审计；8 plot tests | 完成 fused 因果对照与投稿必需结果 |
| CP-20260823-02 | `9672d5a` | 分层项目记忆与 fused/per-CAV 消融合同 | state documents reviewed | 正式 fused 指标仍未验证 |
| CP-20260823-01 | `739842a` | 公平 fused/per-CAV 表示开关与 seed CLI | 135 tests；双卡训练/评测 smoke | 完成 seed-20 fused 闸门 |
| CP-20260822-01 | `ef008c5` | epoch-19 权重注意力图和全量统计 | 3,986 frames；52,597 records | 图不构成精度因果证据 |
