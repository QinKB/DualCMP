# HANDOFF

## Current

- seed-20 fused-BEV训练和独立`online_masked`评测已完成；per-CAV在整体及partner-only ADE/FDE上均优于fused，表示闸门通过。
- 当前主模型仍为`per_cav + attention + geometry + anchored`。下一步只顺序完成三个必要消融：uniform mean、query attention without geometry、direct trajectory。
- 三个可运行开关、20轮命令和恢复边界见`docs/colif_training.md`第5节；正式run ID为`train-mean-s20`、`train-query-s20`、`train-direct-s20`。

## Verified

- fused ADE@1/3/5s=`0.854988/1.428909/1.972203`，FDE=`1.044122/2.209837/3.278118`；per-CAV六个ADE/FDE、三档MR及三个support bucket方向均更优，样本计数完全一致。
- `--source-aggregation {attention,mean}`、`--source-geometry {on,off}`和`--forecast-trajectory-parameterization {anchored,direct}`已进入训练、checkpoint config、phase log和evaluator metrics。
- uniform mean只平均有效source/patch token并冻结attention/geometry；direct由相同proposal head直接回归位移，不使用anchors且不受12 m anchor-residual上限。
- 完整`tests/colif`为141 passed（18个既有warning）；三个变体均完成真实双卡forward/loss/backward/optimizer，unused trainable=0、AMP overflow=0、DDP parameter/buffer checksum delta=0。
- 新旧四种模式state schema一致；per-CAV和fused epoch-19 checkpoint均严格加载为0 missing/0 unexpected。
- 历史 source refiner 对 core proposal 是梯度隔离的：corridor 位置、base trajectory/logit 均 detach，source 随机过程使用独立 RNG fork，proposal/cross-mode 权重仍为 0.5/0.5；迁移后的 core checkpoint 可用于当前推理，但没有 optimizer/scheduler/scaler state。
- `opencood/logs`当前为4个目录、约2.4 GiB、3个checkpoint；fused只保留epoch-19、训练日志和正式metrics，中间epoch 10--18已清理。

## Open

- 运行`train-mean-s20`并固定epoch-19独立评测。
- 运行`train-query-s20`并固定epoch-19独立评测。
- 运行`train-direct-s20`并固定epoch-19独立评测。
- 将三项结果与现有fused/full结果按相同1/3/5 s ADE/FDE、coverage和support counts核对。

## Risks

- 训练入口会复用既有输出目录并追加日志；所有新 run 必须先通过 `check_run_id.py --must-not-exist`，既有任务只用显式 `--resume`。
- `--disable-actor-cooperative-fusion` 是 no-context 诊断，不是 fused-BEV 基线；注意力图也不能替代 fused-vs-per-CAV 因果消融。
- 当前3.2465 m checkpoint只能恢复推理权重；三个消融均必须从Phase-1 epoch-9初始化，不能从该checkpoint恢复优化器状态。
- direct变体初始loss和gradient norm高于anchored，但两步smoke在既有1000 gradient clip下无AMP overflow并完成optimizer step；正式结果仍须20轮训练验证。
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
