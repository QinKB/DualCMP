# HANDOFF

## Current

- seed-20 fused-BEV 20-epoch 闸门已在 2026-08-23 启动，使用既有长目录 `opencood/logs/dualcmp_fused_receiver01_seed20_20e_20260823`；不要重复启动，也不要在运行中改名或移动该目录。
- 等待任务完成事件；完成后一次性核对退出状态、末轮 checkpoint、独立 `online_masked` 评测及 support counts。规范协议见 `docs/colif_training.md` 第 4 节。
- 当前主模型仍为 `per_cav`；`fused` 只改变预测端空间表示，不改变 tracker、query、history、forecast head、loss 或参数量。
- `manuscript.md` 已按当前实现和已验证证据收敛；fused、uniform mean、learned baselines、额外 seeds 与效率结果完成前保持工作稿，不得删除其证据边界或填写推测值。

## Verified

- `--forecast-bev-representation {fused,per_cav}` 与 `--seed` 已接入训练、checkpoint config 和 evaluator；两种表示总参数均为 195,675,723，Phase-2 可训练参数均为 4,754,680。
- fused 双卡 smoke 已完成真实 forward/loss/backward/optimizer，unused trainable=0、AMP overflow=0、DDP parameter/buffer checksum delta=0；checkpoint 严格加载及截断 strict-online evaluator 均通过。
- 正式 fused run 已观察到正常启动，但尚无经核验的完成状态、末轮 checkpoint 或正式指标，不能用于论文结论。
- 旧 epoch-19 per-CAV checkpoint 严格加载为 0 missing / 0 unexpected；完整 `tests/colif` 为 135 passed（16 个既有 warning）。
- Experiments/Conclusion 已对照本地 CMP、V2XPnP、ObjectFusion、IS-FUSION、LoGoNet 与 MTR 收敛为 5 个实验小节、2 张表和 2 幅结果图；epoch-19 数值、52,597 条 attention records、17 个引用键、12 个公式与 4 个交叉引用已通过自动审计，绘图专项 8 passed。
- 历史 source refiner 对 core proposal 是梯度隔离的：corridor 位置、base trajectory/logit 均 detach，source 随机过程使用独立 RNG fork，proposal/cross-mode 权重仍为 0.5/0.5；迁移后的 core checkpoint 可用于当前推理，但没有 optimizer/scheduler/scaler state。

## Open

- 核验 fused run 是否完成，并进行固定末轮独立评测。
- 只有 seed 20 的 overall 与 partner-only ADE/FDE 支持 per-CAV，才补两种表示的 seeds 42、2026；否则停止并拒绝 Dual 必要性主张。
- 若 seed-20 闸门通过，完成 per-CAV uniform-mean、协议匹配 MTR/CMP、单卡 latency/peak-memory，并将工作引用转换为正式 BibTeX；Fig. 1 还需提交前重绘为可编辑矢量图。

## Risks

- 当前正在使用的长 run ID 是启动前遗留命名，必须为恢复兼容保留；后续新 run 使用紧凑命名规则。
- 训练入口会复用既有输出目录并追加日志；所有新 run 必须先通过 `check_run_id.py --must-not-exist`，既有任务只用显式 `--resume`。
- `--disable-actor-cooperative-fusion` 是 no-context 诊断，不是 fused-BEV 基线；注意力图也不能替代 fused-vs-per-CAV 因果消融。
- 当前 3.2465 m checkpoint 只能恢复推理权重；任何新增 seed 必须直接按现行无-refiner core 配方重训，不能从该 checkpoint 恢复优化器状态。

## Checkpoints

| ID | Commit | Outcome | Validation | Risk / next |
| --- | --- | --- | --- | --- |
| CP-20260824-02 | `afa5364` | 论文证据边界、实验、结论与结果图收敛 | 数值/主张审计；8 plot tests | 完成 fused 因果对照与投稿必需结果 |
| CP-20260823-02 | `9672d5a` | 分层项目记忆与 fused/per-CAV 消融合同 | state documents reviewed | 正式 fused 指标仍未验证 |
| CP-20260823-01 | `739842a` | 公平 fused/per-CAV 表示开关与 seed CLI | 135 tests；双卡训练/评测 smoke | 完成 seed-20 fused 闸门 |
| CP-20260822-01 | `ef008c5` | epoch-19 权重注意力图和全量统计 | 3,986 frames；52,597 records | 图不构成精度因果证据 |
