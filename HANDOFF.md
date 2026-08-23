# HANDOFF

## Current

- seed-20 fused-BEV 20-epoch 闸门已在 2026-08-23 启动，使用既有长目录 `opencood/logs/dualcmp_fused_receiver01_seed20_20e_20260823`；不要重复启动，也不要在运行中改名或移动该目录。
- 等待任务完成事件；完成后一次性核对退出状态、末轮 checkpoint、独立 `online_masked` 评测及 support counts。规范协议见 `docs/colif_training.md` 第 4 节。
- 当前主模型仍为 `per_cav`；`fused` 只改变预测端空间表示，不改变 tracker、query、history、forecast head、loss 或参数量。

## Verified

- `--forecast-bev-representation {fused,per_cav}` 与 `--seed` 已接入训练、checkpoint config 和 evaluator；两种表示总参数均为 195,675,723，Phase-2 可训练参数均为 4,754,680。
- fused 双卡 smoke 已完成真实 forward/loss/backward/optimizer，unused trainable=0、AMP overflow=0、DDP parameter/buffer checksum delta=0；checkpoint 严格加载及截断 strict-online evaluator 均通过。
- 正式 fused run 已观察到正常启动，但尚无经核验的完成状态、末轮 checkpoint 或正式指标，不能用于论文结论。
- 旧 epoch-19 per-CAV checkpoint 严格加载为 0 missing / 0 unexpected；完整 `tests/colif` 为 135 passed（16 个既有 warning）。

## Open

- 核验 fused run 是否完成，并进行固定末轮独立评测。
- 只有 seed 20 的 overall 与 partner-only ADE/FDE 支持 per-CAV，才补两种表示的 seeds 42、2026；否则停止并拒绝 Dual 必要性主张。

## Risks

- 当前正在使用的长 run ID 是启动前遗留命名，必须为恢复兼容保留；后续新 run 使用紧凑命名规则。
- 训练入口会复用既有输出目录并追加日志；所有新 run 必须先通过 `check_run_id.py --must-not-exist`，既有任务只用显式 `--resume`。
- `--disable-actor-cooperative-fusion` 是 no-context 诊断，不是 fused-BEV 基线；注意力图也不能替代 fused-vs-per-CAV 因果消融。

## Checkpoints

| ID | Commit | Outcome | Validation | Risk / next |
| --- | --- | --- | --- | --- |
| CP-20260823-02 | `9672d5a` | 分层项目记忆与 fused/per-CAV 消融合同 | state documents reviewed | 正式 fused 指标仍未验证 |
| CP-20260823-01 | `739842a` | 公平 fused/per-CAV 表示开关与 seed CLI | 135 tests；双卡训练/评测 smoke | 完成 seed-20 fused 闸门 |
| CP-20260822-01 | `ef008c5` | epoch-19 权重注意力图和全量统计 | 3,986 frames；52,597 records | 图不构成精度因果证据 |
