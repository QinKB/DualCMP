# HANDOFF

## Current

- 正式结构已改为`dual BEV representation + MTR 64 intention queries`：fused consensus与per-CAV source RoIs同时进入每层intention-conditioned cross-attention。
- Phase 2只加载Phase-1检测/跟踪checkpoint；完整forecast branch从随机初始化训练，禁止导入MTR预测checkpoint。
- 已删除6个velocity-scale anchors、12 m residual bound、pairwise-median motion descriptor、source neighbour pooling、旧MTR adapter和旧motion-anchor diagnostics。
- `dual`为主模型；`fused`和`source`是唯一必要的表示消融。旧`per_cav`、direct、geometry、mean等命令已失效。
- 可执行命令只维护在`docs/colif_training.md`。

## Verified

- `tests/colif`：135 passed、1 skipped。
- 组件测试覆盖fixed-64 intentions、six-layer raw-64 loss、dynamic endpoint update、NMS-6、dual分支干预、fully-masked等价、source permutation和两路梯度。
- 完整模型能严格加载Phase-1 tracker checkpoint（406 tensors）；forecast branch保留当前seed对应的随机初始化。
- 当前配置实例化`raw_modes=64`、`output_modes=6`、`decoder_layers=6`；Phase 2 trainable parameters=`68,800,988`，total parameters=`270,445,615`。
- 双卡真实1-batch smoke完成forward/loss/backward/optimizer；unused trainable=`0`、AMP overflow=`0`，参数和buffer checksum最大差均为`0.0`。
- smoke checkpoint记录`phase=2`、`lr=5e-4`和gradient accumulation；同配置resume成功，不同gradient accumulation被config hash拒绝。
- `git diff --check`、compileall、Python import、实验语言lint和LaTeX构建通过。

## Open

- 按相同seed/budget依次运行`train-dual-s20`、`train-fused-s20`、`train-source-s20`。
- 三组固定末轮checkpoint执行receiver `[0,1]`、`online_masked`完整test；不得按test选epoch。

## Risks

- 从头初始化的forecast branch首个smoke batch梯度范数较大，但已由`phase2_max_grad_norm=1000`裁剪；完整20轮稳定性与最终性能仍需正式训练验证。
- 旧模型结果只能作为历史否证/动机，不能作为当前dual结构性能。
- 本地64-point文件与CMP副本字节相同，但生成split、脚本与k-means seed provenance仍缺失。
- `source geometry`和point support未证明独立增益，按架构合同不进入当前模型。

## Checkpoints

| ID | Commit | Outcome | Validation | Risk / next |
| --- | --- | --- | --- | --- |
| CP-20260827-04 | `cfc5a16` | 旧mode-conditioned velocity-anchor reader | 141 tests；双卡smoke；一轮闸门 | 仅供refactor前恢复 |
| CP-20260828-04 | 同名Git提交 | dual representation + 64-intention MTR | 126 tests；strict dual checkpoint load；独立review | GPU smoke pending |
| CP-20260828-05 | 同名Git提交 | scratch forecast initialization + `5e-4` training contract | 135 tests；双卡smoke；resume正反例；独立review | 正式20轮训练待执行 |
