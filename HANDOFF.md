# HANDOFF

## Current

- 正式结构已改为`dual BEV representation + MTR 64 intention queries`：fused consensus与per-CAV source RoIs同时进入每层intention-conditioned cross-attention。
- 已删除6个velocity-scale anchors、12 m residual bound、pairwise-median motion descriptor、source neighbour pooling、旧MTR adapter和旧motion-anchor diagnostics。
- `dual`为主模型；`fused`和`source`是唯一必要的表示消融。旧`per_cav`、direct、geometry、mean等命令已失效。
- 可执行命令只维护在`docs/colif_training.md`。

## Verified

- `tests/colif`：126 passed、1 skipped。
- 组件测试覆盖fixed-64 intentions、six-layer raw-64 loss、dynamic endpoint update、NMS-6、dual分支干预、fully-masked等价、source permutation和两路梯度。
- 完整模型能严格加载Phase-1 tracker checkpoint（406 tensors）及CMP/MTR motion decoder checkpoint（714 tensors），missing/unexpected均为空。
- 当前配置实例化`raw_modes=64`、`output_modes=6`、`decoder_layers=6`；Phase 2 trainable parameters=`70,114,268`，total parameters=`270,445,615`。
- `git diff --check`、compileall和Python import检查通过。

## Open

- 等当前legacy `train-modeobj-s20` 30轮任务释放两张RTX 3090后，运行`smoke-dual-s20`真实双卡1-batch forward/loss/backward/optimizer smoke。
- smoke通过后，按相同seed/budget依次运行`train-dual-s20`、`train-fused-s20`、`train-source-s20`。
- 三组固定末轮checkpoint执行receiver `[0,1]`、`online_masked`完整test；不得按test选epoch。

## Risks

- 当前GPU 0/1被用户已有legacy训练占用；本次没有中断、修改或争抢该进程，因此真实GPU smoke尚未执行。
- 旧模型结果只能作为历史否证/动机，不能作为当前dual结构性能。
- 本地64-point文件与CMP副本字节相同，但生成split、脚本与k-means seed provenance仍缺失。
- `source geometry`和point support未证明独立增益，按架构合同不进入当前模型。

## Checkpoints

| ID | Commit | Outcome | Validation | Risk / next |
| --- | --- | --- | --- | --- |
| CP-20260827-04 | `cfc5a16` | 旧mode-conditioned velocity-anchor reader | 141 tests；双卡smoke；一轮闸门 | 仅供refactor前恢复 |
| CP-20260828-04 | 同名Git提交 | dual representation + 64-intention MTR | 126 tests；strict dual checkpoint load；独立review | GPU smoke pending |
