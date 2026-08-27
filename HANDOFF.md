# HANDOFF

## Current

- 当前候选结构为`per_cav actor-aligned RoI tokenizer + anchor-aware mode cross-attention + unique anchored GMM head`。
- 每个目标actor保留逐CAV `5x5` RoI tokens直到mode cross-attention；其他active actors只在各CAV内部做标准RoI max pooling，并以actor-center absolute position形成neighbour object tokens。
- 检测、measurement、persistent tracking、query identity和history始终只使用CMP fused BEV；forecast读取source BEV但不写回tracker。
- 当前代码没有point-occupancy support、relative geometry、source ID embedding、availability/router、trajectory refiner或第二套预测输出。

## Verified

- `tests/colif`：141 passed、2 skipped；覆盖source置换、mask、空source backward、mode-dependent attention、neighbour actor intervention及tokenizer到forecast loss梯度。
- 独立reviewer确认target `LxP`证据未提前池化，neighbour actor-source tokens真实进入每个target mode的K/V，无GT/future泄漏或结构阻断。
- 双卡AMP smoke `smoke-modeobj-amp-percav-s20`完成5 batches/2 optimizer steps：unused trainable=0、AMP overflow=0、DDP parameter/buffer checksum delta=0；trainable parameters=4,425,212。
- 第一次CUDA smoke暴露FP16 attention写入FP32 buffer错误；已在写回处显式转换并由第二次真实双卡smoke验证。
- 历史flat reader结果仅作为失败基线：per-CAV ADE@1/3/5s=`0.851468/1.408316/1.935000`，fused=`0.854988/1.428909/1.972203`。不能代表当前结构。

## Open

- 从同一个Phase-1 seed-20 epoch-9初始化，分别完整训练当前per-CAV与fused结构1 epoch；输出目录固定为`train-modeattn-percav-e1-s20`和`train-modeattn-fused-e1-s20`。
- 两个epoch-0 checkpoint均执行完整receiver `[0,1]` strict-online `online_masked` test，比较overall、receiver-only、partner-only和shared，且核对forecast coverage与matched counts。
- 只有per-CAV在同预算下给出可重复且方向一致的增量，才进入长训；一轮结果不进入论文最终表格。
- 完成成对评测后，审查模型中仍可能发生的source/spatial压缩；若差异仍接近0，先做真实partner-source intervention定位，而不是直接跑20 epochs。

## Risks

- 正式1-epoch目录在启动前必须再次通过`check_run_id.py --must-not-exist`；禁止复用失败或已有目录。
- `--disable-actor-cooperative-fusion`是no-context诊断，不是fused-BEV基线。
- attention图只能说明读取分配，不能证明per-CAV比fused更准；因果证据来自同结构、同初始化、同预算的表示消融。
- `opencood/logs/smoke-modeobj-percav-s20`是AMP dtype错误的首batch失败目录；`smoke-modeobj-amp-percav-s20`是成功但截断的smoke，两者均不是性能结果。

## Checkpoints

| ID | Commit | Outcome | Validation | Next |
| --- | --- | --- | --- | --- |
| CP-20260827-03 | `b32740d` | 新reader改动前的paper/results/code状态 | tracked-only checkpoint | 当前结构完成后新建checkpoint |
| CP-20260825-01 | `14cc0a9` | 旧flat reader消融开关 | 141 tests；双卡smoke | 仅供历史复盘 |
