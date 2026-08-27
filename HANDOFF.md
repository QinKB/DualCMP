# HANDOFF

## Current

- 当前候选结构为`per_cav actor-aligned RoI tokenizer + anchor-aware mode cross-attention + unique anchored GMM head`；一轮同预算表示闸门已通过。
- 用户已决定进入20轮训练；唯一可执行命令维护在`docs/colif_training.md`第3、5节。
- 每个目标actor保留逐CAV `5x5` RoI tokens直到mode cross-attention；其他active actors只在各CAV内部做标准RoI max pooling，并以actor-center absolute position形成neighbour object tokens。
- 检测、measurement、persistent tracking、query identity和history始终只使用CMP fused BEV；forecast读取source BEV但不写回tracker。
- 当前代码没有point-occupancy support、relative geometry、source ID embedding、availability/router、trajectory refiner或第二套预测输出。

## Verified

- `tests/colif`：141 passed、2 skipped；覆盖source置换、mask、空source backward、mode-dependent attention、neighbour actor intervention及tokenizer到forecast loss梯度。
- 独立reviewer确认target `LxP`证据未提前池化，neighbour actor-source tokens真实进入每个target mode的K/V，无GT/future泄漏或结构阻断。
- 双卡AMP smoke `smoke-modeobj-amp-percav-s20`完成5 batches/2 optimizer steps：unused trainable=0、AMP overflow=0、DDP parameter/buffer checksum delta=0；trainable parameters=4,425,212。
- 第一次CUDA smoke暴露FP16 attention写入FP32 buffer错误；已在写回处显式转换并由第二次真实双卡smoke验证。
- 历史flat reader结果仅作为失败基线：per-CAV ADE@1/3/5s=`0.851468/1.408316/1.935000`，fused=`0.854988/1.428909/1.972203`。不能代表当前结构。
- 当前结构从同一个Phase-1 seed-20 epoch-9初始化，各训练1 epoch/428 updates。per-CAV相对fused的ADE@1/3/5s为`1.090815/1.724338/2.199316` vs `1.122182/1.763524/2.247769`，分别改善`2.80%/2.22%/2.16%`；FDE为`1.363446/2.427387/3.382485` vs `1.390626/2.454115/3.442292`，分别改善`1.95%/1.09%/1.74%`。
- 两组完整评测均覆盖3,986 frames、52,597 matched actor frames、coverage=`0.804925`。5 s partner-only ADE/FDE改善`0.61%/0.84%`，receiver-only改善`0.73%/1.19%`，shared改善`4.61%/2.85%`；差异不来自tracking population变化。
- per-CAV/fused mean clip loss=`151.4032/154.6647`；两组均epoch complete、global step 428、AMP overflow 0、DDP checksum delta 0。
- 总体六个MR中唯一主指标例外是2.0 m MR@5s：per-CAV=`0.348043`、fused=`0.346864`，相对轻微恶化`0.34%`；其余五个MR更好。结论必须限定为ADE/FDE六项方向一致。
- 当前主机为24 logical CPUs、约128 GiB RAM和2张24 GiB RTX 3090。`smoke-bs4w4-percav-s20`验证每卡batch 4、workers 4、accumulation 1：5 batches/5 updates、unused trainable=0、AMP overflow=0、DDP checksum delta=0。相对已验证`batch 2/workers 2/accumulation 2` smoke，在相同启动/保存开销下按处理样本数吞吐提高约43%，且有效global batch仍为8、每epoch仍为428 updates。

## Open

- 依次运行Full、fused BEV和Direct Regression的20轮seed-20训练；若论文独立主张ordered temporal history，再运行`w/o Temporal History`。
- 当前没有合法的`w/o Neighbour Interaction`开关；需要该表行时必须先实现明确只移除neighbour object tokens的受控开关，不能用no-context或query容量变化替代。
- 每个20轮run完成后用epoch-19 checkpoint独立执行完整receiver`[0,1]`、`online_masked` test；不得根据test选择epoch。

## Risks

- 正式1-epoch目录在启动前必须再次通过`check_run_id.py --must-not-exist`；禁止复用失败或已有目录。
- `--disable-actor-cooperative-fusion`是no-context诊断，不是fused-BEV基线。
- attention图只能说明读取分配，不能证明per-CAV比fused更准；因果证据来自同结构、同初始化、同预算的表示消融。
- `opencood/logs/smoke-modeobj-percav-s20`是AMP dtype错误的首batch失败目录；`smoke-modeobj-amp-percav-s20`是成功但截断的smoke，两者均不是性能结果。
- 推荐`batch 4/workers 4/accumulation 1`只在当前双3090主机完成短训验证；尚未记录峰值显存。若长期运行出现OOM，退回已完整验证的`2/2/2`，保持global batch 8。
- 训练命令必须使用绝对`PYTHONPATH`和绝对配置/checkpoint/output路径。一次用户启动实际停留在`/home/qkb/Project`，相对`PYTHONPATH=.`导致两个rank在创建run目录前立即退出；`smoke-e30-bs4w4-s20`随后用相同30-epoch/batch/worker配置从正确项目路径完成1-batch验证。

## Checkpoints

| ID | Commit | Outcome | Validation | Next |
| --- | --- | --- | --- | --- |
| CP-20260827-03 | `b32740d` | 新reader改动前的paper/results/code状态 | tracked-only checkpoint | 当前结构完成后新建checkpoint |
| CP-20260827-04 | `cfc5a16` | mode-conditioned target/neighbor source object fusion | 141 tests；独立review；双卡AMP smoke | 一轮表示闸门与完整test |
| CP-20260825-01 | `14cc0a9` | 旧flat reader消融开关 | 141 tests；双卡smoke | 仅供历史复盘 |
