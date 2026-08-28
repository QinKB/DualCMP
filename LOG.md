# CoLIF Log

本文件只保留仍影响当前实现、实验或恢复的里程碑。2026-08-24 以前的详细实验 chronology 可从 Git 历史读取；被拒绝路线的根因见 `docs/colif_failure_postmortem.md`。

## 2026-08-28 — Dual representation + MTR intention-query refactor

- `CP-20260828-04`按`模型架构更新.md`把正式预测改为fused consensus actor RoI + per-CAV source RoIs，两路由同一组MTR 64 endpoint intention queries分别读取；raw-64在六层训练，推理endpoint NMS到6。
- 删除velocity-scale anchors、12 m residual、pairwise-median motion descriptor、source-neighbour pooling、旧MTR adapter、residual prototypes及lateral/multiscale hand diagnostics。History改为masked learned temporal encoder；actor interaction只由MTR object decoder负责。
- 独立reviewer发现并推动修复MTR legacy local frame与source patch position错配、dual双残差幅度混杂及attention图字段错误。最终两路按`sqrt(active_paths)`缩放并经共享post-norm，nonzero-yaw position test直接与loss transform对齐。
- 验证：`tests/colif` 126 passed/1 skipped；Phase-1 406 tensors和MTR decoder 714 tensors均strict load且missing/unexpected为空；`dual/fused/source`均为70,114,268 trainable、270,445,615 total parameters和491个trainable tensors。
- 限制：两张GPU被启动于本次refactor之前的legacy `train-modeobj-s20`占用，本次未中断该用户进程；当前结构的真实双卡AMP/DDP smoke和性能实验尚未运行。

## 2026-08-28 — Phase-2 batch/worker资源闸门

- 当前主机资源为24 logical CPUs、约128 GiB RAM和2张24 GiB RTX 3090；检查时GPU空闲、磁盘可用约450 GB。
- `smoke-bs4w4-percav-s20`以每卡batch 4、每rank workers 4、accumulation 1完成5-batch真实双卡AMP/DDP训练：unused trainable=0、AMP overflow=0、parameter/buffer checksum delta=0、无OOM。它保持global batch 8、每epoch 428 updates，与一轮闸门的`batch 2/workers 2/accumulation 2`优化预算一致。
- 旧`2/2/2` smoke处理20个global clips约49.91 s，新`4/4/1`处理40个约69.71 s；包含相同启动与checkpoint开销的样本吞吐提高约43%。因此当前主机20轮命令采用`4/4/1`，`2/2/2`仅作OOM回退。
- 20轮最小实验集合为Full、fused BEV和Direct Regression；`w/o Temporal History`仅在论文主张该组件时运行。当前没有合法的`w/o Neighbour Interaction`开关，不能用no-context或修改query容量伪造。
- 一次正式启动的终端实际cwd为`/home/qkb/Project`而非CoLIF根目录，相对`PYTHONPATH=.`使两个torchrun ranks在run目录创建前立即退出，外层只保留`ChildFailedError`。使用独立`smoke-e30-bs4w4-s20`从正确路径复现相同`epochs=30/batch=4/workers=4`配置，1 batch forward/backward/update完成且AMP/DDP正常。当前规范命令因此改用绝对`PYTHONPATH`和绝对配置、checkpoint、output路径，避免cwd依赖。

## 2026-08-27 — mode-conditioned actor/source object interaction

- 旧flat actor-source reader、trajectory source refiner和双预测出口已替换为单一forecast head。目标actor的逐CAV `5x5` RoI tokens保留到anchor-aware mode cross-attention；其他active actor在每个CAV内部执行标准RoI max pooling，再加入actor-center absolute position并作为neighbour object tokens。
- 代码审查发现raw point occupancy会把无回波但仍有BEV上下文的区域硬屏蔽，因此该实现及接口已完整删除。当前没有support、availability、router、relative geometry或source ID embedding；mask仅由active actor、CAV padding和BEV boundary组成。
- 141 tests通过；synthetic intervention确认改变neighbour actor的partner-source token会改变target mode score，而mask该token后变化无效。独立reviewer确认无剩余阻断或高风险问题。
- 首次真实双卡AMP smoke暴露FP16 attention写入FP32 buffer错误，修复后`smoke-modeobj-amp-percav-s20`完成5 batches/2 optimizer steps：unused trainable=0、AMP overflow=0、DDP parameter/buffer checksum delta=0。
- 下一闸门为同一Phase-1 seed-20 epoch-9初始化下，per-CAV与fused各完整训练1 epoch并执行完整`online_masked` test；旧20-epoch flat结果不能代表新结构。
- 一轮成对闸门已完成。per-CAV相对fused的ADE@1/3/5s为`1.090815/1.724338/2.199316` vs `1.122182/1.763524/2.247769`，改善`2.80%/2.22%/2.16%`；FDE为`1.363446/2.427387/3.382485` vs `1.390626/2.454115/3.442292`，改善`1.95%/1.09%/1.74%`。
- 两组均为428 updates且评测对象完全一致：3,986 frames、52,597 matched actor frames、coverage=`0.804925`，receiver-only/partner-only/shared counts=`15,847/11,902/24,848`。5 s ADE/FDE相对改善分别为receiver-only `0.73%/1.19%`、partner-only `0.61%/0.84%`、shared `4.61%/2.85%`。
- 5 s 2.0 m MR中per-CAV轻微差`0.348043` vs `0.346864`，其余1/3 s 2.0 m MR和全部1/3/5 s 3.6 m MR均更好。因此闸门结论是“source-specific representation产生一致ADE/FDE增量”，不是所有指标无条件占优。
- 原始metrics SHA-256：per-CAV `d3c7fa6a95dea3003e6ee9bc3b5c62def276326130d7f49fe4689b8a8c2e9452`；fused `bd755fbd9e7229dc764b184ef6e420020d07f6b45de72ef84bbfdfdab2229513`。

## 2026-08-27 — relative geometry与motion anchors消融

- geometry-off从epoch-15恢复并完成epoch-19，最终global step 8,560；独立双卡`online_masked`评测取得ADE@1/3/5s=`0.849644/1.414029/1.948706`、FDE=`1.032063/2.190736/3.249115`。相对Full仅ADE@1s改善0.21%，其余五项退化0.08%--0.71%，不支持显著独立geometry贡献；旧epoch-15 metrics被保留但不进入最终表。
- `train-direct-s20/checkpoint_epoch_019.pth`完成20轮；同协议ADE=`1.270013/2.307474/3.119810`、FDE=`1.794235/3.912687/5.568135`，相对Full分别增加49.16/63.85/61.23%和74.03/79.14/71.51%。该checkpoint使用每卡batch 4、214 updates/epoch，而Full/geometry-off使用batch 2；同时direct formulation移除了anchors和12 m residual bound，因此不能把全部退化归因于anchors。
- 两项评测均为exit 0、strict-online、GT-free、world size 2；与Full/fused相同地覆盖3,986 frames、52,597 matched actor frames、80.4925% coverage及15,847/11,902/24,848个receiver-only/partner-only/shared records。正式JSON保存在各自运行目录。
- 经文献审查，uniform mean不再作为论文必要消融；该开关仅保留用于诊断。

## 2026-08-25 — fused闸门与必要消融开关

- seed-20 fused epoch-19在相同3,986 frames、52,597 matched forecasts和support counts下取得ADE@1/3/5s=`0.854988/1.428909/1.972203`、FDE=`1.044122/2.209837/3.278118`；per-CAV在六个ADE/FDE、三档MR及receiver-only/partner-only/shared 5 s ADE/FDE上方向均更优。
- fused目录已只保留epoch-19、`train_log.jsonl`和`metrics_test_online_masked.json`；epoch 10--18中间权重已删除且不可从Git恢复。
- `CP-20260825-01` / `14cc0a9`增加uniform mean、query attention without geometry和direct trajectory开关。141 tests通过；三种模式双卡optimizer smoke均为unused trainable=0、AMP overflow=0、parameter/buffer checksum delta=0；既有per-CAV/fused checkpoint strict load均为0 missing/unexpected。
- 三个正式20轮命令唯一维护在`docs/colif_training.md`第5节；结果完成前不得填写论文消融表。

## 2026-08-24 — 工作区资产收敛

- `opencood/logs` 清理前有 158 个运行目录、约 233.58 GiB；其中绝大部分是旧 phase、smoke、profile、single-receiver、失败结构和已删除 source-refiner 的完整 checkpoint，而不是文本日志。
- 已删除 154 个旧运行目录、49 个冗余 epoch checkpoint、旧 MTR seed-42 中间训练目录、Python/pytest 缓存、论文审计草稿、会话转储、已替代框架图和已完成/拒绝的计划文档。活动任务的 epoch-10 checkpoint 已通过 epoch-complete、ZIP CRC 和 DDP 同步记录核验，epoch 0--9 随后删除。`opencood/logs` 清理后仅保留 4 个目录、约 2.35 GiB；活动任务后续仍会生成 epoch 11--19，完成评测后只保留固定末轮。
- 当前最小恢复链为：Phase-1 epoch-9 初始化、per-CAV core epoch-19、per-CAV train/metrics JSONL、活动 fused seed-20 目录、CMP/MTR 两个官方预测权重和 CoBEVT 预训练权重。精确路径由 `docs/colif_training.md` 维护。
- 保留权重 SHA-256：CoBEVT `60e9e84e...9c920b`；Phase-1 epoch-9 `6c585b69...5bd3b`；per-CAV core epoch-19 `ef6c94af...6be43c`；CMP MTR aggregation `39951eea...4dc12`；MTR no-aggregation `213683c4...d1356`。
- 删除的二进制 checkpoint 不在 Git 中，不能从仓库恢复；其已验证结论仍保留在失败复盘或历史 Git 版本中。未来训练不得从这些旧 checkpoint 恢复。
- CoLIF 清理检查点：`CP-20260824-04` / `f40c26b`。独立 MOTR 依赖仓库删除其历史受跟踪 bytecode 并忽略生成物：`CP-20260824-01` / `42e26ec`。

## 2026-08-24 — 全稿证据审计

- `manuscript.md` 的 Experiments 收敛为 Experimental Setup、Overall Performance、Effect of Source-Specific BEV Features、Source-Aware Analysis、Efficiency and Limitations；主文保留 2 张表和 2 幅与核心表示主张直接相关的结果图。
- epoch-19 指标逐项对回 per-CAV `metrics_test.jsonl`；attention 数量、中位数和距离分组逐项对回 `figures/dualcmp_attention_records.npz` 与 `dualcmp_attention_summary.json`。审计覆盖 3,986 frames 和 52,597 matched forecasts。
- 17 个引用键、12 个公式、4 个图表交叉引用和绘图专项 8 tests 通过。论文明确保留 fused/per-CAV 因果对照、额外 seeds、learned baselines 和效率结果的未完成边界。
- 论文/结果图检查点：`CP-20260824-02` / `afa5364`；状态边界检查点：`CP-20260824-03` / `5c02e6e`。

## 2026-08-23 — 协议匹配的 fused/per-CAV 闸门

- `CP-20260823-01` / `739842a` 增加 `--forecast-bev-representation {fused,per_cav}`。两种表示只改变预测 reader 的空间输入，tracker、query、history、forecast head、loss 和参数量一致。
- seed-20 fused 20-epoch 任务使用恢复兼容的历史长目录运行；完成前不移动、不改名，并始终保留最新完整恢复 checkpoint。完成后固定末轮独立执行 `online_masked` 评测，只有 overall 和 partner-only 方向均支持 per-CAV 才继续 seeds 42/2026。
- 注意力权重只能诊断 source allocation，不能替代 fused-vs-per-CAV 的 ADE/FDE 因果消融。

## 2026-08-22 — 当前 per-CAV core 与注意力证据

- receiver `[0,1]`、20-epoch、seed-20 的 epoch-19 proposal 在 `online_masked` 下取得 ADE@1/3/5s=`0.851468/1.408316/1.935000`，FDE@1/3/5s=`1.030991/2.184138/3.246500`，MOTA=`0.650443`，coverage=`0.804925`。
- 旧 wrapper 的 source-refiner 输入和 base trajectory/logit 均显式 detach，source RNG 独立，proposal/cross-mode loss 权重保持 0.5/0.5；因此删除 45 个 refiner tensors并重命名 `motion_base.*` 后，core checkpoint 可严格加载当前无-refiner模型。该 checkpoint 只供推理/评测，不含可恢复 optimizer/scheduler/scaler state。
- 注意力全量运行覆盖 3,986 receiver-centric test frames、52,597 个有效 5-s records：receiver-only 15,847、partner-only 11,902、shared 24,848。partner attention mass 中位数为 0.3706/0.5988/0.4839；相对距离五组均值从 0.3600 增至 0.5642，95% CI 使用 1,000 次 scene-receiver cluster bootstrap。
- 当前核心模型与 source-refiner 删除检查点：`07a52b1`；注意力图检查点：`CP-20260822-01` / `ef008c5`。

## 仍有效的失败教训

- `--disable-actor-cooperative-fusion` 是 no-context 诊断，不是 fused-BEV baseline。
- 预测不得反馈当前 tracker；历史 future-feedback、trajectory-source refiner 和 availability router 未改善最终指标，已从 active model 删除。
- 第一帧只能建立状态；至少两个在线 state entries 后才具备速度依据并输出长时预测。
- 旧 CMP cache、single-receiver、one-epoch、`cmp_fixed` 和 `online_masked` 结果具有不同 target population 或协议，不能混报。
