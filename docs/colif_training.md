# CoLIF 当前训练、恢复与评测

从 `/home/qkb/Project/CoLIF` 执行，环境为 `coop`。当前正式模型只有两个训练阶段，没有 Phase 3：

```text
Phase 1: CMP fused-BEV persistent-query tracking（forecast关闭）
Phase 2: 冻结感知/跟踪，训练cooperative motion forecast
```

正式协议为 official `train` 优化、固定末轮自动评测 official `test`；不构造 val、不按 test 指标自动选择 best。论文必须披露 repeated-test monitoring。

## 当前最小资产集

项目只保留当前训练、评测和投稿实验仍会使用的权重与原始指标：

```text
pretrained/v2v4real/point_pillar_cobevt_multiego_256x/net_epoch75.pth
opencood/logs/colif_paper_phase1_20e_seed20/checkpoint_epoch_009.pth
opencood/logs/colif_paper_phase1_20e_seed20/{train_log,metrics_test}.jsonl
opencood/logs/dualcmp_core_receiver01_seed20_20e_20260821/checkpoint_epoch_019.pth
opencood/logs/dualcmp_v2_full_receiver01_seed20_20e_20260821/{train_log,metrics_test}.jsonl
opencood/logs/dualcmp_fused_receiver01_seed20_20e_20260823/
MTR/output/v2v4real_multiego_cobevt_c256/ckpt/best_model.pth
MTR/output/v2v4real_multiego_cobevt_c256_no_agg/ckpt/best_model.pth
```

最后一个 `opencood` 目录保存已完成的 fused seed-20 训练、epoch-19 checkpoint 和独立 `online_masked` 评测。其余历史 phase、profile、旧 refiner、single-receiver 和旧协议 checkpoint 已从工作区删除；需要追溯时使用 Git 历史中的 `LOG.md`，不能把历史结果恢复为当前论文证据。

`preprocessed_data/v2v4real/fused_features_point_pillar_cobevt_multiego_256x` 约 105 GiB，但它不是日志或冗余权重。当前 MTR/CMP 配置直接把它作为 `cobevt_fused_features_dir`，投稿前尚需完成 learned baselines，因此暂时保留；删除后必须重新提取整套 fused features。相关 tracking/GT caches 同理保留到 MTR/CMP 协议对照结束。

## Receiver / ego 协议

默认 `receiver_ids: [0]`，即只以 CAV 0 为 receiver。若要严格采用 CMP 的 multi-ego 数据增广，在 **Phase 1、Phase 2、独立评测与推理** 命令中均添加：

```bash
--receiver-ids 0,1
```

此时 CAV 0 和 CAV 1 各自轮流作为 receiver：对每个 receiver，其他 CAV 点云先变换到该 receiver 坐标系，receiver 本身始终位于 source slot 0，online query state 以 `(scene, receiver)` 独立重置。它会增加 receiver-centric clips；必须从 Phase 1 重新训练，并以同一 receiver 协议训练/评测，不能把旧的 `[0]` checkpoint 结果与 `[0,1]` 结果混报。`--resume` 会因配置 hash 不同拒绝跨协议恢复。

## 1. 依赖预检

首次运行或 Python/PyTorch/CUDA 变化后编译固定 MOTR deformable-attention：

```bash
cd /home/qkb/Project/CoLIF/third_party/MOTR/models/ops
conda run --no-capture-output -n coop sh make.sh
cd /home/qkb/Project/CoLIF
```

运行回归：

```bash
conda run --no-capture-output -n coop python -m pytest -q tests/colif
```

## 2. Phase 1：fused query tracking

```bash
ulimit -n 65536
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 -m opencood.tools.train_colif \
  --hypes_yaml opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --phase 1 --batch-size 2 \
  --init-cobevt-checkpoint \
  pretrained/v2v4real/point_pillar_cobevt_multiego_256x/net_epoch75.pth \
  --output-dir opencood/logs/train-p1-s20 \
  --epochs 20 --workers 2 --eval-workers 2 --seed 20 --log-every 50
```

Phase 1 使用 FP32。当前论文实验统一从第 3 节所列的 seed-20 epoch-9 Phase-1 checkpoint 初始化；旧单轮诊断使用的 tracker 已清理，不能与当前 receiver `[0,1]`、20-epoch 协议混用。

## 3. Phase 2：mode-conditioned source feature fusion

当前结构使用 actor-aligned RoI tokenizer 保留目标actor的`[L,P,D]`细粒度
source tokens；其他actor只在各source内部做标准RoI max pooling。带physical-anchor
encoding的motion queries直接cross-attend目标局部tokens与邻居actor-source tokens，
最后进入唯一anchored GMM head。模型没有point-occupancy support、relative geometry、
availability router、trajectory refiner或第二套预测输出。

```bash
ulimit -n 65536
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 -m opencood.tools.train_colif \
  --hypes_yaml opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --phase 2 \
  --init-colif-checkpoint \
  /home/qkb/Project/CoLIF/opencood/logs/colif_paper_phase1_20e_seed20/checkpoint_epoch_009.pth \
  --output-dir opencood/logs/train-modeattn-percav-s20 \
  --epochs 20 --batch-size 2 --grad-accum-steps 2 --workers 2 \
  --receiver-ids 0,1 --forecast-bev-representation per_cav --seed 20 \
  --no-eval --log-every 50

```

旧20轮结果属于已替换的flat actor-source reader；只能作为历史基线，不能加载为
当前结构的训练恢复点：

```text
opencood/logs/dualcmp_core_receiver01_seed20_20e_20260821/checkpoint_epoch_019.pth
online-masked ADE@1/3/5s = 0.8515 / 1.4083 / 1.9350 m
online-masked FDE@1/3/5s = 1.0310 / 2.1841 / 3.2465 m
MOTA=0.6504, forecast coverage=0.8049
```

## 4. 一轮 fused-BEV / per-CAV BEV 结构闸门

`--disable-actor-cooperative-fusion` 会把预测上下文置零，它只能作为
no-context 诊断，不能作为 fused-BEV 基线。正式表示消融必须保持同一个
tokenizer、mode decoder、forecast head、tracker 和训练协议，只改变预测端空间输入：

```bash
# fused-BEV actor RoI tokens
--forecast-bev-representation fused

# aligned per-CAV actor RoI tokens
--forecast-bev-representation per_cav
```

两种模式的总参数量相同。fused 模式把 `fused_bev.unsqueeze(1)` 作为唯一
receiver-frame source；per_cav 模式使用
`regrouped_source_bev[B,L,C,H,W]`。检测、跟踪、query、history、motion
anchors 和 loss 均保持不变。

先用一轮同预算闸门判断新结构是否值得长训。启动前必须分别运行：

```bash
python /home/qkb/.agents/skills/code-implementation/scripts/check_run_id.py \
  opencood/logs/train-modeattn-percav-e1-s20 --must-not-exist
python /home/qkb/.agents/skills/code-implementation/scripts/check_run_id.py \
  opencood/logs/train-modeattn-fused-e1-s20 --must-not-exist
```

per-CAV一轮训练：

```bash
ulimit -n 65536
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 -m opencood.tools.train_colif \
  --hypes_yaml opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --phase 2 \
  --init-colif-checkpoint \
  opencood/logs/colif_paper_phase1_20e_seed20/checkpoint_epoch_009.pth \
  --output-dir opencood/logs/train-modeattn-percav-e1-s20 \
  --epochs 1 --batch-size 2 --grad-accum-steps 2 --workers 2 \
  --receiver-ids 0,1 --forecast-bev-representation per_cav --seed 20 \
  --no-eval --log-every 50
```

fused一轮训练只改变representation与输出目录：

```bash
ulimit -n 65536
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 -m opencood.tools.train_colif \
  --hypes_yaml opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --phase 2 \
  --init-colif-checkpoint \
  opencood/logs/colif_paper_phase1_20e_seed20/checkpoint_epoch_009.pth \
  --output-dir opencood/logs/train-modeattn-fused-e1-s20 \
  --epochs 1 --batch-size 2 --grad-accum-steps 2 --workers 2 \
  --receiver-ids 0,1 --forecast-bev-representation fused --seed 20 \
  --no-eval --log-every 50
```

两组必须使用epoch-0 checkpoint独立完整评测。例如per-CAV：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 \
  -m opencood.tools.eval_colif_v2v4real \
  --hypes_yaml opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --checkpoint \
  opencood/logs/train-modeattn-percav-e1-s20/checkpoint_epoch_000.pth \
  --split test --workers 2 --receiver-ids 0,1 \
  --forecast-metric-protocol online_masked \
  --output \
  opencood/logs/train-modeattn-percav-e1-s20/metrics_test_online_masked.json
```

一轮结果只作为结构筛选，不进入论文最终表格。只有per-CAV在整体和
partner-only方向均优于同结构fused，且训练/coverage完全可比，才进入长训。

训练入口会创建或复用输出目录并追加日志，因此检查失败时必须改用新 run ID；
恢复既有任务只能走第 6 节的显式 `--resume` 路径。

旧 one-epoch no-source/no-history 诊断属于早期 single-receiver 研究阶段，既不进入当前论文表格，也不能替代 fused/per-CAV 表示消融；其日志和 checkpoint 已清理。若未来重新研究 ordered history，必须从当前 Phase-1 checkpoint 按 receiver `[0,1]` 和 20-epoch 协议重新训练。

## 5. 旧flat reader消融记录

`w/o Relative Geometry`与旧`mean`开关属于已经替换的flat actor-source reader。
训练入口现在显式拒绝`--source-aggregation`、`--source-geometry`、
`--disable-proposal-refinement`和`--forecast-inference-layer`，防止新旧网络混训。
旧数值只保留用于失败复盘；不得用下列历史checkpoint恢复当前结构。

历史`online_masked`结果：

```text
w/o Relative Geometry (epoch 19)
ADE@1/3/5s = 0.849644 / 1.414029 / 1.948706
FDE@1/3/5s = 1.032063 / 2.190736 / 3.249115

w/o Motion Anchors (epoch 19)
ADE@1/3/5s = 1.270013 / 2.307474 / 3.119810
FDE@1/3/5s = 1.794235 / 3.912687 / 5.568135
```

现有geometry-off epoch-19来自每卡batch 2、428 updates/epoch，共8,560 updates；
现有direct epoch-19实际来自每卡batch 4、214 updates/epoch，共4,280 updates。因此
direct数值只能作为当前direct-regression formulation的
结果；严格的motion-anchor单模块归因需要按上方匹配命令重训。

对应证据文件为`opencood/logs/train-query-s20/metrics_test_online_masked_epoch019.json`和
`opencood/logs/train-direct-s20/metrics_test_online_masked.json`。

## 6. 同阶段断点恢复

`--resume` 仅用于同一结构、同一阶段、同一配置恢复 model/optimizer/scheduler/scaler/epoch/global step：

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 -m opencood.tools.train_colif \
  --hypes_yaml opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --phase 2 \
  --resume /path/to/checkpoint_epoch_000.pth \
  --output-dir /path/to/resume_output \
  --epochs 2 --workers 2 --eval-workers 2
```

配置 hash 不一致会拒绝 resume。跨阶段只能使用 `--init-colif-checkpoint`。由旧 wrapper 提取的 `dualcmp_core_receiver01_seed20_20e_20260821` 是仅供推理和评测的 checkpoint，不含可恢复的 optimizer/scheduler/scaler state。

## 7. 双卡 smoke

```bash
ulimit -n 65536
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 -m opencood.tools.train_colif \
  --hypes_yaml opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --phase 2 \
  --init-colif-checkpoint \
  opencood/logs/colif_paper_phase1_20e_seed20/checkpoint_epoch_009.pth \
  --output-dir opencood/logs/smoke-modeobj-percav-s21 \
  --epochs 1 --batch-size 2 --grad-accum-steps 2 --workers 2 \
  --receiver-ids 0,1 --forecast-bev-representation per_cav --seed 21 \
  --smoke-batches 5 --no-eval --log-every 1
```

只有 smoke 可用 `--smoke-force-birth`；任何 smoke/截断 test 数字都不能作为论文性能。

## 8. 独立 strict-online test 与 GT-free inference

评测 checkpoint（checkpoint 内配置优先）：

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 \
  -m opencood.tools.eval_colif_v2v4real \
  --hypes_yaml opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --checkpoint /path/to/checkpoint_epoch_000.pth \
  --split test --workers 2 --output /path/to/metrics_test.json
```

评测按完整 `(scene, receiver)` timeline 分配 rank，保持 online state；所有 numerator/count 在写 JSON 前跨卡 all-reduce。`--max-frames-per-rank` 只用于通路 smoke。

GT-free inference：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n coop python \
  -m opencood.tools.inference_colif_v2v4real \
  --hypes_yaml opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --checkpoint /path/to/checkpoint_epoch_000.pth \
  --split test --output results/colif_online/test_runtime.pth
```

runtime dataset 递归拒绝 GT IDs、boxes、labels 与 future targets；`ass_id` 只进入 criterion/evaluator。
