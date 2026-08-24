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

最后一个 `opencood` 目录是正在运行的 fused seed-20 任务，完成独立评测前必须保留目录、训练日志和最新完整 checkpoint；更早的完整 epoch 权重可以在验证最新恢复点后清理。其余历史 phase、smoke、profile、旧 refiner、single-receiver 和旧协议 checkpoint 已从工作区删除；需要追溯时使用 Git 历史中的 `LOG.md`，不能把历史结果恢复为当前论文证据。

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

## 3. Phase 2：agent-specific CAV feature fusion

默认配置为 geometry-conditioned actor-source fusion + ordered history + training-only cross-mode auxiliary，推理输出 proposal layer：

```bash
ulimit -n 65536
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 -m opencood.tools.train_colif \
  --hypes_yaml opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --phase 2 \
  --init-colif-checkpoint \
  /home/qkb/Project/CoLIF/opencood/logs/colif_paper_phase1_20e_seed20/checkpoint_epoch_009.pth \
  --output-dir opencood/logs/train-percav-s20 \
  --epochs 20 --batch-size 2 --grad-accum-steps 2 --workers 2 \
  --receiver-ids 0,1 --forecast-bev-representation per_cav --seed 20 \
  --no-eval --log-every 50

```

已完成的 20 轮 receiver `[0,1]` 结果来自已移除 wrapper 中的 proposal 主干。该主干参数已逐张量原样提取到当前无 source-refiner 结构，严格加载为 0 missing / 0 unexpected：

```text
opencood/logs/dualcmp_core_receiver01_seed20_20e_20260821/checkpoint_epoch_019.pth
online-masked ADE@1/3/5s = 0.8515 / 1.4083 / 1.9350 m
online-masked FDE@1/3/5s = 1.0310 / 2.1841 / 3.2465 m
MOTA=0.6504, forecast coverage=0.8049
```

## 4. 最小必要的 fused-BEV / per-CAV BEV 对照

`--disable-actor-cooperative-fusion` 会把预测上下文置零，它只能作为
no-context 诊断，不能作为 fused-BEV 基线。正式表示消融必须保持同一个
actor reader、forecast head、tracker 和训练协议，只改变预测端空间输入：

```bash
# fused-BEV actor patch
--forecast-bev-representation fused

# DualCMP aligned per-CAV actor patches
--forecast-bev-representation per_cav
```

两种模式的总参数量相同。fused 模式把 `fused_bev.unsqueeze(1)` 作为唯一
receiver-frame source；per_cav 模式使用
`regrouped_source_bev[B,L,C,H,W]`。检测、跟踪、query、history、motion
anchors 和 loss 均保持不变。

先运行 seed-20 闸门。已有 per-CAV checkpoint 为：

```text
opencood/logs/dualcmp_core_receiver01_seed20_20e_20260821/checkpoint_epoch_019.pth
```

只需新增一组 fused 20-epoch 训练：

当前 seed-20 任务已经使用下面的历史长目录启动。为保证运行与恢复兼容，
本轮不得在运行中改名；该例外不再用于新任务。后续新 run 使用紧凑 ID，
例如 `train-fused-s42`、`train-percav-s42`、`train-fused-s2026`。

```bash
ulimit -n 65536
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 -m opencood.tools.train_colif \
  --hypes_yaml opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --phase 2 \
  --init-colif-checkpoint \
  opencood/logs/colif_paper_phase1_20e_seed20/checkpoint_epoch_009.pth \
  --output-dir opencood/logs/dualcmp_fused_receiver01_seed20_20e_20260823 \
  --epochs 20 --batch-size 2 --grad-accum-steps 2 --workers 2 \
  --receiver-ids 0,1 --forecast-bev-representation fused --seed 20 \
  --no-eval --log-every 50
```

固定末轮独立评测：

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 \
  -m opencood.tools.eval_colif_v2v4real \
  --hypes_yaml opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --checkpoint \
  opencood/logs/dualcmp_fused_receiver01_seed20_20e_20260823/checkpoint_epoch_019.pth \
  --split test --workers 2 --receiver-ids 0,1 \
  --forecast-metric-protocol online_masked \
  --output \
  opencood/logs/dualcmp_fused_receiver01_seed20_20e_20260823/metrics_test_online_masked.json
```

只有 seed 20 中 per-CAV 的整体和 partner-only ADE/FDE 方向优于 fused，才继续
补 seed 42 与 2026。后续两种表示都从同一个 Phase-1 checkpoint 初始化，并仅
修改 `--seed`、`--forecast-bev-representation` 和输出目录。最终报告三个 seed
的 mean/std；不得按 test 结果选择 checkpoint。

启动 seed 42/2026 前必须同时满足：seed-20 训练进程已退出；固定末轮 checkpoint
存在且严格加载；独立 `online_masked` 完整评测已生成；overall、partner-only 与
support counts 已和 per-CAV 结果按同一协议核对；方向闸门通过。任一项未验证就
停止，不得提前并发下一 seed。

每个新输出目录还必须在启动前通过命名与冲突检查：

```bash
python /home/qkb/.agents/skills/code-implementation/scripts/check_run_id.py \
  opencood/logs/train-fused-s42 --must-not-exist
python /home/qkb/.agents/skills/code-implementation/scripts/check_run_id.py \
  opencood/logs/train-percav-s42 --must-not-exist
```

训练入口会创建或复用输出目录并追加日志，因此检查失败时必须改用新 run ID；
恢复既有任务只能走第 5 节的显式 `--resume` 路径。

旧 one-epoch no-source/no-history 诊断属于早期 single-receiver 研究阶段，既不进入当前论文表格，也不能替代 fused/per-CAV 表示消融；其日志和 checkpoint 已清理。若未来重新研究 ordered history，必须从当前 Phase-1 checkpoint 按 receiver `[0,1]` 和 20-epoch 协议重新训练。

## 5. 同阶段断点恢复

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

## 6. 双卡 smoke

```bash
ulimit -n 65536
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 -m opencood.tools.train_colif \
  --hypes_yaml opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --phase 2 \
  --init-colif-checkpoint /path/to/phase1_checkpoint.pth \
  --output-dir opencood/logs/smoke-p2-percav-s20 \
  --epochs 1 --workers 2 --smoke-batches 5 --no-eval --log-every 1
```

只有 smoke 可用 `--smoke-force-birth`；任何 smoke/截断 test 数字都不能作为论文性能。

## 7. 独立 strict-online test 与 GT-free inference

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
