# CoLIF 当前训练、恢复与评测

从 `/home/qkb/Project/CoLIF` 执行，环境为 `coop`。当前正式模型只有两个训练阶段，没有 Phase 3：

```text
Phase 1: CMP fused-BEV persistent-query tracking（forecast关闭）
Phase 2: 冻结感知/跟踪，训练cooperative motion forecast
```

正式协议为 official `train` 优化、固定末轮自动评测 official `test`；不构造 val、不按 test 指标自动选择 best。论文必须披露 repeated-test monitoring。

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
  --output-dir opencood/logs/colif_fused_phase1_release \
  --epochs 2 --workers 2 --eval-workers 2 --log-every 50
```

Phase 1 使用 FP32。当前四组公平预测实验共同使用的冻结 tracker 为：

```text
opencood/logs/colif_fused_v7_phase1_predict_correct_2epoch_20260812/checkpoint_epoch_001.pth
```

该 checkpoint 为 epoch 1 固定末轮，不是按 test 挑选。

## 3. Phase 2A：默认整体最佳分支

默认配置为 geometry-conditioned actor-source fusion + ordered history + training-only cross-mode auxiliary，推理输出 proposal layer：

```bash
ulimit -n 65536
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 -m opencood.tools.train_colif \
  --hypes_yaml opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --phase 2 \
  --init-colif-checkpoint \
  opencood/logs/colif_fused_v7_phase1_predict_correct_2epoch_20260812/checkpoint_epoch_001.pth \
  --output-dir opencood/logs/colif_actor_source_release \
  --epochs 1 --workers 2 --eval-workers 2 --log-every 50
```

已完成的严格结果/权重：

```text
opencood/logs/colif_fused_v19_geometry_source_seedfixed_full_1epoch_20260813/
ADE@1/3/5s = 0.9997 / 1.8472 / 2.5870 m
FDE@1/3/5s = 1.3708 / 2.9177 / 4.3804 m
MOTA=0.6333, forecast coverage=0.7894
```

## 4. Phase 2B：可选 trajectory-source 完整分支

该分支保留 `K=6 × T=50` trajectory-state tokens，沿每条 proposal 在 per-source BEV 上重新采样，并部署 refined layer：

```bash
ulimit -n 65536
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 -m opencood.tools.train_colif \
  --hypes_yaml opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --phase 2 \
  --forecast-refinement-type trajectory_source \
  --forecast-inference-layer refined \
  --init-colif-checkpoint \
  opencood/logs/colif_fused_v7_phase1_predict_correct_2epoch_20260812/checkpoint_epoch_001.pth \
  --output-dir opencood/logs/colif_trajectory_source_release \
  --epochs 1 --workers 2 --eval-workers 2 --log-every 50
```

已完成的严格结果/权重：

```text
opencood/logs/colif_fused_v21_identity_trajectory_source_seedfixed_full_restart_1epoch_20260813/
ADE@1/3/5s = 1.0728 / 1.9512 / 2.6714 m
FDE@1/3/5s = 1.4703 / 2.9896 / 4.3836 m
MOTA=0.6333, forecast coverage=0.7894
```

它没有超过默认分支的总体指标，但 partner-only FDE5 为 6.4371 m（默认 6.4486），age 2--3 FDE5 为 12.4346 m（默认 12.4862）。因此代码与命令保留，论文中应作为可选困难目标分支/消融，不写成总体 superiority。

## 5. 公平消融命令

关闭 actor-source fusion：

```bash
# 在Phase 2A命令基础上添加
--disable-actor-cooperative-fusion
```

关闭 ordered temporal history，但保留 robust velocity/dispersion 与 actor-source fusion：

```bash
# 在Phase 2A命令基础上添加
--disable-temporal-history
```

对应完整结果：

```text
no source:
  opencood/logs/colif_fused_v18_no_actor_source_seedfixed_full_1epoch_20260813/
  ADE5/FDE5=2.7437/4.3980 m

no ordered history:
  opencood/logs/colif_fused_v22_no_temporal_history_seedfixed_full_1epoch_20260813/
  ADE5/FDE5=2.6535/4.3827 m
```

训练器使用独立 `torch.Generator(seed+rank)` 和 worker seed，数据随机性不再受模型参数初始化顺序影响。四组的 MOTA、coverage、actor counts 完全相同，证明预测消融没有改变冻结 tracker。

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

不能用默认分支 checkpoint 恢复 trajectory-source 结构，反之亦然；配置 hash 不一致会拒绝 resume。跨阶段只能使用 `--init-colif-checkpoint`。

## 7. 双卡 smoke

```bash
ulimit -n 65536
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 -m opencood.tools.train_colif \
  --hypes_yaml opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --phase 2 \
  --init-colif-checkpoint /path/to/phase1_checkpoint.pth \
  --output-dir opencood/logs/colif_smoke \
  --epochs 1 --workers 2 --smoke-batches 5 --no-eval --log-every 1
```

trajectory-source smoke 添加第 4 节的两个 forecast flags。只有 smoke 可用 `--smoke-force-birth`；任何 smoke/截断 test 数字都不能作为论文性能。

## 8. 独立 strict-online test 与 GT-free inference

评测 checkpoint（checkpoint 内配置优先，trajectory-source 无需重复 CLI override）：

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
