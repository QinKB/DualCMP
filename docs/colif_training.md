# DualCMP Training and Evaluation

本文件是当前结构唯一可执行命令来源。正式模型为`dual`：fused consensus actor RoI与per-CAV source actor RoIs同时进入MTR 64 intention queries。旧`per_cav`、motion-anchor、direct regression、source-refiner和旧MTR adapter命令均已失效。

## 1. Fixed assets

```text
Phase-1 tracker:
opencood/logs/colif_paper_phase1_20e_seed20/checkpoint_epoch_009.pth

CMP/MTR decoder:
MTR/output/v2v4real_multiego_cobevt_c256/ckpt/best_model.pth

64 endpoint intentions:
preprocessed_data/v2v4real/v2v4real_cluster_64_center_dict.pkl

Formal config:
opencood/hypes_yaml/colif/colif_v2v4real.yaml
```

Receiver协议固定为`0,1`。正式test使用`online_masked`，每个run只评测固定末轮checkpoint，不按test选择epoch。

## 2. Preflight

```bash
cd /home/qkb/Project/CoLIF || exit 1
ulimit -n 65536

PYTHONPATH=/home/qkb/Project/CoLIF \
  conda run --no-capture-output -n coop \
  python -m pytest -q tests/colif
```

新run启动前必须检查目录不存在：

```bash
python /home/qkb/.agents/skills/code-implementation/scripts/check_run_id.py \
  /home/qkb/Project/CoLIF/opencood/logs/<run-id> --must-not-exist
```

## 3. Phase 1 tracking

当前论文实验复用上面的固定Phase-1 checkpoint。只有需要重训tracker时才运行：

```bash
cd /home/qkb/Project/CoLIF || exit 1
ulimit -n 65536
PYTHONPATH=/home/qkb/Project/CoLIF CUDA_VISIBLE_DEVICES=0,1 \
  conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 -m opencood.tools.train_colif \
  --hypes_yaml /home/qkb/Project/CoLIF/opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --phase 1 --batch-size 2 --workers 2 --eval-workers 2 \
  --init-cobevt-checkpoint \
  /home/qkb/Project/CoLIF/pretrained/v2v4real/point_pillar_cobevt_multiego_256x/net_epoch75.pth \
  --output-dir /home/qkb/Project/CoLIF/opencood/logs/train-p1-s20 \
  --epochs 20 --receiver-ids 0,1 --seed 20 --log-every 50
```

## 4. Phase 2 formal DualCMP

MTR V2V4Real原配置每GPU batch为1。当前新decoder在完成GPU smoke前也使用保守的每卡batch 1、梯度累积4；双卡effective global batch为8。

```bash
cd /home/qkb/Project/CoLIF || exit 1
ulimit -n 65536
PYTHONPATH=/home/qkb/Project/CoLIF CUDA_VISIBLE_DEVICES=0,1 \
  conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 -m opencood.tools.train_colif \
  --hypes_yaml /home/qkb/Project/CoLIF/opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --phase 2 \
  --init-colif-checkpoint \
  /home/qkb/Project/CoLIF/opencood/logs/colif_paper_phase1_20e_seed20/checkpoint_epoch_009.pth \
  --init-mtr-checkpoint \
  /home/qkb/Project/CoLIF/MTR/output/v2v4real_multiego_cobevt_c256/ckpt/best_model.pth \
  --output-dir /home/qkb/Project/CoLIF/opencood/logs/train-dual-s20 \
  --epochs 20 --batch-size 1 --grad-accum-steps 4 --workers 2 \
  --receiver-ids 0,1 --forecast-bev-representation dual --seed 20 \
  --no-eval --log-every 50
```

不得省略`--init-mtr-checkpoint`：Phase-1 checkpoint故意不导入其未训练的forecast参数。

## 5. Required representation ablations

只做三个同协议实验：

| run | flag | purpose |
| --- | --- | --- |
| `train-dual-s20` | `dual` | fused consensus + source evidence |
| `train-fused-s20` | `fused` | fused consensus only |
| `train-source-s20` | `source` | per-CAV source evidence only |

三者使用同一Phase-1 checkpoint、MTR checkpoint、seed、epoch、batch、optimizer、receiver和test population。运行`fused`或`source`时，只替换主命令中的两项：

```text
--output-dir /home/qkb/Project/CoLIF/opencood/logs/train-fused-s20
--forecast-bev-representation fused

--output-dir /home/qkb/Project/CoLIF/opencood/logs/train-source-s20
--forecast-bev-representation source
```

`--disable-actor-cooperative-fusion`是no-evidence诊断，不是论文中的fused baseline。

## 6. Smoke

GPU空闲后，先用独立目录运行1 batch；成功标准为真实dataset forward/loss/backward/optimizer均完成，64 queries、6 layers、两路tokenizer均有梯度，无AMP overflow、NaN、unused trainable或DDP checksum差异。

```bash
cd /home/qkb/Project/CoLIF || exit 1
ulimit -n 65536
PYTHONPATH=/home/qkb/Project/CoLIF CUDA_VISIBLE_DEVICES=0,1 \
  conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 -m opencood.tools.train_colif \
  --hypes_yaml /home/qkb/Project/CoLIF/opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --phase 2 \
  --init-colif-checkpoint \
  /home/qkb/Project/CoLIF/opencood/logs/colif_paper_phase1_20e_seed20/checkpoint_epoch_009.pth \
  --init-mtr-checkpoint \
  /home/qkb/Project/CoLIF/MTR/output/v2v4real_multiego_cobevt_c256/ckpt/best_model.pth \
  --output-dir /home/qkb/Project/CoLIF/opencood/logs/smoke-dual-s20 \
  --epochs 1 --smoke-batches 1 \
  --batch-size 1 --grad-accum-steps 1 --workers 1 \
  --receiver-ids 0,1 --forecast-bev-representation dual --seed 20 \
  --no-eval --log-every 1
```

Smoke目录不能作为性能结果或正式resume起点。

## 7. Resume

只允许完全相同结构和config hash：

```bash
--resume /home/qkb/Project/CoLIF/opencood/logs/<run-id>/checkpoint_epoch_NNN.pth
```

`--resume`不能与任何`--init-*-checkpoint`同时使用。旧motion-anchor/flat-reader checkpoint禁止resume到当前模型。

## 8. Strict-online test

```bash
cd /home/qkb/Project/CoLIF || exit 1
PYTHONPATH=/home/qkb/Project/CoLIF CUDA_VISIBLE_DEVICES=0,1 \
  conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 \
  -m opencood.tools.eval_colif_v2v4real \
  --hypes_yaml /home/qkb/Project/CoLIF/opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --checkpoint /home/qkb/Project/CoLIF/opencood/logs/<run-id>/checkpoint_epoch_019.pth \
  --split test --receiver-ids 0,1 --workers 2 \
  --forecast-metric-protocol online_masked \
  --output /home/qkb/Project/CoLIF/opencood/logs/<run-id>/metrics_test_online_masked.json
```

报告必须同时包含1/3/5 s ADE、FDE、MR、coverage、matched actor frames及receiver-only/partner-only/shared分桶。Attention只用于解释不同intention读取了哪些表示，不能代替三组受控实验的因果证据。
