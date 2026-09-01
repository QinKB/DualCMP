# DualCMP

DualCMP is a staged cooperative perception, lightweight tracking, and motion
forecasting model for V2V4Real. Fused BEV owns detection, identity, and causal
history; source-preserved BEVs provide per-CAV evidence to the MTR forecast
head.

## Environment

Run from the repository root in the `coop` conda environment. All CUDA commands
below use both local GPUs. `--batch-size` and `--workers` are per GPU process.

```text
data: ../data/V2V4Real/{train,val,test}
test labels: ../Reference/Code/CMP/AB3Dmot/scripts/KITTI/v2v4real_label/test
```

Full-train supervision comes from target-only raw YAML. Evaluator-native
AB3DMOT labels are used only at the fixed test metric boundary.

## Formal epoch schedules

| Stage | Epochs | Batch/workers per GPU | Optimizer and schedule |
|---|---:|---:|---|
| Phase A, T=3 tracking | 20 | 4 / 8 | JDT3D-aligned AdamW; heads `5e-5→5e-4`, trunk `0.1×`; cosine rise for 8 epochs, decay for 12 |
| Phase B, forecast | 30 | 4 / 8 | official CMP/MTR AdamW `1e-4`; `×0.5` at epochs 22/24/26/28 |
| Phase C, joint | 20 | 6 / 8 | AdamW; detector/tracking/forecast `1e-6/1e-5/1e-5`; 500-step warmup + cosine |

`--max-steps` is for bounded preflight only, never the formal schedule.

## Phase A — full tracking

The accepted small-subset Phase-A checkpoint supplies the current 32×
compressor and tracker initialization. The only local standalone CMP detector
checkpoint is a structurally incompatible 256× model, so it is not loaded with
`strict=False`.

```bash
conda activate coop

CUDA_VISIBLE_DEVICES=0,1 \
V2V4REAL_ROOT=../data/V2V4Real \
DUALCMP_TRACKING_LABEL_ROOT=../Reference/Code/CMP/AB3Dmot/scripts/KITTI/v2v4real_label/test \
DUALCMP_PHASE_A_INIT_WEIGHTS=runs/train-track-bind-s20260831/checkpoint.pt \
python -m torch.distributed.run --nproc_per_node=2 \
  -m opencood.tools.dualcmp.train_tracking_ddp \
  --output-dir runs/phase-a-full-s20260902 \
  --full-data --data-split train --epochs 20 \
  --batch-size 4 --workers 8 \
  --lr 5e-5 --max-lr 5e-4 --weight-decay 0.01 \
  --log-every 100 --eval-every 1 --save-every 1
```

`metrics.csv` receives globally synchronized train rows every 100 updates and
one `train+test` row per epoch. Test columns are explicitly prefixed `test_`;
each epoch also writes `test_metrics_epoch_XXX.json`. Checkpoints contain model,
optimizer, scheduler, completed epoch, global step, and provenance for exact
same-budget resume.

## Phase B — full forecast training

Start only after Phase A finishes and passes fresh evaluation.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
V2V4REAL_ROOT=../data/V2V4Real \
DUALCMP_TRACKING_LABEL_ROOT=../Reference/Code/CMP/AB3Dmot/scripts/KITTI/v2v4real_label/test \
DUALCMP_TRACKER_WEIGHTS=runs/phase-a-full-s20260902/checkpoint.pt \
DUALCMP_MTR_WEIGHTS=../data/V2V4Real/derived/CMP/runs/mtr_noagg_epoch2_test_20260807/checkpoint_epoch_5_snapshot.pth \
python -m torch.distributed.run --nproc_per_node=2 \
  -m opencood.tools.dualcmp.train_forecast_ddp \
  --output-dir runs/phase-b-full-s20260902 \
  --full-data --data-split train --epochs 30 \
  --batch-size 4 --workers 8 --lr 1e-4 --save-every 5
```

## Phase C — full joint fine-tuning

The first ten causal frames remain graph-free; the current frame jointly
updates detection, tracking, source representation, and forecast modules.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
V2V4REAL_ROOT=../data/V2V4Real \
DUALCMP_INIT_WEIGHTS=runs/phase-b-full-s20260902/checkpoint.pt \
env -u DUALCMP_TRACKING_LABEL_ROOT \
python -m torch.distributed.run --nproc_per_node=2 \
  -m opencood.tools.dualcmp.train_joint_ddp \
  --output-dir runs/phase-c-full-s20260902 \
  --full-data --data-split train --epochs 20 \
  --batch-size 6 --workers 8 \
  --base-lr 1e-5 --detector-lr-scale 0.1 \
  --forecast-lr-scale 1 --weight-decay 0.01 \
  --lambda-det 1 --lambda-track 1 --lambda-pred 1e-4 \
  --warmup-steps 500 --min-lr-ratio 1e-3 --save-every 2
```

## Fixed two-GPU evaluation

```bash
export V2V4REAL_ROOT=../data/V2V4Real
export DUALCMP_TRACKING_LABEL_ROOT=../Reference/Code/CMP/AB3Dmot/scripts/KITTI/v2v4real_label/test

python -m torch.distributed.run --nproc_per_node=2 \
  -m opencood.tools.dualcmp.eval_forecast_ddp \
  --checkpoint runs/phase-c-full-s20260902/checkpoint.pt \
  --output runs/phase-c-full-s20260902/forecast_metrics.json

python -m torch.distributed.run --nproc_per_node=2 \
  -m opencood.tools.dualcmp.eval_tracking_ddp \
  --checkpoint runs/phase-c-full-s20260902/checkpoint.pt \
  --output runs/phase-c-full-s20260902/tracking_metrics.json
```

Forecast uses the existing CMP 1/3/5-second minADE, minFDE, and MR. Tracking
uses the existing AB3DMOT evaluator. DualCMP does not redefine either metric.
