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
AB3DMOT labels are used only at the complete test metric boundary.

## Formal epoch schedules

| Stage | Epochs | Batch/workers per GPU | Optimizer and schedule |
|---|---:|---:|---|
| Phase A, T=3 tracking | 20 | 4 / 8 | JDT3D-aligned AdamW; heads `5e-5→5e-4`, trunk `0.1×`; cosine rise for 8 epochs, decay for 12 |
| Phase B, forecast | 30 | 4 / 8 | official CMP/MTR AdamW `1e-4`; `×0.5` at epochs 22/24/26/28 |
| Phase C, joint | 20 | 6 / 8 | AdamW; detector/tracking/forecast `1e-6/1e-5/1e-5`; 500-step warmup + cosine |

`--max-steps` is for bounded preflight only, never the formal schedule.

## Phase A — full tracking

Phase A loads the official V2V4Real CMP CoBEVT detector trained on the train
split and initializes every tracker parameter randomly. Its 256× compressor is
kept unchanged. The checkpoint contains 16 graph/post-graph parameters that
are disabled in the official forward; the loader requires exactly those 16
unused keys, then strict-loads all 243 active detector keys.

```bash
conda activate coop

CUDA_VISIBLE_DEVICES=0,1 \
V2V4REAL_ROOT=../data/V2V4Real \
DUALCMP_TRACKING_LABEL_ROOT=../Reference/Code/CMP/AB3Dmot/scripts/KITTI/v2v4real_label/test \
DUALCMP_DETECTOR_WEIGHTS=pretrained/v2v4real/point_pillar_cobevt_multiego_256x/net_epoch75.pth \
DUALCMP_CODE_SHA=$(git rev-parse HEAD) \
env -u DUALCMP_PHASE_A_INIT_WEIGHTS \
python -m torch.distributed.run --nproc_per_node=2 \
  -m opencood.tools.dualcmp.train_tracking_ddp \
  --output-dir runs/train-a-cmp256-full-r2 \
  --full-data --data-split train --epochs 20 \
  --detector-compression 256 \
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
DUALCMP_TRACKER_WEIGHTS=runs/train-a-cmp256-full-r2/checkpoint.pt \
DUALCMP_CODE_SHA=$(git rev-parse HEAD) \
env -u DUALCMP_MTR_WEIGHTS \
python -m torch.distributed.run --nproc_per_node=2 \
  -m opencood.tools.dualcmp.train_forecast_ddp \
  --output-dir runs/train-b-full \
  --full-data --data-split train --epochs 30 \
  --batch-size 4 --workers 8 --lr 1e-4 --log-every 100 --eval-every 1 --save-every 1
```

Phase B copies the official CMP/MTR architecture, loss, intention queries and
30-epoch schedule, but initializes every forecast and source-evidence parameter
randomly. It never loads a V2V4Real-trained MTR checkpoint.

All stages write globally synchronized numeric CSV columns every 100 optimizer
steps and one global train row per epoch. Phase A adds complete-test AB3DMOT
columns, Phase B adds complete-test CMP minADE/minFDE/MR columns, and Phase C
adds both. Every epoch atomically saves model, optimizer, scheduler, epoch,
global step, sample count, and provenance before evaluation.

## Phase C — full joint fine-tuning

The first ten causal frames remain graph-free; the current frame jointly
updates detection, tracking, source representation, and forecast modules.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
V2V4REAL_ROOT=../data/V2V4Real \
DUALCMP_INIT_WEIGHTS=runs/train-b-full/checkpoint.pt \
DUALCMP_TRACKING_LABEL_ROOT=../Reference/Code/CMP/AB3Dmot/scripts/KITTI/v2v4real_label/test \
DUALCMP_CODE_SHA=$(git rev-parse HEAD) \
python -m torch.distributed.run --nproc_per_node=2 \
  -m opencood.tools.dualcmp.train_joint_ddp \
  --output-dir runs/train-c-full \
  --full-data --data-split train --epochs 20 \
  --batch-size 6 --workers 8 \
  --base-lr 1e-5 --detector-lr-scale 0.1 \
  --forecast-lr-scale 1 --weight-decay 0.01 \
  --lambda-det 1 --lambda-track 1 --lambda-pred 1e-4 \
  --warmup-steps 500 --min-lr-ratio 1e-3 --log-every 100 --eval-every 1 --save-every 1
```

## Complete two-GPU test evaluation

```bash
export V2V4REAL_ROOT=../data/V2V4Real
export DUALCMP_TRACKING_LABEL_ROOT=../Reference/Code/CMP/AB3Dmot/scripts/KITTI/v2v4real_label/test

python -m torch.distributed.run --nproc_per_node=2 \
  -m opencood.tools.dualcmp.eval_forecast_ddp \
  --checkpoint runs/train-c-full/checkpoint.pt \
  --output runs/train-c-full/forecast_metrics.json \
  --full-test --batch-size 4 --workers 8

python -m torch.distributed.run --nproc_per_node=2 \
  -m opencood.tools.dualcmp.eval_tracking_ddp \
  --checkpoint runs/train-c-full/checkpoint.pt \
  --output runs/train-c-full/tracking_metrics.json \
  --full-test
```

Forecast uses the existing CMP 1/3/5-second minADE, minFDE, and MR. Tracking
uses the existing AB3DMOT evaluator. DualCMP does not redefine either metric.
