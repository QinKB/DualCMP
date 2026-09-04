# DualCMP Memory

## Current

- Goal: build one end-to-end cooperative perception, lightweight tracking and
  motion-forecasting model around the task-asymmetric dual representation in
  `model_design.md`.
- The tracker owns only actor presence, receiver-local position, runtime ID and
  causal geometry history. Its latent feature is tracking-internal.
- Fused BEV owns detection/tracking/history; source-preserved BEVs own current
  per-CAV evidence used by MTR motion-intention queries.

## Knowledge

- Runtime samples are causal `(scene, receiver)` sequences; runtime payloads
  contain no GT IDs or future. Training/evaluation labels are target-only.
- V2V4Real tracking uses evaluator-native label text and projects receiver 1
  predictions to CAV0 only at the metric boundary.
- Forecasting must use 11 capture-native receiver-local history frames and 50
  target-only future frames to match the released CMP/MTR contract.
- CMP centers history and future labels on the predicted current track, while
  the dataset stores future displacement from GT current. Evaluation must add
  each displacement to its own center before computing errors.
- Keep Reference CMP `MTR/` files byte-identical; source-evidence integration
  belongs in the DualCMP-owned decoder adapter.
- Reuse sources are pinned locally: CMP/CoLIF `6b8788e`, Sparse4D `249ffbb`
  (MIT), JDT3D `b1a5d12` (CC BY-NC-SA), MOTR `8690da3` (MIT), and UniAD
  `609ee08` (Apache-2.0). CMP/OpenCOOD code is academic-research licensed;
  bundled MTR is Apache-2.0.

## Lessons

- Formal long runs require global `metrics.csv` rows every 100 updates, a
  global epoch train mean, and the authoritative test metric every epoch.
  Checkpoints must include optimizer/scheduler; model-only epoch checkpoints
  are recovery artifacts, not resumable formal training.
- A label oracle, syntax check, unit test or decreasing loss is not model
  acceptance. Every learned stage must reach real data, two-GPU training,
  checkpoint, fresh inference and authoritative metrics.
- `velocity_mask` excludes newborn placeholder zero velocity without a prior
  receiver frame; box/existence supervision remains active (two focused tests passed).
- Existing runtime tracks must retain loss-local GT ownership while valid;
  disappeared tracks are negatives and cannot be rebound as newborn actors.
- Do not tune or flatten tracking scores to game AMOTA. Preserve failed frozen
  protocols and record any later user-approved phase decision separately.

## Open

- A/B/C lifecycle acceptance passed two-GPU stop/resume with numeric CSV,
  continuous scheduler/step/sample state, no duplicate keys, and complete epoch
  checkpoints. Complete test evaluation covers 18 tracking timelines/3,986
  frames and 3,626 forecast samples.
- Paper Phase-A initialization is official CMP CoBEVT compression-256 detector
  only: 243 active keys strict-loaded, exactly 16 officially unused graph keys
  ignored, checkpoint SHA256 verified, tracker random. Never initialize a
  paper run from `train-track-bind-s20260831`, which saw the fixed test window.
- Detector compression is checkpoint provenance shared by Phase B, Phase C,
  both evaluators, and online inference; do not restore a fixed global value.
- Formal Phase B reuses CMP/MTR code, loss, intention queries and schedule, not
  V2V4Real-trained MTR weights. Forecast and source-evidence modules start from
  the fixed random seed; stale `DUALCMP_MTR_WEIGHTS` must fail closed.
- Formal schedules are source-backed and epoch-based: Phase A uses JDT3D's
  20-epoch T=3 one-cycle cosine; Phase B uses CMP/MTR's 30 epochs and
  22/24/26/28 milestones; Phase C uses UniAD/UniV2X's 20-epoch warmup-cosine.
- Full-data gates passed with Phase A 14,018 clips, B4/W8, 1,752 steps/epoch,
  20.35 GiB/GPU; and Phase B 12,930 clips, B4/W8, 1,616 steps/epoch,
  6.26 GiB/GPU. `--max-steps` no longer changes a full-data scheduler.
- Random Phase-B initialization learned on the frozen 16-sample gate in 800
  two-GPU steps: 5 s minADE `11.194 -> 2.051` m, minFDE
  `20.845 -> 6.568` m, and MR `0.8298 -> 0.6596`. This proves learning ability,
  not full-split paper quality.
- The next execution is a fresh formal full-data Phase A at
  `runs/train-a-cmp256-full-r2`, followed by Phase B and Phase C only after each
  preceding full-test gate is reasonable.

## Phase-B failure lessons

- Map official epoch milestones to the declared fixed-step budget; multiplying
  milestones by a tiny debug-loader length decayed LR after 56/800 steps and
  prevented overfit.
- DDP forecast losses require global actor/center sum-count normalization;
  equal averaging of rank-local means is wrong when counts differ.
- Calling a module `eval()` does not freeze it. Phase-B must set
  `requires_grad=False` and verify frozen temporal-attention state unchanged.
