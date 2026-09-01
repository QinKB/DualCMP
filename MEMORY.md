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
- Existing runtime tracks must retain loss-local GT ownership while valid;
  disappeared tracks are negatives and cannot be rebound as newborn actors.
- Do not tune or flatten tracking scores to game AMOTA. Preserve failed frozen
  protocols and record any later user-approved phase decision separately.

## Open

- Phase-A CSV/test/resume integration passed a two-GPU two-epoch fixed-window
  run: epoch MOTA `0.9715/0.9778`, no duplicate CSV keys, scheduler budget
  preserved at 44 steps. Restart the formal 20-epoch run from initialization.
- Formal schedules are source-backed and epoch-based: Phase A uses JDT3D's
  20-epoch T=3 one-cycle cosine; Phase B uses CMP/MTR's 30 epochs and
  22/24/26/28 milestones; Phase C uses UniAD/UniV2X's 20-epoch warmup-cosine.
- Full-data gates passed with Phase A 14,018 clips, B4/W8, 1,752 steps/epoch,
  20.35 GiB/GPU; and Phase B 12,930 clips, B4/W8, 1,616 steps/epoch,
  6.26 GiB/GPU. `--max-steps` no longer changes a full-data scheduler.
- Phase B is complete at clean source `d2ee211`; canonical artifacts are under
  `runs/train-official-global-s20260901`. Coverage is 94 assigned/92 eligible;
  the two one-history-step newborns remain explicitly uncovered.
- Final forecast metrics are minADE `0.229/0.294/0.411` m, minFDE
  `0.185/0.379/0.982` m and MR `0.0109`. Tracking regression is unchanged on
  MOTA/recall/precision/IDS/fragments, while its older independent
  `overall_pass=false` remains truthful.
- The next open work is Phase C joint fine-tuning, not more Phase-B tuning.
- The next execution is formal full-data Phase A, followed by Phase B; Phase C
  follows only after both fresh evaluators pass.

## Phase-B failure lessons

- Map official epoch milestones to the declared fixed-step budget; multiplying
  milestones by a tiny debug-loader length decayed LR after 56/800 steps and
  prevented overfit.
- DDP forecast losses require global actor/center sum-count normalization;
  equal averaging of rank-local means is wrong when counts differ.
- Calling a module `eval()` does not freeze it. Phase-B must set
  `requires_grad=False` and verify frozen temporal-attention state unchanged.
