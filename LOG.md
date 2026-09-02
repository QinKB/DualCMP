# DualCMP Log

## 2026-08-31 — Causal data and tracker path established

- Finding/change: added `(scene, receiver)` T=3 data, tensor actor state,
  runtime/target separation, DDP trainer and strict-online evaluation.
- Evidence: `CP-20260831-01` `cc2a828`, `CP-20260831-02` `416c29f`,
  `CP-20260831-03` `5367a64`.
- Validation: real V2V4Real B=2/T=3 data and two-process smoke passed.
- Remaining: learned metric acceptance and evaluator-label alignment.

## 2026-09-01 — Evaluator label and recurrent tracker checkpoints

- Finding/change: aligned evaluator-native labels/coordinates, reused bundled
  AB3DMOT GIoU, DMSTrack tensor KF equations, Sparse4D-style state cache and
  JDT3D identity-ownership supervision.
- Evidence: `CP-20260901-01` `b2d3afe`, `CP-20260901-02` `43e96a5`,
  `CP-20260901-03` `d9fc612`; 55 DualCMP CPU tests passed.
- Experiments: `runs/train-track-kf-s20260831` reached AMOTA 0.5097, MOTA
  0.9662, recall 0.9757, precision 1.0 and IDS 9. The later identity-binding
  run reached AMOTA 0.4834, MOTA 0.9610, recall 0.9705, precision 1.0 and IDS 9.
- Limitation: the original frozen tracker gate did not pass. The user chose to
  preserve that failure and proceed using the source-backed V2V4Real baseline
  comparison; no threshold was rewritten.

## 2026-09-01 — Phase B reference-first plan fixed

- Finding/change: current code has actor/position/runtime ID but only one-frame
  history; `source_bev` has no forecast consumer. `PLAN.md` now fixes exact
  CoLIF/CMP/MTR symbols to copy and adapt before any writer assignment.
- Evidence: PaperMeld catalog `/home/qkb/Project/Reference/Thesis/library.jsonl`
  (325 records) and local JDT3D, Sparse4D v3, S2-Track, MOTR/UniAD source audit.
- Remaining: implement one integrated history → source-token → MTR → loss →
  trainer → online-metric vertical slice and pass two-GPU overfit acceptance.

## 2026-09-01 — Source-preserved forecast preflight checkpoint

- Change: copied/adapted CoLIF actor history, source tokenizer, MTR actor
  adapter/intention decoder and GMM loss; connected two-GPU training, strict
  fresh online inference and CMP metrics.
- Evidence: `CP-20260901-04` `5f84817`; final dirty preflight
  `forecast-smoke-c30aM9/run12`, checkpoint SHA256
  `10106b8217ad10fe41fad5b2e17bfa66ceb8de7ee8af8f82a5e54d0c503f10c7`.
- Validation: two GPU, B4/rank, W4, global batch 8, 8.69 GiB peak/rank,
  0.667 clips/s/rank, DDP sync; fresh evaluation assigned 94/94 complete
  targets and evaluated 92, excluding two active one-history-step newborns.
- Limitation: after two steps the model did not beat stationary. The 800-step
  clean-checkpoint overfit and tracking regression remain pending.

## 2026-09-01 — 800-step forecast gate partially passed

- Experiment: `runs/train-forecast-s20260901`, clean source `5f84817`, two
  GPUs, 800 steps, B4/rank, W4, global batch 8, DDP sync.
- Evidence: checkpoint SHA256
  `3a7b6f27c6d01202609605da6da67efd4993d0cb4c4fe3cac1aa071005ea5378`;
  final `L_gmm=68.1163`, `L_mode=0.8514`; fresh evaluation count 92/94.
- Result: model minADE was `0.877/4.439/8.148` m and minFDE was
  `1.783/10.462/17.413` m at 1/3/5 s, better than stationary and CV at every
  horizon. MR 3.6 was `0.6522`, equal to stationary, so the frozen performance
  gate failed and tracking regression was not run.
- Next: diagnose the long-horizon target/loss/decode/metric path against the
  exact local CoLIF/CMP/MTR references before modifying or retraining.

## 2026-09-01 — Restored fixed CMP heading protocol

- Root cause: implementation copied CoLIF's dirty working-tree radians adapter
  instead of fixed commit `6b8788e`. The existing CMP/CoLIF/DualCMP intention
  pkl SHA256 is `2f244fa6...6810` and expects degree-numeric MTR trigonometry.
- Change: `CP-20260901-05` `fd9189c` restores `rad2deg` at the MTR actor
  adapter, source-token positional frame and forecast-target rotation; tracking
  state and physical BEV sampling remain radians.
- Validation: two-GPU B4/W4 forward/loss/backward/checkpoint and fresh CMP eval
  passed. B8/W6 was 14.3% slower in clips/s; B4/W8 was stable and 0.59% faster
  than B4/W4, so the next long run uses B4/W8.
- Remaining: clean-SHA 800-step performance and tracking regression.

## 2026-09-01 — Official CMP/MTR semantic alignment

- Audit: the 22-D actor adapter matches official CMP within `5.96e-8`, and the
  six-layer intention CE/GMM loss matches exactly. Official `MTR/` files are
  byte-identical to local Reference CMP; the evidence hook now lives only in
  `opencood/models/dualcmp/mtr_vendor.py`.
- Root causes: official dense-future supervision was missing; evaluation mixed
  track-relative predictions with GT-current-relative targets instead of
  restoring both to one absolute coordinate frame.
- Validation: two-GPU no-hook decoder equality is exact on both ranks.
  `runs/official-align-smoke-s20260901` completed B4/W8 forward, main+dense
  loss, backward, optimizer, checkpoint and fresh evaluation; dense loss was
  finite/nonzero and forecast coverage remained 92/94.
- Remaining: frozen 800-step overfit, reasonable fresh CMP metrics and tracking
  regression.

## 2026-09-01 — Phase B accepted on clean source

- Diagnosis: official MTR, DualCMP actor tensors and the source-evidence hook
  each passed separate two-sample 800-step overfit gates. This ruled out a
  non-learning architecture or broken base target contract.
- Root causes fixed: official 30-epoch LR milestones were remapped to the
  800-step budget (`587/640/693/747`); GMM/mode/dense losses now use global DDP
  sum/count normalization; temporal attention is frozen and checked unchanged;
  forecast metrics record checkpoint provenance.
- Evidence: clean checkpoint `d2ee211`, run
  `runs/train-official-global-s20260901`, checkpoint SHA256
  `f9462989...eb710`, two GPUs, B4/rank, W8/rank, global batch 8, 800 steps.
- Forecast result: coverage 94 assigned/92 eligible; minADE
  `0.229/0.294/0.411` m, minFDE `0.185/0.379/0.982` m, MR `0.0109`, all better
  than stationary and CV.
- Tracking regression: MOTA `0.961`, recall `0.9705`, precision `1.0`, IDS `9`,
  fragments `24`, matching Phase A. The original tracker threshold gate remains
  `overall_pass=false`; forecast integration did not rewrite it.

## 2026-09-02 — Joint integration and formal epoch schedules

- Implemented current-frame Phase-C joint gradients, two-GPU tracking eval,
  full-data Phase-A/B entries and clip-consistent sequence augmentation copied
  from OpenCOOD/JDT3D semantics.
- Local-source schedules: Phase A JDT3D 20-epoch one-cycle; Phase B CMP/MTR
  30-epoch milestones; Phase C UniAD/UniV2X 20-epoch warmup-cosine.
- Full-data preflight passed: Phase A 14,018 clips, B4/W8, 1,752 steps/epoch,
  20.35 GiB/GPU; Phase B 12,930 clips, B4/W8, 1,616 steps/epoch, 6.26 GiB/GPU.
- Joint 100-step calibration rejected base LR `1e-4`; base `1e-5` improved
  tracking while retaining reasonable forecast metrics. Formal Phase C keeps
  that LR with 500-step warmup and cosine decay.

## 2026-09-02 — Phase-A global CSV and per-epoch test evaluation

- Stopped the first formal run after epoch 8 because the trainer produced only
  model checkpoints and no step/epoch logs or epoch-end test metrics.
- Reused QCNet's step/epoch logging semantics, OpenCOOD global sum/count,
  MTR resumable checkpoint pattern, and the existing DualCMP AB3DMOT gather.
- Two-GPU stop/resume acceptance completed 2 epochs/44 steps with 12 CSV rows,
  no duplicate keys, full optimizer/scheduler state, and test MOTA
  `0.9715/0.9778`. No loss, metric, or threshold was changed.

## 2026-09-02 — Paper-clean Phase-A initialization

- Stopped the full-tracker-initialized run after epoch 2: its starting MOTA
  came from a checkpoint trained on the same fixed test window and is not a
  paper-clean initialization.
- Formal Phase A uses only the official CMP V2V4Real train-split detector,
  compression 256, while all tracker parameters are random. The loader
  strict-loads 243 active keys and explicitly verifies the 16 unused official
  graph/post-graph keys.
- Two-GPU checks passed on the fixed window and one full-train step; the latter
  used B4/W8 and 20.34 GiB/GPU with a 35,040-step formal scheduler.
- Phase B and Phase C then strict-loaded compression 256 and completed two-GPU
  forward, loss, backward, DDP sync and checkpoint output. Phase C used
  15.00 GiB/GPU with nonzero forecast-to-source and detector gradients.
