# HANDOFF

## Current

- Phase-C joint integration is implemented and full-data launch schedules are
  frozen from local upstream configs: Phase A 20 epochs, Phase B 30 epochs,
  Phase C 20 epochs. Formal Phase A is the next running job; Phase B starts
  only after its fresh tracking evaluation.
- Phase B is functionally accepted at clean commit `d2ee211`: official MTR
  initialization, source-preserved evidence, main+dense loss, global DDP
  normalization, frozen tracker, fresh forecast metrics and tracking
  regression all passed on the fixed small subset.
- Final artifact is `runs/train-official-global-s20260901/checkpoint.pt`, SHA256
  `f9462989...eb710`, bound to clean source and evaluator provenance.
- Next implementation phase is Phase C joint fine-tuning. Do not reopen Phase B
  architecture or thresholds without new contradictory evidence.

## Verified

- Current checkpoint: `CP-20260901-04`, commit
  `5f84817ef98ea3bd57cf96293ee99358e8e6c507`.
- Tracker online path provides current actors, receiver-local boxes and runtime
  IDs plus an 11-frame causal capture-native forecast history.
- `PointPillarFax` returns `source_bev/source_mask/fused_bev`; source BEV is
  consumed by the actor-source tokenizer while fused BEV remains the
  detection/tracking representation. Tracker latent is not a forecast input.
- `runs/train-track-bind-s20260831` is provenance-bound to the checkpoint and
  completed two-GPU training plus CPU 2x24 evaluation; `overall_pass=false`.
- Final dirty-tree preflight `forecast-smoke-c30aM9/run12` completed two-GPU
  forward/loss/backward/optimizer/checkpoint/fresh evaluation. It used B4 per
  rank, W4, 8.69 GiB peak/rank and 0.667 clips/s/rank with DDP sync.
- The frozen target set has 94 complete futures; all 94 are assigned, 92 are
  forecast-eligible. The two excluded active newborns each have one history
  step, below copied MTR's two-step minimum.
- `runs/train-forecast-s20260901` is bound to clean source `5f84817`, completed
  800 steps with DDP sync, and produced checkpoint SHA256 `3a7b6f27...5378`.
  Model ADE/FDE beats stationary and CV at 1/3/5 s, but MR 3.6 equals
  stationary at 0.6522 and 5 s minFDE remains 17.413 m.
- The V2V4Real intention pkl in DualCMP, CoLIF and Reference CMP has identical
  SHA256 `2f244fa6...6810` and was generated under CMP's legacy degree-numeric
  heading contract. The earlier radians adapter came from CoLIF's dirty working
  tree, not fixed commit `6b8788e`.
- Synthetic 22-D adapter comparison against official CMP has max absolute error
  `5.96e-8`; the six-layer intention CE/GMM comparison is exactly zero.
- On both GPUs, the DualCMP decoder with no source hook matches official CMP
  scores, trajectories and dense trajectories with max absolute error zero.
- `runs/official-align-smoke-s20260901` completed two-GPU B4/W8
  forward/main+dense loss/backward/optimizer/checkpoint/fresh inference;
  `L_dense_gmm=8426.87`, count=92 and all metrics are finite.
- Two-sample discriminators established causality: official MTR overfits
  official cache; official MTR overfits DualCMP actor tensors; and the source
  hook overfits DualCMP source tokens (MR `0.6 -> 0.2`). Thus the architecture
  and data contract are learnable.
- The failed 16-sample runs were caused by an incorrectly compressed official
  LR schedule, then exposed two integration bugs: rank-local forecast-loss
  normalization and an unfrozen temporal-attention module.
- Final clean two-GPU run: global batch 8, B4/W8, 800 steps, LR milestones
  `[587,640,693,747]`, unequal rank counts handled globally, temporal attention
  unchanged, 94 assigned/92 eligible trajectories. Model minADE was
  `0.229/0.294/0.411` m, minFDE `0.185/0.379/0.982` m, and MR `0.0109`.
- Tracking regression kept MOTA `0.961`, recall `0.9705`, precision `1.0`, IDS
  `9`, and fragments `24`; the historical tracker `overall_pass=false` remains.

## Open

- Run full Phase A on 14,018 T=3 clips with B4/GPU, W8/GPU and the JDT3D
  20-epoch one-cycle schedule; then evaluate and run full Phase B for 30 epochs.

## Risks

- The only standalone local CMP detector checkpoint is a structurally
  incompatible 256x model. Formal Phase A therefore resumes the accepted 32x
  Phase-A checkpoint rather than using a partial or non-strict load.
- Current one-token tracker attention is not the full Sparse4D temporal
  attention and must not be described as such.
- Learned-box second KF update is a project adaptation, not a verbatim
  DMSTrack path; the saved ablation improved recall but worsened IDS, so retain
  it for Phase B initialization but recheck tracking metrics after integration.
- Forecast history and target coordinates must follow CMP's moving-receiver
  capture-time contract, while KF propagation remains current-receiver state.
- JDT3D/CMP/OpenCOOD/AB3DMOT components carry noncommercial academic license
  constraints; preserve provenance and intended research use.

## Checkpoints

| ID | Commit | Outcome | Validation | Next |
|---|---|---|---|---|
| CP-20260901-01 | `b2d3afe` | evaluator labels/coordinates | label oracle and real T=3 | recurrent tracker |
| CP-20260901-02 | `43e96a5` | recurrent KF tracker preflight | 53+1 CPU checks; two-GPU run | identity supervision |
| CP-20260901-03 | `d9fc612` | persistent identity supervision | 55 CPU tests; two-GPU run/eval | Phase B vertical slice |
| CP-20260901-04 | `5f84817` | source-preserved forecast preflight | two-GPU run12 + fresh CMP metrics | 800-step overfit |
| CP-20260901-05 | `fd9189c` | restore fixed CoLIF/CMP heading boundary | two-GPU smoke + fresh CMP eval | 800-step result |
| CP-20260901-06 | `d2ee211` | official-aligned Phase-B code | global-loss/freeze smoke + clean 800-step gate | Phase C |
| CP-20260902-01 | pending | full-data Phase A/B and Phase-C joint integration | two-GPU epoch-schedule preflights | Phase A 20 epochs |
