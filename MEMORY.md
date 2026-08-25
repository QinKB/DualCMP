# CoLIF Memory

## Current

- Repository: `/home/qkb/Project/CoLIF`.
- Active question: how source preservation, query weighting, relative geometry, and motion anchors contribute under one fixed online protocol.

## Knowledge

- Current architecture and tensor contracts are canonical in `model_design.md`; runnable training/evaluation commands are canonical in `docs/colif_training.md`.
- Main inference uses `actor_cooperative_fusion.representation=per_cav`. The `fused` ablation changes only forecast spatial evidence after fused tracking and has identical model/trainable parameter counts.
- Controlled source ablations use `aggregation={attention,mean}` and `geometry_conditioning={true,false}`; the motion-anchor ablation uses `trajectory_parameterization={anchored,direct}` with an identical parameter schema.
- Recoverable implementation checkpoint: `CP-20260825-01` / `14cc0a9`. Existing per-CAV seed-20 checkpoint: `opencood/logs/dualcmp_core_receiver01_seed20_20e_20260821/checkpoint_epoch_019.pth`.

## Lessons

- `--disable-actor-cooperative-fusion` is a no-context diagnostic, not a fused-BEV baseline. Using it to claim fused information loss is invalid.
- Attention weights diagnose learned evidence allocation but do not establish that per-CAV features improve ADE/FDE.
- Forecast representation comparisons must keep receiver IDs `[0,1]`, Phase-1 initialization, epoch count, batch/accumulation, metric protocol, tracker outputs, and actor support counts fixed.
- New experiment IDs follow `<stage>-<variant>[-sN][-rN]`. Validate a proposed path with `check_run_id.py`, repeat with `--must-not-exist` before launch because the trainer appends to existing directories, and use explicit resume without renaming for an existing run.

## Open

- Unverified: uniform mean, query attention without geometry, and direct trajectory 20-epoch results.
