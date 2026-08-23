# CoLIF Memory

## Current

- Repository: `/home/qkb/Project/CoLIF`.
- Active question: whether aligned per-CAV BEV patches improve forecasting over a protocol-matched fused-BEV actor patch.

## Knowledge

- Current architecture and tensor contracts are canonical in `model_design.md`; runnable training/evaluation commands are canonical in `docs/colif_training.md`.
- Main inference uses `actor_cooperative_fusion.representation=per_cav`. The `fused` ablation changes only forecast spatial evidence after fused tracking and has identical model/trainable parameter counts.
- Recoverable implementation checkpoint: `CP-20260823-01` / `739842a`. Existing per-CAV seed-20 checkpoint: `opencood/logs/dualcmp_core_receiver01_seed20_20e_20260821/checkpoint_epoch_019.pth`.

## Lessons

- `--disable-actor-cooperative-fusion` is a no-context diagnostic, not a fused-BEV baseline. Using it to claim fused information loss is invalid.
- Attention weights diagnose learned evidence allocation but do not establish that per-CAV features improve ADE/FDE.
- Forecast representation comparisons must keep receiver IDs `[0,1]`, Phase-1 initialization, epoch count, batch/accumulation, metric protocol, tracker outputs, and actor support counts fixed.
- New training output paths must not exist before launch because the trainer reuses directories and appends logs; use the explicit resume path for recovery.

## Open

- Unverified: seed-20 fused 20-epoch metrics and the direction of overall/partner-only ADE/FDE relative to per-CAV.
- If seed 20 supports per-CAV, repeat fused/per-CAV for seeds 42 and 2026; otherwise remove the Dual necessity claim.
