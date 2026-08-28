# CoLIF Memory

## Current

- Repository: `/home/qkb/Project/CoLIF`.
- Active question: validate and train the new dual-representation MTR intention-query forecaster without reusing obsolete motion-anchor results.

## Knowledge

- Current architecture and tensor contracts are canonical in `model_design.md`; runnable training/evaluation commands are canonical in `docs/colif_training.md`.
- Main inference uses `actor_cooperative_fusion.representation=dual`: fused consensus actor RoIs and per-CAV source actor RoIs are both read by the same 64 MTR intention queries.
- Controlled representation ablations are `dual`, `fused`, and `source`; they share the tokenizer, MTR decoder, dual attention parameters, tracker, loss and parameter schema.
- Training keeps raw 64 queries at all six decoder layers, assigns the positive query from fixed endpoint intentions, and applies endpoint NMS to expose six public trajectories.
- Last pre-refactor checkpoint is `CP-20260827-04` / `cfc5a16`; the current intention-query refactor has no Git checkpoint until review and runtime validation finish.

## Lessons

- `--disable-actor-cooperative-fusion` is a no-context diagnostic, not a fused-BEV baseline. Using it to claim fused information loss is invalid.
- Attention weights diagnose learned evidence allocation but do not establish that per-CAV features improve ADE/FDE.
- Forecast representation comparisons must keep receiver IDs `[0,1]`, Phase-1 initialization, epoch count, batch/accumulation, metric protocol, tracker outputs, and actor support counts fixed.
- Old velocity-anchor, direct-regression, flat-reader and source-refiner checkpoints/results do not validate the current model and cannot be resumed into it.
- Phase 2 requires both the fixed Phase-1 checkpoint and the bundled CMP/MTR decoder checkpoint; cross-phase loading deliberately preserves newly initialized history/tokenizer/dual-attention parameters.
- New experiment IDs follow `<stage>-<variant>[-sN][-rN]`. Validate a proposed path with `check_run_id.py`, repeat with `--must-not-exist` before launch because the trainer appends to existing directories, and use explicit resume without renaming for an existing run.

## Open

- Unverified: whether `dual` outperforms both `source` and `fused` under a matched formal run.
- Pending: real dual-GPU smoke after the currently running legacy training releases both GPUs.
