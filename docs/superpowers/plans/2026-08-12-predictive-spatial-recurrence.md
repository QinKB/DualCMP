# Predictive Spatial Recurrence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the previous short-horizon multimodal forecast participate in current-frame proposal association and missing-state propagation while preserving strict-online CMP fused-BEV inference and exact same-checkpoint ablations.

**Architecture:** A stateless `ForecastMotionPrior` converts the short prefix of the preceding forecast into current-receiver mixture centers and variances, adds constant velocity as a fallback component, and exposes both a MAP reference and mixture association cost. `PointPillarColif` consumes this result before Hungarian assignment and stores only the next five forecast steps in runtime state.

**Tech Stack:** Python 3.10, PyTorch, SciPy Hungarian assignment, pytest, existing OpenCOOD/CMP CoBEVT, torchrun DDP.

## Global Constraints

- Runtime forward must not receive GT boxes, actor IDs, associations, or future targets.
- Main V2V4Real protocol remains synchronous 10 Hz and receiver 0.
- Phase 1 must be numerically unchanged with spatial-prior mode `none`.
- The previous latent `FutureAdapter` is not used by the new main path.
- Evaluation modes must not modify targets, evaluator matching, or metric denominators.
- Existing uncommitted workspace changes and logs must be preserved.

---

### Task 1: Forecast spatial-prior component

**Files:**
- Create: `opencood/models/colif/forecast_motion_prior.py`
- Modify: `opencood/models/colif/__init__.py`
- Test: `tests/colif/test_forecast_motion_prior.py`

**Interfaces:**
- Consumes: CV boxes/velocities, stored mode probabilities/relative means/variances, prior/current poses and timestamps.
- Produces: `ForecastPriorResult(predicted_box, predicted_velocity, mode_centers, mode_variances, mode_probabilities, valid_mask, selected_forecast_mask, step_index)` and `pair_cost(...)`.

- [ ] Write tests proving invalid forecasts equal CV, uniform forecasts select CV, confident modes select their spatial center, pose conversion is correct, association uses a helpful alternative mode, and recurrent/detached modes have the intended gradients.
- [ ] Run `pytest -q tests/colif/test_forecast_motion_prior.py` and verify import/behavior failures occur because the component is absent.
- [ ] Implement `ForecastMotionPrior` with modes `none`, `association`, `detached`, and `recurrent_bptt`; five-step storage horizon; a 0.35 CV component; isotropic variance bounds; elapsed-time step selection; private pair-validity mask; and score-aware GMM cost.
- [ ] Re-run the focused tests until all pass.

### Task 2: Runtime state contract

**Files:**
- Modify: `opencood/models/colif/state.py`
- Test: `tests/colif/test_forecast_motion_prior.py`
- Test: `tests/colif/test_persistent_query.py`

**Interfaces:**
- Consumes: `future_modes` and `future_horizon_steps` in `GlobalForecastState.empty`.
- Produces: `future_mode_probabilities [B,Q,K]`, `future_relative_means [B,Q,K,S,2]`, `future_mode_variances [B,Q,K,S]`, and `future_spatial_valid_mask [B,Q]` with validation/reset/detach support.

- [ ] Add a failing state-shape/reset/detach test.
- [ ] Run that test and verify the missing state fields cause the failure.
- [ ] Add the four state tensors, constructor dimensions, shape validation, and existing dataclass-wide reset/detach compatibility.
- [ ] Run focused state and persistent-query tests.

### Task 3: Predictive assignment and correction

**Files:**
- Modify: `opencood/models/point_pillar_colif.py`
- Modify: `opencood/models/colif/measurement_state_update.py`
- Test: `tests/colif/test_forecast_motion_prior.py`
- Test: `tests/colif/test_persistent_query.py`

**Interfaces:**
- `_assign_measurements(..., association_cost=None, association_valid=None)` uses explicit private null columns and returns the existing four outputs.
- `MeasurementStateUpdate.forward(..., prior_variance_xy=None)` applies variance-derived x/y correction for observed tracks and preserves the predicted state when unobserved.

- [ ] Add failing tests for private-null assignment, predictive assignment overriding the wrong CV match, high-confidence measurement correction, and missing propagation.
- [ ] Verify each fails for the intended old behavior.
- [ ] Instantiate `ForecastMotionPrior`; compute it before proposal assignment; pass its MAP/CV reference into the decoder; pass mixture costs into assignment; and expose prior diagnostics in outputs/aux.
- [ ] Store the first five forecast means/variances/probabilities in `next_state`; stop writing latent cues into the main recurrence; keep zero-valued checkpoint-compatible legacy cue fields.
- [ ] Re-run focused tests and confirm `none` mode preserves legacy CV behavior.

### Task 4: Training and evaluation ablations

**Files:**
- Modify: `opencood/hypes_yaml/colif/colif_v2v4real.yaml`
- Modify: `opencood/tools/train_colif.py`
- Modify: `opencood/tools/colif_train_utils.py`
- Modify: `opencood/tools/eval_colif_v2v4real.py`
- Modify: `tests/colif/test_train_utils.py`

**Interfaces:**
- Config: `forecast_motion_prior.phase2_mode` and `train_params.phase2_future_prior_mode`.
- CLI: `--future-prior-mode {checkpoint,none,association,detached,recurrent_bptt}` for same-checkpoint strict-online evaluation.
- Phase report: `future_prior_mode` replaces the main-path `future_adapter_mode` claim while retaining the latter only as legacy state.

- [ ] Add failing phase/CLI tests proving Phase 1 forces `none`, Phase 2 enables the configured spatial mode, and the latent adapter stays disabled.
- [ ] Verify failures against current utilities.
- [ ] Update phase configuration, train-mode handling, checkpoint cross-phase retention, logs, YAML, and evaluator override without changing evaluator targets or denominators.
- [ ] Run train-utility and runtime-schema tests.

### Task 5: Static and model-level validation

**Files:**
- Modify only files required by failures in Tasks 1--4.

**Interfaces:**
- Existing model single-frame/clip forward, loss, inference, and checkpoint APIs remain callable.

- [ ] Run `python -m compileall -q opencood/models/colif opencood/models/point_pillar_colif.py opencood/tools/train_colif.py opencood/tools/eval_colif_v2v4real.py`.
- [ ] Run `pytest -q tests/colif` and fix only regressions caused by this change.
- [ ] Run a real V2V4Real CPU data load and GPU single-frame forward, finite loss, and backward check.
- [ ] Run a two-rank one-batch `torchrun` smoke and verify parameter/buffer synchronization, finite gradients, checkpoint save, and strict-online mini-evaluation.

### Task 6: Scientific ablation gate

**Files:**
- Update: `docs/colif_failure_postmortem.md`
- Update: `HANDOFF.md`

**Interfaces:**
- Same trained checkpoint, same test loader, same strict-online evaluator, four runtime prior modes.

- [ ] Train the forecast spatial recurrence from the verified Phase-1 tracker using the fixed two-stage protocol; do not select checkpoints by test metrics.
- [ ] Evaluate the fixed final checkpoint under `none`, `association`, `detached`, and `recurrent_bptt`.
- [ ] Record minADE/minFDE/MR, coverage, MOTA/MOTP, recall/precision, IDS, fragments, FP/FN, and exact checkpoint/config hashes.
- [ ] Accept the future-improves-tracking claim only if, against `none` on the same checkpoint, a spatial-prior mode has strictly higher MOTA, no more IDS, no lower tracking recall, and no lower forecast coverage; its 5 s minFDE must also remain below the fixed reproduced CMP no-aggregation reference `7.5559 m`. Otherwise retain the implementation only as a negative ablation and reject the claim.
