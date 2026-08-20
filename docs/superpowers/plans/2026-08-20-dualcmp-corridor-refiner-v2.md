# DualCMP Corridor Refiner v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the failed v1 dual-source head with a legacy-exact actor-source motion base plus a conservative mode/time/source trajectory-corridor refiner, then train and evaluate it under the historical two-receiver one-epoch protocol.

**Architecture:** `PointPillarColif` keeps its proven `ActorConditionedCooperativeFusion` and passes that context into a wrapped `MotionAnchoredForecastHead`. The wrapper exposes unchanged legacy proposal/base outputs and adds a detached source-only corridor refiner with real-source softmax, an independent null gate, zero-preserving value path, and velocity-integrated residual. Base and source losses and clipping are explicitly separated.

**Tech Stack:** Python 3.10, PyTorch 2.1.1, CUDA 12.1, NCCL DDP, pytest, V2V4Real strict-online evaluator.

**Spec:** `docs/superpowers/specs/2026-08-20-dualcmp-corridor-refiner-design.md`

## Global Constraints

- Preserve `forecast.architecture=motion_anchored` behavior and historical checkpoint loading.
- Dual-source v2 requires `dual_source_version=2` and `forecast_motion_prior.mode/phase2_mode=none`.
- No source-slot embedding and no forecast-to-tracker feedback.
- Source final loss must not reach actor-source fusion or motion-base parameters.
- Use `receiver_ids=[0,1]` for final training; dataset must report3420 train records and3986 test frames.
- Use `python -m pytest`; do not resume v1 A1/A2/A3 checkpoints.
- Initialize v2 from `opencood/logs/colif_paper_phase1_20e_seed20/checkpoint_epoch_009.pth`.

---

### Task 1: Freeze Receiver Protocol and Historical Baseline

**Files:** Modify `tests/colif/test_receiver_protocol.py`, `docs/colif_training.md`.

**Interfaces:** `apply_receiver_override(hypes, "0,1") -> (0,1)` must write `fusion.args.receiver_ids=[0,1]`, which `ColifSequenceDataset` consumes when constructing `(scene,receiver)` clips.

- [ ] Write a failing dataset-level receiver test that retains receivers0/1 and rejects receiver2.
- [ ] Run `/home/qkb/miniconda3/envs/coop/bin/python -m pytest -q tests/colif/test_receiver_protocol.py` and verify RED for the new assertion.
- [ ] Complete the fixture against the real sequence-dataset filtering contract without changing runtime semantics.
- [ ] Re-run and require all receiver tests GREEN.

### Task 2: Implement Metric Trajectory Corridor Sampling

**Files:** Create `opencood/models/colif/trajectory_corridor.py`; create `tests/colif/test_trajectory_corridor.py`.

**Interfaces:** `TrajectoryCorridorSampler.forward(source_bev, actor_boxes, proposal_mean, cav_mask)` samples five offsets sequentially and returns pooled evidence `[B,Q,K,T,L,C]`, valid `[B,Q,K,T,L]`, grid `[B,Q,K,T,5,2]`.

- [ ] Write failing shape, curved-tangent, yaw-fallback, source-mask and out-of-range tests.
- [ ] Run the new test and require an import failure because the sampler does not exist.
- [ ] Implement receiver-frame absolute positions, tangent/normal offsets, five fixed samples, vectorized `grid_sample`, `padding_mode="zeros"`, `align_corners=False`.
- [ ] Re-run and require finite, correctly masked samples.

### Task 3: Implement Zero-Preserving Source Corridor Refiner

**Files:** Modify `trajectory_corridor.py` and `test_trajectory_corridor.py`.

**Interfaces:** `SourceCorridorRefiner.forward(samples, sample_valid, base_trajs, base_logits, geometry, age, availability_logits, force_null=False)` returns final GMM tensors, gate `[B,Q,K]`, selection `[B,Q,K,T,L]`, router `[B,Q,K,T,L+1]`, velocity and mean deltas.

- [ ] Write failing tests for zero context, forced/all-invalid null, joint source permutation, time-varying source selection and velocity integration.
- [ ] Verify RED because the refiner is absent.
- [ ] Implement bias-free source value projection, multiplicative time FiLM and zero-preserving temporal blocks.
- [ ] Implement real-source masked softmax and independent mode gate initialized with `logit(0.1)`.
- [ ] Implement vector-norm-bounded velocity integration (`3m/s`, `4m` position bound) and gated uncertainty/logit residuals.
- [ ] Require exact zero correction for zero source context, router sums of one, and first-step norm at most0.3m.

### Task 4: Rebuild the Dual Head Around the Legacy-Exact Base

**Files:** Rewrite `opencood/models/colif/dual_source_forecast.py`; rewrite `tests/colif/test_dual_source_forecast.py`.

**Interfaces:** `DualSourceMotionAnchoredForecastHead.forward(..., cooperative_context, source_bev, cav_mask, source_to_receiver, delta_t_s)` preserves legacy keys and adds proposal/base/source diagnostics plus explicit loss layer weights.

- [ ] Write failing tests that construct A0 and v2 after the same seed and require `v2.motion_base.state_dict()` to equal A0.
- [ ] Verify v1 fails because it has no `motion_base`, corridor outputs or v2 layer metadata.
- [ ] Construct `MotionAnchoredForecastHead` before every source module and pass unchanged cooperative context into it.
- [ ] Make source-disabled output reuse the legacy loss layers without a detached duplicate.
- [ ] Add detached corridor refinement, geometry/age conditioning, exact masks and `torch.random.fork_rng` around source stochastic execution.
- [ ] Expose base layers `("proposal","cross_mode")/[0.5,0.5]`; full adds `"source_final"/1.0`.
- [ ] Expose disjoint `gradient_parameter_groups()` for base and source.
- [ ] Require all identity, gradient, RNG, mask, permutation and prior-guard tests GREEN.

### Task 5: Correct Loss Weighting and Add Source Regularizers

**Files:** Modify `opencood/loss/colif_loss.py`, `tests/colif/test_colif_loss.py`.

**Interfaces:** Consume `forecast_loss_layer_weights`, `source_gate`, `source_velocity_delta`, `source_mean_delta`; emit weighted GMM/mode loss and matched source regularizers.

- [ ] Write failing tests proving absent source layers cannot halve base gradients and detached final leaves base gradients unchanged.
- [ ] Verify current loss fails because it blindly averages layers.
- [ ] Validate finite nonnegative layer weights with exact layer count; compute unnormalized weighted sums.
- [ ] Add matched, detached-mode-weighted gate KL(prior0.1), velocity smoothness and residual magnitude raw sums/counts with weights0.02/0.05/0.01.
- [ ] Require legacy outputs without metadata to retain their existing mean behavior.

### Task 6: Integrate v2 Without Disabling Proven Actor-Source Context

**Files:** Modify `opencood/models/point_pillar_colif.py`, `opencood/models/colif/__init__.py`, `tests/colif/test_forecast_motion_prior.py`, `tests/colif/test_receiver_protocol.py`.

**Interfaces:** `PointPillarColif` computes actor cooperative context before both legacy and v2 forecast calls; v2 receives that context and source BEV metadata. Tracking remains unchanged.

- [ ] Write failing tests requiring actor cooperative fusion enabled/trainable for v2, version2 validation and prior rejection.
- [ ] Verify RED because current code disables actor cooperative fusion for dual architecture.
- [ ] Implement v2 call path and `[B,Q,K,T,L+1]` diagnostics while preserving A0 branch.
- [ ] Run integration tests and require no tracking/prior regression.

### Task 7: Isolate Gradient Clipping and Log Source Health

**Files:** Modify `opencood/tools/colif_train_utils.py`, `opencood/tools/train_colif.py`, `tests/colif/test_receiver_protocol.py`.

**Interfaces:** `forecast_gradient_parameter_groups(model)` returns disjoint base/source trainable lists; trainer clips each group independently and logs group norms, gate/null and residual statistics.

- [ ] Write a failing test with base norm2 and source norm2000 proving source clipping must not scale base.
- [ ] Verify RED against the current global clip.
- [ ] Implement group discovery, disjointness/coverage validation and independent `clip_grad_norm_`.
- [ ] Make CLI architecture selection set `dual_source_version=2` and add health logging.
- [ ] Require clipping and CLI tests GREEN.

### Task 8: Static and Unit Regression Gate

**Files:** Verify every modified model/loss/train file.

- [ ] Run compileall for model, loss and train entrypoints.
- [ ] Run `git diff --check`.
- [ ] Run focused corridor/dual/loss/receiver/prior tests.
- [ ] Run `/home/qkb/miniconda3/envs/coop/bin/python -m pytest -q tests/colif` and require zero failures.

### Task 9: Real V2V4Real Engineering Gate

**Files:** Write outputs only under new `opencood/logs/dualcmp_v2_*` directories.

- [ ] Run one-batch single-GPU full v2 with receiver0,1 and require finite loss plus nonzero base/source gradients.
- [ ] Run source-disabled same-seed smoke and require base output/gradient checksums equal full v2.
- [ ] Run two-GPU five-batch stability; require no AMP overflow, unused trainable0, checksum delta0, finite group norms and gate mean in `[0.02,0.60]`.
- [ ] Run128-frame strict-online diagnostic; require no NaN/Inf, null non-collapse and first-step residual P90 below0.6m.

### Task 10: Train and Evaluate the Historical One-Epoch Protocol

**Files:** Output `opencood/logs/dualcmp_v2_base_receiver01_seed20_1e_20260820/`, `opencood/logs/dualcmp_v2_full_receiver01_seed20_1e_20260820/`; update `HANDOFF.md`, `docs/colif_training.md`.

- [ ] Train v2 base-only for one epoch from Phase1 epoch9 with two GPUs, receiver0,1, seed20, and historical batch/schedule; require3420 records.
- [ ] Evaluate base-only using the exact historical `online_masked` protocol that produced `minFDE@5s=3.427`, then separately evaluate corrected `cmp_fixed`.
- [ ] Train full v2 using identical initialization, data order, batch contract and schedule.
- [ ] Evaluate full refined and same-checkpoint base output; report net residual contribution and support buckets.
- [ ] Require base reproduction within0.02m of3.427 and full not worse than base by more than0.01m on any 1/3/5s ADE/FDE. If full fails, ship base output and reject source final rather than publish a worse model.
- [ ] Record exact commands, checkpoints, metrics, tests, limitations and recovery state in HANDOFF.
