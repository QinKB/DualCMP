# CoLIF Predictive Spatial Recurrence Design

## Experimental disposition

This document specifies a falsifiable architecture experiment, not the final
accepted main path. Deterministic, same-checkpoint full-test ablations rejected
all tested future-to-tracking variants. Forecast association, joint
association/propagation, and isolated gap imputation each degraded at least
MOTA, IDS, recall, or coverage relative to `none`. Therefore the released main
configuration uses `future_prior_mode=none`; the implementations below remain
available only to reproduce the negative ablations. The accepted CoLIF runtime
is fused persistent-query tracking followed by causal motion forecasting, with
no forecast value written into tracking state.

## Goal

Replace the falsified latent-query feedback path with a causal
predict--correct path in which the previous forecast's short-horizon multimodal
spatial distribution propagates an actor state only when the current frame has
no assigned measurement. Preserve the original constant-velocity association,
CMP fused-BEV perception, persistent runtime IDs, strict GT-free inference,
two-stage training, and the existing forecasting output contract.

## Root cause being corrected

The current runtime first matches constant-velocity boxes to CMP dense proposals
with position-only Hungarian assignment, then applies a forecast-derived latent
to query content. Therefore the forecast cannot affect the current association,
and a missing query keeps the constant-velocity geometry because measurement
gain becomes zero. `measurement_observed_mask=False` is an assignment outcome,
not a visibility label, so missing-only latent gating cannot correct this causal
ordering.

## Selected architecture

At receiver frame `t`, the forecasting head continues to output `K` relative
GMM trajectories. CoLIF stores only the first `S=5` steps (0.1--0.5 s): mode
probabilities, relative means, isotropic per-mode variances, and a valid mask.
It never stores GT future or uses GT identity at runtime.

Before processing frame `t+1`, a new stateless `ForecastMotionPrior`:

1. selects the forecast step nearest to the elapsed receiver time;
2. converts every mode center from the previous receiver frame into the current
   receiver frame;
3. adds the ordinary constant-velocity prediction as an explicit calibrated
   fallback component;
4. uses the maximum-probability component for the single reference box required
   by the query decoder and missing-state propagation, avoiding a physically
   invalid average between incompatible modes;
5. falls back exactly to constant velocity when no valid forecast is available.

The accepted main path does **not** use forecast modes for association. For an
active query `i` and proposal `j`, it retains the tracker baseline

```text
C_ij = d_CV,
valid_ij = [d_CV <= 4 m].
```

The matrix is augmented with one private null column per active query, so
Hungarian never forces an invalid proposal onto a query. Gating is performed
before assignment rather than as a post-hoc rejection. Unassigned proposals
continue to seed newborn queries. The implemented forecast-mixture association
is retained only as a negative ablation: same-checkpoint full-test results show
that even its conservative CV-first fallback reduces MOTA and forecast quality.

The constant-velocity component has prior mass
`max(0.15, 1/(K+1)+1e-4)`, slightly above one uniform forecast component; the remaining
mass is distributed according to the learned forecast probabilities. Thus a
uniform or weakly ranked forecast cannot displace the CV reference, while a
forecast mode with probability above `0.15 / 0.85` can become the spatial MAP
state. This is an explicit motion-model mixture rather than a learned content
gate.

Matched proposals correct the predicted state through
`MeasurementStateUpdate`. For the x/y state, gain is derived from prior and
measurement variance (`P/(P+R)`), lower-bounded by the pretrained tracker's
learned gain, so a future prior can never weaken correction by a current
measurement. Without a measurement, the
state remains the causal prediction prior and its miss count increases; it is
not relabelled as observed.

## Ablation modes

- `none`: constant-velocity reference and Euclidean-distance assignment.
- `association`: forecast mixture changes only assignment; state propagation
  remains constant velocity and the stored forecast is detached.
- `propagation_detached`: assignment is identical to `none`; only a query with
  no assigned measurement keeps the detached forecast MAP spatial state.
- `propagation_bptt`: main mode; inference is identical to
  `propagation_detached`, while next-frame state loss reaches the forecast mean.
- `detached`: forecast mixture changes assignment and missing-state propagation,
  but cross-frame gradients are stopped.
- `recurrent_bptt`: identical inference to `detached`, while next-frame tracking
  loss may backpropagate through the spatial prior into the forecast head.

The same checkpoint can be evaluated under all six modes. Mode changes do not
alter targets, evaluator matching, coverage definitions, or forecast output.

## State and coordinate contract

`GlobalForecastState` adds:

```text
future_mode_probabilities [B,Q,K]
future_relative_means      [B,Q,K,S,2]
future_mode_variances      [B,Q,K,S]
future_spatial_valid_mask  [B,Q]
```

Means are relative displacements in the forecast capture receiver frame.
Variances are isotropic in square metres, which remains valid under receiver yaw
rotation. The previous state box supplies the forecast origin. `receiver_pose`
and `receiver_time_s` determine coordinate transformation and step selection.
Scene resets replace all four tensors together with the rest of the state.

## Training signal

Phase 1 is unchanged and trains fused query tracking with the future-prior mode
forced to `none`. Phase 2 freezes CMP/query-tracker parameters and trains the
forecast head. GMM/mode losses calibrate the multimodal distribution. In
`propagation_bptt`, the next-frame query box/velocity losses additionally reach
the preceding forecast through the selected differentiable forecast-mode mean.
The discrete Hungarian decision remains forecast-independent and is not
differentiated.

The previous `FutureAdapter` remains loadable only for historical checkpoints,
but is forced to `none` in the main path and is not an optimizer target.

## Required tests

1. invalid forecast is numerically identical to constant-velocity propagation;
2. propagation-only mode preserves CV association even when a forecast mode is
   closer to a competing proposal;
3. private null columns prevent forced bad matches;
4. `association` changes matching but not the propagated reference and remains
   available only as an ablation;
5. `propagation_detached` blocks cross-frame forecast gradients;
6. `propagation_bptt` sends a finite nonzero next-frame state loss gradient to
   forecast means/probabilities;
7. a high-confidence measurement corrects an uncertain prior more strongly;
8. no measurement preserves the predicted prior and increments miss count;
9. state reset, detach, dtype, device, AMP, checkpoint and two-rank DDP contracts
   remain valid;
10. the strict-online runtime schema still rejects GT and future targets.

## Acceptance

Implementation is accepted only after focused RED/GREEN tests, the complete
CoLIF unit suite, a real V2V4Real forward/loss/backward smoke, and a two-GPU DDP
smoke pass. Scientific benefit is decided only by same-checkpoint strict-online
test ablations (`none`, `association`, `propagation_detached`,
`propagation_bptt`, `detached`, `recurrent_bptt`) reporting tracking and
forecasting metrics together. The main path is accepted only if
propagation-only improves MOTA and IDS over `none` without changing the target
or evaluator contract; association modes are not promoted when that condition
fails.
