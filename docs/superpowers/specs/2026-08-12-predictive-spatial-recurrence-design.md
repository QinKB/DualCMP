# CoLIF Predictive Spatial Recurrence Design

## Goal

Replace the falsified latent-query feedback path with a causal
predict--associate--correct path in which the previous forecast's short-horizon
multimodal spatial distribution is consumed before current-frame measurement
assignment. Preserve CMP fused-BEV perception, persistent runtime IDs, strict
GT-free inference, two-stage training, and the existing forecasting output
contract.

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
   fallback component and retains the complete mixture for proposal association;
4. uses the maximum-probability component for the single reference box required
   by the query decoder and missing-state propagation, avoiding a physically
   invalid average between incompatible modes;
5. falls back exactly to constant velocity when no valid forecast is available.

For an active query `i` and proposal `j`, predictive association uses

```text
C_ij = -log sum_k pi_ik N(z_j; mu_ik, sigma_ik^2 I)
       + score_cost(z_j)
```

and augments the cost matrix with one private null column per active query.
Hungarian therefore chooses a proposal only when it is cheaper than the null
hypothesis; gating is performed before assignment rather than as a post-hoc
rejection. An absolute safety radius prevents a pathologically broad forecast
from claiming distant proposals. Unassigned proposals continue to seed newborn
queries.

The constant-velocity component has fixed prior mass `0.35`; the remaining
mass is distributed according to the learned forecast probabilities. Thus a
uniform or weakly ranked forecast cannot displace the CV reference, while a
forecast mode with probability above `0.35 / 0.65` can become the spatial MAP
state. This is an explicit motion-model mixture rather than a learned content
gate.

Matched proposals correct the predicted state through
`MeasurementStateUpdate`. For the x/y state, gain is derived from prior and
measurement variance (`P/(P+R)`), so a confident current measurement overrides
the prior while an uncertain measurement does not. Without a measurement, the
state remains the causal prediction prior and its miss count increases; it is
not relabelled as observed.

## Ablation modes

- `none`: constant-velocity reference and Euclidean-distance assignment.
- `association`: forecast mixture changes only assignment; state propagation
  remains constant velocity and the stored forecast is detached.
- `detached`: forecast mixture changes assignment and missing-state propagation,
  but cross-frame gradients are stopped.
- `recurrent_bptt`: identical inference to `detached`, while next-frame tracking
  loss may backpropagate through the spatial prior into the forecast head.

The same checkpoint can be evaluated under all four modes. Mode changes do not
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
`recurrent_bptt`, the next-frame query box/velocity losses additionally reach
the preceding forecast through the selected differentiable forecast-mode mean.
The discrete Hungarian decision is not differentiated; its forecast input is
trained by the GMM objective and evaluated through IDS/MOTA.

The previous `FutureAdapter` remains loadable only for historical checkpoints,
but is forced to `none` in the main path and is not an optimizer target.

## Required tests

1. invalid forecast is numerically identical to constant-velocity propagation;
2. a bimodal forecast can select the correct proposal when CV distance chooses
   the wrong one;
3. private null columns prevent forced bad matches;
4. `association` changes matching but not the propagated reference;
5. `detached` blocks cross-frame forecast gradients;
6. `recurrent_bptt` sends a finite nonzero next-frame state loss gradient to
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
test ablations (`none`, `association`, `detached`, `recurrent_bptt`) reporting
tracking and forecasting metrics together.
