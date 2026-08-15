# Selectable Forecast Metric Protocol Design

## Objective

DualCMP must support both its existing strict-online, variable-validity forecast
metrics and a corrected CMP-style fixed-horizon report.  Changing the reporting
protocol must not change model inference, matching distance, trajectories, or
tracking metrics.

## Verified indexing convention

V2V4Real is sampled at 10 Hz and the future tensor starts at the next frame.
Therefore future tensor indices `0..49` represent `0.1..5.0 s`.

| Horizon | ADE samples | FDE index |
|---|---:|---:|
| 1 s | `[:10]` | `9` |
| 3 s | `[:30]` | `29` |
| 5 s | `[:50]` | `49` |

CMP's local custom evaluator uses indices `10` and `30` for 1 s and 3 s FDE
and excludes index `49` from 5 s ADE.  Those are off-by-one errors and are not
reproduced.

## Protocols

### `online_masked`

This preserves the current implementation.  For each horizon, ADE averages the
valid points within that horizon and FDE uses the last valid point in that
window.  A target is eligible when the window contains at least one valid point.
This protocol is suitable for variable-length strict-online analysis.

### `cmp_fixed`

This reports corrected fixed 1/3/5 s metrics.  A target contributes only when
all 50 future samples are valid, so 1 s, 3 s, and 5 s use the same target
population.  ADE and FDE then use the exact samples and endpoints in the table
above.

The protocol affects forecast displacement metrics and their age/source
breakdowns.  It does not alter model outputs, online actor matching, tracking
metrics, or lifecycle/coverage accounting.  Consequently it improves temporal
definition compatibility with CMP, but does not by itself make the two complete
pipelines identical: their runtime actor construction and evaluated actor
populations must still be reported separately.

## Interface and provenance

Both training-time test evaluation and standalone evaluation expose:

```text
--forecast-metric-protocol {online_masked,cmp_fixed}
```

Every metrics record stores the selected protocol, horizon steps, FDE indices,
sampling interval, and whether a complete 5 s future is required.  The default
remains `online_masked` to preserve existing experiment behavior.

## Validation

- A synthetic full-future case must yield ADE `5.5/15.5/25.5` and FDE
  `10/30/50` for targets at x=`1..50` m under `cmp_fixed`.
- A truncated target must remain eligible under `online_masked` but must be
  excluded from all fixed horizons under `cmp_fixed`.
- Existing strict-online tests must remain unchanged and pass.
- Both command-line entry points must advertise the new selector.

## Recovery boundary

Only the evaluator, its two entry points, focused tests, and task-specific
documentation are changed.  Existing dirty model, manuscript, YAML, HANDOFF,
and training-log files are outside this change and must not enter its commit.
