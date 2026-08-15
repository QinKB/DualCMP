# DualCMP Forecast Metric Protocols

The evaluator supports two reporting protocols without changing the model or
its predictions.

- `online_masked` is the backward-compatible strict-online protocol.  ADE uses
  valid samples inside each window, and FDE uses that window's last valid
  sample.
- `cmp_fixed` is the corrected fixed 1/3/5 s protocol.  It requires a complete
  50-step future, uses ADE slices `[:10]`, `[:30]`, and `[:50]`, and uses FDE
  indices `9`, `29`, and `49`.

CMP's local custom evaluator contains off-by-one endpoint errors at 1 s and
3 s and omits the 5.0 s sample from 5 s ADE.  `cmp_fixed` intentionally does
not reproduce those errors.  It aligns the time definitions; it does not erase
differences in runtime actor construction or actor populations between CMP and
DualCMP.

## Training with automatic test evaluation

Add one of the following to the existing `train_colif` command:

```bash
--forecast-metric-protocol online_masked
```

or:

```bash
--forecast-metric-protocol cmp_fixed
```

## Standalone checkpoint evaluation

Use the same option with `opencood.tools.eval_colif_v2v4real`, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n coop \
  torchrun --standalone --nproc_per_node=2 \
  -m opencood.tools.eval_colif_v2v4real \
  --hypes_yaml opencood/hypes_yaml/colif/colif_v2v4real.yaml \
  --checkpoint /absolute/path/to/checkpoint.pth \
  --output /absolute/path/to/metrics_cmp_fixed.json \
  --receiver-ids 0,1 \
  --forecast-metric-protocol cmp_fixed
```

Each JSON result records `forecast_metric_protocol`,
`forecast_horizon_steps`, `forecast_fde_indices`,
`forecast_fde_endpoint_rule`, `forecast_sample_period_s`, and
`forecast_requires_complete_future`.
