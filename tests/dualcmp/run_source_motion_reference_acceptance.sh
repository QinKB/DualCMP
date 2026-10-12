#!/usr/bin/env bash
# Reuse the previously accepted perception + deterministic CMP-step baseline.
set -euo pipefail
event_dir=$1
output=$2
policy=$3
processes=$4
accumulation=$5
workers=$6
budget=$7
normalization=${8:-batch_norm_v1}
refresh_bn=${9:-0}
dropout=${10:-0.1}
[[ "$policy" == source_temporal || "$policy" == fused_temporal || "$policy" == source_current || "$policy" == geometry_temporal ]]
[[ "$budget" == 3 || "$budget" == 400 ]]
[[ "$normalization" == batch_norm_v1 || "$normalization" == layer_norm_v1 ]]
[[ "$refresh_bn" == 0 || "$refresh_bn" == 1 ]]
[[ $((2 * processes * accumulation)) == 4 ]]
finish() {
    result=$?
    printf '%s\n' "$result" > "$event_dir/exit_code"
    printf '%s\n' "$result" > "$event_dir/complete"
}
trap finish EXIT
export OMP_NUM_THREADS=4 PYTHONPATH=. CUBLAS_WORKSPACE_CONFIG=:4096:8
export V2V4REAL_ROOT=../data/V2V4Real
export DUALCMP_TRACKING_LABEL_ROOT=../Reference/Code/CMP/AB3Dmot/scripts/KITTI/v2v4real_label/test
export DUALCMP_TRAIN_TRACKING_LABEL_ROOT=runs/labels-aligned-cmpids-v2-20260930/v2v4real_train_label
export DUALCMP_TRACKER_WEIGHTS=runs/repair-joint400-20261007-r1/perception.pt
unset DUALCMP_CODE_SHA DUALCMP_MTR_WEIGHTS DUALCMP_STARTUP_FIFO
python_bin=/home/qkb/miniconda3/envs/coop/bin/python
run_ddp() { "$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$processes" "$@"; }
trainer=(-m opencood.tools.dualcmp.train_forecast_ddp)
refresh=()
if [[ "$refresh_bn" == 1 ]]; then
    refresh=(--refresh-forecast-bn)
fi
if [[ "$budget" == 3 ]]; then
    trainer=(tests/dualcmp/check_source_motion_inputs_ddp.py)
fi
run_ddp "${trainer[@]}" \
    --output-dir "$output" --dirty-smoke --delay-smoke --cache-subset --deterministic \
    --forecast-representation source_temporal_aligned --aligned-evidence-policy "$policy" \
    --source-feature-encoder motion_correlation_v1 --forecast-supervision cmp_complete_50_v1 \
    "${refresh[@]}" \
    --forecast-normalization "$normalization" --forecast-variance log_std_clamp_v1 \
    --forecast-dropout "$dropout" \
    --partner-delay-ms 100 --intention-points-file runs/intention-aligned-cmpids-v2-20260930/cluster_64.pkl \
    --epochs "$budget" --max-steps "$budget" --batch-size 2 --accumulation-steps "$accumulation" \
    --workers "$workers" --lr 1e-3 --weight-decay 1e-4 --forecast-mean-weight 5 \
    --lr-schedule cmp_step_v1 --save-every "$budget" --eval-every 1000 --log-every 1000
run_ddp tests/dualcmp/eval_cmp_iou05_ddp.py \
    --checkpoint "$output/checkpoint.pt" --output "$output/train_eval.json" \
    --train-window --partner-delay-ms 100 --batch-size 2 --workers 2
if [[ "$budget" == 400 ]]; then
    CUDA_VISIBLE_DEVICES='' "$python_bin" tests/dualcmp/check_source_motion_acceptance.py --directory "$output"
fi
