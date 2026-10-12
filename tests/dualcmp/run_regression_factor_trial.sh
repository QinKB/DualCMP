#!/usr/bin/env bash
# Frozen one-factor trial; formal trainer and evaluators remain authoritative.
set -euo pipefail
event_dir=$1
output_dir=$2
frontend=$3
association=$4
supervision=$5
processes=$6
accumulation=$7
budget=${8:-400}
case "$frontend" in old) weights=runs/train-a-native/checkpoint_epoch_003.pt ;;
    new) weights=runs/repair-joint400-20261007-r1/perception.pt ;; *) exit 2 ;; esac
case "$association" in legacy_hungarian_v1|iou_causal_v1) ;; *) exit 2 ;; esac
case "$supervision" in partial_future_v1|cmp_complete_50_v1) ;; *) exit 2 ;; esac
[[ $((2 * processes * accumulation)) == 4 && "$budget" -gt 0 ]] || exit 2
finish() {
    result=$?
    printf '%s\n' "$result" > "$event_dir/exit_code"
    printf '%s\n' "$result" > "$event_dir/complete"
}
trap finish EXIT
export OMP_NUM_THREADS=4 PYTHONPATH=.
export V2V4REAL_ROOT=../data/V2V4Real
export DUALCMP_TRACKING_LABEL_ROOT=../Reference/Code/CMP/AB3Dmot/scripts/KITTI/v2v4real_label/test
export DUALCMP_TRAIN_TRACKING_LABEL_ROOT=runs/labels-aligned-cmpids-v2-20260930/v2v4real_train_label
export DUALCMP_TRACKER_WEIGHTS="$weights"
unset DUALCMP_CODE_SHA DUALCMP_MTR_WEIGHTS DUALCMP_STARTUP_FIFO
python_bin=/home/qkb/miniconda3/envs/coop/bin/python
run_ddp() { "$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$processes" "$@"; }
run_ddp -m opencood.tools.dualcmp.train_forecast_ddp \
    --output-dir "$output_dir" --dirty-smoke --delay-smoke --cache-subset \
    --forecast-representation source_temporal_aligned --aligned-evidence-policy source_temporal \
    --forecast-association-policy "$association" --forecast-supervision "$supervision" \
    --forecast-variance log_std_clamp_v1 --forecast-normalization batch_norm_v1 \
    --source-feature-encoder legacy_additive_v1 --partner-delay-ms 100 \
    --intention-points-file runs/intention-aligned-cmpids-v2-20260930/cluster_64.pkl \
    --epochs "$budget" --max-steps "$budget" --batch-size 2 --accumulation-steps "$accumulation" \
    --workers 2 --lr 1e-3 --weight-decay 1e-4 --forecast-mean-weight 5 \
    --lr-schedule cmp_step_v1 --save-every "$budget" --eval-every 1000 --log-every 1000
run_ddp tests/dualcmp/eval_forecast_labels.py \
    --checkpoint "$output_dir/checkpoint.pt" --output "$output_dir/train_eval.json" \
    --delay-window --partner-delay-ms 100 --batch-size 1 --workers 2 \
    --evaluation-association-policy iou_causal_v1 --export-actor-metrics
