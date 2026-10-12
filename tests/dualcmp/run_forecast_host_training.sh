#!/usr/bin/env bash
# Caller creates and registers the completion FIFO before launching in tmux.
set -euo pipefail
event_dir=$1
output_dir=$2
policy=$3
processes=$4
accumulation=$5
workers=$6
mode=$7
resume_checkpoint=${8:-}
finish() {
    result=$?
    printf '%s\n' "$result" > "$event_dir/exit_code"
    printf '%s\n' "$result" > "$event_dir/complete"
}
trap finish EXIT
case "$policy" in source_temporal|fused_temporal) ;; *) exit 2 ;; esac
case "$mode" in preflight|full) ;; *) exit 2 ;; esac
export OMP_NUM_THREADS=4 PYTHONPATH=.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export V2V4REAL_ROOT=../data/V2V4Real
export DUALCMP_TRACKING_LABEL_ROOT=../Reference/Code/CMP/AB3Dmot/scripts/KITTI/v2v4real_label/test
export DUALCMP_TRAIN_TRACKING_LABEL_ROOT=runs/labels-aligned-cmpids-v2-20260930/v2v4real_train_label
export DUALCMP_TRACKER_WEIGHTS=${DUALCMP_TRACKER_WEIGHTS:-runs/repair-joint400-20261007-r1/perception.pt}
unset DUALCMP_CODE_SHA DUALCMP_MTR_WEIGHTS
python_bin=/home/qkb/miniconda3/envs/coop/bin/python
run_ddp() { "$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$processes" "$@"; }
budget_args=()
resume_args=()
if [[ -n "$resume_checkpoint" ]]; then
    [[ "$mode" == full && -f "$resume_checkpoint" ]] || exit 2
    resume_args=(--resume "$resume_checkpoint")
fi
if [[ "$mode" == preflight ]]; then
    unset DUALCMP_STARTUP_FIFO
    budget_args=(--max-steps 3)
fi
run_ddp -m opencood.tools.dualcmp.train_forecast_ddp \
    --output-dir "$output_dir" --full-data --data-split train --dirty-smoke --deterministic \
    --forecast-representation source_temporal_aligned --aligned-evidence-policy "$policy" \
    --intention-points-file runs/intention-aligned-cmpids-v2-20260930/cluster_64.pkl \
    --partner-delay-ms 100 --epochs 30 --batch-size 2 --accumulation-steps "$accumulation" \
    --workers "$workers" --lr 1e-4 --weight-decay 1e-4 --forecast-mean-weight 5 \
    --log-every 100 --save-every 1 --eval-every 1 "${budget_args[@]}" "${resume_args[@]}"
if [[ "$mode" == preflight ]]; then
    run_ddp tests/dualcmp/eval_forecast_labels.py \
        --checkpoint "$output_dir/checkpoint.pt" --output "$output_dir/preflight_eval.json" \
        --delay-window --partner-delay-ms 100 --batch-size 1 --workers 2
fi
