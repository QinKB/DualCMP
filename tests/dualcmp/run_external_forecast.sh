#!/usr/bin/env bash
# Run the fixed two-GPU Source Phase-B external-validation training contract.
set -euo pipefail
event_dir=$1
output_dir=$2
workers=$3
mode=$4
python_bin=/home/qkb/miniconda3/envs/coop/bin/python

finish() {
    result=$?
    trap - EXIT
    if ! "$python_bin" -c 'import sys; from pathlib import Path; from opencood.tools.dualcmp.checkpoint_handoff import atomic_json; output=Path(sys.argv[1]); events=output / "checkpoint_events"; expected=sorted(path.name for path in events.glob("checkpoint_step_*.json") if path.is_file()); atomic_json(output / "training_finished.json", {"exit_code": int(sys.argv[2]), "expected_events": expected})' "$output_dir" "$result"; then
        printf 'Could not publish training_finished.json\n' >&2
        (( result != 0 )) || result=1
    fi
    temporary="$event_dir/.exit_code.$$"
    printf '%s\n' "$result" > "$temporary"
    mv -f -- "$temporary" "$event_dir/exit_code"
    printf '%s\n' "$result" > "$event_dir/complete"
    return "$result"
}
trap finish EXIT
case "$mode" in preflight|full) ;; *) exit 2 ;; esac
[[ "$workers" =~ ^[0-9]+$ ]] && (( workers >= 0 )) || exit 2
export OMP_NUM_THREADS=4 PYTHONPATH=.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export V2V4REAL_ROOT=../data/V2V4Real
export DUALCMP_TRACKING_LABEL_ROOT=../Reference/Code/CMP/AB3Dmot/scripts/KITTI/v2v4real_label/test
export DUALCMP_TRAIN_TRACKING_LABEL_ROOT=runs/labels-aligned-cmpids-v2-20260930/v2v4real_train_label
export DUALCMP_TRACKER_WEIGHTS=${DUALCMP_TRACKER_WEIGHTS:-runs/repair-joint400-20261007-r1/perception.pt}
unset DUALCMP_CODE_SHA DUALCMP_MTR_WEIGHTS
run_ddp() { "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=2 "$@"; }
budget_args=()
if [[ "$mode" == preflight ]]; then
    budget_args=(--delay-smoke --cache-subset --epochs 3 --max-steps 3 --lr 1e-3 --log-every 1)
else
    budget_args=(--full-data --epochs 30 --lr 1e-4 --log-every 100)
fi
run_ddp -m opencood.tools.dualcmp.train_forecast_ddp \
    --output-dir "$output_dir" --data-split train --no-validation --dirty-smoke --deterministic \
    --forecast-representation source_temporal_aligned --aligned-evidence-policy source_temporal \
    --source-feature-encoder motion_correlation_v1 \
    --intention-points-file runs/intention-aligned-cmpids-v2-20260930/cluster_64.pkl \
    --partner-delay-ms 100 --batch-size 2 --accumulation-steps 1 --workers "$workers" \
    --weight-decay 1e-4 --forecast-mean-weight 5 --forecast-dropout 0.1 \
    --save-every 1 "${budget_args[@]}"
