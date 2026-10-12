#!/usr/bin/env bash
set -euo pipefail
event_dir=$1
root=$2
output=$3
proof=$4
mode=$5
remote=$6
batch=${7:-2}
finish() {
    result=$?
    if [[ -n "${readiness_pid:-}" ]]; then
        kill "$readiness_pid" 2>/dev/null || true
    fi
    printf '%s\n' "$result" > "$event_dir/exit_code"
    printf '%s\n' "$result" > "$event_dir/complete"
}
trap finish EXIT
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONPATH=.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export V2V4REAL_ROOT=../data/V2V4Real
export DUALCMP_TRACKING_LABEL_ROOT=../Reference/Code/CMP/AB3Dmot/scripts/KITTI/v2v4real_label/test
export DUALCMP_TRAIN_TRACKING_LABEL_ROOT=runs/labels-aligned-cmpids-v2-20260930/v2v4real_train_label
export DUALCMP_TRACKER_WEIGHTS=runs/repair-joint400-20261007-r1/perception.pt
unset DUALCMP_CODE_SHA DUALCMP_MTR_WEIGHTS
scope=()
[[ "$mode" != preflight ]] || scope=(--train-window)
python_bin=/home/qkb/miniconda3/envs/coop/bin/python
# A separate FIFO forwards readiness without competing with the completion listener.
(IFS= read -r ready < "$event_dir/validator_ready"; \
    [[ "$ready" == ready ]] && printf '{"status":"validator_ready"}\n' > "$event_dir/startup") &
readiness_pid=$!
"$python_bin" -m opencood.tools.dualcmp.watch_checkpoints \
    --run-root "$root" --output "$output" --sync-proof "$proof" \
    --ready-fifo "$event_dir/validator_ready" --remote "$remote" \
    --batch-size "$batch" --workers 4 "${scope[@]}"
