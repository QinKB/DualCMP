#!/usr/bin/env bash
# The caller registers the durable completion listener before launch.
set -euo pipefail
event_dir=$1
checkpoint=$2
output=$3
scope=$4
batch=$5
workers=$6
finish() {
    result=$?
    printf '%s\n' "$result" > "$event_dir/exit_code"
    printf '%s\n' "$result" > "$event_dir/complete"
}
trap finish EXIT
export OMP_NUM_THREADS=4 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0
export V2V4REAL_ROOT=../data/V2V4Real
export DUALCMP_TRACKING_LABEL_ROOT=../Reference/Code/CMP/AB3Dmot/scripts/KITTI/v2v4real_label/test
export DUALCMP_TRAIN_TRACKING_LABEL_ROOT=runs/labels-aligned-cmpids-v2-20260930/v2v4real_train_label
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset DUALCMP_CODE_SHA DUALCMP_MTR_WEIGHTS
scope_args=()
[[ "$scope" != full && "$scope" != iou05-full ]] || scope_args=(--full-test)
entry_args=(-m opencood.tools.dualcmp.eval_forecast_mainstream_ddp)
evaluation_args=(--partner-delay-ms 100 --batch-size "$batch" --workers "$workers")
if [[ "$scope" == iou05-preflight || "$scope" == iou05-full || "$scope" == external-train-window ]]; then
    entry_args=(tests/dualcmp/eval_cmp_iou05_ddp.py)
fi
[[ "$scope" != external-train-window ]] || scope_args=(--train-window --refresh-train-bn)
if [[ "$scope" == audit ]]; then
    entry_args=(tests/dualcmp/audit_cmp_metric_evidence.py)
    evaluation_args=()
fi
/home/qkb/miniconda3/envs/coop/bin/python -m torch.distributed.run \
    --standalone --nproc_per_node=1 "${entry_args[@]}" \
    --checkpoint "$checkpoint" --output "$output" "${evaluation_args[@]}" "${scope_args[@]}"
