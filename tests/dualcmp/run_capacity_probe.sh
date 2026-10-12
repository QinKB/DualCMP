#!/usr/bin/env bash
set -euo pipefail
event_dir=$1
output=$2
batch_size=$3
finish() {
    result=$?
    printf '%s\n' "$result" > "$event_dir/exit_code"
    printf '%s\n' "$result" > "$event_dir/complete"
}
trap finish EXIT
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4
export V2V4REAL_ROOT=../data/V2V4Real
export DUALCMP_TRAIN_TRACKING_LABEL_ROOT=runs/labels-aligned-cmpids-v2-20260930/v2v4real_train_label
unset DUALCMP_CODE_SHA
/home/qkb/miniconda3/envs/coop/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
    tests/dualcmp/probe_capacity_ddp.py --checkpoint runs/accept-source-aligned-query-v1-20260930/checkpoint.pt \
    --output-dir "$output" --batch-size "$batch_size"
