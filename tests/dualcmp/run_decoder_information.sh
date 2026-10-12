#!/usr/bin/env bash
set -euo pipefail
event_dir=$1
checkpoint=$2
output=$3
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
    tests/dualcmp/audit_decoder_information_ddp.py --checkpoint "$checkpoint" --output "$output"
