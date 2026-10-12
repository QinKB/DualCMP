#!/usr/bin/env bash
set -euo pipefail
event_dir=$1
label_root=$2
finish() {
    result=$?
    printf '%s\n' "$result" > "$event_dir/exit_code"
    printf '%s\n' "$result" > "$event_dir/complete"
}
trap finish EXIT
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
/home/qkb/miniconda3/envs/coop/bin/python -m opencood.tools.dualcmp.build_aligned_labels \
    --root ../data/V2V4Real/train --tracker-checkpoint runs/train-a-native/checkpoint_epoch_003.pt \
    --output "$label_root" --workers 8
