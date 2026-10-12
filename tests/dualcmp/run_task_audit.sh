#!/usr/bin/env bash
# Completion FIFO is created by the caller before this durable job starts.
set -euo pipefail
event_dir=$1
output_dir=$2
finish() {
    result=$?
    printf '%s\n' "$result" > "$event_dir/exit_code"
    printf '%s\n' "$result" > "$event_dir/complete"
}
trap finish EXIT
export V2V4REAL_ROOT=../data/V2V4Real
export DUALCMP_TRACKING_LABEL_ROOT=../Reference/Code/CMP/AB3Dmot/scripts/KITTI/v2v4real_label/test
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=.
python_bin=${DUALCMP_PYTHON:-/home/qkb/miniconda3/envs/coop/bin/python}
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node=2 \
    tests/dualcmp/audit_task_fitness_ddp.py \
    --checkpoint runs/train-b-source-temporal-full-d100/checkpoint_epoch_017.pt \
    --output "$output_dir/source.json" --samples 32 --batch-size 2 --workers 4
"$python_bin" -m torch.distributed.run --standalone --nproc_per_node=2 \
    tests/dualcmp/audit_task_fitness_ddp.py \
    --checkpoint runs/train-b-fused/checkpoint_epoch_025.pt \
    --output "$output_dir/fused.json" --samples 32 --batch-size 2 --workers 4
