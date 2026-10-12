#!/usr/bin/env bash
# Execute a hash-verified historical source in an isolated checkout.
set -euo pipefail
event_dir=$1
source_dir=$2
output_dir=$3
budget=${4:-3}
finish() {
    result=$?
    printf '%s\n' "$result" > "$event_dir/exit_code"
    printf '%s\n' "$result" > "$event_dir/complete"
}
trap finish EXIT
cd "$source_dir"
export OMP_NUM_THREADS=4 PYTHONPATH=.
export V2V4REAL_ROOT=/home/qkb/Project/data/V2V4Real
export DUALCMP_TRACKING_LABEL_ROOT=/home/qkb/Project/Reference/Code/CMP/AB3Dmot/scripts/KITTI/v2v4real_label/test
export DUALCMP_TRAIN_TRACKING_LABEL_ROOT=runs/labels-aligned-cmpids-v2-20260930/v2v4real_train_label
export DUALCMP_TRACKER_WEIGHTS=runs/train-a-native/checkpoint_epoch_003.pt
unset DUALCMP_CODE_SHA DUALCMP_MTR_WEIGHTS DUALCMP_STARTUP_FIFO
python_bin=/home/qkb/miniconda3/envs/coop/bin/python
run_ddp() { "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=2 "$@"; }
if [[ "$budget" == 3 ]]; then
    run_ddp tests/dualcmp/eval_forecast_labels.py \
        --checkpoint runs/accept-pipeline-baseline5-20261004-r1/learning-training/checkpoint.pt \
        --output runs/regression-factor-audit-20261007-r1/old_checkpoint_reference_eval.json \
        --delay-window --partner-delay-ms 100 --batch-size 1 --workers 2
fi
run_ddp -m opencood.tools.dualcmp.train_forecast_ddp \
    --output-dir "$output_dir" --dirty-smoke --delay-smoke --cache-subset \
    --forecast-representation source_temporal_aligned --partner-delay-ms 100 \
    --intention-points-file runs/intention-aligned-cmpids-v2-20260930/cluster_64.pkl \
    --epochs "$budget" --max-steps "$budget" --batch-size 2 --workers 4 \
    --lr 1e-3 --weight-decay 1e-4 --forecast-mean-weight 5 \
    --save-every "$budget" --eval-every "$budget" --log-every "$budget"
run_ddp tests/dualcmp/eval_forecast_labels.py \
    --checkpoint "$output_dir/checkpoint.pt" --output "$output_dir/train_eval.json" \
    --delay-window --partner-delay-ms 100 --batch-size 1 --workers 2
