#!/usr/bin/env bash
# The caller registers the durable completion listener before launching.
set -euo pipefail
event_dir=$1
output_dir=$2
policy=$3
processes=$4
accumulation=$5
[[ "$policy" == source_temporal || "$policy" == fused_temporal ]]
[[ $((2 * processes * accumulation)) == 4 ]]
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
export DUALCMP_TRACKER_WEIGHTS=runs/train-a-native/checkpoint_epoch_003.pt
unset DUALCMP_CODE_SHA DUALCMP_MTR_WEIGHTS DUALCMP_STARTUP_FIFO
python_bin=/home/qkb/miniconda3/envs/coop/bin/python
run_ddp() { "$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$processes" "$@"; }
mkdir "$output_dir"
run_ddp -m opencood.tools.dualcmp.train_forecast_ddp \
    --output-dir "$output_dir/training" --dirty-smoke --deterministic --delay-smoke --cache-subset \
    --forecast-representation source_temporal_aligned --aligned-evidence-policy "$policy" \
    --intention-points-file runs/intention-aligned-cmpids-v2-20260930/cluster_64.pkl \
    --partner-delay-ms 100 --epochs 30 --max-steps 1 --batch-size 2 --accumulation-steps "$accumulation" \
    --workers 2 --lr 1e-4 --weight-decay 1e-4 --forecast-mean-weight 5 \
    --log-every 1 --save-every 1 --eval-every 1 2>&1 | tee "$output_dir/job.log"
run_ddp -m opencood.tools.dualcmp.eval_forecast_mainstream_ddp \
    --checkpoint "$output_dir/training/checkpoint.pt" --output "$output_dir/fresh_test.json" \
    --partner-delay-ms 100 --batch-size 2 --workers 2 2>&1 | tee -a "$output_dir/job.log"
cp "$output_dir/job.log" "$output_dir/training/job.log"
"$python_bin" tests/dualcmp/check_mainstream_training_artifacts.py \
    --directory "$output_dir/training" --fresh-report "$output_dir/fresh_test.json"
