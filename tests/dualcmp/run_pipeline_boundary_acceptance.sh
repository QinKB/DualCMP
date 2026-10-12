#!/usr/bin/env bash
set -euo pipefail
event_dir=$1
output_dir=$2
policy=${3:-source_temporal}
processes=${4:-2}
accumulation=${5:-1}
workers=${6:-4}
mode=${7:-all}
supervision=${8:-cmp_complete_50_v1}
schedule=${9:-cmp_step_v1}
case "$schedule" in cmp_step_v1|constant_v1) ;; *) exit 2 ;; esac
case "$supervision" in partial_future_v1|cmp_complete_50_v1) ;; *) exit 2 ;; esac
case "$policy" in source_temporal|fused_temporal) ;; *) exit 2 ;; esac
case "$mode" in preflight|learning|boundary|all) ;; *) exit 2 ;; esac
[[ $((2 * processes * accumulation)) == 4 ]] || exit 2
if [[ "$mode" == boundary || "$mode" == all ]]; then
    [[ "$policy" == source_temporal && "$processes" == 2 && "$accumulation" == 1 ]] || exit 2
fi
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
export DUALCMP_TRACKER_WEIGHTS=${DUALCMP_TRACKER_WEIGHTS:-runs/train-a-native/checkpoint_epoch_003.pt}
unset DUALCMP_CODE_SHA DUALCMP_MTR_WEIGHTS DUALCMP_STARTUP_FIFO
python_bin=/home/qkb/miniconda3/envs/coop/bin/python
prior_file=runs/intention-aligned-cmpids-v2-20260930/cluster_64.pkl
run_ddp() { "$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$processes" "$@"; }
if [[ "$mode" == preflight || "$mode" == learning ]]; then
    budget=3
    [[ "$mode" != learning ]] || budget=400
    run_ddp -m opencood.tools.dualcmp.train_forecast_ddp \
        --output-dir "$output_dir" --dirty-smoke --delay-smoke --cache-subset \
        --forecast-representation source_temporal_aligned --aligned-evidence-policy "$policy" \
        --forecast-supervision "$supervision" \
        --partner-delay-ms 100 --intention-points-file "$prior_file" \
        --epochs "$budget" --max-steps "$budget" --batch-size 2 --accumulation-steps "$accumulation" \
        --workers "$workers" --lr 1e-3 --weight-decay 1e-4 --forecast-mean-weight 5 \
        --lr-schedule "$schedule" \
        --save-every "$budget" --eval-every 1000 --log-every 1000
    evaluation=preflight_eval.json
    [[ "$mode" != learning ]] || evaluation=train_eval.json
    run_ddp tests/dualcmp/eval_forecast_labels.py \
        --checkpoint "$output_dir/checkpoint.pt" --output "$output_dir/$evaluation" \
        --delay-window --partner-delay-ms 100 --batch-size 1 --workers 2
    exit 0
fi
boundary_args=(--output-dir "$output_dir/boundary-training" --dirty-smoke --delay-smoke
    --cache-subset --forecast-representation source_temporal_aligned --partner-delay-ms 100
    --intention-points-file "$prior_file" --epochs 4 --batch-size 2 --workers 0
    --lr 1e-3 --weight-decay 1e-4 --forecast-mean-weight 5 --eval-every 1000 --log-every 1000)
run_ddp tests/dualcmp/check_forecast_empty_ddp.py "${boundary_args[@]}" --partial-only-empty --stop-after-epoch 3
run_ddp tests/dualcmp/check_forecast_empty_ddp.py "${boundary_args[@]}" \
    --resume "$output_dir/boundary-training/checkpoint_epoch_003.pt"
run_ddp tests/dualcmp/eval_forecast_labels.py \
    --checkpoint "$output_dir/boundary-training/checkpoint.pt" --output "$output_dir/boundary_eval.json" \
    --delay-window --partner-delay-ms 100 --batch-size 1 --workers 2
run_ddp tests/dualcmp/check_forecast_gradient_ddp.py \
    --checkpoint "$output_dir/boundary-training/checkpoint.pt" --output "$output_dir/gradient_equivalence.json"
if [[ "$mode" == boundary ]]; then
    export DUALCMP_INIT_WEIGHTS="$output_dir/boundary-training/checkpoint.pt"
    run_ddp tests/dualcmp/check_forecast_empty_ddp.py --joint \
        --output-dir "$output_dir/joint-training" --train-label-smoke \
        --forecast-representation source_temporal_aligned --partner-delay-ms 100 \
        --epochs 1 --max-steps 1 --batch-size 2 --workers 0 --eval-every 1000 --log-every 1000
    run_ddp tests/dualcmp/check_forecast_empty_ddp.py --joint --joint-global-empty --partial-only-empty \
        --output-dir "$output_dir/joint-empty-training" --train-label-smoke \
        --forecast-representation source_temporal_aligned --partner-delay-ms 100 \
        --epochs 2 --max-steps 2 --batch-size 2 --workers 0 --eval-every 1000 --log-every 1000
    exit 0
fi
run_ddp -m opencood.tools.dualcmp.train_forecast_ddp \
    --output-dir "$output_dir/learning-training" --dirty-smoke --delay-smoke --cache-subset \
    --forecast-representation source_temporal_aligned --partner-delay-ms 100 --intention-points-file "$prior_file" \
    --epochs 400 --max-steps 400 --batch-size 2 --workers 4 --lr 1e-3 --weight-decay 1e-4 \
    --forecast-mean-weight 5 --save-every 400 --eval-every 400 --log-every 400
run_ddp tests/dualcmp/eval_forecast_labels.py \
    --checkpoint "$output_dir/learning-training/checkpoint.pt" --output "$output_dir/train_eval.json" \
    --delay-window --partner-delay-ms 100 --batch-size 1 --workers 2
export DUALCMP_INIT_WEIGHTS="$output_dir/learning-training/checkpoint.pt"
run_ddp tests/dualcmp/check_forecast_empty_ddp.py --joint \
    --output-dir "$output_dir/joint-training" --train-label-smoke \
    --forecast-representation source_temporal_aligned --partner-delay-ms 100 \
    --epochs 1 --max-steps 1 --batch-size 2 --workers 0 --eval-every 1000 --log-every 1000
run_ddp tests/dualcmp/eval_forecast_labels.py \
    --checkpoint "$output_dir/joint-training/checkpoint.pt" --output "$output_dir/joint_eval.json" \
    --delay-window --partner-delay-ms 100 --batch-size 1 --workers 2
run_ddp tests/dualcmp/check_forecast_empty_ddp.py --joint --joint-global-empty \
    --output-dir "$output_dir/joint-empty-training" --train-label-smoke \
    --forecast-representation source_temporal_aligned --partner-delay-ms 100 \
    --epochs 2 --max-steps 2 --batch-size 2 --workers 0 --eval-every 1000 --log-every 1000
CUDA_VISIBLE_DEVICES='' "$python_bin" tests/dualcmp/check_task_acceptance.py \
    --phase learning --all-horizons --directory "$output_dir" --training-directory "$output_dir/learning-training"
