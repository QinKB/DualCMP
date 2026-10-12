#!/usr/bin/env bash
set -euo pipefail
event_dir=$1
output_dir=$2
start_phase=${3:-legacy}
training_dir=${4:-runs/accept-source-aligned-cmpids-v2-20260930}
prior_file=${5:-runs/intention-aligned-cmpids-v2-20260930/cluster_64.pkl}
joint_dir=${6:-runs/accept-source-aligned-joint-cmpids-v2-20260930}
label_root=${7:-runs/labels-aligned-cmpids-v2-20260930/v2v4real_train_label}
mean_weight=${8:-5}
legacy_reference=${9:-}
finish() {
    result=$?
    printf '%s\n' "$result" > "$event_dir/exit_code"
    printf '%s\n' "$result" > "$event_dir/complete"
}
trap finish EXIT
export V2V4REAL_ROOT=../data/V2V4Real
export DUALCMP_TRACKING_LABEL_ROOT=../Reference/Code/CMP/AB3Dmot/scripts/KITTI/v2v4real_label/test
export DUALCMP_TRAIN_TRACKING_LABEL_ROOT="$label_root"
export DUALCMP_TRACKER_WEIGHTS=runs/train-a-native/checkpoint_epoch_003.pt
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=.
unset DUALCMP_CODE_SHA DUALCMP_MTR_WEIGHTS DUALCMP_STARTUP_FIFO
python_bin=${DUALCMP_PYTHON:-/home/qkb/miniconda3/envs/coop/bin/python}
run_ddp() { "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=2 "$@"; }
# New artifacts only; legacy label files, priors and runs stay immutable.
if ! test -f "$(dirname "$label_root")/aligned_labels_manifest.json"; then
CUDA_VISIBLE_DEVICES='' "$python_bin" -m opencood.tools.dualcmp.build_aligned_labels \
    --root "$V2V4REAL_ROOT/train" --tracker-checkpoint "$DUALCMP_TRACKER_WEIGHTS" --output "$label_root"
fi
if test "$start_phase" = legacy; then
run_ddp tests/dualcmp/audit_task_fitness_ddp.py \
    --checkpoint runs/train-b-source-temporal-full-d100/checkpoint_epoch_017.pt \
    --output "$output_dir/legacy_source.json" --samples 32 --batch-size 2 --workers 4
run_ddp tests/dualcmp/audit_task_fitness_ddp.py \
    --checkpoint runs/train-b-fused/checkpoint_epoch_025.pt \
    --output "$output_dir/legacy_fused.json" --samples 32 --batch-size 2 --workers 4
fi
if test -n "$legacy_reference"; then
    # Reuse immutable compatibility evidence for an unchanged model source
    # tree; document its origin instead of spending another paired GPU run.
    mkdir -p "$output_dir"
    cp "$legacy_reference/legacy_source.json" "$output_dir/legacy_source.json"
    cp "$legacy_reference/legacy_fused.json" "$output_dir/legacy_fused.json"
fi
if test "$start_phase" != preflight; then
CUDA_VISIBLE_DEVICES='' "$python_bin" tests/dualcmp/check_task_acceptance.py --phase legacy --directory "$output_dir"
fi
if ! test -f "$prior_file"; then
CUDA_VISIBLE_DEVICES='' "$python_bin" -m opencood.tools.dualcmp.build_aligned_intentions \
    --tracker-checkpoint "$DUALCMP_TRACKER_WEIGHTS" \
    --output "$prior_file" --samples 512
fi
if test "$start_phase" = preflight; then
run_ddp -m opencood.tools.dualcmp.train_forecast_ddp \
    --output-dir "$training_dir" --dirty-smoke --delay-smoke --cache-subset \
    --forecast-representation source_temporal_aligned --partner-delay-ms 100 \
    --intention-points-file "$prior_file" --epochs 2 --max-steps 2 \
    --batch-size 2 --workers 8 --lr 1e-3 --weight-decay 1e-4 \
    --forecast-mean-weight "$mean_weight" --save-every 2 --eval-every 2 --log-every 2
run_ddp tests/dualcmp/audit_task_fitness_ddp.py \
    --checkpoint "$training_dir/checkpoint.pt" --allow-dirty-smoke --boundary-stream \
    --output "$output_dir/preflight.json" --samples 4 --batch-size 2 --workers 8
exit 0
fi
run_ddp -m opencood.tools.dualcmp.train_forecast_ddp \
    --output-dir "$training_dir" --dirty-smoke --delay-smoke --cache-subset \
    --forecast-representation source_temporal_aligned --partner-delay-ms 100 \
    --intention-points-file "$prior_file" \
    --epochs 400 --max-steps 400 --batch-size 2 --workers 4 --lr 1e-3 --weight-decay 1e-4 \
    --forecast-mean-weight "$mean_weight" --save-every 400 --eval-every 400 --log-every 400
run_ddp tests/dualcmp/eval_forecast_labels.py \
    --checkpoint "$training_dir/checkpoint.pt" \
    --output "$output_dir/train_eval.json" --delay-window --partner-delay-ms 100 --batch-size 1 --workers 2
CUDA_VISIBLE_DEVICES='' "$python_bin" tests/dualcmp/check_task_acceptance.py --phase learning --directory "$output_dir" --training-directory "$training_dir"
run_ddp tests/dualcmp/audit_task_fitness_ddp.py \
    --checkpoint "$training_dir/checkpoint.pt" --allow-dirty-smoke --boundary-stream \
    --output "$output_dir/aligned.json" --samples 32 --batch-size 2 --workers 8
export DUALCMP_INIT_WEIGHTS="$training_dir/checkpoint.pt"
run_ddp -m opencood.tools.dualcmp.train_joint_ddp \
    --output-dir "$joint_dir" --train-label-smoke \
    --forecast-representation source_temporal_aligned --partner-delay-ms 100 \
    --epochs 1 --max-steps 1 --batch-size 1 --workers 2 --log-every 1
run_ddp tests/dualcmp/eval_forecast_labels.py \
    --checkpoint "$joint_dir/checkpoint.pt" --output "$output_dir/joint_eval.json" \
    --delay-window --partner-delay-ms 100 --batch-size 1 --workers 2
CUDA_VISIBLE_DEVICES='' "$python_bin" tests/dualcmp/check_task_acceptance.py --phase final --directory "$output_dir" --training-directory "$training_dir" --joint-directory "$joint_dir"
