#!/usr/bin/env bash
# Durable callback supervisor. No status/GPU/log polling is performed.
set -euo pipefail
output_dir=$1
event_dir=$2
code_sha=$3
python_bin=${DUALCMP_PYTHON:-/home/qkb/miniconda3/envs/coop/bin/python}
for fifo in lifecycle startup complete; do
    test -p "$event_dir/$fifo"
done
test ! -e "$output_dir"
# Keep both ends open before the worker starts. Lifecycle writes are buffered
# even if an event arrives before this supervisor enters its blocking read.
exec 3<> "$event_dir/lifecycle"
(
    finish_worker() {
        result=$?
        printf '{"status": "training_completed", "exit_code": %s}\n' "$result" >&3
    }
    trap finish_worker EXIT
    export PYTHONPATH=.
    export CUDA_VISIBLE_DEVICES=0,1
    export OMP_NUM_THREADS=4
    export V2V4REAL_ROOT=../data/V2V4Real
    export DUALCMP_TRACKING_LABEL_ROOT=../Reference/Code/CMP/AB3Dmot/scripts/KITTI/v2v4real_label/test
    export DUALCMP_TRAIN_TRACKING_LABEL_ROOT=runs/labels-aligned-cmpids-v2-20260930/v2v4real_train_label
    export DUALCMP_TRACKER_WEIGHTS=runs/train-a-native/checkpoint_epoch_003.pt
    export DUALCMP_CODE_SHA="$code_sha"
    export DUALCMP_STARTUP_FIFO="$event_dir/lifecycle"
    unset DUALCMP_MTR_WEIGHTS
    test "$(git rev-parse HEAD)" = "$code_sha"
    "$python_bin" -m torch.distributed.run --standalone --nproc_per_node=2 \
        -m opencood.tools.dualcmp.train_forecast_ddp \
        --output-dir "$output_dir" --full-data \
        --forecast-representation source_temporal_aligned --partner-delay-ms 100 \
        --intention-points-file runs/intention-aligned-cmpids-v2-20260930/cluster_64.pkl \
        --epochs 30 --batch-size 2 --workers 8 --lr 1e-4 --weight-decay 1e-4 \
        --forecast-mean-weight 5 --save-every 1 --eval-every 1 --log-every 100
) > "$event_dir/train.log" 2>&1 &
worker_pid=$!
started=false
while IFS= read -r message <&3; do
    case "$message" in
        *'"status": "first_optimizer_step_complete"'*)
            started=true
            "$python_bin" -c 'import json,sys; r=json.loads(sys.argv[1]); r.update(source_sha=sys.argv[2], output_dir=sys.argv[3]); print(json.dumps(r))' \
                "$message" "$code_sha" "$output_dir" > "$event_dir/startup.json"
            cp "$event_dir/startup.json" "$output_dir/startup.json"
            mv "$event_dir/train.log" "$output_dir/train.log"
            # Blocking writer latches readiness if the caller is not ready yet.
            printf '%s\n' "$(< "$event_dir/startup.json")" > "$event_dir/startup"
            ;;
        *'"status": "training_completed"'*)
            result=$("$python_bin" -c 'import json,sys; print(json.loads(sys.argv[1])["exit_code"])' "$message")
            wait "$worker_pid" || true
            mkdir -p "$output_dir"
            printf '%s\n' "$result" > "$event_dir/exit_code"
            printf '%s\n' "$message" > "$event_dir/completion.json"
            cp "$event_dir/completion.json" "$output_dir/completion.json"
            if ! "$started"; then
                printf '{"status": "failed_before_first_optimizer_step", "exit_code": %s}\n' "$result" > "$event_dir/startup.json"
                cp "$event_dir/startup.json" "$output_dir/startup.json"
                mv "$event_dir/train.log" "$output_dir/train.log"
                printf '%s\n' "$(< "$event_dir/startup.json")" > "$event_dir/startup"
            fi
            # Caller may exit after startup; this blocks until a later listener
            # consumes completion, preserving a race-free completion latch.
            printf '%s\n' "$message" > "$event_dir/complete"
            exit "$result"
            ;;
    esac
done
