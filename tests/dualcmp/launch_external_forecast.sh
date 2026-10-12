#!/usr/bin/env bash
# Start the lab-lan forecast job after registering durable startup/completion latches.
set -euo pipefail
if [[ $# != 5 ]]; then
    printf 'usage: %s SESSION OUTPUT WORKERS MODE SYNC_PROOF\n' "$0" >&2
    exit 2
fi
session=$1
output=$(realpath -- "$2")
workers=$3
mode=$4
sync_proof=$(realpath -- "$5")
case "$mode" in preflight|full) ;; *) exit 2 ;; esac
[[ "$workers" =~ ^[0-9]+$ ]] && (( workers >= 0 )) || exit 2
[[ -d "$output" && -d "$output/checkpoint_events" && -f "$sync_proof" ]] || {
    printf 'Output reservation and sync proof must exist: %s\n' "$output" >&2
    exit 2
}
if find "$output/checkpoint_events" -mindepth 1 -print -quit | grep -q .; then
    printf 'Checkpoint event directory is not empty: %s\n' "$output/checkpoint_events" >&2
    exit 2
fi
shopt -s nullglob dotglob
for path in "$output"/*; do
    [[ "$path" == "$output/checkpoint_events" || "$path" == "$sync_proof" ]] || {
        printf 'Output already contains a training artifact: %s\n' "$path" >&2
        exit 2
    }
done
python_bin=/home/qkb/miniconda3/envs/coop/bin/python
"$python_bin" tests/dualcmp/verify_training_sync.py --check-proof "$sync_proof"
if tmux has-session -t "$session" 2>/dev/null; then
    printf 'Session already exists: %s\n' "$session" >&2
    exit 2
fi

event_dir=$(mktemp -d /tmp/dualcmp-external-forecast.XXXXXX)
mkfifo "$event_dir/ready" "$event_dir/startup" "$event_dir/complete" "$event_dir/first"
printf -v listener '%q tests/dualcmp/watch_forecast_host_training.py %q > %q 2>&1' \
    "$python_bin" "$event_dir" "$event_dir/listener.log"
tmux new-session -d -s "$session" -n events -c "$PWD" "$listener"
tmux set-option -t "$session" remain-on-exit on
IFS= read -r ready < "$event_dir/ready"
[[ "$ready" == ready ]]
printf -v job 'set -o pipefail; env CUDA_VISIBLE_DEVICES=0,1 DUALCMP_STARTUP_FIFO=%q bash tests/dualcmp/run_external_forecast.sh %q %q %q %q 2>&1 | tee %q' \
    "$event_dir/startup" "$event_dir" "$output" "$workers" "$mode" "$event_dir/job.log"
tmux new-window -d -t "$session" -n train -c "$PWD" "$job"
printf 'tmux_session=%s\nevent_dir=%s\noutput_dir=%s\n' "$session" "$event_dir" "$output"
IFS= read -r first < "$event_dir/first"
if ! printf '%s\n' "$first" | "$python_bin" -c 'import json, sys; event=json.load(sys.stdin); raise SystemExit(0 if event.get("status") == "first_optimizer_step_complete" else 1)'; then
    printf 'Training finished before its first optimizer step: %s\n' "$first" >&2
    exit 1
fi
printf '%s\n' "$first"
# Recovery reads first_event.json/completed.json and exit_code, never a spent FIFO.
