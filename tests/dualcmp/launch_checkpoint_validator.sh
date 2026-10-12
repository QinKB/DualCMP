#!/usr/bin/env bash
# Reserve the inotify listener before the producer is allowed to start.
set -euo pipefail
session=$1
root=$2
output=$3
proof=$4
mode=$5
remote=${6:-lab-lan}
batch=${7:-2}
[[ "$mode" == preflight || "$mode" == full ]]
[[ ! -e "$output" ]] || { printf 'Validation output already exists: %s\n' "$output" >&2; exit 2; }
! tmux has-session -t "$session" 2>/dev/null
mkdir -p "$root/checkpoint_events"
event_dir=$(mktemp -d /tmp/dualcmp-validator.XXXXXX)
mkfifo "$event_dir/ready" "$event_dir/startup" "$event_dir/complete" "$event_dir/first" "$event_dir/validator_ready"
python_bin=/home/qkb/miniconda3/envs/coop/bin/python
printf -v listener '%q tests/dualcmp/watch_forecast_host_training.py %q > %q 2>&1' \
    "$python_bin" "$event_dir" "$event_dir/listener.log"
tmux new-session -d -s "$session" -n events -c "$PWD" "$listener"
tmux set-option -t "$session" remain-on-exit on
IFS= read -r ready < "$event_dir/ready"
[[ "$ready" == ready ]]
printf -v job 'bash tests/dualcmp/run_checkpoint_validator.sh %q %q %q %q %q %q %q > %q 2>&1' \
    "$event_dir" "$root" "$output" "$proof" "$mode" "$remote" "$batch" "$event_dir/job.log"
tmux new-window -d -t "$session" -n validate -c "$PWD" "$job"
# The job sends startup only after inotify registration; failure also wakes first.
IFS= read -r first < "$event_dir/first"
printf 'tmux_session=%s\nevent_dir=%s\noutput_dir=%s\n%s\n' "$session" "$event_dir" "$output" "$first"
[[ "$first" == *'validator_ready'* ]]
