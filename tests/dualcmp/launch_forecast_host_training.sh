#!/usr/bin/env bash
# Registers durable event latches, launches one full job, returns after update 1.
set -euo pipefail
session=$1
output=$2
policy=$3
processes=$4
accumulation=$5
workers=$6
mode=${7:-full}
resume_checkpoint=${8:-}
case "$mode" in preflight|full) ;; *) exit 2 ;; esac
[[ ! -e "$output" ]] || { printf 'Output already exists: %s\n' "$output" >&2; exit 2; }
if tmux has-session -t "$session" 2>/dev/null; then
    printf 'Session already exists: %s\n' "$session" >&2
    exit 2
fi
if [[ -n "$resume_checkpoint" ]]; then
    [[ "$mode" == full && -f "$resume_checkpoint" ]] || exit 2
    mkdir "$output"
    cp --reflink=auto -- "$resume_checkpoint" "$output/checkpoint_resume.pt"
    resume_checkpoint="$output/checkpoint_resume.pt"
fi
event_dir=$(mktemp -d "/tmp/dualcmp-${policy}.XXXXXX")
mkfifo "$event_dir/ready" "$event_dir/startup" "$event_dir/complete" "$event_dir/first"
python_bin=/home/qkb/miniconda3/envs/coop/bin/python
printf -v listener '%q tests/dualcmp/watch_forecast_host_training.py %q > %q 2>&1' \
    "$python_bin" "$event_dir" "$event_dir/listener.log"
tmux new-session -d -s "$session" -n events -c "$PWD" "$listener"
tmux set-option -t "$session" remain-on-exit on
IFS= read -r ready < "$event_dir/ready"
[[ "$ready" == ready ]]
devices=0
[[ "$processes" != 2 ]] || devices=0,1
printf -v job 'set -o pipefail; env CUDA_VISIBLE_DEVICES=%q DUALCMP_STARTUP_FIFO=%q bash tests/dualcmp/run_forecast_host_training.sh %q %q %q %q %q %q %q %q 2>&1 | tee %q' \
    "$devices" "$event_dir/startup" "$event_dir" "$output" "$policy" \
    "$processes" "$accumulation" "$workers" "$mode" "$resume_checkpoint" "$event_dir/job.log"
tmux new-window -d -t "$session" -n train -c "$PWD" "$job"
printf 'tmux_session=%s\nevent_dir=%s\noutput_dir=%s\n' "$session" "$event_dir" "$output"
IFS= read -r first < "$event_dir/first"
printf '%s\n' "$first"
# Recovery reads first_event.json/completed.json or exit_code, not spent FIFOs.
