#!/usr/bin/env bash
# Register the completion listener before starting the detached evaluation.
set -euo pipefail
session=$1
checkpoint=$2
output=$3
scope=$4
batch=$5
workers=$6
[[ "$scope" == preflight || "$scope" == full || "$scope" == audit || "$scope" == iou05-preflight || "$scope" == iou05-full || "$scope" == external-train-window ]]
if [[ "$scope" == external-train-window ]]; then
    /home/qkb/miniconda3/envs/coop/bin/python tests/dualcmp/verify_training_sync.py --check-proof "${7:?sync proof required}"
fi
[[ -f "$checkpoint" && ! -e "$output" && ! -e "${output%.json}.csv" ]]
if tmux has-session -t "$session" 2>/dev/null; then exit 2; fi
event_dir=$(mktemp -d /tmp/dualcmp-cmp-eval.XXXXXX)
mkfifo "$event_dir/ready" "$event_dir/startup" "$event_dir/complete" "$event_dir/first"
python_bin=/home/qkb/miniconda3/envs/coop/bin/python
printf -v listener '%q tests/dualcmp/watch_forecast_host_training.py %q > %q 2>&1' \
    "$python_bin" "$event_dir" "$event_dir/listener.log"
tmux new-session -d -s "$session" -n events -c "$PWD" "$listener"
tmux set-option -t "$session" remain-on-exit on
IFS= read -r ready < "$event_dir/ready"
[[ "$ready" == ready ]]
printf -v job 'bash tests/dualcmp/run_cmp_evaluation.sh %q %q %q %q %q %q > %q 2>&1' \
    "$event_dir" "$checkpoint" "$output" "$scope" "$batch" "$workers" "$event_dir/job.log"
tmux new-window -d -t "$session" -n evaluate -c "$PWD" "$job"
printf 'tmux_session=%s\nevent_dir=%s\noutput=%s\n' "$session" "$event_dir" "$output"
IFS= read -r complete < "$event_dir/first"
printf '%s\n' "$complete"
