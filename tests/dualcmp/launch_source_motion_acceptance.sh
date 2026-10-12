#!/usr/bin/env bash
# No status polling: first event here is the durable completion event.
set -euo pipefail
session=$1
output=$2
policy=$3
processes=$4
accumulation=$5
workers=$6
budget=$7
normalization=${8:-batch_norm_v1}
refresh_bn=${9:-0}
dropout=${10:-0.1}
# A matching directory or an old hash is not proof that sync finished.
python_bin=/home/qkb/miniconda3/envs/coop/bin/python
"$python_bin" tests/dualcmp/verify_training_sync.py \
    --check-proof "${DUALCMP_SYNC_PROOF:?verify both hosts before acceptance}"
[[ ! -e "$output" ]]
! tmux has-session -t "$session" 2>/dev/null
event_dir=$(mktemp -d /tmp/dualcmp-motion.XXXXXX)
mkfifo "$event_dir/ready" "$event_dir/startup" "$event_dir/complete" "$event_dir/first"
printf -v listener '%q tests/dualcmp/watch_forecast_host_training.py %q > %q 2>&1' \
    /home/qkb/miniconda3/envs/coop/bin/python "$event_dir" "$event_dir/events.log"
tmux new-session -d -s "$session" -n events -c "$PWD" "$listener"
IFS= read -r ready < "$event_dir/ready"
[[ "$ready" == ready ]]
printf -v job 'bash tests/dualcmp/run_source_motion_reference_acceptance.sh %q %q %q %q %q %q %q %q %q %q > %q 2>&1' \
    "$event_dir" "$output" "$policy" "$processes" "$accumulation" "$workers" "$budget" "$normalization" "$refresh_bn" "$dropout" "$event_dir/job.log"
tmux new-window -d -t "$session" -n training -c "$PWD" "$job"
printf 'Completion latch: %s\n' "$event_dir"
IFS= read -r complete < "$event_dir/first"
printf '%s\n' "$complete"
exit "$(< "$event_dir/exit_code")"
