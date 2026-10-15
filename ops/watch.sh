#!/bin/bash
# Started by start_runs.sh; runs on this server, apart from any chat or terminal.
# Every 5 minutes it copies the results back (without model files). When the run list has
# printed its summary, it copies everything including the models, then stops the machine.
set -uo pipefail
source "$(dirname "$0")/config.sh"
name="$1"; stop="${2:-yes}"
out="$ROOT/grader/out/"
while true; do
    rsync -a --exclude '*.pt' --exclude '*.tmp' "gpu:$REMOTE/grader/out/" "$out" 2>/dev/null
    if grep -q '^Summary' "$out/queue_$name.log" 2>/dev/null; then
        rsync -a --exclude '*.tmp' --exclude last.pt "gpu:$REMOTE/grader/out/" "$out" && echo "results and models copied"
        [ "$stop" = yes ] && "$VAST" stop instance "$(instance_id)"
        echo "run list $name finished at $(date -u '+%Y-%m-%d %H:%M UTC'); machine stopped: $stop"
        exit 0
    fi
    sleep 300
done
