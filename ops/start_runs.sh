#!/bin/bash
# Start a run list on the rented machine (inside tmux, so it survives a lost connection),
# and start the watcher here that copies results back and stops the machine at the end.
#
#   ops/start_runs.sh grader/session2.txt
#   ops/start_runs.sh grader/session2.txt --no-stop     # keep the machine running afterwards
#
# A run list has one train.py command per line, starting with --name (see grader/session1.txt).
set -euo pipefail
source "$(dirname "$0")/config.sh"
list="${1:?give a run list, e.g. grader/session2.txt}"
name=$(basename "$list" .txt)
rsync -a "$ROOT/$list" "gpu:$REMOTE/grader/"
on_gpu "cd $REMOTE/grader && mkdir -p out && tmux new-session -d -s $name 'python run_queue.py $name.txt 2>&1 | tee -a out/queue_$name.log'"
echo "started $name on machine $(instance_id); progress: ops/status.sh $name"
stop=yes; [ "${2:-}" = --no-stop ] && stop=no
setsid nohup "$ROOT/ops/watch.sh" "$name" "$stop" > "$ROOT/grader/out/watch_$name.log" 2>&1 < /dev/null &
echo "watcher started here (log: grader/out/watch_$name.log); it stops the machine at the end: $stop"
