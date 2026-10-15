#!/bin/bash
# Stop or destroy the machine rented last.
#   stop     keeps its disk (pictures stay; about 0.03 dollars per hour). It may not restart if
#            someone else rents its card meanwhile; then destroy it and rent another.
#   destroy  deletes the machine and its disk. Billing ends. First copies everything back.
#
#   ops/vast_end.sh stop
#   ops/vast_end.sh destroy
set -euo pipefail
source "$(dirname "$0")/config.sh"
id=$(instance_id)
case "${1:?stop or destroy}" in
    stop)    "$VAST" stop instance "$id" ;;
    destroy) if $SSH -o BatchMode=yes gpu true 2>/dev/null; then
                 rsync -a --exclude '*.tmp' --exclude last.pt "gpu:$REMOTE/grader/out/" "$ROOT/grader/out/" && echo "results copied back"
             else
                 echo "machine not reachable: nothing more copied (the watcher copies at the end of every run list)"
             fi
             read -r -p "destroy machine $id and its disk? [y/N] " answer
             [ "$answer" = y ] && "$VAST" destroy instance "$id" -y && rm -f "$INSTANCE_FILE" ;;
    *)       echo "stop or destroy"; exit 1 ;;
esac
"$VAST" show instances --raw | "$VENV/bin/python" -c "import json,sys; print('machines rented now:', [(d['id'], d.get('actual_status')) for d in json.load(sys.stdin)])"
