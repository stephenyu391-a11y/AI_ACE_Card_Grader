#!/bin/bash
# Copy code, models and pictures to the rented machine, install the libraries there,
# and check that every picture arrived (same names and sizes on both sides).
#
#   ops/upload.sh            # everything (pictures take 10 to 35 minutes)
#   ops/upload.sh code       # only the grader code, e.g. after a fix
set -euo pipefail
source "$(dirname "$0")/config.sh"
on_gpu "mkdir -p $REMOTE/models $REMOTE/Data"
rsync -a --exclude out --exclude __pycache__ "$ROOT/grader" "gpu:$REMOTE/"
[ "${1:-}" = code ] && { echo "code copied"; exit 0; }

rsync -a --exclude '*.md' "$ROOT/models/dinov2-base" "$ROOT/models/convnext-base-in22k" "gpu:$REMOTE/models/"
on_gpu "cd $REMOTE/grader && pip install -q --root-user-action=ignore -r requirements.txt && which tmux > /dev/null || (apt-get update -qq && apt-get install -y -qq tmux > /dev/null)"
rsync -a --info=progress2 --exclude original "$ROOT/Data/" "gpu:$REMOTE/Data/"

here=$(cd "$ROOT/Data" && find . -name '*.png' -printf '%P %s\n' | sort | md5sum)
there=$(on_gpu "cd $REMOTE/Data && find . -name '*.png' -printf '%P %s\n' | sort | md5sum")
[ "$here" = "$there" ] && echo "all pictures arrived" || { echo "PICTURES DIFFER: run ops/upload.sh again"; exit 1; }
