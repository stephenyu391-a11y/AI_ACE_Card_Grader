#!/bin/bash
# How far the runs are: the run list, the newest run's passes, how busy the card is, the credit.
#
#   ops/status.sh session2
set -euo pipefail
source "$(dirname "$0")/config.sh"
name="${1:?give the run list name, e.g. session2}"
on_gpu "cd $REMOTE/grader/out && tail -4 queue_$name.log; run=\$(ls -td */ | head -1); echo \"--- \$run\"; cat \${run}fold0/history.csv 2>/dev/null | cut -d, -f1-4,10; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
"$VAST" show user --raw | "$VENV/bin/python" -c "import json,sys; print('credit:', round(json.load(sys.stdin)['credit'], 2), 'dollars')"
