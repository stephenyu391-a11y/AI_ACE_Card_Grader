#!/bin/bash
# Compare finished runs (already copied back by the watcher) in one table, plus the age-shortcut table.
#
#   ops/report.sh r1-base r4-seed1 own-views
set -euo pipefail
source "$(dirname "$0")/config.sh"
cd "$ROOT/grader" && "$VENV/bin/python" compare_runs.py "$@"
