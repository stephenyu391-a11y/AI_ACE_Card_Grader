#!/bin/bash
# List rentable machines that meet the project's needs, cheapest per unit of work first.
#
#   ops/vast_find.sh                     # RTX 5090 and RTX 4090
#   ops/vast_find.sh RTX_4090 RTX_3090   # other cards
set -euo pipefail
source "$(dirname "$0")/config.sh"
cards="${*:-RTX_5090 RTX_4090}"
"$VAST" search offers "gpu_name in [${cards// /,}] num_gpus=1 verified=true rentable=true reliability>0.99 \
    cpu_cores_effective>=12 cpu_ram>=64 disk_space>=100 inet_up>=500 inet_down>=500" \
    -o 'dph+' --storage 100 --raw | "$VENV/bin/python" "$ROOT/ops/offers_table.py"
