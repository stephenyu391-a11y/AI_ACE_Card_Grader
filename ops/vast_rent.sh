#!/bin/bash
# Rent one machine (an offer id from vast_find.sh), wait until it runs, give it our SSH key,
# and point the SSH alias "gpu" at it. Starts billing.
#
#   ops/vast_rent.sh 41246684 [label]
set -euo pipefail
source "$(dirname "$0")/config.sh"
offer="${1:?give an offer id from ops/vast_find.sh}"
label="${2:-sn44}"

id=$("$VAST" create instance "$offer" --image "$IMAGE" --disk 100 --ssh --direct --label "$label" --raw \
     | "$VENV/bin/python" -c "import json,sys; d=json.load(sys.stdin); print(d['new_contract'] if d.get('success') else '')")
[ -n "$id" ] || { echo "renting failed (the offer may be gone; run vast_find.sh again)"; exit 1; }
echo "$id" > "$INSTANCE_FILE"
echo "rented machine $id; waiting for it to start (a host that is still loading after 10 minutes is broken: destroy it)"

for i in $(seq 1 45); do
    read -r status ip port < <("$VAST" show instance "$id" --raw | "$VENV/bin/python" -c "
import json,sys; d=json.load(sys.stdin)
print(d.get('actual_status'), d.get('public_ipaddr') or '-', ((d.get('ports') or {}).get('22/tcp') or [{}])[0].get('HostPort') or '-')")
    [ "$status" = running ] && [ "$port" != - ] && break
    sleep 20
done
[ "$status" = running ] || { echo "machine $id did not start; ops/vast_end.sh destroy"; exit 1; }

"$VAST" attach ssh "$id" "$(cat "$KEY.pub")" > /dev/null   # the account-level key alone did not reach the machine
cat > "$SSH_CONFIG" <<CFG
Host gpu
  HostName $ip
  Port $port
  User root
  IdentityFile $KEY
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
  ServerAliveInterval 30
  ServerAliveCountMax 6
CFG
chmod 600 "$SSH_CONFIG"
for i in $(seq 1 20); do $SSH -o BatchMode=yes gpu true 2>/dev/null && break; sleep 10; done
on_gpu 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; python -c "import torch; print(\"torch\", torch.__version__, \"cuda ok:\", torch.cuda.is_available())"'
echo "ready: ssh -F $SSH_CONFIG gpu"
