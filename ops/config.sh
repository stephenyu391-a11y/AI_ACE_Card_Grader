# Shared settings for the ops scripts. Read by every other script; not run on its own.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the 44 project folder
VENV="$ROOT/.venv"
VAST="$VENV/bin/vastai"
KEY="$HOME/.ssh/vast_ed25519"                 # SSH key used for every rented machine
SSH_CONFIG="$HOME/.ssh/config.vast"           # host alias "gpu" points at the machine rented last
INSTANCE_FILE="$ROOT/ops/.instance"           # id of the machine rented last
REMOTE="/root/44"                             # project folder on the rented machine
IMAGE="pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime"   # CUDA 12.8 is needed for RTX 5090
SSH="ssh -F $SSH_CONFIG -o ConnectTimeout=20"
export RSYNC_RSH="$SSH"

instance_id() {
    [ -s "$INSTANCE_FILE" ] || { echo "no rented machine recorded in $INSTANCE_FILE" >&2; exit 1; }
    cat "$INSTANCE_FILE"
}

on_gpu() {   # run a command on the rented machine, without the Vast.ai greeting
    $SSH gpu "$@" 2> >(grep -v "^Welcome to vast.ai\|^Have fun" >&2)
}
