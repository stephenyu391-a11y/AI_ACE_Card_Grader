#!/bin/bash
# Prepare this server for the SN44 grader: Python environment, libraries, Vast.ai tool, SSH key.
# Safe to run again; finished parts are skipped.
#
#   ops/setup_local.sh                 # everything
#   VAST_API_KEY=... ops/setup_local.sh  # also stores the Vast.ai key (otherwise give it later)
set -euo pipefail
source "$(dirname "$0")/config.sh"

# 1. Python environment. This server has no python3-venv package, so pip is added by hand.
if [ ! -x "$VENV/bin/pip" ]; then
    python3 -m venv --without-pip "$VENV"
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    "$VENV/bin/python" /tmp/get-pip.py -q
fi

# 2. Libraries. PyTorch for the processor only: this server has no graphics card.
"$VENV/bin/pip" install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/pip" install -q -r "$ROOT/grader/requirements.txt"
"$VENV/bin/pip" install -q reportlab vastai            # the work-log PDF and the Vast.ai tool

# 3. Pre-trained models. Downloaded only if missing.
for pair in "dinov2-base facebook/dinov2-base" "convnext-base-in22k timm/convnext_base.fb_in22k"; do
    set -- $pair
    if [ ! -s "$ROOT/models/$1/model.safetensors" ]; then
        "$VENV/bin/hf" download "$2" --local-dir "$ROOT/models/$1" --include "*.json" "model.safetensors"
    fi
done

# 4. Vast.ai key, readable only by this user. Never written into logs.
if [ -n "${VAST_API_KEY:-}" ]; then
    mkdir -p "$HOME/.config/vastai"
    (umask 077; printf '%s' "$VAST_API_KEY" > "$HOME/.config/vastai/vast_api_key")
fi

# 5. SSH key for the rented machines.
[ -f "$KEY" ] || ssh-keygen -q -t ed25519 -N '' -C sn44-vast -f "$KEY"

# 6. Check.
"$VENV/bin/python" -c "import torch, transformers, timm, numpy; print('torch', torch.__version__, '| transformers', transformers.__version__, '| timm', timm.__version__)"
ls "$ROOT/models"
if [ -s "$HOME/.config/vastai/vast_api_key" ]; then
    "$VAST" show user --raw | "$VENV/bin/python" -c "import json,sys; print('Vast.ai credit:', round(json.load(sys.stdin)['credit'], 2), 'dollars')"
else
    echo "No Vast.ai key yet: run again with VAST_API_KEY=... set"
fi
