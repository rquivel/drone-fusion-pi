#!/usr/bin/env bash
# One-shot installer for drone-fusion-pi on a fresh DietPi / Pi 5.
# Idempotent: safe to re-run. Mirrors the "One-time setup" section of the README.
#
# Usage:
#   scripts/install.sh                  # apt deps + venv + python deps
#   scripts/install.sh --with-service   # also install + enable systemd units
#   scripts/install.sh --skip-apt       # skip apt step (e.g. on non-Debian dev box)

set -euo pipefail

WITH_SERVICE=0
SKIP_APT=0
for arg in "$@"; do
    case "$arg" in
        --with-service) WITH_SERVICE=1 ;;
        --skip-apt)     SKIP_APT=1 ;;
        -h|--help)
            sed -n '2,9p' "$0"
            exit 0
            ;;
        *)
            echo "unknown flag: $arg" >&2
            exit 2
            ;;
    esac
done

# Resolve repo root from this script's location, so it works no matter where
# you invoke it from.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

log() { printf '\n=== %s ===\n' "$*"; }

if [[ $SKIP_APT -eq 0 ]]; then
    log "installing apt packages"
    sudo apt update
    sudo apt install -y \
        python3-venv python3-pip python3-dev \
        build-essential swig liblgpio-dev \
        portaudio19-dev libsndfile1 \
        libgl1 libglib2.0-0
else
    log "skipping apt step (--skip-apt)"
fi

if [[ ! -d venv ]]; then
    log "creating venv"
    python3 -m venv venv
else
    log "venv already exists, reusing"
fi

# Use the venv's binaries directly so this script works even if the caller's
# shell hasn't sourced activate.
PIP="$REPO_ROOT/venv/bin/pip"
PY="$REPO_ROOT/venv/bin/python"

log "upgrading pip + wheel"
"$PIP" install --upgrade pip wheel

# Pin torch to the last CPU-only aarch64 build before 2.7 (Jetson/CUDA) wheels
# started shipping. See README step 4 for the full reasoning.
log "installing pinned torch (CPU-only, 2.6.x)"
"$PIP" install "torch==2.6.*" "torchaudio==2.6.*" "torchvision==0.21.*"

log "installing requirements.txt"
"$PIP" install -r requirements.txt

log "verifying torch imports without CUDA"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

if [[ $WITH_SERVICE -eq 1 ]]; then
    log "installing systemd units"
    sudo cp systemd/drone-fusion.service          /etc/systemd/system/
    sudo cp systemd/drone-fusion-selftest.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable drone-fusion-selftest.service
    sudo systemctl enable drone-fusion.service
    echo "service installed but not started — run:"
    echo "  sudo systemctl start drone-fusion.service"
fi

log "done"
echo "next: drop weights into models/ then run scripts/sanity_check.py"
