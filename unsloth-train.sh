#!/usr/bin/env bash
# unsloth pipeline on the training box (venv, no docker)
# usage: ./unsloth-train.sh setup   # one-time venv + install
#        ./unsloth-train.sh check   # masking verification, must pass first
#        ./unsloth-train.sh train   # training (run inside tmux yourself!)
set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv-unsloth

# dataset/prepared is gitignored; rebuild is deterministic (seeded)
[ -f dataset/prepared/train.jsonl ] || python3 prepare_data.py

case "${1:-train}" in
  setup)
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install unsloth
    ;;
  check)
    "$VENV/bin/python" check_masking.py
    ;;
  train)
    # the axolotl container and this run cannot share the 4090
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^cuad-axolotl$'; then
      echo "cuad-axolotl container is still training; refusing to fight it for the GPU"
      exit 1
    fi
    "$VENV/bin/python" train_unsloth.py
    ;;
  *)
    echo "usage: $0 {setup|check|train}"
    exit 1
    ;;
esac
