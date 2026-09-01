#!/usr/bin/env bash
# run axolotl in docker on the training box
# usage: ./docker-train.sh                     # train
#        ./docker-train.sh preprocess --debug  # loss-masking smoke test
set -euo pipefail
cd "$(dirname "$0")"

# dataset/prepared is gitignored; rebuild is deterministic (seeded)
[ -f dataset/prepared/train.jsonl ] || python3 prepare_data.py

cmd="${1:-train}"
shift || true

# --ipc=host: dataloader workers need more shared memory than docker's default
# hf cache mount: keeps the ~8GB model download across container runs
docker run --gpus all --ipc=host --rm -it \
  -v "$PWD":/workspace/repo \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace/repo \
  axolotlai/axolotl:main-latest \
  axolotl "$cmd" cuad-qlora.yaml "$@"
