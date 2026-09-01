#!/usr/bin/env bash
# run axolotl in docker on the training box
# usage: ./docker-train.sh                     # train (detached tmux session)
#        ./docker-train.sh preprocess --debug  # loss-masking smoke test
set -euo pipefail
cd "$(dirname "$0")"

# dataset/prepared is gitignored; rebuild is deterministic (seeded)
[ -f dataset/prepared/train.jsonl ] || python3 prepare_data.py

cmd="${1:-train}"
shift || true

# training must outlive an ssh disconnect: re-launch inside a detached tmux
# session, logging to outputs/train.log; short commands stay in the foreground
if [ "$cmd" = train ] && [ -z "${TMUX:-}" ] && command -v tmux >/dev/null; then
  if tmux has-session -t cuad-train 2>/dev/null; then
    echo "session 'cuad-train' already exists; attach: tmux attach -t cuad-train"
    exit 1
  fi
  mkdir -p outputs
  tmux new-session -d -s cuad-train './docker-train.sh train 2>&1 | tee outputs/train.log'
  echo "training running in tmux session 'cuad-train'"
  echo "  watch: tmux attach -t cuad-train   (detach again: ctrl-b, then d)"
  echo "  log:   tail -f outputs/train.log"
  exit 0
fi

# --ipc=host: dataloader workers need more shared memory than docker's default
# hf cache mount: keeps the ~8GB model download across container runs
# explicit CDI device: --gpus all would also probe the AMD iGPU and fail
docker run --device nvidia.com/gpu=all --ipc=host --rm -it \
  -v "$PWD":/workspace/repo \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /workspace/repo \
  axolotlai/axolotl:main-latest \
  axolotl "$cmd" cuad-qlora.yaml "$@"
