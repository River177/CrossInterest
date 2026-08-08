#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-artifacts/gcr/qwen3}"

find "$ROOT" -type d -name 'offline-run-*' -print0 |
  while IFS= read -r -d '' run_dir; do
    marker="$run_dir/.synced"
    if [[ -f "$marker" ]]; then
      continue
    fi
    wandb sync "$run_dir"
    touch "$marker"
  done
