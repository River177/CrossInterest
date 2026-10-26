#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="Qwen/Qwen3.5-9B"
NUM_GPUS="${SEMANTIC_ID_NUM_GPUS:-4}"
SOURCE_ROOT="${SEMANTIC_ID_SOURCE_ROOT:?SEMANTIC_ID_SOURCE_ROOT is required}"
SOURCE_RESULT_ID="${SEMANTIC_ID_SOURCE_RESULT_ID:?SEMANTIC_ID_SOURCE_RESULT_ID is required}"
STALE_IDENTITY="${SEMANTIC_ID_STALE_IDENTITY:?SEMANTIC_ID_STALE_IDENTITY is required}"

if [[ "$NUM_GPUS" != "4" ]]; then
  echo "Semantic-ID evaluation recovery requires four GPUs; got $NUM_GPUS" >&2
  exit 2
fi
if [[ "$(basename "$SOURCE_ROOT")" != "$SOURCE_RESULT_ID" ]]; then
  echo "Semantic-ID source result path does not match its pinned identity" >&2
  exit 2
fi

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME="${HF_HOME:-/mnt/default/huggingface-cache}"
mkdir -p "$HF_HOME"

python - "$MODEL_NAME" <<'PY'
import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

name = sys.argv[1]
token = os.environ.get("HF_TOKEN")


def validate_snapshot(value: str) -> Path:
    path = Path(value)
    weight_files: set[str] = set()
    for index_path in (
        path / "model.safetensors.index.json",
        path / "pytorch_model.bin.index.json",
    ):
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_files.update(payload.get("weight_map", {}).values())
    if not weight_files:
        raise RuntimeError("Cached Qwen snapshot has no sharded weight index")
    missing = [
        value
        for value in sorted(weight_files)
        if not (path / value).is_file() or (path / value).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            "Cached Qwen snapshot has missing weights: " + ", ".join(missing[:5])
        )
    AutoConfig.from_pretrained(path, local_files_only=True)
    AutoTokenizer.from_pretrained(path, local_files_only=True)
    return path


try:
    snapshot = validate_snapshot(snapshot_download(name, local_files_only=True))
    source = "persistent-cache"
except Exception:
    snapshot = validate_snapshot(snapshot_download(name, token=token))
    source = "downloaded-or-repaired"
print("SEMANTIC_ID_EVAL_MODEL_READY", name, snapshot, source)
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DATA_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_semantic_id_token_200_h50_v1"
EVAL_CONFIG="$AMLT_CODE_DIR/configs/id_token/static_sud_qwen35_interest_full.json"
EVAL_FILE="$DATA_ROOT/eval/interest_only/dev_full_grouped.parquet"
MAPPING_FILE="$DATA_ROOT/interest_token_mapping.parquet"
ADAPTER_PATH="$SOURCE_ROOT/sft/final_adapter"
RECOVERY_ROOT="$AMLT_OUTPUT_DIR/recovery"
EVAL_ROOT="$RECOVERY_ROOT/eval"

for path in \
  "$SOURCE_ROOT/sft/run_summary.json" \
  "$ADAPTER_PATH/adapter_model.safetensors" \
  "$ADAPTER_PATH/adapter_config.json" \
  "$EVAL_CONFIG" \
  "$EVAL_FILE" \
  "$MAPPING_FILE"; do
  if [[ ! -s "$path" ]]; then
    echo "Required recovery input is missing or empty: $path" >&2
    exit 1
  fi
done

mkdir -p "$RECOVERY_ROOT"
python -m new_interest.experiments.prepare_semantic_id_token_eval_recovery \
  --source-root "$SOURCE_ROOT" \
  --expected-source-result-id "$SOURCE_RESULT_ID" \
  --expected-stale-identity "$STALE_IDENTITY" \
  --output "$RECOVERY_ROOT/stale_cache_cleanup_manifest.json"

python - "$SOURCE_ROOT/sft/run_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
checks = {
    "expected_optimizer_steps": 2025,
    "train_samples": 21600,
    "world_size": 4,
}
for key, expected in checks.items():
    actual = int(summary.get(key, -1))
    if actual != expected:
        raise RuntimeError(f"Source training gate failed for {key}: {actual}")
if float(summary.get("configured_epochs", -1)) != 3.0:
    raise RuntimeError("Source training did not complete exactly three epochs")
print("SEMANTIC_ID_SOURCE_TRAINING_GATE_OK", checks)
PY

mkdir -p "$EVAL_ROOT"
PIDS=()
SHARD_DIRS=()
for SHARD_INDEX in 0 1 2 3; do
  SHARD_NAME="$(printf 'shard-%02d' "$SHARD_INDEX")"
  SHARD_DIR="$EVAL_ROOT/$SHARD_NAME"
  SHARD_DIRS+=("$SHARD_DIR")
  (
    export CUDA_VISIBLE_DEVICES="$SHARD_INDEX"
    python -m new_interest.evaluation.evaluate_id_token_ranking \
      --config "$EVAL_CONFIG" \
      --adapter-path "$ADAPTER_PATH" \
      --eval-file "$EVAL_FILE" \
      --interest-token-mapping-file "$MAPPING_FILE" \
      --output-dir "$SHARD_DIR" \
      --run-name "static-sud-semantic-id-qwen35-interest-eval-recovery-$SHARD_NAME" \
      --num-shards 4 \
      --shard-index "$SHARD_INDEX" \
      >"$EVAL_ROOT/${SHARD_NAME}.log" 2>&1
  ) &
  PIDS+=("$!")
done

EVAL_FAILURE=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    EVAL_FAILURE=1
  fi
done
if [[ "$EVAL_FAILURE" -ne 0 ]]; then
  echo "At least one recovered ID-token evaluation shard failed" >&2
  for log in "$EVAL_ROOT"/shard-*.log; do
    echo "===== $log =====" >&2
    tail -n 80 "$log" >&2 || true
  done
  exit 1
fi

python -m new_interest.evaluation.merge_id_token_shards \
  --shard-dirs "${SHARD_DIRS[@]}" \
  --eval-file "$EVAL_FILE" \
  --interest-token-mapping-file "$MAPPING_FILE" \
  --output-dir "$EVAL_ROOT/merged" \
  --expected-records 31392

python - "$EVAL_ROOT/merged/metrics.json" "$EVAL_FILE" <<'PY'
import json
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {
    "samples": 31392,
    "supported_interest_count": 104,
    "label_vocabulary_size": 108,
    "top_k": 20,
}
for key, value in expected.items():
    if int(metrics.get(key, -1)) != value:
        raise RuntimeError(f"Recovered evaluation gate failed for {key}")
if metrics.get("history_prediction_filtering") is not False:
    raise RuntimeError("Recovered evaluation unexpectedly filtered History labels")
eval_rows = pq.read_table(
    Path(sys.argv[2]), columns=["acceptable_subcategories"]
).column(0)
target_occurrences = sum(len(value.as_py()) for value in eval_rows)
if target_occurrences != 37728:
    raise RuntimeError(
        "Recovered evaluation input does not contain 37,728 targets: "
        f"{target_occurrences}"
    )
for key in ("interest_macro_recall_at_10", "interest_macro_ndcg_at_10"):
    if not math.isfinite(float(metrics[key])):
        raise RuntimeError(f"Recovered evaluation produced non-finite {key}")
print(
    "SEMANTIC_ID_EVAL_RECOVERY_GATE_OK",
    metrics["interest_macro_recall_at_10"],
    metrics["interest_macro_ndcg_at_10"],
    target_occurrences,
)
PY
