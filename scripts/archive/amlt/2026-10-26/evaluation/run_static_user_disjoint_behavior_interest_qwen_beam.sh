#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${STATIC_SUD_NUM_GPUS:-4}"
if [[ "$NUM_GPUS" != "4" ]]; then
  echo "Behavior+Interest Qwen full Beam evaluation requires exactly 4 GPUs" >&2
  exit 2
fi

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true

DATASET_DIR="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_balanced_200_h50_behavior_interest_v1"
EVAL_FILE="$DATASET_DIR/eval/behavior_interest/dev_full_grouped.parquet"
LABELS_FILE="$AMLT_CODE_DIR/data/training/generative_new_interest_static_user_disjoint_balanced_200_h50_v1/retained_interests.parquet"
BEAM_CONFIG="$AMLT_CODE_DIR/configs/beam/static_sud_qwen35_behavior_interest_full.json"
SOURCE_SFT_DIR="${AMLT_MAP_INPUT_DIR:?AMLT_MAP_INPUT_DIR is required}/behavior_interest_serial/qwen35/sft"
ADAPTER_PATH="$SOURCE_SFT_DIR/final_adapter"
PARALLEL_ROOT="$AMLT_OUTPUT_DIR/behavior_interest_parallel/qwen35"
BEAM_ROOT="$PARALLEL_ROOT/beam"
RUN_NAME="static-sud-qwen35-behavior-interest-21600-seed42-beam-full-g4"
mkdir -p "$BEAM_ROOT"

for required in \
  "$EVAL_FILE" \
  "$LABELS_FILE" \
  "$BEAM_CONFIG" \
  "$DATASET_DIR/statistics.json" \
  "$DATASET_DIR/metadata.json" \
  "$SOURCE_SFT_DIR/run_summary.json" \
  "$ADAPTER_PATH/adapter_model.safetensors" \
  "$ADAPTER_PATH/adapter_config.json"; do
  if [[ ! -s "$required" ]]; then
    echo "Required Behavior+Interest Qwen Beam input is missing or empty: $required" >&2
    exit 1
  fi
done

python - "$DATASET_DIR" "$EVAL_FILE" "$LABELS_FILE" "$PARALLEL_ROOT/dataset_audit.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

dataset_dir = Path(sys.argv[1])
eval_file = Path(sys.argv[2])
labels_file = Path(sys.argv[3])
output = Path(sys.argv[4])
statistics = json.loads((dataset_dir / "statistics.json").read_text())
metadata = json.loads((dataset_dir / "metadata.json").read_text())

expected_dev = {
    "impressions": 31392,
    "users": 27082,
    "target_occurrences": 37728,
    "supported_interests": 104,
}
actual_dev = {key: int(statistics["dev"][key]) for key in expected_dev}
if actual_dev != expected_dev:
    raise RuntimeError(f"Behavior+Interest Dev mismatch: {actual_dev}")
if int(statistics["pairing"]["train_dev_user_overlap"]) != 0:
    raise RuntimeError("Train/Dev users are not disjoint")
if pq.ParquetFile(eval_file).metadata.num_rows != 31392:
    raise RuntimeError("Dev Parquet does not contain 31,392 rows")
if pq.ParquetFile(labels_file).metadata.num_rows != 108:
    raise RuntimeError("Retained-interest vocabulary does not contain 108 labels")

eval_digest = hashlib.sha256(eval_file.read_bytes()).hexdigest()
if eval_digest != metadata["artifact_sha256"]["eval/behavior_interest/dev_full_grouped.parquet"]:
    raise RuntimeError("Dev Parquet SHA-256 does not match metadata")

audit = {
    "dataset_rule_version": metadata["rule_version"],
    "history_limit": int(metadata["history_limit"]),
    "history_item_format": metadata["history_item_format"],
    "history_mode": "behavior_interest",
    "dev": actual_dev,
    "eval_sha256": eval_digest,
    "labels_sha256": hashlib.sha256(labels_file.read_bytes()).hexdigest(),
    "label_vocabulary_size": 108,
    "train_dev_user_overlap": 0,
}
output.write_text(json.dumps(audit, indent=2) + "\n")
print("BEHAVIOR_INTEREST_QWEN_DATASET_GATE_OK", eval_digest)
PY

python - "$SOURCE_SFT_DIR/run_summary.json" "$ADAPTER_PATH" "$PARALLEL_ROOT/adapter_audit.json" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
adapter_path = Path(sys.argv[2])
output = Path(sys.argv[3])
summary = json.loads(summary_path.read_text())
checks = {
    "train_samples": int(summary["train_samples"]),
    "world_size": int(summary["world_size"]),
    "effective_global_batch_size": int(summary["effective_global_batch_size"]),
    "expected_optimizer_steps": int(summary["expected_optimizer_steps"]),
    "token_audit_samples": int(summary["token_audit"]["samples"]),
    "over_max_seq_length": int(summary["token_audit"]["over_max_seq_length"]),
}
expected = {
    "train_samples": 21600,
    "world_size": 4,
    "effective_global_batch_size": 32,
    "expected_optimizer_steps": 2025,
    "token_audit_samples": 21600,
    "over_max_seq_length": 0,
}
if checks != expected:
    raise RuntimeError(f"Qwen training contract mismatch: {checks}")
if float(summary["configured_epochs"]) != 3.0:
    raise RuntimeError("Qwen did not train for exactly three epochs")
loss = float(summary["train_metrics"]["train_loss"])
if not math.isfinite(loss):
    raise RuntimeError(f"Qwen has non-finite train loss: {loss}")
adapter_file = adapter_path / "adapter_model.safetensors"
adapter_audit = {
    "source_run_summary": str(summary_path),
    "train_loss": loss,
    "optimizer_steps": 2025,
    "configured_epochs": 3.0,
    "adapter_path": str(adapter_path),
    "adapter_size_bytes": adapter_file.stat().st_size,
    "adapter_sha256": hashlib.sha256(adapter_file.read_bytes()).hexdigest(),
}
output.write_text(json.dumps(adapter_audit, indent=2) + "\n")
print(
    "BEHAVIOR_INTEREST_QWEN_ADAPTER_GATE_OK",
    loss,
    adapter_audit["adapter_size_bytes"],
    adapter_audit["adapter_sha256"],
)
PY

: "${HF_HOME:=/mnt/default/huggingface-cache}"
export HF_HOME
mkdir -p "$HF_HOME"
python - <<'PY'
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

name = "Qwen/Qwen3.5-9B"
snapshot = snapshot_download(name)
config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
print(
    "QWEN35_BEHAVIOR_INTEREST_SNAPSHOT_READY",
    snapshot,
    type(config).__name__,
    type(tokenizer).__name__,
)
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PIDS=()
SHARD_DIRS=()
for SHARD_INDEX in $(seq 0 $((NUM_GPUS - 1))); do
  SHARD_NAME="$(printf 'shard-%02d' "$SHARD_INDEX")"
  SHARD_DIR="$BEAM_ROOT/$SHARD_NAME"
  SHARD_DIRS+=("$SHARD_DIR")
  (
    export CUDA_VISIBLE_DEVICES="$SHARD_INDEX"
    python -m new_interest.evaluation.evaluate_beam \
      --config "$BEAM_CONFIG" \
      --adapter-path "$ADAPTER_PATH" \
      --output-dir "$SHARD_DIR" \
      --run-name "${RUN_NAME}-${SHARD_NAME}" \
      --num-shards "$NUM_GPUS" \
      --shard-index "$SHARD_INDEX" \
      --wandb-mode disabled \
      >"$BEAM_ROOT/${SHARD_NAME}.log" 2>&1
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
  echo "At least one Qwen Behavior+Interest Beam shard failed" >&2
  for log in "$BEAM_ROOT"/shard-*.log; do
    echo "===== ${log##*/} (tail) =====" >&2
    tail -40 "$log" >&2 || true
  done
  exit 1
fi

python -m new_interest.evaluation.merge_beam_shards \
  --shard-dirs "${SHARD_DIRS[@]}" \
  --eval-file "$EVAL_FILE" \
  --retained-interests-file "$LABELS_FILE" \
  --output-dir "$BEAM_ROOT/merged" \
  --expected-records 31392

python - "$PARALLEL_ROOT" <<'PY'
import json
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
adapter_audit = json.loads((root / "adapter_audit.json").read_text())
metrics = json.loads((root / "beam/merged/metrics.json").read_text())
predictions = pq.read_table(
    root / "beam/merged/beam_predictions.parquet",
    columns=["record_index", "impression_id"],
).to_pylist()
if int(metrics["samples"]) != 31392 or len(predictions) != 31392:
    raise RuntimeError("Merged Beam prediction count is not 31,392")
if len({int(row["record_index"]) for row in predictions}) != 31392:
    raise RuntimeError("Merged Beam record_index values are not unique")
if len({str(row["impression_id"]) for row in predictions}) != 31392:
    raise RuntimeError("Merged Beam impression_id values are not unique")
if int(metrics["supported_interest_count"]) != 104:
    raise RuntimeError("Interest Macro denominator is not 104")
if int(metrics["label_vocabulary_size"]) != 108:
    raise RuntimeError("Constrained vocabulary size is not 108")

def check_finite(value, path="metrics"):
    if isinstance(value, dict):
        for key, child in value.items():
            check_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            check_finite(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"Non-finite metric: {path}={value}")

check_finite(metrics)
shard_counts = []
for shard_index in range(4):
    manifest = json.loads(
        (root / f"beam/shard-{shard_index:02d}/shard_manifest.json").read_text()
    )
    shard_counts.append(int(manifest["shard_record_count"]))
if shard_counts != [7848, 7848, 7848, 7848]:
    raise RuntimeError(f"Unexpected Qwen Beam shard counts: {shard_counts}")

summary = {
    "status": "complete",
    "history_mode": "behavior_interest",
    "train_loss": float(adapter_audit["train_loss"]),
    "optimizer_steps": int(adapter_audit["optimizer_steps"]),
    "configured_epochs": float(adapter_audit["configured_epochs"]),
    "final_adapter": adapter_audit["adapter_path"],
    "adapter_size_bytes": int(adapter_audit["adapter_size_bytes"]),
    "adapter_sha256": adapter_audit["adapter_sha256"],
    "beam_metrics": str(root / "beam/merged/metrics.json"),
    "beam_predictions": str(root / "beam/merged/beam_predictions.parquet"),
    "shard_record_counts": shard_counts,
    "merged_records": 31392,
    "unique_record_indices": 31392,
    "unique_impression_ids": 31392,
    "supported_interest_count": 104,
    "label_vocabulary_size": 108,
    "interest_macro_recall_at_10": float(metrics["interest_macro_recall_at_10"]),
    "interest_macro_ndcg_at_10": float(metrics["interest_macro_ndcg_at_10"]),
}
(root / "parallel_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(
    "BEHAVIOR_INTEREST_QWEN_FULL_EVAL_GATE_OK",
    metrics["interest_macro_recall_at_10"],
    metrics["interest_macro_ndcg_at_10"],
)
print("BEHAVIOR_INTEREST_QWEN_PARALLEL_COMPLETE")
PY
