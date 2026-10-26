#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="Qwen/Qwen3.5-9B"
NUM_GPUS="${STATIC_SUD_CROSSFIT_NUM_GPUS:-8}"
SOURCE_ROOT="${CROSSFIT_SOURCE_ROOT:?CROSSFIT_SOURCE_ROOT is required}"
SOURCE_RESULT_ID="${CROSSFIT_SOURCE_RESULT_ID:?CROSSFIT_SOURCE_RESULT_ID is required}"

if [[ "$NUM_GPUS" != "8" ]]; then
  echo "Behavior evaluation recovery requires eight GPUs; got $NUM_GPUS" >&2
  exit 2
fi
if [[ "$(basename "$SOURCE_ROOT")" != "$SOURCE_RESULT_ID" ]]; then
  echo "Recovery source path does not match its pinned result identity" >&2
  exit 2
fi
case "$AMLT_JOB_NAME" in
  *behavior-vanilla*) CONDITION="vanilla" ;;
  *behavior-weighted*) CONDITION="weighted" ;;
  *)
    echo "Cannot infer Behavior recovery condition from $AMLT_JOB_NAME" >&2
    exit 2
    ;;
esac

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME="${HF_HOME:-/mnt/default/huggingface-cache}"
export NEW_INTEREST_PARQUET_STAGE_DIR="${NEW_INTEREST_PARQUET_STAGE_DIR:-/scratch/new-interest-parquet-stage}"
mkdir -p "$HF_HOME" "$NEW_INTEREST_PARQUET_STAGE_DIR" "$AMLT_OUTPUT_DIR"

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
    weights: set[str] = set()
    for index_path in (
        path / "model.safetensors.index.json",
        path / "pytorch_model.bin.index.json",
    ):
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weights.update(payload.get("weight_map", {}).values())
    if not weights:
        raise RuntimeError("Cached Qwen snapshot has no sharded weight index")
    missing = [
        name
        for name in sorted(weights)
        if not (path / name).is_file() or (path / name).stat().st_size == 0
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
print("CROSSFIT_BEHAVIOR_EVAL_MODEL_READY", name, snapshot, source)
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DATA_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_crossfit_reweighting_qwen35_v1"
LABELS_FILE="$AMLT_CODE_DIR/data/training/generative_new_interest_static_user_disjoint_balanced_200_h50_v1/retained_interests.parquet"
BEAM_CONFIG="$AMLT_CODE_DIR/configs/beam/static_sud_qwen35_behavior_full.json"
ADAPTER_PATH="$SOURCE_ROOT/sft/final_adapter"
SOURCE_SUMMARY="$SOURCE_ROOT/sft/run_summary.json"
SOURCE_CHECKPOINT="$SOURCE_ROOT/sft/checkpoints/checkpoint-2025"

for path in \
  "$SOURCE_SUMMARY" \
  "$SOURCE_CHECKPOINT/checkpoint_complete.json" \
  "$ADAPTER_PATH/adapter_model.safetensors" \
  "$ADAPTER_PATH/adapter_config.json" \
  "$ADAPTER_PATH/tokenizer.json" \
  "$BEAM_CONFIG" \
  "$LABELS_FILE"; do
  if [[ ! -s "$path" ]]; then
    echo "Required recovery input is missing or empty: $path" >&2
    exit 1
  fi
done

python - \
  "$SOURCE_ROOT" \
  "$SOURCE_RESULT_ID" \
  "$CONDITION" \
  "$AMLT_OUTPUT_DIR/source_training_audit.json" <<'PY'
import json
import math
import sys
from pathlib import Path

from new_interest.evaluation.evaluate_beam import (
    ADAPTER_SEMANTIC_HASH_VERSION,
    sha256_adapter_semantics,
    sha256_file,
)
from new_interest.training.checkpoint_persistence import (
    validate_persistent_checkpoint,
)

root = Path(sys.argv[1]).resolve(strict=True)
expected_result_id = sys.argv[2]
condition = sys.argv[3]
output = Path(sys.argv[4])
if root.name != expected_result_id:
    raise RuntimeError("Pinned source result identity mismatch")
summary_path = root / "sft" / "run_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
expected = {
    "train_samples": 21600,
    "expected_optimizer_steps": 2025,
    "world_size": 8,
    "effective_global_batch_size": 32,
}
for key, value in expected.items():
    if int(summary.get(key, -1)) != value:
        raise RuntimeError(f"Source training gate failed for {key}")
if float(summary.get("configured_epochs", -1)) != 3.0:
    raise RuntimeError("Source training did not complete exactly three epochs")
loss = float(summary.get("train_metrics", {}).get("train_loss", float("nan")))
if not math.isfinite(loss):
    raise RuntimeError("Source training loss is not finite")

checkpoint = root / "sft" / "checkpoints" / "checkpoint-2025"
checkpoint_marker = validate_persistent_checkpoint(
    checkpoint,
    summary["checkpoint_identity"],
)
adapter = root / "sft" / "final_adapter"
adapter_files = {
    path.relative_to(adapter).as_posix(): {
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    for path in sorted(adapter.rglob("*"))
    if path.is_file()
}
adapter_model_hash = adapter_files["adapter_model.safetensors"]["sha256"]
checkpoint_model_hash = checkpoint_marker["files"][
    "adapter_model.safetensors"
]["sha256"]
if adapter_model_hash != checkpoint_model_hash:
    raise RuntimeError("Final adapter weights differ from checkpoint-2025")
payload = {
    "source_result_id": expected_result_id,
    "condition": condition,
    "source_run_summary_sha256": sha256_file(summary_path),
    "source_checkpoint_marker_sha256": sha256_file(
        checkpoint / "checkpoint_complete.json"
    ),
    "source_checkpoint_step": int(checkpoint_marker["global_step"]),
    "source_train_samples": int(summary["train_samples"]),
    "source_expected_optimizer_steps": int(summary["expected_optimizer_steps"]),
    "source_world_size": int(summary["world_size"]),
    "source_effective_global_batch_size": int(
        summary["effective_global_batch_size"]
    ),
    "source_configured_epochs": float(summary["configured_epochs"]),
    "source_train_loss": loss,
    "adapter_hash_version": ADAPTER_SEMANTIC_HASH_VERSION,
    "adapter_semantic_sha256": sha256_adapter_semantics(adapter),
    "adapter_files": adapter_files,
}
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_name(f".{output.name}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(output)
print(
    "CROSSFIT_BEHAVIOR_SOURCE_TRAINING_GATE_OK",
    condition,
    expected_result_id,
    payload["adapter_semantic_sha256"],
)
PY

python - "$AMLT_OUTPUT_DIR/environment.json" <<'PY'
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

packages = ("accelerate", "peft", "pyarrow", "torch", "transformers", "wandb")
payload = {
    "python": sys.version,
    "platform": platform.platform(),
    "packages": {
        package: importlib.metadata.version(package) for package in packages
    },
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
PY

for EVAL_NAME in real shuffled_20260818 shuffled_20260819 shuffled_20260820; do
  EVAL_FILE="$DATA_ROOT/eval/behavior_only/$EVAL_NAME.parquet"
  BEAM_ROOT="$AMLT_OUTPUT_DIR/eval/$EVAL_NAME"
  if [[ ! -s "$EVAL_FILE" ]]; then
    echo "Required evaluation file is missing or empty: $EVAL_FILE" >&2
    exit 1
  fi
  mkdir -p "$BEAM_ROOT"
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
        --eval-file "$EVAL_FILE" \
        --output-dir "$SHARD_DIR" \
        --run-name "static-sud-crossfit-behavior-${CONDITION}-eval-recovery-${EVAL_NAME}-${SHARD_NAME}" \
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
    echo "At least one recovered Beam shard failed for $EVAL_NAME" >&2
    for log in "$BEAM_ROOT"/shard-*.log; do
      echo "===== $log =====" >&2
      tail -n 80 "$log" >&2 || true
    done
    exit 1
  fi
  python -m new_interest.evaluation.merge_beam_shards \
    --shard-dirs "${SHARD_DIRS[@]}" \
    --eval-file "$EVAL_FILE" \
    --retained-interests-file "$LABELS_FILE" \
    --output-dir "$BEAM_ROOT/merged" \
    --expected-records 31392
done

python - "$AMLT_OUTPUT_DIR/eval" "$CONDITION" <<'PY'
import json
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
condition = sys.argv[2]
identities = {}
for name in (
    "real",
    "shuffled_20260818",
    "shuffled_20260819",
    "shuffled_20260820",
):
    merged = root / name / "merged"
    metrics = json.loads((merged / "metrics.json").read_text(encoding="utf-8"))
    expected = {
        "samples": 31392,
        "merged_prediction_count": 31392,
        "supported_interest_count": 104,
        "top_k": 20,
        "logical_num_shards": 8,
    }
    for key, value in expected.items():
        if int(metrics.get(key, -1)) != value:
            raise RuntimeError(f"{name}: recovered gate failed for {key}")
    for key in ("interest_macro_recall_at_10", "interest_macro_ndcg_at_10"):
        if not math.isfinite(float(metrics[key])):
            raise RuntimeError(f"{name}: non-finite {key}")
    rows = pq.read_table(
        merged / "beam_predictions.parquet",
        columns=["record_index", "evaluation_identity"],
    ).to_pylist()
    if len({int(row["record_index"]) for row in rows}) != 31392:
        raise RuntimeError(f"{name}: predictions are not unique")
    row_identities = {str(row["evaluation_identity"]) for row in rows}
    if row_identities != {str(metrics["evaluation_identity"])}:
        raise RuntimeError(f"{name}: prediction identity mismatch")
    identities[name] = metrics["evaluation_identity"]
print("CROSSFIT_BEHAVIOR_EVAL_RECOVERY_GATE_OK", condition, identities)
PY

python - "$AMLT_OUTPUT_DIR" "$SOURCE_RESULT_ID" "$CONDITION" <<'PY'
import json
import sys
from pathlib import Path

from new_interest.evaluation.evaluate_beam import _atomic_write_json, sha256_file

root = Path(sys.argv[1])
source_result_id = sys.argv[2]
condition = sys.argv[3]
manifest_path = root / "artifact_manifest.json"
files = sorted(
    path for path in root.rglob("*") if path.is_file() and path != manifest_path
)
hashes = {
    path.relative_to(root).as_posix(): {
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    for path in files
}
_atomic_write_json(
    manifest_path,
    {
        "manifest_version": "crossfit-behavior-eval-recovery-v1",
        "source_result_id": source_result_id,
        "condition": condition,
        "files": hashes,
    },
)
print("CROSSFIT_BEHAVIOR_RECOVERY_MANIFEST_OK", condition, len(hashes))
PY
