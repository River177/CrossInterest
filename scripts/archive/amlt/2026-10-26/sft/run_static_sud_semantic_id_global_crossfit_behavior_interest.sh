#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: run_static_sud_semantic_id_global_crossfit_behavior_interest.sh <smoke|full>}"
case "$MODE" in
  smoke)
    EXPECTED_JOB_NAME="qwen35-behavior-interest-semantic-id-global-crossfit-smoke"
    ;;
  full)
    EXPECTED_JOB_NAME="qwen35-behavior-interest-semantic-id-global-crossfit-final"
    ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    exit 2
    ;;
esac

NUM_GPUS="${SEMANTIC_ID_GLOBAL_CROSSFIT_NUM_GPUS:-4}"
if [[ "$NUM_GPUS" != "4" ]]; then
  echo "Behavior+Interest Semantic-ID global Cross-Fitted training requires four GPUs; got $NUM_GPUS" >&2
  exit 2
fi
if [[ "${AMLT_JOB_NAME:?AMLT_JOB_NAME is required}" != "$EXPECTED_JOB_NAME" ]]; then
  echo "Unexpected combined Semantic-ID job name: $AMLT_JOB_NAME" >&2
  exit 2
fi

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export WANDB_DISABLED=true
export HF_HOME="${HF_HOME:-/mnt/default/huggingface-cache}"
mkdir -p "$HF_HOME" "$AMLT_OUTPUT_DIR"

MODEL_NAME="Qwen/Qwen3.5-9B"
CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_semantic_id_global_crossfit_qwen35_behavior_interest_weighted_v1.yaml"
EVAL_CONFIG="$AMLT_CODE_DIR/configs/id_token/static_sud_qwen35_behavior_interest_full.json"
SEMANTIC_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_v1"
TRAIN_FILE="$SEMANTIC_ROOT/train/behavior_interest/21600.parquet"
EVAL_FILE="$SEMANTIC_ROOT/eval/behavior_interest/dev_full_grouped.parquet"
MAPPING_FILE="$SEMANTIC_ROOT/interest_token_mapping.parquet"
WEIGHT_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_semantic_id_global_crossfit_qwen35_behavior_interest_v1"
WEIGHT_FILE="$WEIGHT_ROOT/sample_weights.parquet"
WEIGHT_SUMMARY="$WEIGHT_ROOT/weight_summary.json"
FOLD_MANIFEST="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_crossfit_reweighting_qwen35_v1/fold_manifest.parquet"
SFT_DIR="$AMLT_OUTPUT_DIR/sft"
EVAL_ROOT="$AMLT_OUTPUT_DIR/real_eval"
MERGED_DIR="$EVAL_ROOT/merged"
LOCAL_CHECKPOINT_ROOT="/scratch/semantic-id-global-crossfit-behavior-interest-${MODE}-checkpoints"
PERSISTENT_CHECKPOINT_ROOT="$AMLT_OUTPUT_DIR/checkpoints"
SUCCESS_MARKER="$AMLT_OUTPUT_DIR/${MODE}_success.json"

TRAIN_SHA256="dc8f04c172bf2481f4414d2c7a0ca510ae2bc0e125c2a03d984072b72b4a8140"
EVAL_SHA256="c05b20033ce0d1a0c2d4b7bb97ca0cd6b7cffb7f198a7265cce9e54046b665fd"
MAPPING_SHA256="45397451ac2ae2e64d8156b2b47be3f454cc309a8d30a83bc23c2d3267653a0a"
FOLD_MANIFEST_SHA256="3d25e36b4c34ae2516ac6fccee3e2c4d6596ac610076c6ef0f8b8505cd3a4b3a"
WEIGHT_SHA256="4d41c6daae4f03a968e5e05effa232dda593a0ba8055523df9bed78c148c33e0"
WEIGHT_SUMMARY_SHA256="2828bed1d6f1a6a485275272070673419f6f53535b4e0e7074a4b2d71336c3cc"

if [[ "$WEIGHT_SHA256" == PENDING_* || "$WEIGHT_SUMMARY_SHA256" == PENDING_* ]]; then
  echo "Behavior+Interest OOF weights have not been frozen yet" >&2
  exit 1
fi

for required_path in \
  "$CONFIG" \
  "$EVAL_CONFIG" \
  "$TRAIN_FILE" \
  "$EVAL_FILE" \
  "$MAPPING_FILE" \
  "$WEIGHT_FILE" \
  "$WEIGHT_SUMMARY" \
  "$FOLD_MANIFEST"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing combined experiment input: $required_path" >&2
    exit 1
  fi
done

if [[ -f "$SUCCESS_MARKER" ]]; then
  python - "$AMLT_OUTPUT_DIR" "$SUCCESS_MARKER" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
marker_path = Path(sys.argv[2])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


marker = json.loads(marker_path.read_text(encoding="utf-8"))
manifest_path = root / "artifact_manifest.json"
if sha256(manifest_path) != marker["artifact_manifest_sha256"]:
    raise RuntimeError("Committed artifact manifest hash mismatch")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for relative, expected in manifest["files"].items():
    path = root / relative
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"Committed artifact is missing or corrupt: {relative}")
print("COMBINED_SEMANTIC_ID_GLOBAL_CROSSFIT_REUSE_OK", len(manifest["files"]))
PY
  exit 0
fi

python - "$AMLT_OUTPUT_DIR/environment.json" <<'PY'
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch

packages = ("accelerate", "peft", "pyarrow", "torch", "transformers", "wandb")
payload = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages": {name: importlib.metadata.version(name) for name in packages},
    "cuda": torch.version.cuda,
    "cuda_devices": torch.cuda.device_count(),
}
if payload["packages"]["transformers"] != "5.14.1":
    raise RuntimeError("Unexpected Transformers version")
if payload["packages"]["peft"] != "0.18.0":
    raise RuntimeError("Unexpected PEFT version")
if payload["packages"]["accelerate"] != "1.12.0":
    raise RuntimeError("Unexpected Accelerate version")
if payload["cuda_devices"] != 4:
    raise RuntimeError("Expected exactly four visible GPUs")
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("COMBINED_SEMANTIC_ID_GLOBAL_CROSSFIT_ENV_OK", json.dumps(payload, sort_keys=True))
PY

python - \
  "$CONFIG" "$EVAL_CONFIG" \
  "$TRAIN_FILE" "$TRAIN_SHA256" \
  "$EVAL_FILE" "$EVAL_SHA256" \
  "$MAPPING_FILE" "$MAPPING_SHA256" \
  "$FOLD_MANIFEST" "$FOLD_MANIFEST_SHA256" \
  "$WEIGHT_FILE" "$WEIGHT_SHA256" \
  "$WEIGHT_SUMMARY" "$WEIGHT_SUMMARY_SHA256" \
  "$AMLT_OUTPUT_DIR/input_audit.json" <<'PY'
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml

config_path = Path(sys.argv[1])
eval_config_path = Path(sys.argv[2])
train_path = Path(sys.argv[3])
eval_path = Path(sys.argv[5])
mapping_path = Path(sys.argv[7])
fold_manifest_path = Path(sys.argv[9])
weight_path = Path(sys.argv[11])
weight_summary_path = Path(sys.argv[13])
output_path = Path(sys.argv[15])
expected_hashes = {
    train_path: sys.argv[4],
    eval_path: sys.argv[6],
    mapping_path: sys.argv[8],
    fold_manifest_path: sys.argv[10],
    weight_path: sys.argv[12],
    weight_summary_path: sys.argv[14],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


for path, expected in expected_hashes.items():
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"Frozen input hash mismatch for {path}: {actual}")

config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
expected_train = (
    "data/training/"
    "sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_v1/"
    "train/behavior_interest/21600.parquet"
)
expected_mapping = (
    "data/training/"
    "sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_v1/"
    "interest_token_mapping.parquet"
)
expected_weights = (
    "data/training/"
    "sft_static_user_disjoint_semantic_id_global_crossfit_"
    "qwen35_behavior_interest_v1/"
    "sample_weights.parquet"
)
if config["model_name"] != "Qwen/Qwen3.5-9B":
    raise RuntimeError("Config model identity mismatch")
if config["train_file"] != expected_train:
    raise RuntimeError("Config Train identity mismatch")
if config["sample_weight_file"] != expected_weights:
    raise RuntimeError("Config weight identity mismatch")
if config["semantic_id_token_mapping_file"] != expected_mapping:
    raise RuntimeError("Config mapping identity mismatch")
if config.get("enable_thinking") is not False:
    raise RuntimeError("Config unexpectedly enables thinking")
if config.get("gradient_checkpointing_use_reentrant") is not False:
    raise RuntimeError("Config must use non-reentrant checkpointing")
if int(config["epochs"]) != 3:
    raise RuntimeError("Config epoch mismatch")
if int(config["per_device_train_batch_size"]) != 1:
    raise RuntimeError("Config per-device batch mismatch")
if int(config["gradient_accumulation_steps"]) != 8:
    raise RuntimeError("Config accumulation mismatch")

eval_config = json.loads(eval_config_path.read_text(encoding="utf-8"))
if eval_config["model_name"] != "Qwen/Qwen3.5-9B":
    raise RuntimeError("Evaluation model identity mismatch")
if eval_config["eval_file"] != expected_train.replace(
    "train/behavior_interest/21600.parquet",
    "eval/behavior_interest/dev_full_grouped.parquet",
):
    raise RuntimeError("Evaluation file identity mismatch")
if eval_config["interest_token_mapping_file"] != expected_mapping:
    raise RuntimeError("Evaluation mapping identity mismatch")
if eval_config["top_k"] != 20 or eval_config["cutoffs"] != [1, 5, 10, 20]:
    raise RuntimeError("Evaluation cutoff identity mismatch")

mapping_rows = pq.read_table(mapping_path).to_pylist()
token_by_label = {
    str(row["target_subcategory"]): str(row["interest_token"])
    for row in mapping_rows
}
if len(token_by_label) != 108 or len(set(token_by_label.values())) != 108:
    raise RuntimeError("Mapping is not one-to-one over exactly 108 Semantic IDs")

train_rows = pq.read_table(train_path).to_pylist()
if len(train_rows) != 21600:
    raise RuntimeError("Train is not exactly 21,600 rows")
train_by_source = {str(row["source_sample_id"]): row for row in train_rows}
if len(train_by_source) != 21600:
    raise RuntimeError("Train source IDs are not unique")
if len({str(row["user_id"]) for row in train_rows}) != 21600:
    raise RuntimeError("Train is not one row per user")
if {str(row["task_type"]) for row in train_rows} != {"behavior_interest"}:
    raise RuntimeError("Train input is not Behavior+Interest")
if {str(row["target_subcategory"]) for row in train_rows} != set(token_by_label):
    raise RuntimeError("Train does not cover all mapped labels")
for row in train_rows:
    response = str(row["response"])
    target = str(row["target_subcategory"])
    messages = row["messages"]
    if response != token_by_label[target] or not re.fullmatch(r"<interest_\d{3}>", response):
        raise RuntimeError("Train contains a non-Semantic-ID response")
    if len(messages) != 2 or messages[0]["role"] != "user" or messages[1] != {"role": "assistant", "content": response}:
        raise RuntimeError("Train message contract mismatch")

weights = pq.read_table(weight_path).to_pylist()
weight_by_source = {str(row["source_sample_id"]): row for row in weights}
if len(weights) != 21600 or set(weight_by_source) != set(train_by_source):
    raise RuntimeError("Weights do not exactly cover combined Train")
manifest_rows = pq.read_table(fold_manifest_path).to_pylist()
manifest_by_source = {str(row["source_sample_id"]): row for row in manifest_rows}
if len(manifest_by_source) != 21600 or set(manifest_by_source) != set(train_by_source):
    raise RuntimeError("Fold manifest does not exactly cover combined Train")

fold_counts = Counter()
fold_rows = defaultdict(list)
class_values = defaultdict(list)
values = []
for source_id, row in weight_by_source.items():
    train_row = train_by_source[source_id]
    manifest_row = manifest_by_source[source_id]
    label = str(row["target_subcategory"])
    if str(row["user_id"]) != str(train_row["user_id"]) or label != str(train_row["target_subcategory"]):
        raise RuntimeError("Weight identity differs from combined Train")
    if str(manifest_row["user_id"]) != str(train_row["user_id"]) or str(manifest_row["target_subcategory"]) != label:
        raise RuntimeError("Fold identity differs from combined Train")
    fold = int(row["fold"])
    if int(manifest_row["fold"]) != fold:
        raise RuntimeError("Weight fold differs from frozen fold manifest")
    rank = int(row["fold_global_rank"])
    quantile = float(row["fold_global_difficulty_quantile"])
    raw_weight = float(row["raw_weight"])
    weight = float(row["sample_weight"])
    if not 0 <= rank < 4320 or not math.isclose(quantile, rank / 4319, abs_tol=1e-15):
        raise RuntimeError("Fold-global rank/quantile contract mismatch")
    if not math.isclose(raw_weight, 1.0 / (1.0 + quantile), abs_tol=1e-15):
        raise RuntimeError("Fold-global raw-weight formula mismatch")
    if not math.isfinite(weight) or weight <= 0:
        raise RuntimeError("Sample weight is not positive and finite")
    if not math.isfinite(float(row["prior_adjusted_nll"])) or not math.isfinite(float(row["prior_adjusted_margin"])):
        raise RuntimeError("Weights contain non-finite OOF scores")
    fold_counts[fold] += 1
    fold_rows[fold].append(row)
    class_values[label].append(weight)
    values.append(weight)
if fold_counts != Counter({fold: 4320 for fold in range(5)}):
    raise RuntimeError(f"Weight fold counts mismatch: {fold_counts}")
if len(class_values) != 108 or {len(rows) for rows in class_values.values()} != {200}:
    raise RuntimeError("Weights are not exactly 200 rows for each of 108 classes")
for rows in fold_rows.values():
    if {int(row["fold_global_rank"]) for row in rows} != set(range(4320)):
        raise RuntimeError("Fold-global ranks are not complete and unique")
class_means = {
    label: math.fsum(rows) / len(rows)
    for label, rows in class_values.items()
}
if all(math.isclose(mean, 1.0, abs_tol=1e-12) for mean in class_means.values()):
    raise RuntimeError("Global weights were unexpectedly normalized within class")
if not math.isclose(math.fsum(values) / len(values), 1.0, abs_tol=1e-12):
    raise RuntimeError("Global sample-weight mean is not one")
if not math.isclose(max(values) / min(values), 2.0, abs_tol=1e-12):
    raise RuntimeError("Sample-weight max/min ratio is not two")

weight_summary = json.loads(weight_summary_path.read_text(encoding="utf-8"))
if weight_summary["weight_version"] != "semantic-id-fold-global-quantile-v1":
    raise RuntimeError("Weight version mismatch")
if weight_summary["formula"]["class_normalization"] is not False:
    raise RuntimeError("Weight summary unexpectedly enables class normalization")
if weight_summary["output_sha256"]["sample_weights"] != expected_hashes[weight_path]:
    raise RuntimeError("Weight summary output identity mismatch")
if not math.isclose(weight_summary["class_weight_mean_min"], min(class_means.values()), abs_tol=1e-15):
    raise RuntimeError("Weight summary class-mean minimum mismatch")
if not math.isclose(weight_summary["class_weight_mean_max"], max(class_means.values()), abs_tol=1e-15):
    raise RuntimeError("Weight summary class-mean maximum mismatch")

eval_rows = pq.read_table(eval_path).to_pylist()
if len(eval_rows) != 31392 or len({str(row["impression_id"]) for row in eval_rows}) != 31392:
    raise RuntimeError("Real Dev is not exactly 31,392 unique impressions")
supported = {
    str(label)
    for row in eval_rows
    for label in row["acceptable_subcategories"]
}
if len(supported) != 104:
    raise RuntimeError("Real Dev does not have exactly 104 supported interests")
if any(
    not set(map(str, row["acceptable_responses"])).issubset(
        set(token_by_label.values())
    )
    for row in eval_rows
):
    raise RuntimeError("Real Dev contains an unmapped Semantic-ID response")

audit = {
    "audit_version": "combined-semantic-id-global-crossfit-input-v1",
    "analysis_scope": "qwen35-behavior-plus-interest",
    "train_rows": len(train_rows),
    "unique_train_users": len({str(row["user_id"]) for row in train_rows}),
    "classes": len(token_by_label),
    "weight_rows": len(weights),
    "weight_source_view": "behavior-plus-interest-semantic-id-oof",
    "weight_application_view": "behavior-plus-interest-semantic-id",
    "weight_rank_scope": "within-heldout-fold-across-target-classes",
    "class_normalization": False,
    "fold_counts": dict(sorted(fold_counts.items())),
    "global_weight_mean": math.fsum(values) / len(values),
    "class_weight_mean_min": min(class_means.values()),
    "class_weight_mean_max": max(class_means.values()),
    "global_weight_max_min_ratio": max(values) / min(values),
    "real_eval_rows": len(eval_rows),
    "real_eval_supported_interests": len(supported),
    "input_sha256": {path.name: expected for path, expected in expected_hashes.items()},
    "shuffled_evaluation_used": False,
    "history_prediction_filtering": False,
}
output_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
print("COMBINED_SEMANTIC_ID_GLOBAL_CROSSFIT_INPUT_OK", json.dumps(audit, sort_keys=True))
PY

python - "$MODEL_NAME" <<'PY'
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

name = sys.argv[1]


def validate_snapshot(value: str) -> Path:
    root = Path(value)
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        raise RuntimeError("Qwen snapshot has no safetensors index")
    filenames = set(
        json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()
    )
    missing = [
        filename
        for filename in sorted(filenames)
        if not (root / filename).is_file()
        or (root / filename).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"Qwen snapshot is incomplete: {missing[:3]}")
    AutoConfig.from_pretrained(root, local_files_only=True)
    AutoTokenizer.from_pretrained(root, local_files_only=True)
    return root


try:
    snapshot = validate_snapshot(snapshot_download(name, local_files_only=True))
    source = "persistent-cache"
except Exception:
    snapshot = validate_snapshot(snapshot_download(name))
    source = "downloaded-or-repaired"
print("COMBINED_SEMANTIC_ID_GLOBAL_CROSSFIT_MODEL_READY", name, snapshot, source)
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ACTIVE_TRAIN_FILE="$TRAIN_FILE"
ACTIVE_WEIGHT_FILE="$WEIGHT_FILE"
EPOCHS=3
SAVE_STEPS=100
if [[ "$MODE" == "smoke" ]]; then
  SMOKE_DATA_DIR="$AMLT_OUTPUT_DIR/smoke_data"
  mkdir -p "$SMOKE_DATA_DIR"
  ACTIVE_TRAIN_FILE="$SMOKE_DATA_DIR/train.parquet"
  ACTIVE_WEIGHT_FILE="$SMOKE_DATA_DIR/sample_weights.parquet"
  python - "$TRAIN_FILE" "$WEIGHT_FILE" "$ACTIVE_TRAIN_FILE" "$ACTIVE_WEIGHT_FILE" <<'PY'
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

train_source = Path(sys.argv[1])
weight_source = Path(sys.argv[2])
train_output = Path(sys.argv[3])
weight_output = Path(sys.argv[4])
train_rows = pq.read_table(train_source).to_pylist()
weight_rows = pq.read_table(weight_source).to_pylist()
weight_by_source = {
    str(row["source_sample_id"]): row for row in weight_rows
}
selected = {}
for row in sorted(train_rows, key=lambda value: str(value["source_sample_id"])):
    selected.setdefault(str(row["target_subcategory"]), row)
if len(selected) != 108:
    raise RuntimeError("Smoke selection does not cover all 108 classes")
rows = [selected[label] for label in sorted(selected)]
weights = [weight_by_source[str(row["source_sample_id"])] for row in rows]
if len({str(row["user_id"]) for row in rows}) != 108:
    raise RuntimeError("Smoke selection does not contain 108 distinct users")
pq.write_table(
    pa.Table.from_pylist(rows, schema=pq.ParquetFile(train_source).schema_arrow),
    train_output,
    compression="zstd",
    use_dictionary=True,
)
pq.write_table(
    pa.Table.from_pylist(weights, schema=pq.ParquetFile(weight_source).schema_arrow),
    weight_output,
    compression="zstd",
    use_dictionary=True,
)
print("COMBINED_SEMANTIC_ID_GLOBAL_CROSSFIT_SMOKE_DATA_OK", len(rows))
PY
  EPOCHS=2
  SAVE_STEPS=675
fi

TRAIN_ARGS=(
  --config "$CONFIG"
  --train-file "$ACTIVE_TRAIN_FILE"
  --sample-weight-file "$ACTIVE_WEIGHT_FILE"
  --output-dir "$SFT_DIR"
  --checkpoint-dir "$LOCAL_CHECKPOINT_ROOT"
  --run-name "static-sud-semantic-id-global-crossfit-qwen35-behavior-interest-${MODE}"
  --epochs "$EPOCHS"
  --per-device-train-batch-size 1
  --gradient-accumulation-steps 8
  --save-steps "$SAVE_STEPS"
  --wandb-mode disabled
)
if [[ "$MODE" == "full" ]]; then
  TRAIN_ARGS+=(
    --persistent-checkpoint-dir "$PERSISTENT_CHECKPOINT_ROOT"
    --resume-from-checkpoint auto
  )
fi

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NUM_GPUS" \
  -m new_interest.training.train_sft \
  "${TRAIN_ARGS[@]}"

python - \
  "$SFT_DIR/run_summary.json" \
  "$ACTIVE_TRAIN_FILE" "$ACTIVE_WEIGHT_FILE" \
  "$MODE" "$PERSISTENT_CHECKPOINT_ROOT" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

from new_interest.training.checkpoint_persistence import (
    select_latest_complete_checkpoint,
)

summary_path = Path(sys.argv[1])
train_path = Path(sys.argv[2])
weight_path = Path(sys.argv[3])
mode = sys.argv[4]
persistent_root = Path(sys.argv[5])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


summary = json.loads(summary_path.read_text(encoding="utf-8"))
expected_samples = 108 if mode == "smoke" else 21600
expected_epochs = 2 if mode == "smoke" else 3
expected_steps = 8 if mode == "smoke" else 2025
if summary["model_name"] != "Qwen/Qwen3.5-9B":
    raise RuntimeError("Training model identity mismatch")
if summary["world_size"] != 4 or summary["train_samples"] != expected_samples:
    raise RuntimeError("Training DDP/data identity mismatch")
if summary["configured_epochs"] != expected_epochs:
    raise RuntimeError("Training epoch identity mismatch")
if summary["effective_global_batch_size"] != 32:
    raise RuntimeError("Training effective batch mismatch")
if summary["expected_optimizer_steps"] != expected_steps:
    raise RuntimeError("Training optimizer-step identity mismatch")
if summary["gradient_checkpointing_kwargs"] != {"use_reentrant": False}:
    raise RuntimeError("Training gradient checkpointing is not non-reentrant")
if summary["train_file_sha256"] != sha256(train_path):
    raise RuntimeError("Trainer consumed the wrong Train file")
if summary["sample_weight_file_sha256"] != sha256(weight_path):
    raise RuntimeError("Trainer consumed the wrong sample weights")
if not math.isfinite(float(summary["train_metrics"]["train_loss"])):
    raise RuntimeError("Training loss is not finite")
if summary["token_audit"]["samples"] != expected_samples:
    raise RuntimeError("Token audit sample count mismatch")
if summary["token_audit"]["over_max_seq_length"] != 0:
    raise RuntimeError("Token audit found over-length samples")
semantic = summary["model_audit"]["semantic_id_tokens"]
if not semantic["enabled"] or semantic["count"] != 108:
    raise RuntimeError("Semantic-ID model audit mismatch")
if semantic["save_embedding_layers"] is not False:
    raise RuntimeError("Full embedding saving was enabled")
if summary["model_audit"]["forbidden_trainable_parameters"]:
    raise RuntimeError("Forbidden model parameters are trainable")
drift = summary["semantic_token_drift"]
if not drift["all_input_rows_changed"] or not drift["all_output_rows_changed"]:
    raise RuntimeError("At least one Semantic-ID row did not change")
if mode == "full":
    selected, diagnostics = select_latest_complete_checkpoint(
        persistent_root, summary["checkpoint_identity"]
    )
    if diagnostics:
        raise RuntimeError(f"Full checkpoint diagnostics: {diagnostics}")
    if selected is None or selected.name != "checkpoint-2025":
        raise RuntimeError(f"Expected checkpoint-2025, got {selected}")
print("COMBINED_SEMANTIC_ID_GLOBAL_CROSSFIT_TRAIN_OK", mode, expected_samples, expected_steps)
PY

python - "$SFT_DIR" <<'PY'
import json
import sys
from pathlib import Path

from safetensors import safe_open

root = Path(sys.argv[1])
summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
adapter = root / "final_adapter/adapter_model.safetensors"
mapping = root / "final_adapter/interest_token_mapping.parquet"
if not adapter.is_file() or adapter.stat().st_size == 0 or not mapping.is_file():
    raise RuntimeError("Final Semantic-ID adapter is incomplete")
with safe_open(adapter, framework="pt", device="cpu") as handle:
    keys = list(handle.keys())
    shapes = {key: list(handle.get_slice(key).get_shape()) for key in keys}
token_keys = [key for key in keys if "trainable_tokens_delta" in key]
if len(token_keys) != 2 or any(shapes[key][0] != 108 for key in token_keys):
    raise RuntimeError("Semantic-ID compact tensor contract mismatch")
resized_vocab = int(summary["semantic_token_initialization"]["resized_vocabulary_size"])
full_embeddings = [
    key
    for key, shape in shapes.items()
    if len(shape) == 2 and shape[0] == resized_vocab and shape[1] > 108
]
if full_embeddings:
    raise RuntimeError(f"Full embedding matrices were saved: {full_embeddings}")
if not any("lora_" in key for key in keys):
    raise RuntimeError("Final adapter has no LoRA tensors")
audit = {
    "adapter_bytes": adapter.stat().st_size,
    "tensor_count": len(keys),
    "trainable_token_tensor_count": len(token_keys),
    "trainable_token_shapes": {key: shapes[key] for key in token_keys},
    "full_embedding_tensors": full_embeddings,
}
(root / "semantic_adapter_state_audit.json").write_text(
    json.dumps(audit, indent=2) + "\n", encoding="utf-8"
)
print("COMBINED_SEMANTIC_ID_GLOBAL_CROSSFIT_ADAPTER_OK", json.dumps(audit, sort_keys=True))
PY

CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.verify_semantic_id_token_adapter \
  --model-name "$MODEL_NAME" \
  --adapter-path "$SFT_DIR/final_adapter" \
  --mapping-file "$MAPPING_FILE" \
  --reference-file "$SFT_DIR/semantic_reload_reference.json" \
  --output "$SFT_DIR/semantic_adapter_reload_audit.json"

python - "$SFT_DIR/semantic_adapter_reload_audit.json" <<'PY'
import json
import sys
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not audit["all_close"] or audit["sample_count"] != 4:
    raise RuntimeError("Adapter reload logits audit failed")
integrity = audit["adapter_state_integrity"]
if integrity["tensor_count"] != 498 or not integrity["all_tensors_exact"]:
    raise RuntimeError("Exact compact adapter state audit failed")
if integrity["saved_state_sha256"] != integrity["loaded_state_sha256"]:
    raise RuntimeError("Reloaded adapter state hash mismatch")
print("COMBINED_SEMANTIC_ID_GLOBAL_CROSSFIT_RELOAD_OK", integrity["tensor_count"])
PY

mkdir -p "$EVAL_ROOT"
if [[ "$MODE" == "smoke" ]]; then
  CUDA_VISIBLE_DEVICES=0 python -m new_interest.evaluation.evaluate_id_token_ranking \
    --config "$EVAL_CONFIG" \
    --adapter-path "$SFT_DIR/final_adapter" \
    --eval-file "$EVAL_FILE" \
    --interest-token-mapping-file "$MAPPING_FILE" \
    --output-dir "$MERGED_DIR" \
    --run-name "qwen35-behavior-interest-semantic-id-global-crossfit-smoke" \
    --max-examples 64 \
    --num-shards 1 \
    --shard-index 0 \
    --batch-size 8
else
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
        --adapter-path "$SFT_DIR/final_adapter" \
        --eval-file "$EVAL_FILE" \
        --interest-token-mapping-file "$MAPPING_FILE" \
        --output-dir "$SHARD_DIR" \
        --run-name "qwen35-behavior-interest-semantic-id-global-crossfit-real-${SHARD_NAME}" \
        --num-shards 4 \
        --shard-index "$SHARD_INDEX" \
        --batch-size 8 \
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
    echo "At least one real ID-token evaluation shard failed" >&2
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
    --output-dir "$MERGED_DIR" \
    --expected-records 31392
fi

python - "$MERGED_DIR" "$MAPPING_FILE" "$MODE" <<'PY'
import json
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
mapping_file = Path(sys.argv[2])
mode = sys.argv[3]
metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
expected = 64 if mode == "smoke" else 31392
if metrics["samples"] != expected:
    raise RuntimeError(f"Evaluation sample count mismatch: {metrics['samples']}")
supported = int(metrics["supported_interest_count"])
if mode == "full" and supported != 104:
    raise RuntimeError("Full evaluation supported-interest count mismatch")
if mode == "smoke" and not 1 <= supported <= 104:
    raise RuntimeError("Smoke evaluation supported-interest count is invalid")
if metrics["label_vocabulary_size"] != 108:
    raise RuntimeError("Evaluation vocabulary size mismatch")
if metrics["top_k"] != 20 or metrics["cutoffs"] != [1, 5, 10, 20]:
    raise RuntimeError("Evaluation Top-K contract mismatch")
if metrics["history_prediction_filtering"] is not False:
    raise RuntimeError("History filtering was enabled unexpectedly")
if mode == "full":
    if metrics["merged_prediction_count"] != 31392:
        raise RuntimeError("Merged prediction count mismatch")
    if metrics["full_record_count"] != 31392:
        raise RuntimeError("Merged full-record count mismatch")
    if metrics["logical_num_shards"] != 4:
        raise RuntimeError("Merged logical-shard count mismatch")
for cutoff in (1, 5, 10, 20):
    for prefix in ("recall", "ndcg", "interest_macro_recall", "interest_macro_ndcg"):
        key = f"{prefix}_at_{cutoff}"
        if not math.isfinite(float(metrics[key])):
            raise RuntimeError(f"Non-finite evaluation metric: {key}")

labels = {
    str(row["target_subcategory"])
    for row in pq.read_table(mapping_file).to_pylist()
}
rows = pq.read_table(root / "id_token_predictions.parquet").to_pylist()
if len(rows) != expected:
    raise RuntimeError("Prediction row count mismatch")
if len({int(row["record_index"]) for row in rows}) != expected:
    raise RuntimeError("Prediction record indices are not unique")
if len({str(row["impression_id"]) for row in rows}) != expected:
    raise RuntimeError("Prediction impression IDs are not unique")
for row in rows:
    ranked_labels = list(map(str, row["ranked_labels"]))
    ranked_scores = list(map(float, row["ranked_scores"]))
    if len(ranked_labels) != 20 or len(set(ranked_labels)) != 20:
        raise RuntimeError("Prediction is not a unique Top-20 ranking")
    if not set(ranked_labels).issubset(labels):
        raise RuntimeError("Prediction contains a label outside the vocabulary")
    if len(ranked_scores) != 20 or not all(math.isfinite(value) for value in ranked_scores):
        raise RuntimeError("Prediction contains invalid restricted scores")
print(
    "COMBINED_SEMANTIC_ID_GLOBAL_CROSSFIT_EVAL_OK",
    mode,
    metrics["interest_macro_recall_at_10"],
    metrics["interest_macro_ndcg_at_10"],
)
PY

python - \
  "$AMLT_OUTPUT_DIR" "$MODE" \
  "$TRAIN_SHA256" "$EVAL_SHA256" "$MAPPING_SHA256" "$WEIGHT_SHA256" <<'PY'
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]
identity = {
    "train_sha256": sys.argv[3],
    "eval_sha256": sys.argv[4],
    "mapping_sha256": sys.argv[5],
    "weight_sha256": sys.argv[6],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


required = [
    "environment.json",
    "input_audit.json",
    "sft/token_audit.json",
    "sft/semantic_token_initialization.json",
    "sft/model_audit.json",
    "sft/semantic_token_drift.json",
    "sft/semantic_reload_reference.json",
    "sft/run_summary.json",
    "sft/resolved_config.json",
    "sft/semantic_adapter_state_audit.json",
    "sft/semantic_adapter_reload_audit.json",
    "real_eval/merged/id_token_predictions.parquet",
    "real_eval/merged/metrics.json",
]
if mode == "full":
    required.extend(
        [
            "real_eval/merged/per_interest_metrics.parquet",
            "real_eval/merged/merge_manifest.json",
        ]
    )
    required.extend(f"real_eval/shard-{index:02d}.log" for index in range(4))
else:
    unexpected = [
        path
        for path in root.rglob("*")
        if path.is_file() and "checkpoints" in path.parts
    ]
    if unexpected:
        raise RuntimeError(f"Smoke output contains checkpoint files: {unexpected[:3]}")

adapter_dir = root / "sft/final_adapter"
adapter_files = sorted(path for path in adapter_dir.rglob("*") if path.is_file())
if not adapter_files or not (adapter_dir / "adapter_model.safetensors").is_file():
    raise RuntimeError("Final adapter artifact is incomplete")
files = {}
for relative in required:
    path = root / relative
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing required artifact: {relative}")
    files[relative] = sha256(path)
for path in adapter_files:
    relative = path.relative_to(root).as_posix()
    files[relative] = sha256(path)
metrics = json.loads(
    (root / "real_eval/merged/metrics.json").read_text(encoding="utf-8")
)
manifest = {
    "manifest_version": "combined-semantic-id-global-crossfit-v1",
    "mode": mode,
    "analysis_scope": "qwen35-behavior-plus-interest",
    "model_name": "Qwen/Qwen3.5-9B",
    "input_representation": "behavior-headline-plus-interest-category",
    "output_representation": "single-semantic-id-token",
    "weighted_training_used": True,
    "weight_source_view": "behavior-plus-interest-semantic-id-oof",
    "weight_rank_scope": "within-heldout-fold-across-target-classes",
    "class_normalization_used": False,
    "evaluation_scope": "real-dev-only",
    "shuffled_evaluation_used": False,
    "history_prediction_filtering": False,
    "input_identity": identity,
    "evaluation_identity": metrics["evaluation_identity"],
    "files": files,
    "excluded_prefixes": ["checkpoints/", "real_eval/shard-*/shards/"],
}
manifest_path = root / "artifact_manifest.json"
atomic_json(manifest_path, manifest)
marker = {
    "marker_version": "combined-semantic-id-global-crossfit-commit-v1",
    "mode": mode,
    "identity": identity,
    "artifact_manifest_sha256": sha256(manifest_path),
    "evaluation_identity": metrics["evaluation_identity"],
    "file_count": len(files),
}
atomic_json(root / f"{mode}_success.json", marker)
print("COMBINED_SEMANTIC_ID_GLOBAL_CROSSFIT_GATE_OK", mode, len(files))
PY
