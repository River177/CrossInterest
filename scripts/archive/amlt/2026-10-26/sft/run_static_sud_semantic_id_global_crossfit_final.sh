#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${SEMANTIC_ID_GLOBAL_CROSSFIT_NUM_GPUS:-4}"
if [[ "$NUM_GPUS" != "4" ]]; then
  echo "Semantic-ID global cross-fit final requires four GPUs; got $NUM_GPUS" >&2
  exit 2
fi
if [[ "${AMLT_JOB_NAME:?AMLT_JOB_NAME is required}" != "qwen35-interest-semantic-id-global-crossfit-final" ]]; then
  echo "Unexpected Semantic-ID final job name: $AMLT_JOB_NAME" >&2
  exit 2
fi

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export WANDB_DISABLED=true
export HF_HOME="${HF_HOME:-/mnt/default/huggingface-cache}"
mkdir -p "$HF_HOME"

MODEL_NAME="Qwen/Qwen3.5-9B"
CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_semantic_id_global_crossfit_qwen35_interest_weighted_v1.yaml"
EVAL_CONFIG="$AMLT_CODE_DIR/configs/id_token/static_sud_qwen35_interest_full.json"
SEMANTIC_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_semantic_id_token_200_h50_v1"
TRAIN_FILE="$SEMANTIC_ROOT/train/interest_only/21600.parquet"
EVAL_FILE="$SEMANTIC_ROOT/eval/interest_only/dev_full_grouped.parquet"
MAPPING_FILE="$SEMANTIC_ROOT/interest_token_mapping.parquet"
WEIGHT_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_semantic_id_global_crossfit_qwen35_v1"
WEIGHT_FILE="$WEIGHT_ROOT/sample_weights.parquet"
WEIGHT_SUMMARY="$WEIGHT_ROOT/weight_summary.json"
FOLD_MANIFEST="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_crossfit_reweighting_qwen35_v1/fold_manifest.parquet"
SFT_DIR="$AMLT_OUTPUT_DIR/sft"
EVAL_ROOT="$AMLT_OUTPUT_DIR/real_eval"
MERGED_DIR="$EVAL_ROOT/merged"
LOCAL_CHECKPOINT_ROOT="/scratch/semantic-id-global-crossfit-final-checkpoints"
PERSISTENT_CHECKPOINT_ROOT="$AMLT_OUTPUT_DIR/checkpoints"
FINAL_MARKER="$AMLT_OUTPUT_DIR/final_success.json"

TRAIN_SHA256="c823d34cad5a921b887e2f8952ca8d117a54f2114b732e4d60c80ca649423c64"
EVAL_SHA256="d5d7b8a9a759447212e7bac8a78b894bd9a4564ccb10ed60d6971d4530ee4ac5"
MAPPING_SHA256="45397451ac2ae2e64d8156b2b47be3f454cc309a8d30a83bc23c2d3267653a0a"
FOLD_MANIFEST_SHA256="3d25e36b4c34ae2516ac6fccee3e2c4d6596ac610076c6ef0f8b8505cd3a4b3a"
WEIGHT_SHA256="685657ea1893d8cf5ca088ceca88c619e45f16e95631bd11edf83b434a66ff4f"
WEIGHT_SUMMARY_SHA256="82b05e168376b570a4d7113ed5becc06cc5b78eb7bb6696872811ba69cd302fd"

for path in \
  "$CONFIG" \
  "$EVAL_CONFIG" \
  "$TRAIN_FILE" \
  "$EVAL_FILE" \
  "$MAPPING_FILE" \
  "$WEIGHT_FILE" \
  "$WEIGHT_SUMMARY" \
  "$FOLD_MANIFEST"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing final-run input: $path" >&2
    exit 1
  fi
done

if [[ -f "$FINAL_MARKER" ]]; then
  python - "$AMLT_OUTPUT_DIR" "$TRAIN_SHA256" "$EVAL_SHA256" "$MAPPING_SHA256" "$WEIGHT_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "train_sha256": sys.argv[2],
    "eval_sha256": sys.argv[3],
    "mapping_sha256": sys.argv[4],
    "weight_sha256": sys.argv[5],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


marker = json.loads((root / "final_success.json").read_text(encoding="utf-8"))
if marker.get("identity") != expected:
    raise RuntimeError("Final commit marker identity mismatch")
manifest_path = root / "artifact_manifest.json"
if sha256(manifest_path) != marker.get("artifact_manifest_sha256"):
    raise RuntimeError("Final artifact manifest hash mismatch")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for relative, expected_hash in manifest["compact_files"].items():
    path = root / relative
    if not path.is_file() or sha256(path) != expected_hash:
        raise RuntimeError(f"Final committed artifact is missing or corrupt: {relative}")
for relative, expected_hash in manifest["final_adapter_files"].items():
    path = root / "sft/final_adapter" / relative
    if not path.is_file() or sha256(path) != expected_hash:
        raise RuntimeError(f"Final committed adapter is missing or corrupt: {relative}")
print("SEMANTIC_ID_GLOBAL_CROSSFIT_FINAL_REUSE_OK", len(manifest["compact_files"]))
PY
  echo "SEMANTIC_ID_GLOBAL_CROSSFIT_FINAL_GATE_OK"
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
    "packages": {package: importlib.metadata.version(package) for package in packages},
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
print("SEMANTIC_ID_GLOBAL_CROSSFIT_FINAL_ENV_OK", json.dumps(payload, sort_keys=True))
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
    files = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
    missing = [
        filename
        for filename in sorted(files)
        if not (root / filename).is_file() or (root / filename).stat().st_size == 0
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
print("SEMANTIC_ID_GLOBAL_CROSSFIT_FINAL_MODEL_READY", name, snapshot, source)
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python - \
  "$CONFIG" \
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
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml

config_path = Path(sys.argv[1])
train_path = Path(sys.argv[2])
eval_path = Path(sys.argv[4])
mapping_path = Path(sys.argv[6])
fold_manifest_path = Path(sys.argv[8])
weight_path = Path(sys.argv[10])
weight_summary_path = Path(sys.argv[12])
output_path = Path(sys.argv[14])
expected_hashes = {
    train_path: sys.argv[3],
    eval_path: sys.argv[5],
    mapping_path: sys.argv[7],
    fold_manifest_path: sys.argv[9],
    weight_path: sys.argv[11],
    weight_summary_path: sys.argv[13],
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
if config["model_name"] != "Qwen/Qwen3.5-9B":
    raise RuntimeError("Final config model identity mismatch")
if config["train_file"] != "data/training/sft_static_user_disjoint_semantic_id_token_200_h50_v1/train/interest_only/21600.parquet":
    raise RuntimeError("Final config train file mismatch")
if config["sample_weight_file"] != "data/training/sft_static_user_disjoint_semantic_id_global_crossfit_qwen35_v1/sample_weights.parquet":
    raise RuntimeError("Final config weight file mismatch")
if config["semantic_id_token_mapping_file"] != "data/training/sft_static_user_disjoint_semantic_id_token_200_h50_v1/interest_token_mapping.parquet":
    raise RuntimeError("Final config mapping file mismatch")
if config.get("enable_thinking") is not False:
    raise RuntimeError("Final config unexpectedly enables thinking")
if config.get("gradient_checkpointing_use_reentrant") is not False:
    raise RuntimeError("Final config must use non-reentrant checkpointing")
if int(config["epochs"]) != 3:
    raise RuntimeError("Final config epoch mismatch")
if int(config["per_device_train_batch_size"]) != 1 or int(config["gradient_accumulation_steps"]) != 8:
    raise RuntimeError("Final config batch identity mismatch")

mapping_rows = pq.read_table(mapping_path).to_pylist()
if len(mapping_rows) != 108:
    raise RuntimeError("Mapping does not contain exactly 108 Semantic IDs")
token_by_label = {
    str(row["target_subcategory"]): str(row["interest_token"])
    for row in mapping_rows
}
if len(token_by_label) != 108 or len(set(token_by_label.values())) != 108:
    raise RuntimeError("Semantic-ID mapping is not one-to-one")

train_rows = pq.read_table(train_path).to_pylist()
if len(train_rows) != 21600:
    raise RuntimeError("Final Train is not exactly 21,600 rows")
source_ids = [str(row["source_sample_id"]) for row in train_rows]
user_ids = [str(row["user_id"]) for row in train_rows]
if len(set(source_ids)) != 21600 or len(set(user_ids)) != 21600:
    raise RuntimeError("Final Train must have one row and one target per user")
if set(str(row["task_type"]) for row in train_rows) != {"interest_only"}:
    raise RuntimeError("Final Train is not Interest-only")
if set(str(row["target_subcategory"]) for row in train_rows) != set(token_by_label):
    raise RuntimeError("Final Train does not cover all 108 mapped labels")
for row in train_rows:
    target = str(row["target_subcategory"])
    response = str(row["response"])
    messages = row["messages"]
    if response != token_by_label[target] or not re.fullmatch(r"<interest_\d{3}>", response):
        raise RuntimeError("Final Train has a non-Semantic-ID target")
    if len(messages) != 2 or messages[0]["role"] != "user" or messages[1] != {"role": "assistant", "content": response}:
        raise RuntimeError("Final Train message contract mismatch")

weights = pq.read_table(weight_path).to_pylist()
if len(weights) != 21600:
    raise RuntimeError("Global weight file is not exactly 21,600 rows")
weight_by_source = {str(row["source_sample_id"]): row for row in weights}
if set(weight_by_source) != set(source_ids) or len(weight_by_source) != 21600:
    raise RuntimeError("Global weights do not exactly cover final Train")
manifest_rows = pq.read_table(fold_manifest_path).to_pylist()
manifest_by_source = {
    str(row["source_sample_id"]): row for row in manifest_rows
}
if len(manifest_by_source) != 21600 or set(manifest_by_source) != set(source_ids):
    raise RuntimeError("Frozen fold manifest does not exactly cover final Train")
train_by_source = {str(row["source_sample_id"]): row for row in train_rows}
fold_counts = Counter()
fold_class_counts = Counter()
fold_ranks = {fold: set() for fold in range(5)}
values = []
for source_id, row in weight_by_source.items():
    train_row = train_by_source[source_id]
    manifest_row = manifest_by_source[source_id]
    if str(row["user_id"]) != str(train_row["user_id"]) or str(row["target_subcategory"]) != str(train_row["target_subcategory"]):
        raise RuntimeError("Global weight identity differs from Train")
    fold = int(row["fold"])
    if (
        str(manifest_row["user_id"]) != str(train_row["user_id"])
        or str(manifest_row["target_subcategory"]) != str(train_row["target_subcategory"])
        or int(manifest_row["fold"]) != fold
    ):
        raise RuntimeError("Global weight identity differs from frozen fold manifest")
    fold_counts[fold] += 1
    fold_class_counts[(fold, str(row["target_subcategory"]))] += 1
    rank = int(row["fold_global_rank"])
    fold_ranks[fold].add(rank)
    quantile = float(row["fold_global_difficulty_quantile"])
    raw_weight = float(row["raw_weight"])
    value = float(row["sample_weight"])
    if not 0 <= rank < 4320 or not math.isclose(quantile, rank / 4319, abs_tol=1e-15):
        raise RuntimeError("Global fold rank/quantile contract mismatch")
    if not math.isclose(raw_weight, 1.0 / (1.0 + quantile), abs_tol=1e-15):
        raise RuntimeError("Global raw weight formula mismatch")
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError("Global sample weight is not positive and finite")
    if not math.isfinite(float(row["prior_adjusted_nll"])) or not math.isfinite(float(row["prior_adjusted_margin"])):
        raise RuntimeError("Global weights contain non-finite adjusted scores")
    values.append(value)
if fold_counts != Counter({fold: 4320 for fold in range(5)}):
    raise RuntimeError(f"Global weight fold counts mismatch: {fold_counts}")
if any(ranks != set(range(4320)) for ranks in fold_ranks.values()):
    raise RuntimeError("Global within-fold ranks are not complete and unique")
if len(fold_class_counts) != 540 or set(fold_class_counts.values()) != {40}:
    raise RuntimeError("Global weights are not exactly 40 rows per class per fold")
if not math.isclose(sum(values) / len(values), 1.0, abs_tol=1e-12):
    raise RuntimeError("Global sample-weight mean is not one")
if not math.isclose(max(values) / min(values), 2.0, abs_tol=1e-12):
    raise RuntimeError("Global sample-weight ratio is not two")
class_means = {}
for label in token_by_label:
    selected = [float(row["sample_weight"]) for row in weights if str(row["target_subcategory"]) == label]
    class_means[label] = sum(selected) / len(selected)
if math.isclose(min(class_means.values()), max(class_means.values()), abs_tol=1e-12):
    raise RuntimeError("Class means do not vary; class normalization may have been applied")

weight_summary = json.loads(weight_summary_path.read_text(encoding="utf-8"))
if weight_summary["formula"]["class_normalization"] is not False:
    raise RuntimeError("Weight summary enables class normalization")
if weight_summary["output_sha256"]["sample_weights"] != expected_hashes[weight_path]:
    raise RuntimeError("Weight summary output identity mismatch")

eval_rows = pq.read_table(eval_path).to_pylist()
if len(eval_rows) != 31392 or len({str(row["impression_id"]) for row in eval_rows}) != 31392:
    raise RuntimeError("Real Dev is not exactly 31,392 unique impressions")
if any(not set(map(str, row["acceptable_responses"])).issubset(set(token_by_label.values())) for row in eval_rows):
    raise RuntimeError("Real Dev contains an unmapped Semantic-ID response")
supported = set()
for row in eval_rows:
    supported.update(map(str, row["acceptable_subcategories"]))
if len(supported) != 104:
    raise RuntimeError("Real Dev does not have exactly 104 supported interests")

audit = {
    "audit_version": "semantic-id-global-crossfit-final-input-v1",
    "analysis_scope": "qwen35-interest-only",
    "train_rows": len(train_rows),
    "unique_train_users": len(set(user_ids)),
    "classes": len(token_by_label),
    "weight_rows": len(weights),
    "fold_counts": dict(sorted(fold_counts.items())),
    "global_weight_mean": sum(values) / len(values),
    "global_weight_max_min_ratio": max(values) / min(values),
    "class_weight_mean_min": min(class_means.values()),
    "class_weight_mean_max": max(class_means.values()),
    "class_normalization": False,
    "real_eval_rows": len(eval_rows),
    "real_eval_supported_interests": len(supported),
    "input_sha256": {path.name: expected for path, expected in expected_hashes.items()},
    "shuffled_evaluation_used": False,
    "history_prediction_filtering": False,
}
output_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
print("SEMANTIC_ID_GLOBAL_CROSSFIT_FINAL_INPUT_OK", json.dumps(audit, sort_keys=True))
PY

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NUM_GPUS" \
  -m new_interest.training.train_sft \
  --config "$CONFIG" \
  --train-file "$TRAIN_FILE" \
  --sample-weight-file "$WEIGHT_FILE" \
  --output-dir "$SFT_DIR" \
  --checkpoint-dir "$LOCAL_CHECKPOINT_ROOT" \
  --persistent-checkpoint-dir "$PERSISTENT_CHECKPOINT_ROOT" \
  --resume-from-checkpoint auto \
  --run-name "static-sud-semantic-id-global-crossfit-qwen35-interest-weighted-seed42" \
  --epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --save-steps 100 \
  --wandb-mode disabled

python - "$SFT_DIR/run_summary.json" "$SFT_DIR/checkpoint_audit.json" "$TRAIN_SHA256" "$WEIGHT_SHA256" <<'PY'
import json
import math
import sys
from pathlib import Path

from new_interest.training.checkpoint_persistence import select_latest_complete_checkpoint

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
audit_path = Path(sys.argv[2])
train_sha = sys.argv[3]
weight_sha = sys.argv[4]
if summary["model_name"] != "Qwen/Qwen3.5-9B":
    raise RuntimeError("Final model identity mismatch")
if summary["world_size"] != 4 or summary["train_samples"] != 21600:
    raise RuntimeError("Final DDP/data identity mismatch")
if summary["configured_epochs"] != 3 or summary["effective_global_batch_size"] != 32:
    raise RuntimeError("Final epoch/effective-batch identity mismatch")
if summary["expected_optimizer_steps"] != 2025:
    raise RuntimeError("Final optimizer-step identity mismatch")
if summary["gradient_checkpointing_kwargs"] != {"use_reentrant": False}:
    raise RuntimeError("Final gradient checkpointing is not non-reentrant")
if summary["train_file_sha256"] != train_sha:
    raise RuntimeError("Final trainer consumed the wrong Train file")
if summary["sample_weight_file_sha256"] != weight_sha:
    raise RuntimeError("Final trainer consumed the wrong global weight file")
if not summary["sample_weight_file"]:
    raise RuntimeError("Final trainer did not use sample weights")
if not math.isfinite(float(summary["train_metrics"]["train_loss"])):
    raise RuntimeError("Final training loss is not finite")
if summary["token_audit"]["samples"] != 21600 or summary["token_audit"]["over_max_seq_length"] != 0:
    raise RuntimeError("Final token audit mismatch")
semantic = summary["model_audit"]["semantic_id_tokens"]
if not semantic["enabled"] or semantic["count"] != 108 or semantic["save_embedding_layers"] is not False:
    raise RuntimeError("Final Semantic-ID model audit mismatch")
if summary["model_audit"]["forbidden_trainable_parameters"]:
    raise RuntimeError("Final model has forbidden trainable parameters")
drift = summary["semantic_token_drift"]
if not drift["all_input_rows_changed"] or not drift["all_output_rows_changed"]:
    raise RuntimeError("Final training left a Semantic-ID row unchanged")
persistent_root = Path(summary["persistent_checkpoint_dir"])
selected, diagnostics = select_latest_complete_checkpoint(persistent_root, summary["checkpoint_identity"])
if diagnostics:
    raise RuntimeError(f"Final checkpoint diagnostics: {diagnostics}")
if selected is None or selected.name != "checkpoint-2025":
    raise RuntimeError(f"Expected complete checkpoint-2025, got {selected}")
audit = {
    "checkpoint_name": selected.name,
    "checkpoint_identity": summary["checkpoint_identity"],
    "resumed_from_checkpoint": summary["resumed_from_checkpoint"],
    "identity_safe_persistence": True,
    "checkpoints_excluded_from_download": True,
}
audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
print("SEMANTIC_ID_GLOBAL_CROSSFIT_FINAL_TRAIN_OK", 21600, 2025)
PY

python - "$SFT_DIR" <<'PY'
import json
import sys
from pathlib import Path

from safetensors import safe_open

root = Path(sys.argv[1])
summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
adapter = root / "final_adapter/adapter_model.safetensors"
if not adapter.is_file() or adapter.stat().st_size == 0:
    raise RuntimeError("Missing or empty final adapter")
if not (root / "final_adapter/interest_token_mapping.parquet").is_file():
    raise RuntimeError("Final adapter misses the interest-token mapping")
with safe_open(adapter, framework="pt", device="cpu") as handle:
    keys = list(handle.keys())
    shapes = {key: list(handle.get_slice(key).get_shape()) for key in keys}
token_keys = [key for key in keys if "trainable_tokens_delta" in key]
if len(token_keys) != 2 or any(shapes[key][0] != 108 for key in token_keys):
    raise RuntimeError("Final adapter Semantic-ID tensor contract mismatch")
resized_vocab = int(summary["semantic_token_initialization"]["resized_vocabulary_size"])
full_embeddings = [
    key for key, shape in shapes.items()
    if len(shape) == 2 and shape[0] == resized_vocab and shape[1] > 108
]
if full_embeddings:
    raise RuntimeError(f"Full embedding matrices were saved: {full_embeddings}")
if not any("lora_" in key for key in keys):
    raise RuntimeError("Final adapter contains no LoRA tensors")
audit = {
    "adapter_bytes": adapter.stat().st_size,
    "tensor_count": len(keys),
    "trainable_token_tensor_count": len(token_keys),
    "trainable_token_shapes": {key: shapes[key] for key in token_keys},
    "full_embedding_tensors": full_embeddings,
}
(root / "semantic_adapter_state_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
print("SEMANTIC_ID_GLOBAL_CROSSFIT_FINAL_ADAPTER_OK", json.dumps(audit, sort_keys=True))
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
    raise RuntimeError("Final adapter reload logits audit failed")
integrity = audit["adapter_state_integrity"]
if integrity["tensor_count"] != 498 or not integrity["all_tensors_exact"]:
    raise RuntimeError("Final exact 498-tensor reload audit failed")
if integrity["saved_state_sha256"] != integrity["loaded_state_sha256"]:
    raise RuntimeError("Final reloaded adapter state hash mismatch")
print("SEMANTIC_ID_GLOBAL_CROSSFIT_FINAL_RELOAD_OK", integrity["tensor_count"])
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
      --adapter-path "$SFT_DIR/final_adapter" \
      --eval-file "$EVAL_FILE" \
      --interest-token-mapping-file "$MAPPING_FILE" \
      --output-dir "$SHARD_DIR" \
      --run-name "qwen35-interest-semantic-id-global-crossfit-real-${SHARD_NAME}" \
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

python - "$MERGED_DIR" "$MAPPING_FILE" <<'PY'
import json
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
mapping_file = Path(sys.argv[2])
metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
if metrics["samples"] != 31392 or metrics["merged_prediction_count"] != 31392 or metrics["full_record_count"] != 31392:
    raise RuntimeError("Merged real prediction count is not exactly 31,392")
if metrics["logical_num_shards"] != 4:
    raise RuntimeError("Real evaluation did not use four logical shards")
if metrics["supported_interest_count"] != 104 or metrics["label_vocabulary_size"] != 108:
    raise RuntimeError("Real evaluation support/vocabulary identity mismatch")
if metrics["top_k"] != 20 or metrics["cutoffs"] != [1, 5, 10, 20]:
    raise RuntimeError("Real evaluation did not preserve @1/@5/@10/@20 Top-20")
if metrics["history_prediction_filtering"] is not False:
    raise RuntimeError("History filtering was enabled unexpectedly")
for cutoff in (1, 5, 10, 20):
    for prefix in ("recall", "ndcg", "interest_macro_recall", "interest_macro_ndcg"):
        key = f"{prefix}_at_{cutoff}"
        if not math.isfinite(float(metrics[key])):
            raise RuntimeError(f"Non-finite real metric: {key}")

labels = {
    str(row["target_subcategory"])
    for row in pq.read_table(mapping_file).to_pylist()
}
rows = pq.read_table(root / "id_token_predictions.parquet").to_pylist()
if len(rows) != 31392:
    raise RuntimeError("Merged real prediction file has the wrong row count")
if {int(row["record_index"]) for row in rows} != set(range(31392)):
    raise RuntimeError("Merged real prediction record indices are not unique and complete")
if len({str(row["impression_id"]) for row in rows}) != 31392:
    raise RuntimeError("Merged real predictions do not contain unique impressions")
for row in rows:
    ranked_labels = list(map(str, row["ranked_labels"]))
    ranked_scores = list(map(float, row["ranked_scores"]))
    if len(ranked_labels) != 20 or len(set(ranked_labels)) != 20 or not set(ranked_labels).issubset(labels):
        raise RuntimeError("Merged real row is not a legal unique Top-20 ranking")
    if len(ranked_scores) != 20 or not all(math.isfinite(value) for value in ranked_scores):
        raise RuntimeError("Merged real row has invalid restricted scores")
print(
    "SEMANTIC_ID_GLOBAL_CROSSFIT_FINAL_REAL_EVAL_OK",
    metrics["interest_macro_recall_at_10"],
    metrics["interest_macro_ndcg_at_10"],
)
PY

python - \
  "$AMLT_OUTPUT_DIR" \
  "$TRAIN_SHA256" "$EVAL_SHA256" "$MAPPING_SHA256" "$WEIGHT_SHA256" <<'PY'
import hashlib
import json
import os
import shutil
import sys
import tarfile
import uuid
from pathlib import Path

root = Path(sys.argv[1])
identity = {
    "train_sha256": sys.argv[2],
    "eval_sha256": sys.argv[3],
    "mapping_sha256": sys.argv[4],
    "weight_sha256": sys.argv[5],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_bytes(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.partial-{uuid.uuid4().hex}")
    with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    os.replace(temporary, destination)


def atomic_json(destination: Path, payload: object) -> None:
    temporary = destination.with_name(
        f".{destination.name}.partial-{uuid.uuid4().hex}"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


exports = {
    root / "real_eval/merged/id_token_predictions.parquet": root / "real_dev_id_token_predictions.parquet",
    root / "real_eval/merged/metrics.json": root / "real_dev_metrics.json",
    root / "real_eval/merged/per_interest_metrics.parquet": root / "real_dev_per_interest_metrics.parquet",
    root / "real_eval/merged/merge_manifest.json": root / "real_dev_merge_manifest.json",
    root / "sft/token_audit.json": root / "final_sft_token_audit.json",
    root / "sft/semantic_token_initialization.json": root / "final_sft_semantic_token_initialization.json",
    root / "sft/model_audit.json": root / "final_sft_model_audit.json",
    root / "sft/semantic_token_drift.json": root / "final_sft_semantic_token_drift.json",
    root / "sft/semantic_reload_reference.json": root / "final_sft_semantic_reload_reference.json",
    root / "sft/run_summary.json": root / "final_sft_run_summary.json",
    root / "sft/resolved_config.json": root / "final_sft_resolved_config.json",
    root / "sft/checkpoint_audit.json": root / "final_sft_checkpoint_audit.json",
    root / "sft/semantic_adapter_state_audit.json": root / "final_sft_semantic_adapter_state_audit.json",
    root / "sft/semantic_adapter_reload_audit.json": root / "final_sft_semantic_adapter_reload_audit.json",
}
for source, destination in exports.items():
    copy_bytes(source, destination)
copy_bytes(
    Path(os.environ["AMLT_CODE_DIR"])
    / "data/training/sft_static_user_disjoint_semantic_id_global_crossfit_qwen35_v1/weight_summary.json",
    root / "global_weight_summary.json",
)
for index in range(4):
    copy_bytes(root / f"real_eval/shard-{index:02d}.log", root / f"real_eval_shard_{index:02d}.log")

adapter_dir = root / "sft/final_adapter"
adapter_files = {
    path.relative_to(adapter_dir).as_posix(): sha256(path)
    for path in sorted(adapter_dir.rglob("*"))
    if path.is_file()
}
if not adapter_files or "adapter_model.safetensors" not in adapter_files:
    raise RuntimeError("Final adapter export is incomplete")
tar_path = root / "final_adapter.tar"
temporary_tar = root / f".final_adapter.tar.partial-{uuid.uuid4().hex}"
with tarfile.open(temporary_tar, "w") as archive:
    archive.add(adapter_dir, arcname="final_adapter", recursive=True)
os.replace(temporary_tar, tar_path)

relative_files = [
    "environment.json",
    "input_audit.json",
    "global_weight_summary.json",
    "final_sft_token_audit.json",
    "final_sft_semantic_token_initialization.json",
    "final_sft_model_audit.json",
    "final_sft_semantic_token_drift.json",
    "final_sft_semantic_reload_reference.json",
    "final_sft_run_summary.json",
    "final_sft_resolved_config.json",
    "final_sft_checkpoint_audit.json",
    "final_sft_semantic_adapter_state_audit.json",
    "final_sft_semantic_adapter_reload_audit.json",
    "real_dev_id_token_predictions.parquet",
    "real_dev_metrics.json",
    "real_dev_per_interest_metrics.parquet",
    "real_dev_merge_manifest.json",
    "real_eval_shard_00.log",
    "real_eval_shard_01.log",
    "real_eval_shard_02.log",
    "real_eval_shard_03.log",
    "final_adapter.tar",
]
compact_files = {}
for relative in relative_files:
    path = root / relative
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing final compact artifact: {relative}")
    compact_files[relative] = sha256(path)
metrics = json.loads((root / "real_dev_metrics.json").read_text(encoding="utf-8"))
manifest = {
    "manifest_version": "semantic-id-global-crossfit-final-real-v1",
    "analysis_scope": "qwen35-interest-only",
    "model_name": "Qwen/Qwen3.5-9B",
    "weighted_training_used": True,
    "global_class_normalization_used": False,
    "evaluation_scope": "real-dev-only",
    "logical_eval_shards": 4,
    "shuffled_evaluation_used": False,
    "history_prediction_filtering": False,
    "input_identity": identity,
    "evaluation_identity": metrics["evaluation_identity"],
    "compact_files": compact_files,
    "final_adapter_files": adapter_files,
    "excluded_prefixes": [
        "checkpoints/",
        "real_eval/shard-00/shards/",
        "real_eval/shard-01/shards/",
        "real_eval/shard-02/shards/",
        "real_eval/shard-03/shards/",
    ],
}
manifest_path = root / "artifact_manifest.json"
atomic_json(manifest_path, manifest)
marker = {
    "marker_version": "semantic-id-global-crossfit-final-commit-v1",
    "identity": identity,
    "artifact_manifest_sha256": sha256(manifest_path),
    "evaluation_identity": metrics["evaluation_identity"],
    "compact_file_count": len(compact_files),
    "final_adapter_file_count": len(adapter_files),
}
atomic_json(root / "final_success.json", marker)
print("SEMANTIC_ID_GLOBAL_CROSSFIT_FINAL_ARTIFACTS_OK", len(compact_files), len(adapter_files))
PY

echo "SEMANTIC_ID_GLOBAL_CROSSFIT_FINAL_GATE_OK"
