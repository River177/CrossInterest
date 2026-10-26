#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: run_static_sud_semantic_id_vanilla_scaling.sh <smoke|full> <train-size>}"
TRAIN_SIZE="${2:?Usage: run_static_sud_semantic_id_vanilla_scaling.sh <smoke|full> <train-size>}"
case "$TRAIN_SIZE" in
  5400) EXPECTED_FULL_STEPS=507 ;;
  10800) EXPECTED_FULL_STEPS=1014 ;;
  16200) EXPECTED_FULL_STEPS=1521 ;;
  21600) EXPECTED_FULL_STEPS=2025 ;;
  *)
    echo "Unsupported Vanilla scaling Train size: $TRAIN_SIZE" >&2
    exit 2
    ;;
esac
case "$MODE" in
  smoke)
    if [[ "$TRAIN_SIZE" != "5400" ]]; then
      echo "The scaling smoke is fixed to the nested 5,400-row subset" >&2
      exit 2
    fi
    EXPECTED_JOB_NAME="qwen35-bi-semantic-id-vanilla-scaling-smoke"
    ;;
  full)
    EXPECTED_JOB_NAME="qwen35-bi-semantic-id-vanilla-${TRAIN_SIZE}"
    ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    exit 2
    ;;
esac

NUM_GPUS="${SEMANTIC_ID_VANILLA_SCALING_NUM_GPUS:-4}"
if [[ "$NUM_GPUS" != "4" ]]; then
  echo "Vanilla scaling is fixed to four GPUs; got $NUM_GPUS" >&2
  exit 2
fi
if [[ "${AMLT_JOB_NAME:?AMLT_JOB_NAME is required}" != "$EXPECTED_JOB_NAME" ]]; then
  echo "Unexpected Vanilla scaling job name: $AMLT_JOB_NAME" >&2
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
CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_semantic_id_qwen35_behavior_interest_vanilla_${TRAIN_SIZE}_v1.yaml"
EVAL_CONFIG="$AMLT_CODE_DIR/configs/id_token/static_sud_qwen35_behavior_interest_vanilla_scaling_v1.json"
DATA_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_scaling_v1"
TRAIN_FILE="$DATA_ROOT/train/behavior_interest/${TRAIN_SIZE}.parquet"
EVAL_FILE="$DATA_ROOT/eval/behavior_interest/dev_full_grouped.parquet"
MAPPING_FILE="$DATA_ROOT/interest_token_mapping.parquet"
STATISTICS_FILE="$DATA_ROOT/statistics.json"
METADATA_FILE="$DATA_ROOT/metadata.json"
SUBSET_MANIFEST="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_balanced_200_h50_v1/subset_manifest.parquet"
SFT_DIR="$AMLT_OUTPUT_DIR/sft"
EVAL_ROOT="$AMLT_OUTPUT_DIR/real_eval"
MERGED_DIR="$EVAL_ROOT/merged"
LOCAL_CHECKPOINT_ROOT="/scratch/semantic-id-vanilla-scaling-${TRAIN_SIZE}-${MODE}-checkpoints"
PERSISTENT_CHECKPOINT_ROOT="$AMLT_OUTPUT_DIR/checkpoints"
SUCCESS_MARKER="$AMLT_OUTPUT_DIR/${MODE}_success.json"

case "$TRAIN_SIZE" in
  5400) TRAIN_SHA256="cc1a5921ccaf83bfcef17bdecbcf756561e97448be54e28c3b8e3a673333f72c" ;;
  10800) TRAIN_SHA256="ebcb2f322fc994c19c630cdb16b910e92ad56ebfd012c148b2aa9680a492559a" ;;
  16200) TRAIN_SHA256="188ce84cca2f90afca88a6338bb461c5d40a85dcae0c2a5e6076865d9feb5479" ;;
  21600) TRAIN_SHA256="e76acd152ed6e0e92f8b8015dfbc599603267a480d9321b41b5acb95d10487f0" ;;
esac
EVAL_SHA256="c05b20033ce0d1a0c2d4b7bb97ca0cd6b7cffb7f198a7265cce9e54046b665fd"
MAPPING_SHA256="45397451ac2ae2e64d8156b2b47be3f454cc309a8d30a83bc23c2d3267653a0a"
STATISTICS_SHA256="558584597d2cbfe30159371433afee079535f0cc236da58529fd44ce8fd0f04c"
METADATA_SHA256="5d1a07b3b450083426e7469292159b2ce01edb8584c466210ccd0adc2e6d8c48"
SUBSET_MANIFEST_SHA256="bcd67a3d0b200bfea5ccbd1ace47fa6ec19dc22e3b84fa992376ba78fb018e4b"

for required_file in \
  "$CONFIG" "$EVAL_CONFIG" "$TRAIN_FILE" "$EVAL_FILE" \
  "$MAPPING_FILE" "$STATISTICS_FILE" "$METADATA_FILE" \
  "$SUBSET_MANIFEST"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing Vanilla scaling input: $required_file" >&2
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
marker_file = Path(sys.argv[2])


def sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


marker = json.loads(marker_file.read_text(encoding="utf-8"))
manifest_file = root / "artifact_manifest.json"
if sha256(manifest_file) != marker["artifact_manifest_sha256"]:
    raise RuntimeError("Committed artifact manifest hash mismatch")
manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
for relative, expected in manifest["files"].items():
    artifact_file = root / relative
    if not artifact_file.is_file() or sha256(artifact_file) != expected:
        raise RuntimeError(f"Committed artifact is missing or corrupt: {relative}")
print("SEMANTIC_ID_VANILLA_SCALING_REUSE_OK", len(manifest["files"]))
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
expected = {"transformers": "5.14.1", "peft": "0.18.0", "accelerate": "1.12.0"}
for name, version in expected.items():
    if payload["packages"][name] != version:
        raise RuntimeError(f"Unexpected {name} version: {payload['packages'][name]}")
if payload["cuda_devices"] != 4:
    raise RuntimeError("Expected exactly four visible GPUs")
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("SEMANTIC_ID_VANILLA_SCALING_ENV_OK", json.dumps(payload, sort_keys=True))
PY

python - \
  "$CONFIG" "$EVAL_CONFIG" \
  "$TRAIN_FILE" "$TRAIN_SHA256" \
  "$EVAL_FILE" "$EVAL_SHA256" \
  "$MAPPING_FILE" "$MAPPING_SHA256" \
  "$STATISTICS_FILE" "$STATISTICS_SHA256" \
  "$METADATA_FILE" "$METADATA_SHA256" \
  "$SUBSET_MANIFEST" "$SUBSET_MANIFEST_SHA256" \
  "$TRAIN_SIZE" "$EXPECTED_FULL_STEPS" \
  "$AMLT_OUTPUT_DIR/input_audit.json" <<'PY'
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml

config_file = Path(sys.argv[1])
eval_config_file = Path(sys.argv[2])
train_file = Path(sys.argv[3])
eval_file = Path(sys.argv[5])
mapping_file = Path(sys.argv[7])
statistics_file = Path(sys.argv[9])
metadata_file = Path(sys.argv[11])
subset_manifest_file = Path(sys.argv[13])
train_size = int(sys.argv[15])
expected_steps = int(sys.argv[16])
output_file = Path(sys.argv[17])
expected_hashes = {
    train_file: sys.argv[4],
    eval_file: sys.argv[6],
    mapping_file: sys.argv[8],
    statistics_file: sys.argv[10],
    metadata_file: sys.argv[12],
    subset_manifest_file: sys.argv[14],
}


def sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


for input_file, expected in expected_hashes.items():
    actual = sha256(input_file)
    if actual != expected:
        raise RuntimeError(f"Frozen input hash mismatch for {input_file}: {actual}")

config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
expected_train = (
    "data/training/sft_static_user_disjoint_semantic_id_token_200_h50_"
    f"behavior_interest_scaling_v1/train/behavior_interest/{train_size}.parquet"
)
expected_mapping = (
    "data/training/sft_static_user_disjoint_semantic_id_token_200_h50_"
    "behavior_interest_scaling_v1/interest_token_mapping.parquet"
)
if config["model_name"] != "Qwen/Qwen3.5-9B":
    raise RuntimeError("Config model identity mismatch")
if config["train_file"] != expected_train:
    raise RuntimeError("Config Train identity mismatch")
if config["semantic_id_token_mapping_file"] != expected_mapping:
    raise RuntimeError("Config mapping identity mismatch")
if "sample_weight_file" in config:
    raise RuntimeError("Vanilla config unexpectedly has sample weights")
if config.get("enable_thinking") is not False:
    raise RuntimeError("Vanilla config unexpectedly enables thinking")
if config.get("gradient_checkpointing_use_reentrant") is not False:
    raise RuntimeError("Vanilla config must use non-reentrant checkpointing")
if int(config["epochs"]) != 3 or int(config["seed"]) != 42:
    raise RuntimeError("Vanilla config epoch/seed mismatch")
if int(config["per_device_train_batch_size"]) != 1:
    raise RuntimeError("Vanilla per-device batch mismatch")
if int(config["gradient_accumulation_steps"]) != 8:
    raise RuntimeError("Vanilla accumulation mismatch")
if float(config["learning_rate"]) != 2e-5:
    raise RuntimeError("Vanilla learning-rate mismatch")

mapping_rows = pq.read_table(mapping_file).to_pylist()
token_by_label = {
    str(row["target_subcategory"]): str(row["interest_token"])
    for row in mapping_rows
}
if len(token_by_label) != 108 or len(set(token_by_label.values())) != 108:
    raise RuntimeError("Mapping is not one-to-one over 108 Semantic IDs")
train_rows = pq.read_table(train_file).to_pylist()
if len(train_rows) != train_size:
    raise RuntimeError("Train row count mismatch")
if len({str(row["source_sample_id"]) for row in train_rows}) != train_size:
    raise RuntimeError("Train source IDs are not unique")
if len({str(row["user_id"]) for row in train_rows}) != train_size:
    raise RuntimeError("Train is not one row per user")
if {str(row["task_type"]) for row in train_rows} != {"behavior_interest"}:
    raise RuntimeError("Train input is not Behavior+Interest")
counts = Counter(str(row["target_subcategory"]) for row in train_rows)
if len(counts) != 108 or set(counts.values()) != {train_size // 108}:
    raise RuntimeError("Train is not exactly class-balanced")
for row in train_rows:
    label = str(row["target_subcategory"])
    response = str(row["response"])
    messages = row["messages"]
    if response != token_by_label[label] or not re.fullmatch(r"<interest_\d{3}>", response):
        raise RuntimeError("Train response is not its Semantic ID")
    if len(messages) != 2 or messages[0]["role"] != "user" or messages[1] != {"role": "assistant", "content": response}:
        raise RuntimeError("Train message contract mismatch")

manifest_rows = pq.read_table(
    subset_manifest_file,
    columns=["source_sample_id", "within_class_rank"],
).to_pylist()
selected = {
    str(row["source_sample_id"])
    for row in manifest_rows
    if int(row["within_class_rank"]) < train_size // 108
}
if selected != {str(row["source_sample_id"]) for row in train_rows}:
    raise RuntimeError("Train does not match its deterministic nested prefix")

eval_rows = pq.read_table(eval_file).to_pylist()
if len(eval_rows) != 31392 or len({str(row["impression_id"]) for row in eval_rows}) != 31392:
    raise RuntimeError("Real Dev is not 31,392 unique impressions")
supported = {
    str(label)
    for row in eval_rows
    for label in row["acceptable_subcategories"]
}
if len(supported) != 104:
    raise RuntimeError("Real Dev supported-interest count mismatch")

eval_config = json.loads(eval_config_file.read_text(encoding="utf-8"))
if eval_config["model_name"] != "Qwen/Qwen3.5-9B":
    raise RuntimeError("Evaluation model identity mismatch")
if eval_config["top_k"] != 20 or eval_config["cutoffs"] != [1, 5, 10, 20]:
    raise RuntimeError("Evaluation Top-K identity mismatch")

statistics = json.loads(statistics_file.read_text(encoding="utf-8"))
metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
if statistics["task_type"] != "behavior_interest":
    raise RuntimeError("Statistics task identity mismatch")
if statistics["train"][str(train_size)]["samples"] != train_size:
    raise RuntimeError("Statistics Train count mismatch")
if metadata["train_sizes"] != [5400, 10800, 16200, 21600]:
    raise RuntimeError("Metadata scale list mismatch")

audit = {
    "audit_version": "semantic-id-vanilla-scaling-input-v1",
    "model_name": config["model_name"],
    "task_type": "behavior_interest",
    "output_type": "semantic_id",
    "sample_weight_file": None,
    "train_size": train_size,
    "samples_per_interest": train_size // 108,
    "expected_optimizer_steps": expected_steps,
    "seed": 42,
    "real_eval_rows": len(eval_rows),
    "real_eval_supported_interests": len(supported),
    "input_sha256": {
        input_file.name: expected
        for input_file, expected in expected_hashes.items()
    },
}
output_file.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
print("SEMANTIC_ID_VANILLA_SCALING_INPUT_OK", json.dumps(audit, sort_keys=True))
PY

python - "$MODEL_NAME" <<'PY'
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

name = sys.argv[1]


def validate_snapshot(snapshot_value: str) -> Path:
    snapshot_root = Path(snapshot_value)
    index_file = snapshot_root / "model.safetensors.index.json"
    if not index_file.is_file():
        raise RuntimeError("Qwen snapshot has no safetensors index")
    filenames = set(
        json.loads(index_file.read_text(encoding="utf-8"))["weight_map"].values()
    )
    missing = [
        filename
        for filename in sorted(filenames)
        if not (snapshot_root / filename).is_file()
        or (snapshot_root / filename).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"Qwen snapshot is incomplete: {missing[:3]}")
    AutoConfig.from_pretrained(snapshot_root, local_files_only=True)
    AutoTokenizer.from_pretrained(snapshot_root, local_files_only=True)
    return snapshot_root


try:
    snapshot = validate_snapshot(snapshot_download(name, local_files_only=True))
    source = "persistent-cache"
except Exception:
    snapshot = validate_snapshot(snapshot_download(name))
    source = "downloaded-or-repaired"
print("SEMANTIC_ID_VANILLA_SCALING_MODEL_READY", name, snapshot, source)
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

validate_training_summary() {
  local summary_file="$1"
  local active_train_file="$2"
  local expected_samples="$3"
  local expected_epochs="$4"
  local expected_steps="$5"
  local require_all_rows="$6"
  local expected_token_audit_samples="$7"
  python - \
    "$summary_file" "$active_train_file" "$expected_samples" \
    "$expected_epochs" "$expected_steps" "$require_all_rows" \
    "$expected_token_audit_samples" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
train_file = Path(sys.argv[2])
expected_samples = int(sys.argv[3])
expected_epochs = float(sys.argv[4])
expected_steps = int(sys.argv[5])
require_all_rows = sys.argv[6] == "true"
expected_token_audit_samples = int(sys.argv[7])


def sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if summary["model_name"] != "Qwen/Qwen3.5-9B":
    raise RuntimeError("Training model identity mismatch")
if summary["train_samples"] != expected_samples:
    raise RuntimeError("Training sample count mismatch")
if summary["configured_epochs"] != expected_epochs:
    raise RuntimeError("Training epoch identity mismatch")
if summary["expected_optimizer_steps"] != expected_steps:
    raise RuntimeError("Training optimizer-step identity mismatch")
if summary["train_file_sha256"] != sha256(train_file):
    raise RuntimeError("Trainer consumed the wrong Train file")
if summary["sample_weight_file"] is not None or summary["sample_weight_file_sha256"] is not None:
    raise RuntimeError("Vanilla training unexpectedly consumed sample weights")
if not math.isfinite(float(summary["train_metrics"]["train_loss"])):
    raise RuntimeError("Training loss is non-finite")
if summary["token_audit"]["samples"] != expected_token_audit_samples:
    raise RuntimeError("Token-audit sample count mismatch")
if summary["token_audit"]["over_max_seq_length"] != 0:
    raise RuntimeError("Token audit found over-length examples")
semantic = summary["model_audit"]["semantic_id_tokens"]
if not semantic["enabled"] or semantic["count"] != 108:
    raise RuntimeError("Semantic-ID model audit mismatch")
if semantic["save_embedding_layers"] is not False:
    raise RuntimeError("Full embedding saving was enabled")
if summary["model_audit"]["forbidden_trainable_parameters"]:
    raise RuntimeError("Forbidden model parameters are trainable")
drift = summary["semantic_token_drift"]
if require_all_rows:
    if not drift["all_input_rows_changed"] or not drift["all_output_rows_changed"]:
        raise RuntimeError("At least one Semantic-ID row did not change")
else:
    if not 0 < int(drift["changed_input_rows"]) <= 108:
        raise RuntimeError("No Semantic-ID input row changed")
    if not 0 < int(drift["changed_output_rows"]) <= 108:
        raise RuntimeError("No Semantic-ID output row changed")
if expected_samples == 32:
    if summary["world_size"] != 1:
        raise RuntimeError("Single-GPU smoke world size mismatch")
    history = summary["loss_history"]
    if len(history) < 2 or not float(history[-1]["loss"]) < float(history[0]["loss"]):
        raise RuntimeError(f"Single-GPU smoke loss did not decrease: {history}")
else:
    if summary["world_size"] != 4:
        raise RuntimeError("DDP world size mismatch")
    if summary["effective_global_batch_size"] != 32:
        raise RuntimeError("DDP effective global batch mismatch")
    if summary["gradient_checkpointing_kwargs"] != {"use_reentrant": False}:
        raise RuntimeError("DDP gradient checkpointing is not non-reentrant")
print("SEMANTIC_ID_VANILLA_SCALING_TRAIN_OK", expected_samples, expected_steps)
PY
}

validate_adapter_state() {
  local active_sft_dir="$1"
  local audit_output="${2:-$active_sft_dir/semantic_adapter_state_audit.json}"
  python - "$active_sft_dir" "$audit_output" <<'PY'
import json
import sys
from pathlib import Path

from safetensors import safe_open

root = Path(sys.argv[1])
audit_output = Path(sys.argv[2])
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
    key for key, shape in shapes.items()
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
audit_output.parent.mkdir(parents=True, exist_ok=True)
audit_output.write_text(
    json.dumps(audit, indent=2) + "\n", encoding="utf-8"
)
print("SEMANTIC_ID_VANILLA_SCALING_ADAPTER_OK", json.dumps(audit, sort_keys=True))
PY
}

verify_reload() {
  local active_sft_dir="$1"
  local audit_output="${2:-$active_sft_dir/semantic_adapter_reload_audit.json}"
  CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.verify_semantic_id_token_adapter \
    --model-name "$MODEL_NAME" \
    --adapter-path "$active_sft_dir/final_adapter" \
    --mapping-file "$MAPPING_FILE" \
    --reference-file "$active_sft_dir/semantic_reload_reference.json" \
    --output "$audit_output"
  python - "$audit_output" <<'PY'
import json
import sys
from pathlib import Path

audit = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not audit["all_close"] or audit["sample_count"] != 4:
    raise RuntimeError("Adapter reload logits audit failed")
integrity = audit["adapter_state_integrity"]
if integrity["tensor_count"] != 498 or not integrity["all_tensors_exact"]:
    raise RuntimeError("Exact compact adapter-state audit failed")
if integrity["saved_state_sha256"] != integrity["loaded_state_sha256"]:
    raise RuntimeError("Reloaded adapter-state hash mismatch")
print("SEMANTIC_ID_VANILLA_SCALING_RELOAD_OK", integrity["tensor_count"])
PY
}

ACTIVE_TRAIN_FILE="$TRAIN_FILE"
TRAIN_EPOCHS=3
EXPECTED_TRAIN_SAMPLES="$TRAIN_SIZE"
EXPECTED_TRAIN_STEPS="$EXPECTED_FULL_STEPS"
if [[ "$MODE" == "smoke" ]]; then
  REUSE_SINGLE_ROOT="${SEMANTIC_ID_VANILLA_SCALING_REUSE_SINGLE_ROOT:-}"
  if [[ -n "$REUSE_SINGLE_ROOT" ]]; then
    for reused_file in \
      "$REUSE_SINGLE_ROOT/input_audit.json" \
      "$REUSE_SINGLE_ROOT/gradient/gradient_audit.json" \
      "$REUSE_SINGLE_ROOT/single/run_summary.json" \
      "$REUSE_SINGLE_ROOT/single/semantic_reload_reference.json" \
      "$REUSE_SINGLE_ROOT/single/final_adapter/adapter_model.safetensors" \
      "$REUSE_SINGLE_ROOT/single/final_adapter/adapter_config.json" \
      "$REUSE_SINGLE_ROOT/single/final_adapter/interest_token_mapping.parquet"; do
      if [[ ! -s "$reused_file" ]]; then
        echo "Missing prior single-smoke artifact: $reused_file" >&2
        exit 1
      fi
    done
    python - \
      "$AMLT_OUTPUT_DIR/input_audit.json" \
      "$REUSE_SINGLE_ROOT/input_audit.json" \
      "$REUSE_SINGLE_ROOT/gradient/gradient_audit.json" \
      "$REUSE_SINGLE_ROOT/single/run_summary.json" \
      "$AMLT_OUTPUT_DIR/reused_single_smoke_audit.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

current_input = Path(sys.argv[1])
previous_input = Path(sys.argv[2])
gradient_file = Path(sys.argv[3])
summary_file = Path(sys.argv[4])
output_file = Path(sys.argv[5])


def sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


current = json.loads(current_input.read_text(encoding="utf-8"))
previous = json.loads(previous_input.read_text(encoding="utf-8"))
for key in (
    "model_name", "task_type", "output_type", "sample_weight_file",
    "train_size", "samples_per_interest", "seed", "input_sha256",
):
    if current[key] != previous[key]:
        raise RuntimeError(f"Previous smoke identity differs in {key}")
gradient = json.loads(gradient_file.read_text(encoding="utf-8"))
if gradient["batch_size"] != 1 or gradient["representative_samples"] != 108:
    raise RuntimeError("Previous gradient audit did not use 108 batch-1 representatives")
if not gradient["all_token_rows_finite_and_nonzero"]:
    raise RuntimeError("Previous gradient audit did not pass")
for row in gradient["gradient_parameters"]:
    if row["finite_rows"] != 108 or row["nonzero_rows"] != 108:
        raise RuntimeError("Previous gradient audit has an invalid token row")
summary = json.loads(summary_file.read_text(encoding="utf-8"))
payload = {
    "reuse_version": "semantic-id-vanilla-scaling-single-reuse-v1",
    "previous_input_audit_sha256": sha256(previous_input),
    "gradient_audit_sha256": sha256(gradient_file),
    "single_run_summary_sha256": sha256(summary_file),
    "single_train_loss": summary["train_metrics"]["train_loss"],
    "single_expected_optimizer_steps": summary["expected_optimizer_steps"],
}
output_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("SEMANTIC_ID_VANILLA_SCALING_SINGLE_REUSE_INPUT_OK", json.dumps(payload, sort_keys=True))
PY
    validate_training_summary \
      "$REUSE_SINGLE_ROOT/single/run_summary.json" "$TRAIN_FILE" \
      32 5 20 false 5400
    validate_adapter_state \
      "$REUSE_SINGLE_ROOT/single" \
      "$AMLT_OUTPUT_DIR/reused_single_adapter_state_audit.json"
    verify_reload \
      "$REUSE_SINGLE_ROOT/single" \
      "$AMLT_OUTPUT_DIR/reused_single_adapter_reload_audit.json"
  else
    GRADIENT_DIR="$AMLT_OUTPUT_DIR/gradient"
    SINGLE_DIR="$AMLT_OUTPUT_DIR/single"
    mkdir -p "$GRADIENT_DIR"
    CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.audit_semantic_id_token_gradients \
      --config "$CONFIG" \
      --train-file "$TRAIN_FILE" \
      --output "$GRADIENT_DIR/gradient_audit.json" \
      --batch-size 1

    CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.train_sft \
      --config "$CONFIG" \
      --train-file "$TRAIN_FILE" \
      --output-dir "$SINGLE_DIR" \
      --run-name "qwen35-bi-semantic-id-vanilla-scaling-single-smoke" \
      --max-samples 32 \
      --epochs 5 \
      --per-device-train-batch-size 1 \
      --gradient-accumulation-steps 8 \
      --wandb-mode disabled
    validate_training_summary \
      "$SINGLE_DIR/run_summary.json" "$TRAIN_FILE" 32 5 20 false 5400
    validate_adapter_state "$SINGLE_DIR"
    verify_reload "$SINGLE_DIR"
  fi

  SMOKE_DATA_DIR="$AMLT_OUTPUT_DIR/smoke_data"
  mkdir -p "$SMOKE_DATA_DIR"
  ACTIVE_TRAIN_FILE="$SMOKE_DATA_DIR/train.parquet"
  python - "$TRAIN_FILE" "$ACTIVE_TRAIN_FILE" <<'PY'
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

source_file = Path(sys.argv[1])
output_file = Path(sys.argv[2])
rows = pq.read_table(source_file).to_pylist()
selected = {}
for row in rows:
    selected.setdefault(str(row["target_subcategory"]), row)
if len(selected) != 108:
    raise RuntimeError("Smoke selection does not cover all 108 interests")
smoke_rows = [selected[label] for label in sorted(selected)]
pq.write_table(
    pa.Table.from_pylist(smoke_rows, schema=pq.ParquetFile(source_file).schema_arrow),
    output_file,
    compression="zstd",
    use_dictionary=True,
)
print("SEMANTIC_ID_VANILLA_SCALING_SMOKE_DATA_OK", len(smoke_rows))
PY
  TRAIN_EPOCHS=2
  EXPECTED_TRAIN_SAMPLES=108
  EXPECTED_TRAIN_STEPS=8
fi

TRAIN_ARGS=(
  --config "$CONFIG"
  --train-file "$ACTIVE_TRAIN_FILE"
  --output-dir "$SFT_DIR"
  --checkpoint-dir "$LOCAL_CHECKPOINT_ROOT"
  --run-name "qwen35-bi-semantic-id-vanilla-${TRAIN_SIZE}-${MODE}-seed42"
  --epochs "$TRAIN_EPOCHS"
  --per-device-train-batch-size 1
  --gradient-accumulation-steps 8
  --save-steps 100
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

validate_training_summary \
  "$SFT_DIR/run_summary.json" "$ACTIVE_TRAIN_FILE" \
  "$EXPECTED_TRAIN_SAMPLES" "$TRAIN_EPOCHS" "$EXPECTED_TRAIN_STEPS" true \
  "$EXPECTED_TRAIN_SAMPLES"
if [[ "$MODE" == "full" ]]; then
  python - \
    "$SFT_DIR/run_summary.json" "$PERSISTENT_CHECKPOINT_ROOT" \
    "$EXPECTED_FULL_STEPS" <<'PY'
import json
import sys
from pathlib import Path

from new_interest.training.checkpoint_persistence import (
    select_latest_complete_checkpoint,
)

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
persistent_root = Path(sys.argv[2])
expected_steps = int(sys.argv[3])
selected, diagnostics = select_latest_complete_checkpoint(
    persistent_root, summary["checkpoint_identity"]
)
if diagnostics:
    raise RuntimeError(f"Persistent-checkpoint diagnostics: {diagnostics}")
if selected is None or selected.name != f"checkpoint-{expected_steps}":
    raise RuntimeError(
        f"Expected checkpoint-{expected_steps}, got {selected}"
    )
print("SEMANTIC_ID_VANILLA_SCALING_CHECKPOINT_OK", selected)
PY
fi
validate_adapter_state "$SFT_DIR"
verify_reload "$SFT_DIR"

mkdir -p "$EVAL_ROOT"
if [[ "$MODE" == "smoke" ]]; then
  CUDA_VISIBLE_DEVICES=0 python -m new_interest.evaluation.evaluate_id_token_ranking \
    --config "$EVAL_CONFIG" \
    --adapter-path "$SFT_DIR/final_adapter" \
    --eval-file "$EVAL_FILE" \
    --interest-token-mapping-file "$MAPPING_FILE" \
    --output-dir "$MERGED_DIR" \
    --run-name "qwen35-bi-semantic-id-vanilla-scaling-smoke" \
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
        --run-name "qwen35-bi-semantic-id-vanilla-${TRAIN_SIZE}-${SHARD_NAME}" \
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
    echo "At least one Vanilla scaling evaluation shard failed" >&2
    for log_file in "$EVAL_ROOT"/shard-*.log; do
      echo "===== $log_file =====" >&2
      tail -n 80 "$log_file" >&2 || true
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
    raise RuntimeError("Evaluation sample count mismatch")
if mode == "full" and metrics["supported_interest_count"] != 104:
    raise RuntimeError("Full supported-interest count mismatch")
if metrics["label_vocabulary_size"] != 108:
    raise RuntimeError("Evaluation vocabulary-size mismatch")
if metrics["top_k"] != 20 or metrics["cutoffs"] != [1, 5, 10, 20]:
    raise RuntimeError("Evaluation Top-K contract mismatch")
if metrics["history_prediction_filtering"] is not False:
    raise RuntimeError("History filtering was enabled unexpectedly")
for cutoff in (1, 5, 10, 20):
    for prefix in ("recall", "ndcg", "interest_macro_recall", "interest_macro_ndcg"):
        if not math.isfinite(float(metrics[f"{prefix}_at_{cutoff}"])):
            raise RuntimeError("Evaluation contains a non-finite metric")
labels = {
    str(row["target_subcategory"])
    for row in pq.read_table(mapping_file).to_pylist()
}
rows = pq.read_table(root / "id_token_predictions.parquet").to_pylist()
if len(rows) != expected or len({int(row["record_index"]) for row in rows}) != expected:
    raise RuntimeError("Prediction row/index count mismatch")
for row in rows:
    ranked = list(map(str, row["ranked_labels"]))
    scores = list(map(float, row["ranked_scores"]))
    if len(ranked) != 20 or len(set(ranked)) != 20 or not set(ranked).issubset(labels):
        raise RuntimeError("Prediction is not a legal unique Top-20")
    if len(scores) != 20 or not all(math.isfinite(value) for value in scores):
        raise RuntimeError("Prediction contains invalid scores")
print(
    "SEMANTIC_ID_VANILLA_SCALING_EVAL_OK",
    mode,
    metrics["recall_at_10"],
    metrics["interest_macro_recall_at_10"],
)
PY

python - \
  "$AMLT_OUTPUT_DIR" "$MODE" "$TRAIN_SIZE" \
  "$TRAIN_SHA256" "$EVAL_SHA256" "$MAPPING_SHA256" <<'PY'
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]
train_size = int(sys.argv[3])
identity = {
    "model_name": "Qwen/Qwen3.5-9B",
    "task_type": "behavior_interest",
    "output_type": "semantic_id",
    "training": "vanilla",
    "sample_weight_file": None,
    "train_size": train_size,
    "seed": 42,
    "train_sha256": sys.argv[4],
    "eval_sha256": sys.argv[5],
    "mapping_sha256": sys.argv[6],
}


def sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(file_path: Path, payload: object) -> None:
    temporary = file_path.with_name(f".{file_path.name}.partial-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, file_path)


required = [
    "environment.json",
    "input_audit.json",
    "sft/token_audit.json",
    "sft/semantic_token_initialization.json",
    "sft/model_audit.json",
    "sft/semantic_token_drift.json",
    "sft/semantic_reload_reference.json",
    "sft/semantic_adapter_state_audit.json",
    "sft/semantic_adapter_reload_audit.json",
    "sft/run_summary.json",
    "sft/final_adapter/adapter_config.json",
    "sft/final_adapter/adapter_model.safetensors",
    "sft/final_adapter/interest_token_mapping.parquet",
    "sft/final_adapter/tokenizer.json",
    "real_eval/merged/metrics.json",
    "real_eval/merged/id_token_predictions.parquet",
]
if mode == "smoke":
    if os.environ.get("SEMANTIC_ID_VANILLA_SCALING_REUSE_SINGLE_ROOT"):
        required.extend(
            [
                "reused_single_smoke_audit.json",
                "reused_single_adapter_state_audit.json",
                "reused_single_adapter_reload_audit.json",
            ]
        )
    else:
        required.extend(
            [
                "gradient/gradient_audit.json",
                "single/run_summary.json",
                "single/semantic_adapter_state_audit.json",
                "single/semantic_adapter_reload_audit.json",
            ]
        )
    required.extend(
        [
            "smoke_data/train.parquet",
            "real_eval/merged/shard_manifest.json",
        ]
    )
else:
    required.extend(
        [
            "real_eval/merged/per_interest_metrics.parquet",
            "real_eval/merged/merge_manifest.json",
        ]
    )
    for shard_index in range(4):
        required.extend(
            [
                f"real_eval/shard-{shard_index:02d}/id_token_predictions.parquet",
                f"real_eval/shard-{shard_index:02d}/metrics.json",
                f"real_eval/shard-{shard_index:02d}/shard_manifest.json",
            ]
        )

files = {}
for relative in required:
    artifact_file = root / relative
    if not artifact_file.is_file() or artifact_file.stat().st_size == 0:
        raise RuntimeError(f"Required artifact is missing or empty: {relative}")
    files[relative] = sha256(artifact_file)
manifest = {
    "manifest_version": "semantic-id-vanilla-scaling-artifact-v1",
    "mode": mode,
    "identity": identity,
    "files": dict(sorted(files.items())),
}
manifest_file = root / "artifact_manifest.json"
atomic_json(manifest_file, manifest)
marker = {
    "mode": mode,
    "identity": identity,
    "artifact_manifest_sha256": sha256(manifest_file),
}
atomic_json(root / f"{mode}_success.json", marker)
print("SEMANTIC_ID_VANILLA_SCALING_GATE_OK", mode, train_size, len(files))
PY
