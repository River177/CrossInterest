#!/usr/bin/env bash
set -euo pipefail

METHOD="${SEMANTIC_ID_PILOT_METHOD:?SEMANTIC_ID_PILOT_METHOD is required}"
case "$METHOD" in
  taxonomy)
    EXPECTED_JOB_NAME="qwen35-bi-taxonomy-softmax-5400"
    CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_semantic_id_qwen35_behavior_interest_taxonomy_softmax_5400_v1.yaml"
    LOSS_SCOPE="taxonomy"
    WEIGHT_FILE=""
    WEIGHT_SHA256=""
    ;;
  frontier)
    EXPECTED_JOB_NAME="qwen35-bi-frontier-reweighting-5400"
    CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_semantic_id_qwen35_behavior_interest_frontier_5400_v1.yaml"
    LOSS_SCOPE="full_vocabulary"
    WEIGHT_FILE="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_semantic_id_frontier_qwen35_behavior_interest_5400_v1/sample_weights.parquet"
    WEIGHT_SHA256="f3781e7dd263df037a1764815b04c1666edc3560ad22685a0285130b64ad9ef0"
    ;;
  *)
    echo "Unsupported independent pilot method: $METHOD" >&2
    exit 2
    ;;
esac
if [[ "${AMLT_JOB_NAME:?AMLT_JOB_NAME is required}" != "$EXPECTED_JOB_NAME" ]]; then
  echo "Unexpected independent-pilot job name: $AMLT_JOB_NAME" >&2
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
EXPECTED_MODEL_REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
LOCALIZE_MODEL_CACHE="${SEMANTIC_ID_LOCALIZE_MODEL_CACHE:-0}"
if [[ "$LOCALIZE_MODEL_CACHE" != "0" && "$LOCALIZE_MODEL_CACHE" != "1" ]]; then
  echo "SEMANTIC_ID_LOCALIZE_MODEL_CACHE must be 0 or 1" >&2
  exit 2
fi
DATA_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_scaling_v1"
TRAIN_FILE="$DATA_ROOT/train/behavior_interest/5400.parquet"
EVAL_FILE="$DATA_ROOT/eval/behavior_interest/dev_full_grouped.parquet"
MAPPING_FILE="$DATA_ROOT/interest_token_mapping.parquet"
EVAL_CONFIG="$AMLT_CODE_DIR/configs/id_token/static_sud_qwen35_behavior_interest_vanilla_scaling_v1.json"
TRAIN_SHA256="cc1a5921ccaf83bfcef17bdecbcf756561e97448be54e28c3b8e3a673333f72c"
EVAL_SHA256="c05b20033ce0d1a0c2d4b7bb97ca0cd6b7cffb7f198a7265cce9e54046b665fd"
MAPPING_SHA256="45397451ac2ae2e64d8156b2b47be3f454cc309a8d30a83bc23c2d3267653a0a"
SFT_DIR="$AMLT_OUTPUT_DIR/sft"
EVAL_ROOT="$AMLT_OUTPUT_DIR/real_eval"
MERGED_DIR="$EVAL_ROOT/merged"
CHECKPOINT_DIR="/scratch/semantic-id-${METHOD}-5400-checkpoints"
PERSISTENT_CHECKPOINT_DIR="$AMLT_OUTPUT_DIR/checkpoints"
SUCCESS_MARKER="$AMLT_OUTPUT_DIR/pilot_success.json"

if [[ -f "$SUCCESS_MARKER" ]]; then
  python - "$AMLT_OUTPUT_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

marker = json.loads((root / "pilot_success.json").read_text(encoding="utf-8"))
manifest_file = root / "artifact_manifest.json"
if sha256(manifest_file) != marker["artifact_manifest_sha256"]:
    raise RuntimeError("Committed pilot manifest hash mismatch")
manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
for relative, expected in manifest["files"].items():
    path = root / relative
    if not path.is_file() or sha256(path) != expected:
        raise RuntimeError(f"Committed pilot artifact is corrupt: {relative}")
print("SEMANTIC_ID_INDEPENDENT_PILOT_REUSE_OK", manifest["identity"]["method"])
PY
  exit 0
fi

for required in "$CONFIG" "$TRAIN_FILE" "$EVAL_FILE" "$MAPPING_FILE" "$EVAL_CONFIG"; do
  if [[ ! -s "$required" ]]; then
    echo "Missing independent-pilot input: $required" >&2
    exit 1
  fi
done
if [[ -n "$WEIGHT_FILE" && ! -s "$WEIGHT_FILE" ]]; then
  echo "Missing frontier weights: $WEIGHT_FILE" >&2
  exit 1
fi

python - "$AMLT_OUTPUT_DIR/environment.json" <<'PY'
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch

payload = {
    "python": platform.python_version(),
    "packages": {
        name: importlib.metadata.version(name)
        for name in ("accelerate", "peft", "pyarrow", "torch", "transformers")
    },
    "cuda": torch.version.cuda,
    "cuda_devices": torch.cuda.device_count(),
}
for name, expected in {
    "transformers": "5.14.1",
    "peft": "0.18.0",
    "accelerate": "1.12.0",
}.items():
    if payload["packages"][name] != expected:
        raise RuntimeError(f"Unexpected {name}: {payload['packages'][name]}")
if payload["cuda_devices"] != 4:
    raise RuntimeError("Independent pilot requires four A100 GPUs")
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("SEMANTIC_ID_INDEPENDENT_PILOT_ENV_OK", json.dumps(payload, sort_keys=True))
PY

python - \
  "$METHOD" "$LOSS_SCOPE" "$CONFIG" \
  "$TRAIN_FILE" "$TRAIN_SHA256" "$EVAL_FILE" "$EVAL_SHA256" \
  "$MAPPING_FILE" "$MAPPING_SHA256" "$WEIGHT_FILE" "$WEIGHT_SHA256" \
  "$AMLT_OUTPUT_DIR/input_audit.json" <<'PY'
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml

method, loss_scope = sys.argv[1:3]
config_file = Path(sys.argv[3])
train_file, train_sha = Path(sys.argv[4]), sys.argv[5]
eval_file, eval_sha = Path(sys.argv[6]), sys.argv[7]
mapping_file, mapping_sha = Path(sys.argv[8]), sys.argv[9]
weight_file = Path(sys.argv[10]) if sys.argv[10] else None
weight_sha = sys.argv[11] or None
output_file = Path(sys.argv[12])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

for path, expected in (
    (train_file, train_sha), (eval_file, eval_sha), (mapping_file, mapping_sha)
):
    if sha256(path) != expected:
        raise RuntimeError(f"Frozen input hash mismatch: {path}")
config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
if config["model_name"] != "Qwen/Qwen3.5-9B":
    raise RuntimeError("Pilot model identity mismatch")
if config.get("semantic_id_loss_scope", "full_vocabulary") != loss_scope:
    raise RuntimeError("Pilot loss scope mismatch")
if config.get("enable_thinking") is not False or int(config["seed"]) != 42:
    raise RuntimeError("Pilot thinking/seed mismatch")
if int(config["epochs"]) != 3 or int(config["per_device_train_batch_size"]) != 1:
    raise RuntimeError("Pilot training schedule mismatch")
if int(config["gradient_accumulation_steps"]) != 8:
    raise RuntimeError("Pilot accumulation mismatch")

train_rows = pq.read_table(train_file).to_pylist()
train_ids = [str(row["source_sample_id"]) for row in train_rows]
counts = Counter(str(row["target_subcategory"]) for row in train_rows)
if len(train_rows) != 5400 or len(set(train_ids)) != 5400:
    raise RuntimeError("Pilot Train identity mismatch")
if len(counts) != 108 or set(counts.values()) != {50}:
    raise RuntimeError("Pilot Train is not 50 rows per class")
if pq.ParquetFile(eval_file).metadata.num_rows != 31392:
    raise RuntimeError("Pilot Dev row count mismatch")
if pq.ParquetFile(mapping_file).metadata.num_rows != 108:
    raise RuntimeError("Pilot mapping row count mismatch")

weight_summary = None
if method == "frontier":
    if weight_file is None or sha256(weight_file) != weight_sha:
        raise RuntimeError("Frontier weight hash mismatch")
    rows = pq.read_table(weight_file).to_pylist()
    if {str(row["source_sample_id"]) for row in rows} != set(train_ids):
        raise RuntimeError("Frontier weights do not exactly cover Pilot Train")
    by_class = {}
    for row in rows:
        by_class.setdefault(str(row["target_subcategory"]), []).append(
            float(row["sample_weight"])
        )
    means = [math.fsum(values) / len(values) for values in by_class.values()]
    if len(rows) != 5400 or len(by_class) != 108:
        raise RuntimeError("Frontier weight scope mismatch")
    if any(abs(value - 1.0) > 1e-12 for value in means):
        raise RuntimeError("Frontier class normalization mismatch")
    if not all(0.0 < float(row["sample_weight"]) for row in rows):
        raise RuntimeError("Frontier weights must be positive")
    weight_summary = {
        "sha256": weight_sha,
        "minimum": min(float(row["sample_weight"]) for row in rows),
        "maximum": max(float(row["sample_weight"]) for row in rows),
        "class_means_all_one": True,
    }
elif weight_file is not None or "sample_weight_file" in config:
    raise RuntimeError("Taxonomy pilot unexpectedly has sample weights")

payload = {
    "audit_version": "independent-semantic-id-pilot-input-v1",
    "method": method,
    "model": config["model_name"],
    "input": "behavior_plus_interest",
    "output": "semantic_id_108",
    "loss_scope": loss_scope,
    "train_rows": 5400,
    "train_rows_per_class": 50,
    "eval_rows": 31392,
    "expected_optimizer_steps": 507,
    "sha256": {
        "train": train_sha,
        "eval": eval_sha,
        "mapping": mapping_sha,
        "weights": weight_sha,
        "config": sha256(config_file),
    },
    "weight_summary": weight_summary,
}
output_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("SEMANTIC_ID_INDEPENDENT_PILOT_INPUT_OK", json.dumps(payload, sort_keys=True))
PY

python - "$MODEL_NAME" "$EXPECTED_MODEL_REVISION" <<'PY'
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

root = Path(snapshot_download(sys.argv[1]))
if root.name != sys.argv[2]:
    raise RuntimeError(f"Unexpected Qwen snapshot revision: {root.name}")
index = root / "model.safetensors.index.json"
if not index.is_file():
    raise RuntimeError("Qwen snapshot has no index")
files = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
if any(not (root / name).is_file() for name in files):
    raise RuntimeError("Qwen snapshot is incomplete")
AutoConfig.from_pretrained(root, local_files_only=True)
AutoTokenizer.from_pretrained(root, local_files_only=True)
print("SEMANTIC_ID_INDEPENDENT_PILOT_MODEL_READY", root)
PY

PERSISTENT_HF_HOME="$HF_HOME"
if [[ "$LOCALIZE_MODEL_CACHE" == "1" ]]; then
  PERSISTENT_MODEL_CACHE="$PERSISTENT_HF_HOME/hub/models--Qwen--Qwen3.5-9B"
  LOCAL_HF_HOME="/scratch/huggingface-cache-taxonomy-pilot-v2"
  LOCAL_MODEL_CACHE="$LOCAL_HF_HOME/hub/models--Qwen--Qwen3.5-9B"
  if [[ ! -d "$PERSISTENT_MODEL_CACHE" ]]; then
    echo "Validated persistent Qwen cache disappeared before localization" >&2
    exit 1
  fi
  command -v rsync >/dev/null
  mkdir -p "$LOCAL_MODEL_CACHE"
  rsync -a "$PERSISTENT_MODEL_CACHE/" "$LOCAL_MODEL_CACHE/"
  export HF_HOME="$LOCAL_HF_HOME"
fi

python - \
  "$MODEL_NAME" "$EXPECTED_MODEL_REVISION" "$LOCALIZE_MODEL_CACHE" \
  "$PERSISTENT_HF_HOME" "$HF_HOME" \
  "$AMLT_OUTPUT_DIR/model_cache_audit.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

model_name = sys.argv[1]
expected_revision = sys.argv[2]
localized = sys.argv[3] == "1"
persistent_home = Path(sys.argv[4])
active_home = Path(sys.argv[5])
output_file = Path(sys.argv[6])
root = Path(snapshot_download(model_name, local_files_only=True))
if root.name != expected_revision:
    raise RuntimeError(f"Localized Qwen revision mismatch: {root.name}")
index = root / "model.safetensors.index.json"
weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
shard_names = sorted(set(weight_map.values()))
shards = [root / name for name in shard_names]
if any(not path.is_file() or path.stat().st_size == 0 for path in shards):
    raise RuntimeError("Active Qwen cache is incomplete after localization")
AutoConfig.from_pretrained(root, local_files_only=True)
AutoTokenizer.from_pretrained(root, local_files_only=True)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "audit_version": "semantic-id-model-cache-localization-v1",
    "model_name": model_name,
    "revision": expected_revision,
    "localized": localized,
    "persistent_hf_home": str(persistent_home),
    "active_hf_home": str(active_home),
    "active_snapshot": str(root),
    "snapshot_is_under_active_home": root.is_relative_to(active_home),
    "index_sha256": sha256(index),
    "shard_count": len(shards),
    "shard_names": shard_names,
    "total_shard_bytes": sum(path.stat().st_size for path in shards),
}
if localized and persistent_home == active_home:
    raise RuntimeError("Localized model cache still points to persistent storage")
if not payload["snapshot_is_under_active_home"]:
    raise RuntimeError("Resolved Qwen snapshot is outside the active HF_HOME")
output_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("SEMANTIC_ID_INDEPENDENT_PILOT_LOCAL_CACHE_OK", json.dumps(payload, sort_keys=True))
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TRAIN_ARGS=(
  --config "$CONFIG"
  --train-file "$TRAIN_FILE"
  --output-dir "$SFT_DIR"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --persistent-checkpoint-dir "$PERSISTENT_CHECKPOINT_DIR"
  --resume-from-checkpoint auto
  --run-name "qwen35-bi-${METHOD}-5400-seed42"
  --epochs 3
  --per-device-train-batch-size 1
  --gradient-accumulation-steps 8
  --save-steps 100
  --wandb-mode disabled
)
if [[ -n "$WEIGHT_FILE" ]]; then
  TRAIN_ARGS+=(--sample-weight-file "$WEIGHT_FILE")
fi
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m new_interest.training.train_sft "${TRAIN_ARGS[@]}"

python - "$SFT_DIR/run_summary.json" "$METHOD" "$LOSS_SCOPE" "$TRAIN_FILE" "$WEIGHT_FILE" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
method, loss_scope = sys.argv[2:4]
train_file = Path(sys.argv[4])
weight_file = Path(sys.argv[5]) if sys.argv[5] else None

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if summary["train_samples"] != 5400 or summary["expected_optimizer_steps"] != 507:
    raise RuntimeError("Pilot training scope mismatch")
if summary["world_size"] != 4 or summary["effective_global_batch_size"] != 32:
    raise RuntimeError("Pilot DDP scope mismatch")
if summary["configured_epochs"] != 3 or summary["semantic_id_loss_scope"] != loss_scope:
    raise RuntimeError("Pilot objective/schedule mismatch")
if summary["train_file_sha256"] != sha256(train_file):
    raise RuntimeError("Pilot trainer consumed the wrong Train file")
if not math.isfinite(float(summary["train_metrics"]["train_loss"])):
    raise RuntimeError("Pilot training loss is non-finite")
if summary["gradient_checkpointing_kwargs"] != {"use_reentrant": False}:
    raise RuntimeError("Pilot checkpointing is not non-reentrant")
if summary["model_audit"]["forbidden_trainable_parameters"]:
    raise RuntimeError("Pilot trained a forbidden parameter")
indices = summary["model_audit"]["semantic_id_tokens"]["trainable_token_indices"]
drift = summary["semantic_token_drift"]
if method == "taxonomy":
    if weight_file is not None or summary["sample_weight_file"] is not None:
        raise RuntimeError("Taxonomy pilot consumed weights")
    if len(indices) != 1 or not any("lm_head" in key for key in indices):
        raise RuntimeError("Taxonomy pilot is not output-row-only")
    if drift["changed_input_rows"] != 0 or not drift["all_output_rows_changed"]:
        raise RuntimeError("Taxonomy pilot token drift mismatch")
else:
    if weight_file is None or summary["sample_weight_file_sha256"] != sha256(weight_file):
        raise RuntimeError("Frontier pilot weight identity mismatch")
    if len(indices) != 2:
        raise RuntimeError("Frontier pilot token-row scope mismatch")
    if not drift["all_input_rows_changed"] or not drift["all_output_rows_changed"]:
        raise RuntimeError("Frontier pilot token drift mismatch")
print("SEMANTIC_ID_INDEPENDENT_PILOT_TRAIN_OK", method, 507)
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
    raise RuntimeError("Pilot fresh adapter reload failed")
if not audit["adapter_state_integrity"]["all_tensors_exact"]:
    raise RuntimeError("Pilot adapter state is not bit-exact")
print("SEMANTIC_ID_INDEPENDENT_PILOT_RELOAD_OK")
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
      --run-name "qwen35-bi-${METHOD}-5400-${SHARD_NAME}" \
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
  for LOG_FILE in "$EVAL_ROOT"/shard-*.log; do
    tail -n 80 "$LOG_FILE" >&2 || true
  done
  exit 1
fi
python -m new_interest.evaluation.merge_id_token_shards \
  --shard-dirs "${SHARD_DIRS[@]}" \
  --eval-file "$EVAL_FILE" \
  --interest-token-mapping-file "$MAPPING_FILE" \
  --output-dir "$MERGED_DIR" \
  --expected-records 31392

python - "$AMLT_OUTPUT_DIR" "$METHOD" "$LOSS_SCOPE" "$TRAIN_SHA256" "$EVAL_SHA256" "$MAPPING_SHA256" "$WEIGHT_SHA256" <<'PY'
import hashlib
import json
import math
import os
import sys
import uuid
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
method, loss_scope = sys.argv[2:4]
metrics_file = root / "real_eval/merged/metrics.json"
predictions_file = root / "real_eval/merged/id_token_predictions.parquet"
metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
if metrics["samples"] != 31392 or metrics["supported_interest_count"] != 104:
    raise RuntimeError("Pilot real-Dev scope mismatch")
if metrics["label_vocabulary_size"] != 108 or metrics["top_k"] != 20:
    raise RuntimeError("Pilot ranking scope mismatch")
for key in ("interest_macro_recall_at_10", "interest_macro_ndcg_at_10"):
    if not math.isfinite(float(metrics[key])):
        raise RuntimeError(f"Pilot metric is non-finite: {key}")
if pq.ParquetFile(predictions_file).metadata.num_rows != 31392:
    raise RuntimeError("Pilot prediction row count mismatch")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

required = [
    "environment.json",
    "input_audit.json",
    "model_cache_audit.json",
    "sft/run_summary.json",
    "sft/model_audit.json",
    "sft/semantic_token_drift.json",
    "sft/semantic_reload_reference.json",
    "sft/semantic_adapter_reload_audit.json",
    "sft/final_adapter/adapter_config.json",
    "sft/final_adapter/adapter_model.safetensors",
    "sft/final_adapter/interest_token_mapping.parquet",
    "real_eval/merged/metrics.json",
    "real_eval/merged/id_token_predictions.parquet",
    "real_eval/merged/per_interest_metrics.parquet",
    "real_eval/merged/merge_manifest.json",
]
for shard_index in range(4):
    required.extend(
        [
            f"real_eval/shard-{shard_index:02d}/metrics.json",
            f"real_eval/shard-{shard_index:02d}/id_token_predictions.parquet",
            f"real_eval/shard-{shard_index:02d}/shard_manifest.json",
        ]
    )
files = {}
for relative in required:
    path = root / relative
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Pilot artifact is missing: {relative}")
    files[relative] = sha256(path)
identity = {
    "method": method,
    "model": "Qwen/Qwen3.5-9B",
    "input": "behavior_plus_interest",
    "output": "semantic_id_108",
    "loss_scope": loss_scope,
    "train_rows": 5400,
    "seed": 42,
    "train_sha256": sys.argv[4],
    "eval_sha256": sys.argv[5],
    "mapping_sha256": sys.argv[6],
    "weight_sha256": sys.argv[7] or None,
}
manifest = {
    "manifest_version": "independent-semantic-id-pilot-v1",
    "identity": identity,
    "metrics": {
        "interest_macro_recall_at_10": metrics["interest_macro_recall_at_10"],
        "interest_macro_ndcg_at_10": metrics["interest_macro_ndcg_at_10"],
    },
    "files": dict(sorted(files.items())),
}

def atomic_json(path: Path, payload: object) -> None:
    pending = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    pending.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(pending, path)

manifest_file = root / "artifact_manifest.json"
atomic_json(manifest_file, manifest)
atomic_json(
    root / "pilot_success.json",
    {
        "identity": identity,
        "artifact_manifest_sha256": sha256(manifest_file),
        "metrics": manifest["metrics"],
    },
)
print(
    "SEMANTIC_ID_INDEPENDENT_PILOT_GATE_OK",
    method,
    metrics["interest_macro_recall_at_10"],
    metrics["interest_macro_ndcg_at_10"],
)
PY
