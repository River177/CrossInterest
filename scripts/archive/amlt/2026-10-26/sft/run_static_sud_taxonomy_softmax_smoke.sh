#!/usr/bin/env bash
set -euo pipefail

METHOD="${SEMANTIC_ID_SMOKE_METHOD:-taxonomy}"
case "$METHOD" in
  taxonomy) EXPECTED_JOB_NAME="qwen35-bi-taxonomy-softmax-smoke" ;;
  frontier) EXPECTED_JOB_NAME="qwen35-bi-frontier-reweighting-smoke" ;;
  *) echo "Unsupported independent smoke method: $METHOD" >&2; exit 2 ;;
esac
if [[ "${AMLT_JOB_NAME:?AMLT_JOB_NAME is required}" != "$EXPECTED_JOB_NAME" ]]; then
  echo "Unexpected job name: $AMLT_JOB_NAME" >&2
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
EVAL_CONFIG="$AMLT_CODE_DIR/configs/id_token/static_sud_qwen35_behavior_interest_vanilla_scaling_v1.json"
DATA_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_scaling_v1"
TRAIN_FILE="$DATA_ROOT/train/behavior_interest/21600.parquet"
EVAL_FILE="$DATA_ROOT/eval/behavior_interest/dev_full_grouped.parquet"
MAPPING_FILE="$DATA_ROOT/interest_token_mapping.parquet"
WEIGHT_FILE=""
case "$METHOD" in
  taxonomy)
    CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_semantic_id_qwen35_behavior_interest_taxonomy_softmax_21600_v1.yaml"
    ;;
  frontier)
    CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_semantic_id_qwen35_behavior_interest_frontier_21600_v1.yaml"
    WEIGHT_FILE="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_semantic_id_frontier_qwen35_behavior_interest_21600_v1/sample_weights.parquet"
    ;;
esac

for required in "$CONFIG" "$EVAL_CONFIG" "$TRAIN_FILE" "$EVAL_FILE" "$MAPPING_FILE"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing taxonomy-softmax input: $required" >&2
    exit 1
  fi
done
if [[ -n "$WEIGHT_FILE" && ! -f "$WEIGHT_FILE" ]]; then
  echo "Missing frontier weight file: $WEIGHT_FILE" >&2
  exit 1
fi

python - "$CONFIG" "$TRAIN_FILE" "$EVAL_FILE" "$MAPPING_FILE" "$WEIGHT_FILE" "$METHOD" "$AMLT_OUTPUT_DIR/input_audit.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

config_file = Path(sys.argv[1])
train_file = Path(sys.argv[2])
eval_file = Path(sys.argv[3])
mapping_file = Path(sys.argv[4])
weight_file = Path(sys.argv[5]) if sys.argv[5] else None
method = sys.argv[6]
output_file = Path(sys.argv[7])
config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
if config["model_name"] != "Qwen/Qwen3.5-9B":
    raise RuntimeError("Unexpected model")
expected_scope = "taxonomy" if method == "taxonomy" else "full_vocabulary"
if config.get("semantic_id_loss_scope", "full_vocabulary") != expected_scope:
    raise RuntimeError("Semantic-ID loss scope is not frozen")
if config.get("enable_thinking") is not False:
    raise RuntimeError("Thinking must remain disabled")
if "mera_relations_file" in config:
    raise RuntimeError("Smoke must not include MeRA")
if method == "taxonomy" and "sample_weight_file" in config:
    raise RuntimeError("Taxonomy smoke must not include sample weights")
if method == "frontier":
    if weight_file is None or not weight_file.is_file():
        raise RuntimeError("Frontier smoke is missing weights")
    configured_weight = Path(config["sample_weight_file"])
    if configured_weight.name != weight_file.name:
        raise RuntimeError("Frontier config weight identity mismatch")
if config.get("gradient_checkpointing_use_reentrant") is not False:
    raise RuntimeError("DDP smoke requires non-reentrant checkpointing")

train = pq.ParquetFile(train_file)
mapping = pq.read_table(mapping_file).to_pylist()
evaluation = pq.ParquetFile(eval_file)
if train.metadata.num_rows != 21600 or evaluation.metadata.num_rows != 31392:
    raise RuntimeError("Frozen row-count mismatch")
if len(mapping) != 108:
    raise RuntimeError("Semantic-ID mapping does not contain 108 rows")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "audit_version": "independent-semantic-id-smoke-input-v1",
    "model_name": config["model_name"],
    "input_view": "behavior_plus_interest",
    "output_view": "semantic_id_108",
    "method": method,
    "loss_scope": expected_scope,
    "train_rows": train.metadata.num_rows,
    "eval_rows": evaluation.metadata.num_rows,
    "mapping_rows": len(mapping),
    "sha256": {
        "train": sha256(train_file),
        "eval": sha256(eval_file),
        "mapping": sha256(mapping_file),
        "config": sha256(config_file),
        "sample_weights": sha256(weight_file) if weight_file else None,
    },
}
output_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("SEMANTIC_ID_INDEPENDENT_INPUT_OK", json.dumps(payload, sort_keys=True))
PY

python - "$AMLT_OUTPUT_DIR/environment.json" <<'PY'
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch

packages = ("accelerate", "peft", "pyarrow", "torch", "transformers")
payload = {
    "python": platform.python_version(),
    "packages": {name: importlib.metadata.version(name) for name in packages},
    "cuda": torch.version.cuda,
    "cuda_devices": torch.cuda.device_count(),
}
expected = {"transformers": "5.14.1", "peft": "0.18.0", "accelerate": "1.12.0"}
for name, version in expected.items():
    if payload["packages"][name] != version:
        raise RuntimeError(f"Unexpected {name}: {payload['packages'][name]}")
if payload["cuda_devices"] != 4:
    raise RuntimeError("Expected four A100 GPUs")
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("TAXONOMY_SOFTMAX_ENV_OK", json.dumps(payload, sort_keys=True))
PY

python - "$TRAIN_FILE" "$WEIGHT_FILE" "$AMLT_OUTPUT_DIR/smoke_data" <<'PY'
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

source = Path(sys.argv[1])
weight_file = Path(sys.argv[2]) if sys.argv[2] else None
root = Path(sys.argv[3])
root.mkdir(parents=True, exist_ok=True)
table = pq.read_table(source)
rows = table.to_pylist()
selected = {}
for row in rows:
    selected.setdefault(str(row["target_subcategory"]), row)
if len(selected) != 108:
    raise RuntimeError("Training data does not cover all 108 interests")
representatives = [selected[label] for label in sorted(selected)]
pq.write_table(
    pa.Table.from_pylist(representatives, schema=table.schema),
    root / "ddp_108.parquet",
    compression="zstd",
)
pq.write_table(
    pa.Table.from_pylist(representatives[:32], schema=table.schema),
    root / "single_32.parquet",
    compression="zstd",
)
if weight_file is not None:
    weights = pq.read_table(weight_file)
    by_source = {
        str(row["source_sample_id"]): row for row in weights.to_pylist()
    }
    representative_ids = [str(row["source_sample_id"]) for row in representatives]
    if not set(representative_ids).issubset(by_source):
        raise RuntimeError("Frontier weights do not cover smoke rows")
    for filename, ids in (
        ("ddp_108_weights.parquet", representative_ids),
        ("single_32_weights.parquet", representative_ids[:32]),
    ):
        pq.write_table(
            pa.Table.from_pylist(
                [by_source[source_id] for source_id in ids],
                schema=weights.schema,
            ),
            root / filename,
            compression="zstd",
        )
print("TAXONOMY_SOFTMAX_DATA_OK", len(representatives), 32)
PY

python - "$MODEL_NAME" <<'PY'
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

root = Path(snapshot_download(sys.argv[1]))
index = root / "model.safetensors.index.json"
if not index.is_file():
    raise RuntimeError("Incomplete Qwen snapshot")
files = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
if any(not (root / name).is_file() for name in files):
    raise RuntimeError("Qwen model shard is missing")
AutoConfig.from_pretrained(root, local_files_only=True)
AutoTokenizer.from_pretrained(root, local_files_only=True)
print("TAXONOMY_SOFTMAX_MODEL_READY", root)
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.audit_semantic_id_token_gradients \
  --config "$CONFIG" \
  --train-file "$TRAIN_FILE" \
  --output "$AMLT_OUTPUT_DIR/gradient_audit.json" \
  --batch-size 1

SINGLE_WEIGHT_ARGS=()
DDP_WEIGHT_ARGS=()
if [[ "$METHOD" == "frontier" ]]; then
  SINGLE_WEIGHT_ARGS=(--sample-weight-file "$AMLT_OUTPUT_DIR/smoke_data/single_32_weights.parquet")
  DDP_WEIGHT_ARGS=(--sample-weight-file "$AMLT_OUTPUT_DIR/smoke_data/ddp_108_weights.parquet")
fi

CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.train_sft \
  --config "$CONFIG" \
  --train-file "$AMLT_OUTPUT_DIR/smoke_data/single_32.parquet" \
  --output-dir "$AMLT_OUTPUT_DIR/single" \
  --run-name "qwen35-bi-taxonomy-softmax-single-smoke" \
  --epochs 5 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --wandb-mode disabled \
  "${SINGLE_WEIGHT_ARGS[@]}"

CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.verify_semantic_id_token_adapter \
  --model-name "$MODEL_NAME" \
  --adapter-path "$AMLT_OUTPUT_DIR/single/final_adapter" \
  --mapping-file "$MAPPING_FILE" \
  --reference-file "$AMLT_OUTPUT_DIR/single/semantic_reload_reference.json" \
  --output "$AMLT_OUTPUT_DIR/single_reload_audit.json"

python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m new_interest.training.train_sft \
  --config "$CONFIG" \
  --train-file "$AMLT_OUTPUT_DIR/smoke_data/ddp_108.parquet" \
  --output-dir "$AMLT_OUTPUT_DIR/ddp" \
  --checkpoint-dir "/scratch/taxonomy-softmax-ddp-checkpoints" \
  --run-name "qwen35-bi-taxonomy-softmax-ddp-smoke" \
  --epochs 2 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --wandb-mode disabled \
  "${DDP_WEIGHT_ARGS[@]}"

CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.verify_semantic_id_token_adapter \
  --model-name "$MODEL_NAME" \
  --adapter-path "$AMLT_OUTPUT_DIR/ddp/final_adapter" \
  --mapping-file "$MAPPING_FILE" \
  --reference-file "$AMLT_OUTPUT_DIR/ddp/semantic_reload_reference.json" \
  --output "$AMLT_OUTPUT_DIR/ddp_reload_audit.json"

CUDA_VISIBLE_DEVICES=0 python -m new_interest.evaluation.evaluate_id_token_ranking \
  --config "$EVAL_CONFIG" \
  --adapter-path "$AMLT_OUTPUT_DIR/ddp/final_adapter" \
  --eval-file "$EVAL_FILE" \
  --interest-token-mapping-file "$MAPPING_FILE" \
  --output-dir "$AMLT_OUTPUT_DIR/real_eval" \
  --run-name "qwen35-bi-taxonomy-softmax-smoke" \
  --max-examples 64 \
  --num-shards 1 \
  --shard-index 0 \
  --batch-size 8

python - "$AMLT_OUTPUT_DIR" "$METHOD" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
method = sys.argv[2]
gradient = json.loads((root / "gradient_audit.json").read_text(encoding="utf-8"))
single = json.loads((root / "single/run_summary.json").read_text(encoding="utf-8"))
ddp = json.loads((root / "ddp/run_summary.json").read_text(encoding="utf-8"))
single_reload = json.loads((root / "single_reload_audit.json").read_text(encoding="utf-8"))
ddp_reload = json.loads((root / "ddp_reload_audit.json").read_text(encoding="utf-8"))
metrics = json.loads((root / "real_eval/metrics.json").read_text(encoding="utf-8"))

expected_scope = "taxonomy" if method == "taxonomy" else "full_vocabulary"
if gradient["semantic_id_loss_scope"] != expected_scope or not gradient["all_token_rows_finite_and_nonzero"]:
    raise RuntimeError("Semantic-ID gradient audit failed")
for summary, samples, world_size, steps in ((single, 32, 1, 20), (ddp, 108, 4, 8)):
    if summary["semantic_id_loss_scope"] != expected_scope:
        raise RuntimeError("Trainer used the wrong loss scope")
    if (summary["sample_weight_file"] is not None) != (method == "frontier"):
        raise RuntimeError("Trainer sample-weight identity mismatch")
    if summary["train_samples"] != samples or summary["world_size"] != world_size:
        raise RuntimeError("Training scope mismatch")
    if summary["expected_optimizer_steps"] != steps:
        raise RuntimeError("Optimizer-step mismatch")
    if not math.isfinite(float(summary["train_metrics"]["train_loss"])):
        raise RuntimeError("Non-finite train loss")
    drift = summary["semantic_token_drift"]
    if method == "taxonomy":
        if drift["changed_input_rows"] != 0 or not drift["all_output_rows_changed"]:
            raise RuntimeError(
                "Taxonomy loss must train output Semantic-ID rows only"
            )
    elif samples == 32:
        if drift["changed_input_rows"] != 32 or not drift["all_output_rows_changed"]:
            raise RuntimeError(
                "Frontier single smoke must change its 32 input rows and all output rows"
            )
    elif not drift["all_input_rows_changed"] or not drift["all_output_rows_changed"]:
        raise RuntimeError("Not all Semantic-ID rows changed")
for audit in (single_reload, ddp_reload):
    if not audit["all_close"] or not audit["adapter_state_integrity"]["all_tensors_exact"]:
        raise RuntimeError("Fresh adapter reload failed")
if metrics["samples"] != 64 or metrics["label_vocabulary_size"] != 108:
    raise RuntimeError("Real evaluator scope mismatch")
rows = pq.read_table(root / "real_eval/id_token_predictions.parquet").to_pylist()
if len(rows) != 64 or any(len(row["ranked_labels"]) != 20 for row in rows):
    raise RuntimeError("Prediction artifact mismatch")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

required = [
    "environment.json",
    "input_audit.json",
    "gradient_audit.json",
    "single/run_summary.json",
    "single/final_adapter/adapter_model.safetensors",
    "single_reload_audit.json",
    "ddp/run_summary.json",
    "ddp/final_adapter/adapter_model.safetensors",
    "ddp_reload_audit.json",
    "real_eval/metrics.json",
    "real_eval/id_token_predictions.parquet",
]
files = {relative: sha256(root / relative) for relative in required}
manifest = {
    "manifest_version": "independent-semantic-id-smoke-v1",
    "identity": {
        "method": method,
        "model": "Qwen/Qwen3.5-9B",
        "input": "behavior_plus_interest",
        "output": "semantic_id_108",
        "loss": expected_scope,
        "train_rows": 108,
        "world_size": 4,
        "optimizer_steps": 8,
        "eval_rows": 64,
    },
    "files": files,
}
manifest_path = root / "artifact_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
marker = {
    "artifact_manifest_sha256": sha256(manifest_path),
    "macro_recall_at_10": metrics["interest_macro_recall_at_10"],
}
(root / "smoke_success.json").write_text(
    json.dumps(marker, indent=2) + "\n", encoding="utf-8"
)
print("SEMANTIC_ID_INDEPENDENT_SMOKE_GATE_OK", method, len(files), metrics["interest_macro_recall_at_10"])
PY
