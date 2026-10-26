#!/usr/bin/env bash
set -euo pipefail

if [[ "${AMLT_JOB_NAME:?AMLT_JOB_NAME is required}" != "qwen35-bi-preference-state-reasoning-sft-smoke" ]]; then
  echo "Unexpected job name: $AMLT_JOB_NAME" >&2
  exit 2
fi

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export WANDB_DISABLED=true
export HF_HOME="${HF_HOME:-/mnt/default/huggingface-cache-preference-state-sft-v1}"
mkdir -p "$HF_HOME" "$AMLT_OUTPUT_DIR"

MODEL_NAME="Qwen/Qwen3.5-9B"
MODEL_REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_preference_state_reasoning_qwen35_behavior_interest_1080_v1.yaml"
TRAIN_FILE="$AMLT_CODE_DIR/data/training/preference_state_reasoning_sft_gpt55_qwen35_behavior_interest_1080_v1/train.parquet"
TRAIN_METADATA="$AMLT_CODE_DIR/data/training/preference_state_reasoning_sft_gpt55_qwen35_behavior_interest_1080_v1/metadata.json"
MAPPING_FILE="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_scaling_v1/interest_token_mapping.parquet"
REMOTE_BASE_ADAPTER="/mnt/output/projects/sft4safety/amlt-results/7212029499.70786-e4538499-b9a2/sft/final_adapter"
LOCAL_BASE_ADAPTER="/scratch/preference-state-base-adapter"
EXPECTED_TRAIN_SHA="311fdfae86efb61d5b907da89bc553895058f49475abb84f7a3e228953ead7bc"
EXPECTED_MAPPING_SHA="45397451ac2ae2e64d8156b2b47be3f454cc309a8d30a83bc23c2d3267653a0a"
EXPECTED_BASE_ADAPTER_SHA="b1d698724a90d929f28973433ba41ca8b95f2a5dc15523b4aead627e3c901f2f"

for required in "$CONFIG" "$TRAIN_FILE" "$TRAIN_METADATA" "$MAPPING_FILE"; do
  if [[ ! -s "$required" ]]; then
    echo "Missing Reasoning-SFT input: $required" >&2
    exit 1
  fi
done
for required in \
  "$REMOTE_BASE_ADAPTER/adapter_model.safetensors" \
  "$REMOTE_BASE_ADAPTER/adapter_config.json" \
  "$REMOTE_BASE_ADAPTER/tokenizer.json"; do
  if [[ ! -s "$required" ]]; then
    echo "Missing 21.6k Vanilla base adapter input: $required" >&2
    exit 1
  fi
done

python - \
  "$CONFIG" "$TRAIN_FILE" "$TRAIN_METADATA" "$MAPPING_FILE" \
  "$REMOTE_BASE_ADAPTER" "$EXPECTED_TRAIN_SHA" "$EXPECTED_MAPPING_SHA" \
  "$EXPECTED_BASE_ADAPTER_SHA" "$AMLT_OUTPUT_DIR/input_audit.json" <<'PY'
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml

config_file, train_file, metadata_file, mapping_file, adapter = map(
    Path, sys.argv[1:6]
)
expected_train, expected_mapping, expected_adapter = sys.argv[6:9]
output_file = Path(sys.argv[9])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if sha256(train_file) != expected_train:
    raise RuntimeError("Reasoning-SFT Train hash mismatch")
if sha256(mapping_file) != expected_mapping:
    raise RuntimeError("Semantic-ID mapping hash mismatch")
if sha256(adapter / "adapter_model.safetensors") != expected_adapter:
    raise RuntimeError("21.6k Vanilla base adapter hash mismatch")
config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
if config["model_name"] != "Qwen/Qwen3.5-9B":
    raise RuntimeError("Reasoning-SFT model mismatch")
if config.get("model_revision") != "c202236235762e1c871ad0ccb60c8ee5ba337b9a":
    raise RuntimeError("Reasoning-SFT model revision mismatch")
if config["enable_thinking"] is not True:
    raise RuntimeError("Reasoning-SFT must enable native thinking template")
if int(config["max_seq_length"]) != 2304:
    raise RuntimeError("Reasoning-SFT max sequence length mismatch")
if float(config["learning_rate"]) != 1e-5 or int(config["epochs"]) != 2:
    raise RuntimeError("Reasoning-SFT schedule mismatch")
metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
if metadata["target_disclosure"] != "final-answer-only":
    raise RuntimeError("Reasoning-SFT target disclosure contract mismatch")
rows = pq.read_table(train_file).to_pylist()
counts = Counter(str(row["target_subcategory"]) for row in rows)
token_by_target = {
    str(row["target_subcategory"]): str(row["interest_token"])
    for row in pq.read_table(mapping_file).to_pylist()
}
if len(rows) != 1080 or len(counts) != 108 or set(counts.values()) != {10}:
    raise RuntimeError("Reasoning-SFT data is not 108 x 10")
for row in rows:
    completion = str(row["response"])
    target = str(row["target_subcategory"])
    if not re.fullmatch(
        r"<think>\n[\s\S]+\n</think>\n<answer><interest_\d{3}></answer>",
        completion,
    ):
        raise RuntimeError("Reasoning-SFT completion format mismatch")
    expected_token = token_by_target[target]
    if not completion.endswith(f"<answer>{expected_token}</answer>"):
        raise RuntimeError("Reasoning-SFT final Semantic ID mismatch")
    if (
        row["messages"][0] != {"role": "user", "content": row["prompt"]}
        or row["messages"][1] != {"role": "assistant", "content": completion}
    ):
        raise RuntimeError("Reasoning-SFT message mismatch")
    reasoning = completion.split("</think>", 1)[0]
    if target.casefold() in reasoning.casefold():
        raise RuntimeError("Reasoning-SFT reasoning reveals target")
payload = {
    "audit_version": "preference-state-reasoning-sft-input-v1",
    "model": config["model_name"],
    "base_adapter_sha256": expected_adapter,
    "train_sha256": expected_train,
    "mapping_sha256": expected_mapping,
    "train_rows": len(rows),
    "classes": len(counts),
    "per_class": 10,
    "enable_thinking": True,
    "max_seq_length": 2304,
}
output_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("PREFERENCE_STATE_REASONING_SFT_INPUT_OK", json.dumps(payload, sort_keys=True))
PY

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
    raise RuntimeError("Reasoning-SFT smoke requires four A100 GPUs")
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("PREFERENCE_STATE_REASONING_SFT_ENV_OK", json.dumps(payload, sort_keys=True))
PY

python - "$MODEL_NAME" "$MODEL_REVISION" <<'PY'
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

root = Path(snapshot_download(sys.argv[1], revision=sys.argv[2]))
if root.name != sys.argv[2]:
    raise RuntimeError(f"Unexpected Qwen revision: {root.name}")
index = root / "model.safetensors.index.json"
shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
if len(shards) != 4 or any(not (root / name).is_file() for name in shards):
    raise RuntimeError("Persistent Qwen snapshot is incomplete")
AutoConfig.from_pretrained(root, local_files_only=True)
AutoTokenizer.from_pretrained(root, local_files_only=True)
print("PREFERENCE_STATE_REASONING_SFT_MODEL_READY", root)
PY

PERSISTENT_HF_HOME="$HF_HOME"
LOCAL_HF_HOME="/scratch/huggingface-cache-preference-state-sft-v1"
PERSISTENT_MODEL_CACHE="$PERSISTENT_HF_HOME/hub/models--Qwen--Qwen3.5-9B"
LOCAL_MODEL_CACHE="$LOCAL_HF_HOME/hub/models--Qwen--Qwen3.5-9B"
mkdir -p "$LOCAL_MODEL_CACHE" "$LOCAL_BASE_ADAPTER"
rsync -a "$PERSISTENT_MODEL_CACHE/" "$LOCAL_MODEL_CACHE/"
rsync -a "$REMOTE_BASE_ADAPTER/" "$LOCAL_BASE_ADAPTER/"
export HF_HOME="$LOCAL_HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if [[ "$(shasum -a 256 "$LOCAL_BASE_ADAPTER/adapter_model.safetensors" | awk '{print $1}')" != "$EXPECTED_BASE_ADAPTER_SHA" ]]; then
  echo "Localized base adapter hash mismatch" >&2
  exit 1
fi

python - "$MODEL_NAME" "$MODEL_REVISION" "$AMLT_OUTPUT_DIR/model_cache_audit.json" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path(snapshot_download(sys.argv[1], local_files_only=True))
if root.name != sys.argv[2] or not root.is_relative_to(Path(os.environ["HF_HOME"])):
    raise RuntimeError("Qwen model did not resolve from localized cache")
index = root / "model.safetensors.index.json"
shards = [root / name for name in sorted(set(json.loads(index.read_text())["weight_map"].values()))]
if len(shards) != 4 or any(not path.is_file() or path.stat().st_size == 0 for path in shards):
    raise RuntimeError("Localized Qwen snapshot is incomplete")
payload = {
    "audit_version": "preference-state-reasoning-sft-local-cache-v1",
    "revision": root.name,
    "snapshot": str(root),
    "shard_count": len(shards),
    "total_shard_bytes": sum(path.stat().st_size for path in shards),
}
Path(sys.argv[3]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("PREFERENCE_STATE_REASONING_SFT_LOCAL_CACHE_OK", json.dumps(payload, sort_keys=True))
PY

python - "$MODEL_NAME" "$MAPPING_FILE" "$TRAIN_FILE" "$AMLT_OUTPUT_DIR/token_audit_full.json" <<'PY'
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from new_interest.training.semantic_id_tokens import register_semantic_interest_tokens
from new_interest.training.sft_data import audit_token_lengths

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
register_semantic_interest_tokens(tokenizer, Path(sys.argv[2]), expected_count=108)
audit = audit_token_lengths(
    Path(sys.argv[3]),
    tokenizer,
    max_seq_length=2304,
    enable_thinking=True,
    output_path=Path(sys.argv[4]),
)
if audit["samples"] != 1080 or audit["over_max_seq_length"] != 0:
    raise RuntimeError(f"Reasoning-SFT token audit failed: {audit}")
print("PREFERENCE_STATE_REASONING_SFT_TOKEN_AUDIT_OK", json.dumps(audit, sort_keys=True))
PY

mkdir -p "$AMLT_OUTPUT_DIR/smoke_data"
python - "$TRAIN_FILE" "$AMLT_OUTPUT_DIR/smoke_data" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq

source = pq.read_table(Path(sys.argv[1]))
root = Path(sys.argv[2])
rows = source.to_pylist()
first = rows[:108]
if len({str(row["target_subcategory"]) for row in first}) != 108:
    raise RuntimeError("First Reasoning-SFT block is not one row per class")
pq.write_table(source.slice(0, 32), root / "single_32.parquet", compression="zstd")
pq.write_table(source.slice(0, 108), root / "ddp_108.parquet", compression="zstd")
pq.write_table(source.slice(108, 16), root / "format_holdout_16.parquet", compression="zstd")
print("PREFERENCE_STATE_REASONING_SFT_SMOKE_DATA_OK", 32, 108, 16)
PY

CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.train_sft \
  --config "$CONFIG" \
  --adapter-path "$LOCAL_BASE_ADAPTER" \
  --train-file "$AMLT_OUTPUT_DIR/smoke_data/single_32.parquet" \
  --output-dir "$AMLT_OUTPUT_DIR/single" \
  --run-name "preference-state-reasoning-sft-single-smoke" \
  --epochs 2 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --wandb-mode disabled

CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.verify_semantic_id_token_adapter \
  --model-name "$MODEL_NAME" \
  --adapter-path "$AMLT_OUTPUT_DIR/single/final_adapter" \
  --mapping-file "$MAPPING_FILE" \
  --reference-file "$AMLT_OUTPUT_DIR/single/semantic_reload_reference.json" \
  --output "$AMLT_OUTPUT_DIR/single_reload_audit.json" \
  --enable-thinking

python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m new_interest.training.train_sft \
  --config "$CONFIG" \
  --adapter-path "$LOCAL_BASE_ADAPTER" \
  --train-file "$AMLT_OUTPUT_DIR/smoke_data/ddp_108.parquet" \
  --output-dir "$AMLT_OUTPUT_DIR/ddp" \
  --checkpoint-dir "/scratch/preference-state-reasoning-sft-ddp-checkpoints" \
  --run-name "preference-state-reasoning-sft-ddp-smoke" \
  --epochs 2 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --wandb-mode disabled

CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.verify_semantic_id_token_adapter \
  --model-name "$MODEL_NAME" \
  --adapter-path "$AMLT_OUTPUT_DIR/ddp/final_adapter" \
  --mapping-file "$MAPPING_FILE" \
  --reference-file "$AMLT_OUTPUT_DIR/ddp/semantic_reload_reference.json" \
  --output "$AMLT_OUTPUT_DIR/ddp_reload_audit.json" \
  --enable-thinking

CUDA_VISIBLE_DEVICES=0 python -m new_interest.evaluation.evaluate_preference_state_format \
  --model-name "$MODEL_NAME" \
  --adapter-path "$AMLT_OUTPUT_DIR/ddp/final_adapter" \
  --eval-file "$TRAIN_FILE" \
  --mapping-file "$MAPPING_FILE" \
  --output-dir "$AMLT_OUTPUT_DIR/format_eval" \
  --start-index 108 \
  --max-examples 16 \
  --max-new-tokens 256

python - "$AMLT_OUTPUT_DIR" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
token_audit = json.loads((root / "token_audit_full.json").read_text())
single = json.loads((root / "single/run_summary.json").read_text())
ddp = json.loads((root / "ddp/run_summary.json").read_text())
single_reload = json.loads((root / "single_reload_audit.json").read_text())
ddp_reload = json.loads((root / "ddp_reload_audit.json").read_text())
format_summary = json.loads((root / "format_eval/format_summary.json").read_text())
if token_audit["samples"] != 1080 or token_audit["over_max_seq_length"] != 0:
    raise RuntimeError("Full token audit failed")
for summary, samples, world_size, steps in (
    (single, 32, 1, 8),
    (ddp, 108, 4, 8),
):
    if summary["train_samples"] != samples or summary["world_size"] != world_size:
        raise RuntimeError("Reasoning-SFT training scope mismatch")
    if summary["expected_optimizer_steps"] != steps:
        raise RuntimeError("Reasoning-SFT optimizer-step mismatch")
    if summary["enable_thinking"] is not True:
        raise RuntimeError("Reasoning-SFT disabled thinking unexpectedly")
    if not math.isfinite(float(summary["train_metrics"]["train_loss"])):
        raise RuntimeError("Reasoning-SFT produced non-finite loss")
    if summary["checkpoint_identity"]["input_adapter_sha256"] != "b1d698724a90d929f28973433ba41ca8b95f2a5dc15523b4aead627e3c901f2f":
        raise RuntimeError("Reasoning-SFT consumed the wrong base adapter")
for audit in (single_reload, ddp_reload):
    if not audit["all_close"] or not audit["adapter_state_integrity"]["all_tensors_exact"]:
        raise RuntimeError("Reasoning-SFT adapter reload failed")
if format_summary["samples"] != 16:
    raise RuntimeError("Reasoning-SFT format probe scope mismatch")
if format_summary["valid_format"] < 12 or format_summary["legal_token"] < 12:
    raise RuntimeError(f"Reasoning-SFT format gate failed: {format_summary}")
if format_summary["reasoning_hides_prediction"] < 12:
    raise RuntimeError(f"Reasoning disclosure gate failed: {format_summary}")

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
    "token_audit_full.json",
    "single/run_summary.json",
    "single/final_adapter/adapter_model.safetensors",
    "single_reload_audit.json",
    "ddp/run_summary.json",
    "ddp/final_adapter/adapter_model.safetensors",
    "ddp_reload_audit.json",
    "format_eval/format_summary.json",
    "format_eval/format_predictions.parquet",
]
manifest = {
    "manifest_version": "preference-state-reasoning-sft-smoke-v1",
    "identity": {
        "model": "Qwen/Qwen3.5-9B",
        "model_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "base_adapter_sha256": "b1d698724a90d929f28973433ba41ca8b95f2a5dc15523b4aead627e3c901f2f",
        "train_sha256": "311fdfae86efb61d5b907da89bc553895058f49475abb84f7a3e228953ead7bc",
        "reasoning_dataset_rows": 1080,
        "single_train_rows": 32,
        "ddp_train_rows": 108,
        "world_size": 4,
        "optimizer_steps": 8,
        "format_probe_rows": 16,
    },
    "token_audit": token_audit,
    "format_gate": {
        "valid_format": format_summary["valid_format"],
        "legal_token": format_summary["legal_token"],
        "reasoning_hides_prediction": format_summary["reasoning_hides_prediction"],
        "exact": format_summary["exact"],
    },
    "files": {relative: sha256(root / relative) for relative in required},
}
manifest_file = root / "artifact_manifest.json"
manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
(root / "smoke_success.json").write_text(
    json.dumps({"artifact_manifest_sha256": sha256(manifest_file)}, indent=2) + "\n",
    encoding="utf-8",
)
print("PREFERENCE_STATE_REASONING_SFT_SMOKE_GATE_OK", format_summary["valid_format"], format_summary["exact"])
PY
