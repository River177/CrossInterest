#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${SEMANTIC_ID_BEHAVIOR_INTEREST_CROSSFIT_NUM_GPUS:-4}"
if [[ "$NUM_GPUS" != "4" ]]; then
  echo "Behavior+Interest Semantic-ID class cross-fit OOF requires four GPUs; got $NUM_GPUS" >&2
  exit 2
fi

case "$AMLT_JOB_NAME" in
  qwen35-behavior-interest-semantic-id-class-crossfit-fold-0) FOLD=0 ;;
  qwen35-behavior-interest-semantic-id-class-crossfit-fold-1) FOLD=1 ;;
  qwen35-behavior-interest-semantic-id-class-crossfit-fold-2) FOLD=2 ;;
  qwen35-behavior-interest-semantic-id-class-crossfit-fold-3) FOLD=3 ;;
  qwen35-behavior-interest-semantic-id-class-crossfit-fold-4) FOLD=4 ;;
  *)
    echo "Unexpected Semantic-ID OOF job name: $AMLT_JOB_NAME" >&2
    exit 2
    ;;
esac

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export WANDB_DISABLED=true
export HF_HOME="${HF_HOME:-/mnt/default/huggingface-cache}"
mkdir -p "$HF_HOME"

MODEL_NAME="Qwen/Qwen3.5-9B"
CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_semantic_id_qwen35_behavior_interest_21600.yaml"
SEMANTIC_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_v1"
SEMANTIC_TRAIN="$SEMANTIC_ROOT/train/behavior_interest/21600.parquet"
MAPPING_FILE="$SEMANTIC_ROOT/interest_token_mapping.parquet"
FOLD_MANIFEST="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_crossfit_reweighting_qwen35_v1/fold_manifest.parquet"
SPLIT_DIR="$AMLT_OUTPUT_DIR/data/fold_${FOLD}"
SFT_DIR="$AMLT_OUTPUT_DIR/sft"
SCORE_DIR="$AMLT_OUTPUT_DIR/oof"
LOCAL_CHECKPOINT_ROOT="/scratch/semantic-id-class-crossfit-behavior-interest-oof-fold-${FOLD}-checkpoints"
PERSISTENT_CHECKPOINT_ROOT="$AMLT_OUTPUT_DIR/checkpoints"
TRAIN_SHA256="dc8f04c172bf2481f4414d2c7a0ca510ae2bc0e125c2a03d984072b72b4a8140"
MAPPING_SHA256="45397451ac2ae2e64d8156b2b47be3f454cc309a8d30a83bc23c2d3267653a0a"
FOLD_MANIFEST_SHA256="3d25e36b4c34ae2516ac6fccee3e2c4d6596ac610076c6ef0f8b8505cd3a4b3a"

for path in "$CONFIG" "$SEMANTIC_TRAIN" "$MAPPING_FILE" "$FOLD_MANIFEST"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing OOF input: $path" >&2
    exit 1
  fi
done

python - \
  "$SEMANTIC_TRAIN" "$TRAIN_SHA256" \
  "$MAPPING_FILE" "$MAPPING_SHA256" \
  "$FOLD_MANIFEST" "$FOLD_MANIFEST_SHA256" <<'PY'
import hashlib
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


for index in range(1, len(sys.argv), 2):
    path = Path(sys.argv[index])
    expected = sys.argv[index + 1]
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"Frozen OOF input hash mismatch for {path}: {actual}")
print("COMBINED_SEMANTIC_ID_CLASS_CROSSFIT_OOF_INPUT_OK")
PY

python - "$AMLT_OUTPUT_DIR/environment.json" <<'PY'
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch

packages = (
    "accelerate",
    "peft",
    "pyarrow",
    "torch",
    "transformers",
    "wandb",
)
payload = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages": {
        package: importlib.metadata.version(package) for package in packages
    },
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
print("COMBINED_SEMANTIC_ID_CLASS_CROSSFIT_OOF_ENV_OK", json.dumps(payload, sort_keys=True))
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
print("COMBINED_SEMANTIC_ID_CLASS_CROSSFIT_OOF_MODEL_READY", name, snapshot, source)
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m new_interest.experiments.semantic_id_crossfit prepare \
  --semantic-train-file "$SEMANTIC_TRAIN" \
  --fold-manifest "$FOLD_MANIFEST" \
  --output-dir "$SPLIT_DIR" \
  --fold "$FOLD"

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NUM_GPUS" \
  -m new_interest.training.train_sft \
  --config "$CONFIG" \
  --train-file "$SPLIT_DIR/train.parquet" \
  --output-dir "$SFT_DIR" \
  --checkpoint-dir "$LOCAL_CHECKPOINT_ROOT" \
  --persistent-checkpoint-dir "$PERSISTENT_CHECKPOINT_ROOT" \
  --resume-from-checkpoint auto \
  --run-name "qwen35-behavior-interest-semantic-id-class-crossfit-fold-${FOLD}" \
  --epochs 3 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --save-steps 100 \
  --wandb-mode disabled

python - "$SPLIT_DIR/split_summary.json" "$SFT_DIR/run_summary.json" "$SFT_DIR/checkpoint_audit.json" "$FOLD" <<'PY'
import json
import math
import sys
from pathlib import Path

from new_interest.training.checkpoint_persistence import (
    select_latest_complete_checkpoint,
)

split = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
summary = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
audit_path = Path(sys.argv[3])
fold = int(sys.argv[4])
if split["fold"] != fold:
    raise RuntimeError("OOF split fold mismatch")
if split["classes"] != 108 or split["train_rows"] != 17280:
    raise RuntimeError(f"OOF teacher split mismatch: {split}")
if split["heldout_rows"] != 4320 or split["overlapping_source_ids"] != 0:
    raise RuntimeError(f"OOF held-out split mismatch: {split}")
if summary["model_name"] != "Qwen/Qwen3.5-9B":
    raise RuntimeError("OOF model identity mismatch")
if summary["world_size"] != 4 or summary["train_samples"] != 17280:
    raise RuntimeError("OOF DDP/data identity mismatch")
if summary["configured_epochs"] != 3:
    raise RuntimeError("OOF epoch identity mismatch")
if summary["effective_global_batch_size"] != 32:
    raise RuntimeError("OOF effective batch mismatch")
if summary["expected_optimizer_steps"] != 1620:
    raise RuntimeError("OOF optimizer-step identity mismatch")
if summary["gradient_checkpointing_kwargs"] != {"use_reentrant": False}:
    raise RuntimeError("OOF gradient checkpointing is not non-reentrant")
if summary["sample_weight_file"] is not None:
    raise RuntimeError("OOF teacher unexpectedly used sample weights")
if summary["train_file_sha256"] != split["output_sha256"]["train"]:
    raise RuntimeError("OOF teacher consumed the wrong split")
if not math.isfinite(float(summary["train_metrics"]["train_loss"])):
    raise RuntimeError("OOF training loss is not finite")
if summary["token_audit"]["over_max_seq_length"] != 0:
    raise RuntimeError("OOF token audit found over-length samples")
semantic = summary["model_audit"]["semantic_id_tokens"]
if not semantic["enabled"] or semantic["count"] != 108:
    raise RuntimeError("OOF Semantic-ID model audit mismatch")
if summary["model_audit"]["forbidden_trainable_parameters"]:
    raise RuntimeError("OOF teacher has forbidden trainable parameters")
persistent_root = Path(summary["persistent_checkpoint_dir"])
selected, diagnostics = select_latest_complete_checkpoint(
    persistent_root,
    summary["checkpoint_identity"],
)
if diagnostics:
    raise RuntimeError(f"OOF checkpoint diagnostics: {diagnostics}")
if selected is None or selected.name != "checkpoint-1620":
    raise RuntimeError(f"Expected complete checkpoint-1620, got {selected}")
checkpoint_audit = {
    "fold": fold,
    "checkpoint_name": selected.name,
    "checkpoint_identity": summary["checkpoint_identity"],
    "resumed_from_checkpoint": summary["resumed_from_checkpoint"],
    "identity_safe_persistence": True,
}
audit_path.write_text(json.dumps(checkpoint_audit, indent=2) + "\n", encoding="utf-8")
print("COMBINED_SEMANTIC_ID_CLASS_CROSSFIT_OOF_TRAIN_OK", fold, 17280, 1620)
PY

CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.verify_semantic_id_token_adapter \
  --model-name "$MODEL_NAME" \
  --adapter-path "$SFT_DIR/final_adapter" \
  --mapping-file "$MAPPING_FILE" \
  --reference-file "$SFT_DIR/semantic_reload_reference.json" \
  --output "$SFT_DIR/semantic_adapter_reload_audit.json"

CUDA_VISIBLE_DEVICES=0 python -m new_interest.experiments.semantic_id_crossfit score \
  --model-name "$MODEL_NAME" \
  --adapter-path "$SFT_DIR/final_adapter" \
  --heldout-file "$SPLIT_DIR/heldout.parquet" \
  --mapping-file "$MAPPING_FILE" \
  --output-dir "$SCORE_DIR" \
  --fold "$FOLD" \
  --batch-size 8

python - "$AMLT_OUTPUT_DIR" "$FOLD" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
fold = int(sys.argv[2])
split = json.loads((root / f"data/fold_{fold}/split_summary.json").read_text())
reload = json.loads((root / "sft/semantic_adapter_reload_audit.json").read_text())
score = json.loads((root / "oof/score_summary.json").read_text())
rows = pq.read_table(root / "oof/oof_scores.parquet").to_pylist()
if not reload["all_close"] or reload["sample_count"] != 4:
    raise RuntimeError("OOF adapter reload audit failed")
integrity = reload["adapter_state_integrity"]
if integrity["tensor_count"] != 498 or not integrity["all_tensors_exact"]:
    raise RuntimeError("OOF compact adapter state audit failed")
if integrity["saved_state_sha256"] != integrity["loaded_state_sha256"]:
    raise RuntimeError("OOF reloaded adapter state hash mismatch")
if score["model_name"] != "Qwen/Qwen3.5-9B":
    raise RuntimeError("OOF scorer model identity mismatch")
if score["fold"] != fold or score["rows"] != 4320 or score["classes"] != 108:
    raise RuntimeError("OOF score identity mismatch")
if score["heldout_sha256"] != split["output_sha256"]["heldout"]:
    raise RuntimeError("OOF scorer consumed the wrong held-out split")
if not math.isclose(float(score["fold_prior_sum"]), 1.0, abs_tol=1e-12):
    raise RuntimeError("OOF fold output prior is not normalized")
if float(score["fold_prior_min"]) <= 0:
    raise RuntimeError("OOF fold output prior is not strictly positive")
if len(rows) != 4320:
    raise RuntimeError("OOF score row count mismatch")
if len({row["source_sample_id"] for row in rows}) != 4320:
    raise RuntimeError("OOF scores contain duplicate source IDs")
if len({row["user_id"] for row in rows}) != 4320:
    raise RuntimeError("OOF scores contain duplicate users")
counts = {}
for row in rows:
    if int(row["fold"]) != fold:
        raise RuntimeError("OOF score fold mismatch")
    counts[row["target_subcategory"]] = counts.get(row["target_subcategory"], 0) + 1
    if float(row["fold_target_prior"]) <= 0:
        raise RuntimeError("OOF score has non-positive prior")
    for key in (
        "target_raw_logprob",
        "prior_adjusted_target_logprob",
        "prior_adjusted_nll",
        "prior_adjusted_margin",
    ):
        if not math.isfinite(float(row[key])):
            raise RuntimeError(f"OOF score has non-finite {key}")
if len(counts) != 108 or set(counts.values()) != {40}:
    raise RuntimeError("OOF scores are not exactly 40 rows per class")

relative_files = [
    "environment.json",
    f"data/fold_{fold}/split_summary.json",
    "sft/token_audit.json",
    "sft/semantic_token_initialization.json",
    "sft/model_audit.json",
    "sft/semantic_token_drift.json",
    "sft/semantic_reload_reference.json",
    "sft/run_summary.json",
    "sft/resolved_config.json",
    "sft/checkpoint_audit.json",
    "sft/semantic_adapter_reload_audit.json",
    "oof/oof_scores.parquet",
    "oof/score_summary.json",
]
hashes = {}
for relative in relative_files:
    path = root / relative
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing compact OOF artifact: {relative}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    hashes[relative] = digest.hexdigest()
manifest = {
    "manifest_version": "semantic-id-class-crossfit-behavior-interest-oof-v1",
    "analysis_scope": "qwen35-behavior-plus-interest",
    "fold": fold,
    "shuffled_evaluation_used": False,
    "weighted_training_used": False,
    "compact_files": hashes,
    "excluded_prefixes": [
        "checkpoints/",
        f"data/fold_{fold}/train.parquet",
        f"data/fold_{fold}/heldout.parquet",
        "sft/final_adapter/",
    ],
}
(root / "artifact_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)
print("COMBINED_SEMANTIC_ID_CLASS_CROSSFIT_OOF_GATE_OK", fold, len(hashes))
PY
