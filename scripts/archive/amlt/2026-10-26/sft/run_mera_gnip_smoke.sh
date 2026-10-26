#!/usr/bin/env bash
set -euo pipefail

EXPECTED_JOB_NAME="qwen35-mera-faithful-t005-1gpu-smoke"
if [[ "${AMLT_JOB_NAME:?AMLT_JOB_NAME is required}" != "$EXPECTED_JOB_NAME" ]]; then
  echo "Unexpected MeRA smoke job name: $AMLT_JOB_NAME" >&2
  exit 2
fi
if [[ "${MERA_GNIP_NUM_GPUS:-1}" != "1" ]]; then
  echo "MeRA Gate-C smoke requires exactly one GPU" >&2
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
CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_mera_faithful_qwen35_interest_sid_21600_t005_v1.yaml"
DATA_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_mera_semantic_id_108_200_h50_v1"
TRAIN_FILE="$DATA_ROOT/train/interest_sid/21600.parquet"
MAPPING_FILE="$DATA_ROOT/interest_token_mapping.parquet"
DATA_STATS="$DATA_ROOT/statistics.json"
DATA_METADATA="$DATA_ROOT/metadata.json"
RELATION_ROOT="$AMLT_CODE_DIR/data/training/mera_static_user_disjoint_behavior_interest_200_h50_v1"
RELATIONS_FILE="$RELATION_ROOT/relations.json"
RELATION_STATS="$RELATION_ROOT/statistics.json"
RELATION_METADATA="$RELATION_ROOT/metadata.json"
LOCAL_ROOT="/scratch/mera-gnip-faithful-t005-smoke"
SFT_DIR="$LOCAL_ROOT/sft"
CHECKPOINT_DIR="$LOCAL_ROOT/checkpoints"
GRADIENT_AUDIT="$LOCAL_ROOT/mera_gradient_audit.json"
RELOAD_AUDIT="$LOCAL_ROOT/adapter_reload_audit.json"

TRAIN_SHA256="95ed024252b5fffa2f166c2d0f4d4114968f7b21e0685043c00951e07bccab20"
MAPPING_SHA256="45397451ac2ae2e64d8156b2b47be3f454cc309a8d30a83bc23c2d3267653a0a"
DATA_STATS_SHA256="59fd668a49acc7ce6850aac54771660b7e7a6984ef12741ba605334c2b0957e6"
DATA_METADATA_SHA256="6c054404ae3e3471e881ddad664d417ee68c0b3afb4f4449abd684423af64bd3"
RELATIONS_SHA256="e81323ddeeff065469a0bb91cbd694b4feb92976bccc8d549ee012a874e70813"
RELATION_STATS_SHA256="235e5fb4f889be23c4df705728c44c5736421a50769fd2c98e5bdf0fd8f3fee8"
RELATION_METADATA_SHA256="537fec8cec09beeb888126ad5e2a243167ea1975621aff88be0b3cd5db3c1577"

for required_file in \
  "$CONFIG" "$TRAIN_FILE" "$MAPPING_FILE" "$DATA_STATS" \
  "$DATA_METADATA" "$RELATIONS_FILE" \
  "$RELATION_STATS" "$RELATION_METADATA"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing MeRA smoke input: $required_file" >&2
    exit 1
  fi
done

if [[ -f "$AMLT_OUTPUT_DIR/smoke_success.json" ]]; then
  python - "$AMLT_OUTPUT_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
marker = json.loads((root / "smoke_success.json").read_text(encoding="utf-8"))
manifest_file = root / "artifact_manifest.json"
digest = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
if digest != marker["artifact_manifest_sha256"]:
    raise RuntimeError("Committed MeRA smoke manifest hash mismatch")
manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
for relative, expected in manifest["files"].items():
    path = root / "validated" / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Committed MeRA smoke artifact mismatch: {relative}")
print("MERA_GNIP_SMOKE_REUSE_OK", len(manifest["files"]))
PY
  exit 0
fi

python - \
  "$CONFIG" \
  "$TRAIN_FILE" "$TRAIN_SHA256" \
  "$MAPPING_FILE" "$MAPPING_SHA256" \
  "$DATA_STATS" "$DATA_STATS_SHA256" \
  "$DATA_METADATA" "$DATA_METADATA_SHA256" \
  "$RELATIONS_FILE" "$RELATIONS_SHA256" \
  "$RELATION_STATS" "$RELATION_STATS_SHA256" \
  "$RELATION_METADATA" "$RELATION_METADATA_SHA256" \
  "$AMLT_OUTPUT_DIR/input_audit.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml

config_file = Path(sys.argv[1])
files = [Path(sys.argv[index]) for index in (2, 4, 6, 8, 10, 12, 14)]
hashes = [sys.argv[index] for index in (3, 5, 7, 9, 11, 13, 15)]
output_file = Path(sys.argv[16])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


for path, expected in zip(files, hashes):
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"Frozen input hash mismatch for {path}: {actual}")

config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
expected = {
    "model_name": "Qwen/Qwen3.5-9B",
    "semantic_id_token_count": 108,
    "semantic_id_initialization_mode": "mean",
    "mera_structure_weight": 0.05,
    "mera_collaborative_weight": 0.1,
    "mera_negative_samples": 10,
    "mera_cooccurrence_threshold": 0.05,
    "mera_min_collaborative_anchor_fraction": 0.0,
    "enable_thinking": False,
    "gradient_checkpointing_use_reentrant": False,
}
for key, value in expected.items():
    if config.get(key) != value:
        raise RuntimeError(f"MeRA config mismatch for {key}: {config.get(key)!r}")
if "sample_weight_file" in config:
    raise RuntimeError("MeRA baseline must not consume Cross-Fitted weights")

(
    train_file,
    mapping_file,
    data_stats_file,
    data_metadata_file,
    relations_file,
    stats_file,
    metadata_file,
) = files
if pq.ParquetFile(train_file).metadata.num_rows != 21600:
    raise RuntimeError("MeRA Train must contain 21,600 rows")
mapping = pq.read_table(mapping_file).to_pylist()
if len(mapping) != 108:
    raise RuntimeError("MeRA mapping must contain exactly 108 categories")
labels = [str(row["target_subcategory"]) for row in mapping]
tokens = [str(row["interest_token"]) for row in mapping]
relations = json.loads(relations_file.read_text(encoding="utf-8"))
if relations["labels"] != labels or relations["interest_tokens"] != tokens:
    raise RuntimeError("Relation order does not match Semantic-ID mapping")
if len(relations["cooccurrence_counts"]) != 108 or any(
    len(row) != 108 for row in relations["cooccurrence_counts"]
):
    raise RuntimeError("Relation matrix is not 108 by 108")
data_stats = json.loads(data_stats_file.read_text(encoding="utf-8"))
if data_stats["task_type"] != "interest_sid":
    raise RuntimeError("MeRA Train input is not Interest-SID")
if data_stats["prompt_contract"] != {
    "history": "chronological Semantic-ID tokens from 108 categories",
    "out_of_vocabulary_history_categories": "dropped",
    "headlines_present": False,
    "category_names_present": False,
    "response": "one Semantic-ID token from the same 108 categories",
}:
    raise RuntimeError("MeRA SID-only prompt contract drifted")
if data_stats["category_tokens"] != 108:
    raise RuntimeError("MeRA data token counts drifted")
if data_stats["train"]["empty_histories_after_filter"] != 1:
    raise RuntimeError("MeRA Train empty-History count drifted")
if data_stats["dev"]["empty_histories_after_filter"] != 6:
    raise RuntimeError("MeRA Dev empty-History count drifted")
data_metadata = json.loads(data_metadata_file.read_text(encoding="utf-8"))
if data_metadata["artifact_sha256"]["interest_token_mapping.parquet"] != hashes[1]:
    raise RuntimeError("MeRA data metadata does not commit the mapping hash")
stats = json.loads(stats_file.read_text(encoding="utf-8"))
coverage = {
    float(row["threshold"]): row
    for row in stats["cooccurrence_threshold_coverage"]
}
if coverage[0.02]["eligible_anchors"] != 108:
    raise RuntimeError("T=0.02 coverage diagnostic drifted")
if coverage[0.05]["eligible_anchors"] != 4:
    raise RuntimeError("Paper T=0.05 diagnostic drifted")
metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
if metadata["artifact_sha256"]["relations.json"] != hashes[4]:
    raise RuntimeError("Relation metadata does not commit the relation hash")

audit = {
    "audit_version": "mera-gnip-faithful-smoke-input-v1",
    "model_name": config["model_name"],
    "task_type": "interest_sid",
    "train_rows": 21600,
    "category_token_count": 108,
    "relation_parent_groups": stats["parent_groups"],
    "structure_eligible_anchors": stats["structure_eligible_anchors"],
    "t005_collaborative_eligible_anchors": coverage[0.05]["eligible_anchors"],
    "input_sha256": {
        "train.parquet": hashes[0],
        "interest_token_mapping.parquet": hashes[1],
        "data_statistics.json": hashes[2],
        "data_metadata.json": hashes[3],
        "relations.json": hashes[4],
        "relation_statistics.json": hashes[5],
        "relation_metadata.json": hashes[6],
    },
}
output_file.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
print("MERA_GNIP_INPUT_OK", json.dumps(audit, sort_keys=True))
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
    index_file = root / "model.safetensors.index.json"
    if not index_file.is_file():
        raise RuntimeError("Qwen snapshot has no safetensors index")
    filenames = set(
        json.loads(index_file.read_text(encoding="utf-8"))["weight_map"].values()
    )
    if any(
        not (root / name).is_file() or (root / name).stat().st_size == 0
        for name in filenames
    ):
        raise RuntimeError("Qwen snapshot is incomplete")
    AutoConfig.from_pretrained(root, local_files_only=True)
    AutoTokenizer.from_pretrained(root, local_files_only=True)
    return root


try:
    snapshot = validate_snapshot(snapshot_download(name, local_files_only=True))
    source = "persistent-cache"
except Exception:
    snapshot = validate_snapshot(snapshot_download(name))
    source = "downloaded-or-repaired"
print("MERA_GNIP_MODEL_READY", name, snapshot, source)
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ -e "$LOCAL_ROOT" ]]; then
  echo "Refusing to reuse non-empty local MeRA smoke root: $LOCAL_ROOT" >&2
  exit 1
fi
mkdir -p "$LOCAL_ROOT"
python -m new_interest.training.audit_mera_regularization_gradients \
  --config "$CONFIG" \
  --output "$GRADIENT_AUDIT"

python -m new_interest.training.train_sft \
  --config "$CONFIG" \
  --output-dir "$SFT_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --run-name "static-sud-mera-faithful-qwen35-interest-sid-t005-1gpu-smoke" \
  --max-samples 128 \
  --epochs 1

python - \
  "$SFT_DIR/run_summary.json" "$GRADIENT_AUDIT" \
  "$TRAIN_FILE" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
gradient = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
train_file = Path(sys.argv[3])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if summary["model_name"] != "Qwen/Qwen3.5-9B":
    raise RuntimeError("Training model identity mismatch")
if summary["train_samples"] != 128 or summary["configured_epochs"] != 1:
    raise RuntimeError("Smoke Train scope mismatch")
if summary["expected_optimizer_steps"] != 16:
    raise RuntimeError("Smoke optimizer-step mismatch")
if summary["token_audit"]["samples"] != 21600:
    raise RuntimeError("Full-file token audit scope mismatch")
if summary["token_audit"]["over_max_seq_length"] != 0:
    raise RuntimeError("Token audit found over-length samples")
if summary["train_file_sha256"] != sha256(train_file):
    raise RuntimeError("Trainer consumed the wrong Train file")
if not math.isfinite(float(summary["train_metrics"]["train_loss"])):
    raise RuntimeError("Training loss is non-finite")
if len(summary["loss_history"]) < 4:
    raise RuntimeError("Training did not emit enough loss observations")
losses = [float(row["loss"]) for row in summary["loss_history"]]
if not all(math.isfinite(value) for value in losses):
    raise RuntimeError("Loss history contains non-finite values")
first = sum(losses[:4]) / 4
last = sum(losses[-4:]) / 4
if last >= first:
    raise RuntimeError(f"Smoke loss did not decrease: {first} -> {last}")
mera = summary["mera_regularization_audit"]
if mera["interest_count"] != 108:
    raise RuntimeError("MeRA token count changed during training")
if mera["collaborative_eligible_anchors"] != 4:
    raise RuntimeError("MeRA collaborative coverage changed during training")
if summary["mera_regularization_training"]["calls"] <= 0:
    raise RuntimeError("MeRA regularizer was never called")
drift = summary["semantic_token_drift"]
if not drift["all_input_rows_changed"] or not drift["all_output_rows_changed"]:
    raise RuntimeError("MeRA smoke did not update every Semantic-ID row")
if not gradient["all_token_rows_finite_and_nonzero"]:
    raise RuntimeError("MeRA regularization gradient audit failed")
if len(gradient["gradient_parameters"]) != 2:
    raise RuntimeError("Expected input and output trainable-token gradients")
if any(row["shape"][0] != 108 for row in gradient["gradient_parameters"]):
    raise RuntimeError("MeRA gradient audit did not cover all 108 token rows")
print("MERA_GNIP_TRAIN_OK", first, last)
PY

python -m new_interest.training.verify_semantic_id_token_adapter \
  --model-name "$MODEL_NAME" \
  --adapter-path "$SFT_DIR/final_adapter" \
  --mapping-file "$MAPPING_FILE" \
  --reference-file "$SFT_DIR/semantic_reload_reference.json" \
  --output "$RELOAD_AUDIT"

python - \
  "$SFT_DIR" "$GRADIENT_AUDIT" "$RELOAD_AUDIT" \
  "$AMLT_OUTPUT_DIR" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

sft_dir = Path(sys.argv[1])
gradient_file = Path(sys.argv[2])
reload_file = Path(sys.argv[3])
output_dir = Path(sys.argv[4])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


reload = json.loads(reload_file.read_text(encoding="utf-8"))
if not reload["all_close"]:
    raise RuntimeError("Fresh adapter reload failed")
if not reload["adapter_state_integrity"]["all_tensors_exact"]:
    raise RuntimeError("Adapter state tensors did not reload bit-exactly")

publish = output_dir / "validated"
publish.mkdir(parents=True, exist_ok=True)
sources = [
    gradient_file,
    reload_file,
    sft_dir / "run_summary.json",
    sft_dir / "resolved_config.json",
    sft_dir / "model_audit.json",
    sft_dir / "mera_regularization_audit.json",
    sft_dir / "semantic_token_initialization.json",
    sft_dir / "semantic_token_drift.json",
    sft_dir / "semantic_reload_reference.json",
]
sources.extend(
    path for path in (sft_dir / "final_adapter").rglob("*") if path.is_file()
)
files = {}
for source in sources:
    relative = (
        Path("final_adapter") / source.relative_to(sft_dir / "final_adapter")
        if (sft_dir / "final_adapter") in source.parents
        else Path(source.name)
    )
    destination = publish / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_handle, destination.open("wb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
    files[relative.as_posix()] = sha256(destination)

manifest = {
    "manifest_version": "mera-gnip-smoke-artifacts-v1",
    "files": dict(sorted(files.items())),
}
manifest_file = output_dir / "artifact_manifest.json"
manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
marker = {
    "status": "validated",
    "gate": "single-gpu-gradient-train-save-reload",
    "artifact_manifest_sha256": sha256(manifest_file),
    "artifact_count": len(files),
}
(output_dir / "smoke_success.json").write_text(
    json.dumps(marker, indent=2) + "\n", encoding="utf-8"
)
print("MERA_GNIP_SMOKE_GATE_OK", len(files))
PY
