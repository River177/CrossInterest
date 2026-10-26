#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: run_mera_gnip_ablation.sh <smoke|full> <sid|behavior-interest> <vanilla|structure|collaborative|full>}"
VIEW="${2:?Missing input view}"
CONDITION="${3:?Missing ablation condition}"
NUM_GPUS="${MERA_GNIP_ABLATION_NUM_GPUS:-4}"
if [[ "$NUM_GPUS" != "4" ]]; then
  echo "MeRA ablations require four GPUs; got $NUM_GPUS" >&2
  exit 2
fi
case "$MODE" in
  smoke)
    TRAIN_EPOCHS=2
    EXPECTED_SAMPLES=108
    EXPECTED_STEPS=8
    EXPECTED_EVAL=64
    JOB_SUFFIX="ddp-eval-smoke"
    ;;
  full)
    TRAIN_EPOCHS=3
    EXPECTED_SAMPLES=21600
    EXPECTED_STEPS=2025
    EXPECTED_EVAL=31392
    JOB_SUFFIX="full"
    ;;
  *)
    echo "Unsupported MeRA ablation mode: $MODE" >&2
    exit 2
    ;;
esac
case "$CONDITION" in
  vanilla)
    STRUCTURE_WEIGHT="0.0"
    COLLABORATIVE_WEIGHT="0.0"
    REGULARIZER_ENABLED="false"
    ;;
  structure)
    STRUCTURE_WEIGHT="0.05"
    COLLABORATIVE_WEIGHT="0.0"
    REGULARIZER_ENABLED="true"
    ;;
  collaborative)
    STRUCTURE_WEIGHT="0.0"
    COLLABORATIVE_WEIGHT="0.1"
    REGULARIZER_ENABLED="true"
    ;;
  full)
    STRUCTURE_WEIGHT="0.05"
    COLLABORATIVE_WEIGHT="0.1"
    REGULARIZER_ENABLED="true"
    ;;
  *)
    echo "Unsupported MeRA ablation condition: $CONDITION" >&2
    exit 2
    ;;
esac

MODEL_NAME="Qwen/Qwen3.5-9B"
MAPPING_SHA256="45397451ac2ae2e64d8156b2b47be3f454cc309a8d30a83bc23c2d3267653a0a"
RELATIONS_FILE="$AMLT_CODE_DIR/data/training/mera_static_user_disjoint_behavior_interest_200_h50_v1/relations.json"
RELATIONS_SHA256="e81323ddeeff065469a0bb91cbd694b4feb92976bccc8d549ee012a874e70813"
case "$VIEW" in
  sid)
    VIEW_TAG="sid"
    DATA_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_mera_semantic_id_108_200_h50_v1"
    TRAIN_FILE="$DATA_ROOT/train/interest_sid/21600.parquet"
    EVAL_FILE="$DATA_ROOT/eval/interest_sid/dev_full_grouped.parquet"
    MAPPING_FILE="$DATA_ROOT/interest_token_mapping.parquet"
    EVAL_CONFIG="$AMLT_CODE_DIR/configs/id_token/static_sud_mera_faithful_qwen35_interest_sid_t005_v1.json"
    TRAIN_SHA256="95ed024252b5fffa2f166c2d0f4d4114968f7b21e0685043c00951e07bccab20"
    EVAL_SHA256="afdd1997b6139583e4b0e44b50bba963996bea9cf9325e8f04d77fa8ed1b2bbe"
    case "$CONDITION" in
      vanilla) CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_mera_qwen35_sid_sid_mean_vanilla_21600_v1.yaml" ;;
      structure) CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_mera_qwen35_sid_sid_mean_structure_21600_v1.yaml" ;;
      collaborative) CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_mera_qwen35_sid_sid_mean_collaborative_21600_v1.yaml" ;;
      full) CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_mera_faithful_qwen35_interest_sid_21600_t005_v1.yaml" ;;
    esac
    ;;
  behavior-interest)
    VIEW_TAG="bi"
    DATA_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_scaling_v1"
    TRAIN_FILE="$DATA_ROOT/train/behavior_interest/21600.parquet"
    EVAL_FILE="$DATA_ROOT/eval/behavior_interest/dev_full_grouped.parquet"
    MAPPING_FILE="$DATA_ROOT/interest_token_mapping.parquet"
    EVAL_CONFIG="$AMLT_CODE_DIR/configs/id_token/static_sud_mera_qwen35_behavior_interest_mean_v1.json"
    TRAIN_SHA256="e76acd152ed6e0e92f8b8015dfbc599603267a480d9321b41b5acb95d10487f0"
    EVAL_SHA256="c05b20033ce0d1a0c2d4b7bb97ca0cd6b7cffb7f198a7265cce9e54046b665fd"
    CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_mera_qwen35_behavior_interest_mean_${CONDITION}_21600_v1.yaml"
    ;;
  *)
    echo "Unsupported MeRA ablation input view: $VIEW" >&2
    exit 2
    ;;
esac

EXPECTED_JOB_NAME="qwen35-mera-${VIEW_TAG}-mean-${CONDITION}-${JOB_SUFFIX}"
if [[ "${AMLT_JOB_NAME:?AMLT_JOB_NAME is required}" != "$EXPECTED_JOB_NAME" ]]; then
  echo "Unexpected MeRA ablation job name: $AMLT_JOB_NAME" >&2
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

SFT_DIR="$AMLT_OUTPUT_DIR/sft"
EVAL_ROOT="$AMLT_OUTPUT_DIR/real_eval"
MERGED_DIR="$EVAL_ROOT/merged"
LOCAL_CHECKPOINT_ROOT="/scratch/mera-ablation-${VIEW_TAG}-${CONDITION}-${MODE}-checkpoints"
PERSISTENT_CHECKPOINT_ROOT="$AMLT_OUTPUT_DIR/checkpoints"
SUCCESS_MARKER="$AMLT_OUTPUT_DIR/${MODE}_success.json"

if [[ -f "$SUCCESS_MARKER" ]]; then
  python - "$AMLT_OUTPUT_DIR" "$SUCCESS_MARKER" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, marker_file = Path(sys.argv[1]), Path(sys.argv[2])
marker = json.loads(marker_file.read_text())
manifest_file = root / "artifact_manifest.json"
if hashlib.sha256(manifest_file.read_bytes()).hexdigest() != marker["artifact_manifest_sha256"]:
    raise RuntimeError("Committed MeRA ablation manifest hash mismatch")
manifest = json.loads(manifest_file.read_text())
for relative, expected in manifest["files"].items():
    path = root / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Committed MeRA ablation artifact mismatch: {relative}")
print("MERA_GNIP_ABLATION_REUSE_OK", len(manifest["files"]))
PY
  exit 0
fi

for required_file in "$CONFIG" "$EVAL_CONFIG" "$TRAIN_FILE" "$EVAL_FILE" "$MAPPING_FILE" "$RELATIONS_FILE"; do
  [[ -f "$required_file" ]] || { echo "Missing MeRA ablation input: $required_file" >&2; exit 1; }
done

python - \
  "$CONFIG" "$EVAL_CONFIG" "$TRAIN_FILE" "$TRAIN_SHA256" \
  "$EVAL_FILE" "$EVAL_SHA256" "$MAPPING_FILE" "$MAPPING_SHA256" \
  "$RELATIONS_FILE" "$RELATIONS_SHA256" "$VIEW" "$CONDITION" \
  "$STRUCTURE_WEIGHT" "$COLLABORATIVE_WEIGHT" "$REGULARIZER_ENABLED" \
  "$AMLT_OUTPUT_DIR/input_audit.json" <<'PY'
import hashlib, json, platform, re, sys
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
import torch, transformers, peft, accelerate, yaml

config_file, eval_config_file = Path(sys.argv[1]), Path(sys.argv[2])
train_file, train_sha = Path(sys.argv[3]), sys.argv[4]
eval_file, eval_sha = Path(sys.argv[5]), sys.argv[6]
mapping_file, mapping_sha = Path(sys.argv[7]), sys.argv[8]
relations_file, relations_sha = Path(sys.argv[9]), sys.argv[10]
view, condition = sys.argv[11], sys.argv[12]
structure_weight, collaborative_weight = float(sys.argv[13]), float(sys.argv[14])
regularizer_enabled = sys.argv[15] == "true"
output_file = Path(sys.argv[16])
expected_hashes = {
    train_file: train_sha, eval_file: eval_sha, mapping_file: mapping_sha,
    relations_file: relations_sha,
}
for path, expected in expected_hashes.items():
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Frozen input hash mismatch for {path}: {actual}")
config = yaml.safe_load(config_file.read_text())
expected_common = {
    "model_name": "Qwen/Qwen3.5-9B",
    "semantic_id_token_count": 108,
    "semantic_id_initialization_mode": "mean",
    "gradient_checkpointing_use_reentrant": False,
    "epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "seed": 42,
}
for key, value in expected_common.items():
    if config.get(key) != value:
        raise RuntimeError(f"Config mismatch for {key}: {config.get(key)!r}")
if Path(config["train_file"]).as_posix() not in train_file.as_posix():
    raise RuntimeError("Config Train path does not match selected view")
if "sample_weight_file" in config:
    raise RuntimeError("MeRA ablation unexpectedly uses sample weights")
if regularizer_enabled:
    if config.get("mera_relations_file") is None:
        raise RuntimeError("Regularized condition has no relation file")
    if float(config.get("mera_structure_weight")) != structure_weight:
        raise RuntimeError("Structure weight mismatch")
    if float(config.get("mera_collaborative_weight")) != collaborative_weight:
        raise RuntimeError("Collaborative weight mismatch")
    if config.get("mera_negative_samples") != 10 or config.get("mera_cooccurrence_threshold") != 0.05:
        raise RuntimeError("MeRA N/T mismatch")
else:
    if "mera_relations_file" in config:
        raise RuntimeError("Vanilla condition must not instantiate MeRA regularization")
eval_config = json.loads(eval_config_file.read_text())
if eval_config.get("semantic_id_initialization_mode") != "mean":
    raise RuntimeError("Evaluator does not reconstruct pure-mean SID rows")
mapping = pq.read_table(mapping_file).to_pylist()
token_by_label = {str(row["target_subcategory"]): str(row["interest_token"]) for row in mapping}
if len(token_by_label) != 108 or len(set(token_by_label.values())) != 108:
    raise RuntimeError("Mapping is not one-to-one over 108 SIDs")
relations = json.loads(relations_file.read_text())
if relations["labels"] != list(token_by_label) or relations["interest_tokens"] != list(token_by_label.values()):
    raise RuntimeError("Relation order differs from SID mapping")
train_rows = pq.read_table(train_file).to_pylist()
if len(train_rows) != 21600 or len({str(row["user_id"]) for row in train_rows}) != 21600:
    raise RuntimeError("Train is not 21,600 one-user/one-target rows")
expected_task = "interest_sid" if view == "sid" else "behavior_interest"
if {str(row["task_type"]) for row in train_rows} != {expected_task}:
    raise RuntimeError("Train task type differs from selected view")
counts = Counter(str(row["target_subcategory"]) for row in train_rows)
if len(counts) != 108 or set(counts.values()) != {200}:
    raise RuntimeError("Train is not 108-class balanced")
for row in train_rows:
    label, response = str(row["target_subcategory"]), str(row["response"])
    if response != token_by_label[label] or not re.fullmatch(r"<interest_\d{3}>", response):
        raise RuntimeError("Response is not the mapped SID")
eval_rows = pq.read_table(eval_file, columns=["impression_id", "acceptable_subcategories"]).to_pylist()
if len(eval_rows) != 31392 or len({str(row["impression_id"]) for row in eval_rows}) != 31392:
    raise RuntimeError("Dev is not 31,392 unique impressions")
supported = {str(label) for row in eval_rows for label in row["acceptable_subcategories"]}
if len(supported) != 104:
    raise RuntimeError("Dev supported-interest count mismatch")
versions = {"transformers": transformers.__version__, "peft": peft.__version__, "accelerate": accelerate.__version__}
if versions != {"transformers": "5.14.1", "peft": "0.18.0", "accelerate": "1.12.0"}:
    raise RuntimeError(f"Worker version mismatch: {versions}")
if torch.cuda.device_count() != 4:
    raise RuntimeError("Expected exactly four visible GPUs")
payload = {
    "audit_version": "mera-gnip-ablation-input-v1",
    "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
    "packages": versions, "cuda_devices": torch.cuda.device_count(),
    "view": view, "condition": condition, "train_rows": len(train_rows),
    "eval_rows": len(eval_rows), "category_tokens": 108,
    "semantic_id_initialization_mode": "mean", "structure_weight": structure_weight,
    "collaborative_weight": collaborative_weight, "negative_samples": 10,
    "cooccurrence_threshold": 0.05, "sample_weight_file": None,
    "input_sha256": {path.name: value for path, value in expected_hashes.items()},
}
output_file.write_text(json.dumps(payload, indent=2) + "\n")
print("MERA_GNIP_ABLATION_INPUT_OK", json.dumps(payload, sort_keys=True))
PY

python - "$MODEL_NAME" <<'PY'
import json, sys
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer
name = sys.argv[1]
def validate(value):
    root = Path(value); index = root / "model.safetensors.index.json"
    if not index.is_file(): raise RuntimeError("No model index")
    files = set(json.loads(index.read_text())["weight_map"].values())
    if any(not (root / file).is_file() or (root / file).stat().st_size == 0 for file in files):
        raise RuntimeError("Incomplete model snapshot")
    AutoConfig.from_pretrained(root, local_files_only=True)
    AutoTokenizer.from_pretrained(root, local_files_only=True)
    return root
try:
    snapshot = validate(snapshot_download(name, local_files_only=True)); source = "persistent-cache"
except Exception:
    snapshot = validate(snapshot_download(name)); source = "downloaded-or-repaired"
print("MERA_GNIP_ABLATION_MODEL_READY", snapshot, source)
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ACTIVE_TRAIN_FILE="$TRAIN_FILE"
if [[ "$MODE" == "smoke" ]]; then
  mkdir -p "$AMLT_OUTPUT_DIR/smoke_data"
  ACTIVE_TRAIN_FILE="$AMLT_OUTPUT_DIR/smoke_data/train.parquet"
  python - "$TRAIN_FILE" "$ACTIVE_TRAIN_FILE" <<'PY'
import sys
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq
source, output = Path(sys.argv[1]), Path(sys.argv[2])
selected = {}
for row in pq.read_table(source).to_pylist():
    selected.setdefault(str(row["target_subcategory"]), row)
if len(selected) != 108: raise RuntimeError("Smoke does not cover 108 classes")
rows = [selected[label] for label in sorted(selected)]
pq.write_table(pa.Table.from_pylist(rows, schema=pq.ParquetFile(source).schema_arrow), output, compression="zstd")
print("MERA_GNIP_ABLATION_SMOKE_DATA_OK", len(rows))
PY
fi

TRAIN_ARGS=(
  --config "$CONFIG"
  --train-file "$ACTIVE_TRAIN_FILE"
  --output-dir "$SFT_DIR"
  --checkpoint-dir "$LOCAL_CHECKPOINT_ROOT"
  --run-name "mera-${VIEW_TAG}-mean-${CONDITION}-${MODE}-seed42"
  --epochs "$TRAIN_EPOCHS"
  --per-device-train-batch-size 1
  --gradient-accumulation-steps 8
  --save-steps 100
  --wandb-mode disabled
)
if [[ "$MODE" == "full" ]]; then
  TRAIN_ARGS+=(--persistent-checkpoint-dir "$PERSISTENT_CHECKPOINT_ROOT" --resume-from-checkpoint auto)
fi
python -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" \
  -m new_interest.training.train_sft "${TRAIN_ARGS[@]}"

python - \
  "$SFT_DIR/run_summary.json" "$ACTIVE_TRAIN_FILE" "$EXPECTED_SAMPLES" \
  "$TRAIN_EPOCHS" "$EXPECTED_STEPS" "$REGULARIZER_ENABLED" \
  "$STRUCTURE_WEIGHT" "$COLLABORATIVE_WEIGHT" <<'PY'
import hashlib, json, math, sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text()); train_file = Path(sys.argv[2])
expected_samples, expected_epochs, expected_steps = int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
regularizer_enabled = sys.argv[6] == "true"
structure_weight, collaborative_weight = float(sys.argv[7]), float(sys.argv[8])
if summary["world_size"] != 4 or summary["effective_global_batch_size"] != 32:
    raise RuntimeError("DDP identity mismatch")
if summary["gradient_checkpointing_kwargs"] != {"use_reentrant": False}:
    raise RuntimeError("DDP checkpointing must be non-reentrant")
if summary["train_samples"] != expected_samples or summary["configured_epochs"] != expected_epochs or summary["expected_optimizer_steps"] != expected_steps:
    raise RuntimeError("Train scope/step mismatch")
if summary["train_file_sha256"] != hashlib.sha256(train_file.read_bytes()).hexdigest():
    raise RuntimeError("Trainer consumed wrong Train file")
if summary["sample_weight_file"] is not None:
    raise RuntimeError("Ablation unexpectedly consumed sample weights")
if summary["semantic_token_initialization"]["initialization_mode"] != "mean":
    raise RuntimeError("Training did not use pure-mean SID initialization")
if not math.isfinite(float(summary["train_metrics"]["train_loss"])):
    raise RuntimeError("Non-finite training loss")
semantic = summary["model_audit"]["semantic_id_tokens"]
if semantic["count"] != 108 or semantic["save_embedding_layers"] is not False:
    raise RuntimeError("Semantic-ID compact contract mismatch")
if not summary["semantic_token_drift"]["all_input_rows_changed"] or not summary["semantic_token_drift"]["all_output_rows_changed"]:
    raise RuntimeError("Not all SID rows changed")
if regularizer_enabled:
    audit = summary["mera_regularization_audit"]
    training = summary["mera_regularization_training"]
    if audit["structure_weight"] != structure_weight or audit["collaborative_weight"] != collaborative_weight:
        raise RuntimeError("MeRA regularizer weight mismatch")
    if audit["negative_samples"] != 10 or audit["cooccurrence_threshold"] != 0.05:
        raise RuntimeError("MeRA regularizer N/T mismatch")
    if training["calls"] <= 0:
        raise RuntimeError("MeRA regularizer was not called")
    if structure_weight == 0 and training["mean_structure_loss"] != 0:
        raise RuntimeError("Structure-only ablation boundary drifted")
    if collaborative_weight == 0 and training["mean_collaborative_loss"] != 0:
        raise RuntimeError("Collaborative-only ablation boundary drifted")
else:
    if summary["mera_regularization_audit"] is not None or summary["mera_regularization_training"] is not None:
        raise RuntimeError("Vanilla condition instantiated MeRA regularization")
if expected_samples == 108:
    losses = [float(row["loss"]) for row in summary["loss_history"]]
    if len(losses) < 4 or sum(losses[-2:]) / 2 >= sum(losses[:2]) / 2:
        raise RuntimeError(f"DDP smoke loss did not decrease: {losses}")
print("MERA_GNIP_ABLATION_TRAIN_OK", expected_samples, expected_steps)
PY

if [[ "$MODE" == "full" ]]; then
  python - "$SFT_DIR/run_summary.json" "$PERSISTENT_CHECKPOINT_ROOT" "$EXPECTED_STEPS" <<'PY'
import json, sys
from pathlib import Path
from new_interest.training.checkpoint_persistence import select_latest_complete_checkpoint
summary = json.loads(Path(sys.argv[1]).read_text())
selected, diagnostics = select_latest_complete_checkpoint(Path(sys.argv[2]), summary["checkpoint_identity"])
if diagnostics: raise RuntimeError(f"Persistent checkpoint diagnostics: {diagnostics}")
if selected is None or selected.name != f"checkpoint-{int(sys.argv[3])}":
    raise RuntimeError(f"Final persistent checkpoint mismatch: {selected}")
print("MERA_GNIP_ABLATION_CHECKPOINT_OK", selected)
PY
fi

CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.verify_semantic_id_token_adapter \
  --model-name "$MODEL_NAME" --adapter-path "$SFT_DIR/final_adapter" \
  --mapping-file "$MAPPING_FILE" --reference-file "$SFT_DIR/semantic_reload_reference.json" \
  --output "$SFT_DIR/semantic_adapter_reload_audit.json"
python - "$SFT_DIR/semantic_adapter_reload_audit.json" <<'PY'
import json, sys
from pathlib import Path
audit = json.loads(Path(sys.argv[1]).read_text())
integrity = audit["adapter_state_integrity"]
if not audit["all_close"] or audit["sample_count"] != 4:
    raise RuntimeError("Adapter reload logits audit failed")
if integrity["tensor_count"] != 498 or not integrity["all_tensors_exact"]:
    raise RuntimeError("Exact compact adapter-state audit failed")
print("MERA_GNIP_ABLATION_RELOAD_OK", integrity["tensor_count"])
PY

mkdir -p "$EVAL_ROOT"
if [[ "$MODE" == "smoke" ]]; then
  CUDA_VISIBLE_DEVICES=0 python -m new_interest.evaluation.evaluate_id_token_ranking \
    --config "$EVAL_CONFIG" --adapter-path "$SFT_DIR/final_adapter" \
    --eval-file "$EVAL_FILE" --interest-token-mapping-file "$MAPPING_FILE" \
    --output-dir "$MERGED_DIR" --run-name "mera-${VIEW_TAG}-mean-${CONDITION}-smoke" \
    --max-examples 64 --num-shards 1 --shard-index 0 --batch-size 8
else
  PIDS=(); SHARD_DIRS=()
  for SHARD_INDEX in 0 1 2 3; do
    SHARD_NAME="$(printf 'shard-%02d' "$SHARD_INDEX")"; SHARD_DIR="$EVAL_ROOT/$SHARD_NAME"
    SHARD_DIRS+=("$SHARD_DIR")
    (CUDA_VISIBLE_DEVICES="$SHARD_INDEX" python -m new_interest.evaluation.evaluate_id_token_ranking \
      --config "$EVAL_CONFIG" --adapter-path "$SFT_DIR/final_adapter" \
      --eval-file "$EVAL_FILE" --interest-token-mapping-file "$MAPPING_FILE" \
      --output-dir "$SHARD_DIR" --run-name "mera-${VIEW_TAG}-mean-${CONDITION}-${SHARD_NAME}" \
      --num-shards 4 --shard-index "$SHARD_INDEX" --batch-size 8 >"$EVAL_ROOT/${SHARD_NAME}.log" 2>&1) &
    PIDS+=("$!")
  done
  FAILURE=0; for PID in "${PIDS[@]}"; do wait "$PID" || FAILURE=1; done
  [[ "$FAILURE" == "0" ]] || { tail -n 80 "$EVAL_ROOT"/shard-*.log >&2; exit 1; }
  python -m new_interest.evaluation.merge_id_token_shards \
    --shard-dirs "${SHARD_DIRS[@]}" --eval-file "$EVAL_FILE" \
    --interest-token-mapping-file "$MAPPING_FILE" --output-dir "$MERGED_DIR" \
    --expected-records 31392
fi

python - "$MERGED_DIR" "$EXPECTED_EVAL" "$MAPPING_FILE" <<'PY'
import json, math, sys
from pathlib import Path
import pyarrow.parquet as pq
root, expected, mapping_file = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
metrics = json.loads((root / "metrics.json").read_text())
if metrics["samples"] != expected or metrics["label_vocabulary_size"] != 108 or metrics["top_k"] != 20:
    raise RuntimeError("Evaluation identity mismatch")
if expected == 31392 and metrics["supported_interest_count"] != 104:
    raise RuntimeError("Full supported-interest count mismatch")
for cutoff in (1, 5, 10, 20):
    for prefix in ("recall", "ndcg", "interest_macro_recall", "interest_macro_ndcg"):
        if not math.isfinite(float(metrics[f"{prefix}_at_{cutoff}"])):
            raise RuntimeError("Non-finite evaluation metric")
labels = {str(row["target_subcategory"]) for row in pq.read_table(mapping_file).to_pylist()}
rows = pq.read_table(root / "id_token_predictions.parquet").to_pylist()
if len(rows) != expected or len({int(row["record_index"]) for row in rows}) != expected:
    raise RuntimeError("Prediction row/index mismatch")
for row in rows:
    ranked = list(map(str, row["ranked_labels"])); scores = list(map(float, row["ranked_scores"]))
    if len(ranked) != 20 or len(set(ranked)) != 20 or not set(ranked).issubset(labels):
        raise RuntimeError("Illegal Top-20 labels")
    if len(scores) != 20 or not all(math.isfinite(value) for value in scores):
        raise RuntimeError("Illegal Top-20 scores")
print("MERA_GNIP_ABLATION_EVAL_OK", expected, metrics["recall_at_10"], metrics["interest_macro_recall_at_10"])
PY

python - "$AMLT_OUTPUT_DIR" "$MODE" "$VIEW" "$CONDITION" "$SUCCESS_MARKER" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, mode, view, condition, marker_file = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], Path(sys.argv[5])
files = {}
include_roots = [root / "input_audit.json", root / "sft", root / "real_eval/merged"]
if mode == "smoke": include_roots.append(root / "smoke_data")
for value in include_roots:
    paths = [value] if value.is_file() else sorted(path for path in value.rglob("*") if path.is_file())
    for path in paths:
        if "checkpoints" in path.parts: continue
        files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {
    "manifest_version": "mera-gnip-ablation-v1", "view": view, "condition": condition,
    "files": dict(sorted(files.items())),
}
manifest_file = root / "artifact_manifest.json"
manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
marker = {
    "status": "validated", "mode": mode, "view": view, "condition": condition,
    "artifact_manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
    "artifact_count": len(files),
}
marker_file.write_text(json.dumps(marker, indent=2) + "\n")
print("MERA_GNIP_ABLATION_GATE_OK", mode, view, condition, len(files))
PY
