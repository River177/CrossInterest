#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: run_mera_gnip_distributed.sh <smoke|full>}"
NUM_GPUS="${MERA_GNIP_NUM_GPUS:-4}"
if [[ "$NUM_GPUS" != "4" ]]; then
  echo "MeRA distributed run requires four GPUs; got $NUM_GPUS" >&2
  exit 2
fi
case "$MODE" in
  smoke)
    EXPECTED_JOB_NAME="qwen35-mera-faithful-t005-ddp-eval-smoke"
    TRAIN_EPOCHS=2
    EXPECTED_SAMPLES=108
    EXPECTED_STEPS=8
    EXPECTED_EVAL=64
    ;;
  full)
    EXPECTED_JOB_NAME="qwen35-mera-faithful-t005-full"
    TRAIN_EPOCHS=3
    EXPECTED_SAMPLES=21600
    EXPECTED_STEPS=2025
    EXPECTED_EVAL=31392
    ;;
  *)
    echo "Unsupported MeRA mode: $MODE" >&2
    exit 2
    ;;
esac
if [[ "${AMLT_JOB_NAME:?AMLT_JOB_NAME is required}" != "$EXPECTED_JOB_NAME" ]]; then
  echo "Unexpected MeRA job name: $AMLT_JOB_NAME" >&2
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
EVAL_CONFIG="$AMLT_CODE_DIR/configs/id_token/static_sud_mera_faithful_qwen35_interest_sid_t005_v1.json"
DATA_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_mera_semantic_id_108_200_h50_v1"
TRAIN_FILE="$DATA_ROOT/train/interest_sid/21600.parquet"
EVAL_FILE="$DATA_ROOT/eval/interest_sid/dev_full_grouped.parquet"
MAPPING_FILE="$DATA_ROOT/interest_token_mapping.parquet"
RELATIONS_FILE="$AMLT_CODE_DIR/data/training/mera_static_user_disjoint_behavior_interest_200_h50_v1/relations.json"
SFT_DIR="$AMLT_OUTPUT_DIR/sft"
EVAL_ROOT="$AMLT_OUTPUT_DIR/real_eval"
MERGED_DIR="$EVAL_ROOT/merged"
LOCAL_CHECKPOINT_ROOT="/scratch/mera-gnip-${MODE}-checkpoints"
PERSISTENT_CHECKPOINT_ROOT="$AMLT_OUTPUT_DIR/checkpoints"
SUCCESS_MARKER="$AMLT_OUTPUT_DIR/${MODE}_success.json"

TRAIN_SHA256="95ed024252b5fffa2f166c2d0f4d4114968f7b21e0685043c00951e07bccab20"
EVAL_SHA256="afdd1997b6139583e4b0e44b50bba963996bea9cf9325e8f04d77fa8ed1b2bbe"
MAPPING_SHA256="45397451ac2ae2e64d8156b2b47be3f454cc309a8d30a83bc23c2d3267653a0a"
RELATIONS_SHA256="e81323ddeeff065469a0bb91cbd694b4feb92976bccc8d549ee012a874e70813"

if [[ -f "$SUCCESS_MARKER" ]]; then
  python - "$AMLT_OUTPUT_DIR" "$SUCCESS_MARKER" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, marker_file = Path(sys.argv[1]), Path(sys.argv[2])
marker = json.loads(marker_file.read_text())
manifest_file = root / "artifact_manifest.json"
if hashlib.sha256(manifest_file.read_bytes()).hexdigest() != marker["artifact_manifest_sha256"]:
    raise RuntimeError("Committed MeRA manifest hash mismatch")
manifest = json.loads(manifest_file.read_text())
for relative, expected in manifest["files"].items():
    path = root / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Committed MeRA artifact mismatch: {relative}")
print("MERA_GNIP_DISTRIBUTED_REUSE_OK", len(manifest["files"]))
PY
  exit 0
fi

for path in "$CONFIG" "$EVAL_CONFIG" "$TRAIN_FILE" "$EVAL_FILE" "$MAPPING_FILE" "$RELATIONS_FILE"; do
  [[ -f "$path" ]] || { echo "Missing MeRA input: $path" >&2; exit 1; }
done

python - "$CONFIG" "$EVAL_CONFIG" "$TRAIN_FILE" "$EVAL_FILE" "$MAPPING_FILE" "$RELATIONS_FILE" "$AMLT_OUTPUT_DIR/input_audit.json" <<'PY'
import hashlib, json, platform, sys
from pathlib import Path
import pyarrow.parquet as pq
import torch, transformers, peft, accelerate, yaml

config_file, eval_config_file = Path(sys.argv[1]), Path(sys.argv[2])
train_file, eval_file, mapping_file, relations_file = map(Path, sys.argv[3:7])
output_file = Path(sys.argv[7])
expected_hashes = {
    train_file: "95ed024252b5fffa2f166c2d0f4d4114968f7b21e0685043c00951e07bccab20",
    eval_file: "afdd1997b6139583e4b0e44b50bba963996bea9cf9325e8f04d77fa8ed1b2bbe",
    mapping_file: "45397451ac2ae2e64d8156b2b47be3f454cc309a8d30a83bc23c2d3267653a0a",
    relations_file: "e81323ddeeff065469a0bb91cbd694b4feb92976bccc8d549ee012a874e70813",
}
for path, expected in expected_hashes.items():
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Frozen input hash mismatch for {path}: {actual}")
config = yaml.safe_load(config_file.read_text())
eval_config = json.loads(eval_config_file.read_text())
expected = {
    "model_name": "Qwen/Qwen3.5-9B",
    "semantic_id_token_count": 108,
    "semantic_id_initialization_mode": "mean",
    "mera_structure_weight": 0.05,
    "mera_collaborative_weight": 0.1,
    "mera_negative_samples": 10,
    "mera_cooccurrence_threshold": 0.05,
    "gradient_checkpointing_use_reentrant": False,
}
for key, value in expected.items():
    if config.get(key) != value:
        raise RuntimeError(f"Config mismatch for {key}")
if "sample_weight_file" in config:
    raise RuntimeError("MeRA unexpectedly uses sample weights")
if eval_config.get("semantic_id_initialization_mode") != "mean":
    raise RuntimeError("Evaluator does not reconstruct MeRA mean initialization")
if pq.ParquetFile(train_file).metadata.num_rows != 21600:
    raise RuntimeError("Train row count mismatch")
if pq.ParquetFile(eval_file).metadata.num_rows != 31392:
    raise RuntimeError("Dev row count mismatch")
mapping = pq.read_table(mapping_file).to_pylist()
relations = json.loads(relations_file.read_text())
if len(mapping) != 108 or relations["labels"] != [str(row["target_subcategory"]) for row in mapping]:
    raise RuntimeError("108-class mapping/relation mismatch")
versions = {
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "accelerate": accelerate.__version__,
}
if versions != {"transformers": "5.14.1", "peft": "0.18.0", "accelerate": "1.12.0"}:
    raise RuntimeError(f"Worker version mismatch: {versions}")
if torch.cuda.device_count() != 4:
    raise RuntimeError("Expected exactly four visible GPUs")
payload = {
    "audit_version": "mera-gnip-distributed-input-v1",
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "packages": versions,
    "cuda_devices": torch.cuda.device_count(),
    "train_rows": 21600,
    "eval_rows": 31392,
    "category_tokens": 108,
    "input_sha256": {path.name: value for path, value in expected_hashes.items()},
}
output_file.write_text(json.dumps(payload, indent=2) + "\n")
print("MERA_GNIP_DISTRIBUTED_INPUT_OK", json.dumps(payload, sort_keys=True))
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
print("MERA_GNIP_DISTRIBUTED_MODEL_READY", snapshot, source)
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
print("MERA_GNIP_DDP_SMOKE_DATA_OK", len(rows))
PY
fi

TRAIN_ARGS=(
  --config "$CONFIG"
  --train-file "$ACTIVE_TRAIN_FILE"
  --output-dir "$SFT_DIR"
  --checkpoint-dir "$LOCAL_CHECKPOINT_ROOT"
  --run-name "static-sud-mera-faithful-qwen35-${MODE}-seed42"
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

python - "$SFT_DIR/run_summary.json" "$ACTIVE_TRAIN_FILE" "$EXPECTED_SAMPLES" "$TRAIN_EPOCHS" "$EXPECTED_STEPS" <<'PY'
import hashlib, json, math, sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text()); train_file = Path(sys.argv[2])
expected_samples, expected_epochs, expected_steps = int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
if summary["world_size"] != 4 or summary["effective_global_batch_size"] != 32:
    raise RuntimeError("DDP identity mismatch")
if summary["gradient_checkpointing_kwargs"] != {"use_reentrant": False}:
    raise RuntimeError("DDP checkpointing must be non-reentrant")
if summary["train_samples"] != expected_samples or summary["configured_epochs"] != expected_epochs or summary["expected_optimizer_steps"] != expected_steps:
    raise RuntimeError("Train scope/step mismatch")
if summary["train_file_sha256"] != hashlib.sha256(train_file.read_bytes()).hexdigest():
    raise RuntimeError("Trainer consumed wrong Train file")
if not math.isfinite(float(summary["train_metrics"]["train_loss"])):
    raise RuntimeError("Non-finite training loss")
semantic = summary["model_audit"]["semantic_id_tokens"]
if semantic["count"] != 108 or semantic["save_embedding_layers"] is not False:
    raise RuntimeError("Semantic-ID compact contract mismatch")
mera = summary["mera_regularization_audit"]
if mera["interest_count"] != 108 or mera["cooccurrence_threshold"] != 0.05 or mera["collaborative_eligible_anchors"] != 4:
    raise RuntimeError("MeRA relation identity mismatch")
if summary["mera_regularization_training"]["calls"] <= 0:
    raise RuntimeError("MeRA regularizer was not called")
if not summary["semantic_token_drift"]["all_input_rows_changed"] or not summary["semantic_token_drift"]["all_output_rows_changed"]:
    raise RuntimeError("Not all Semantic-ID rows changed")
if expected_samples == 108:
    losses = [float(row["loss"]) for row in summary["loss_history"]]
    if len(losses) < 4 or sum(losses[-2:]) / 2 >= sum(losses[:2]) / 2:
        raise RuntimeError(f"DDP smoke loss did not decrease: {losses}")
print("MERA_GNIP_DISTRIBUTED_TRAIN_OK", expected_samples, expected_steps)
PY

CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.verify_semantic_id_token_adapter \
  --model-name "$MODEL_NAME" --adapter-path "$SFT_DIR/final_adapter" \
  --mapping-file "$MAPPING_FILE" --reference-file "$SFT_DIR/semantic_reload_reference.json" \
  --output "$SFT_DIR/semantic_adapter_reload_audit.json"

mkdir -p "$EVAL_ROOT"
if [[ "$MODE" == "smoke" ]]; then
  CUDA_VISIBLE_DEVICES=0 python -m new_interest.evaluation.evaluate_id_token_ranking \
    --config "$EVAL_CONFIG" --adapter-path "$SFT_DIR/final_adapter" \
    --eval-file "$EVAL_FILE" --interest-token-mapping-file "$MAPPING_FILE" \
    --output-dir "$MERGED_DIR" --run-name "mera-faithful-ddp-eval-smoke" \
    --max-examples 64 --num-shards 1 --shard-index 0 --batch-size 8
else
  PIDS=(); SHARD_DIRS=()
  for SHARD_INDEX in 0 1 2 3; do
    SHARD_NAME="$(printf 'shard-%02d' "$SHARD_INDEX")"; SHARD_DIR="$EVAL_ROOT/$SHARD_NAME"
    SHARD_DIRS+=("$SHARD_DIR")
    (CUDA_VISIBLE_DEVICES="$SHARD_INDEX" python -m new_interest.evaluation.evaluate_id_token_ranking \
      --config "$EVAL_CONFIG" --adapter-path "$SFT_DIR/final_adapter" \
      --eval-file "$EVAL_FILE" --interest-token-mapping-file "$MAPPING_FILE" \
      --output-dir "$SHARD_DIR" --run-name "mera-faithful-full-${SHARD_NAME}" \
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

python - "$MERGED_DIR" "$EXPECTED_EVAL" <<'PY'
import json, math, sys
from pathlib import Path
import pyarrow.parquet as pq
root, expected = Path(sys.argv[1]), int(sys.argv[2])
metrics = json.loads((root / "metrics.json").read_text())
if metrics["samples"] != expected or metrics["label_vocabulary_size"] != 108 or metrics["top_k"] != 20:
    raise RuntimeError("Evaluation identity mismatch")
if expected == 31392 and metrics["supported_interest_count"] != 104:
    raise RuntimeError("Full supported-interest count mismatch")
for cutoff in (1, 5, 10, 20):
    for prefix in ("recall", "ndcg", "interest_macro_recall", "interest_macro_ndcg"):
        if not math.isfinite(float(metrics[f"{prefix}_at_{cutoff}"])):
            raise RuntimeError("Non-finite evaluation metric")
rows = pq.read_table(root / "id_token_predictions.parquet").to_pylist()
if len(rows) != expected or len({int(row["record_index"]) for row in rows}) != expected:
    raise RuntimeError("Prediction row/index mismatch")
print("MERA_GNIP_DISTRIBUTED_EVAL_OK", expected, metrics["recall_at_10"], metrics["interest_macro_recall_at_10"])
PY

python - "$AMLT_OUTPUT_DIR" "$MODE" "$SUCCESS_MARKER" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, mode, marker_file = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
files = {}
include_roots = [root / "input_audit.json", root / "sft", root / "real_eval/merged"]
if mode == "smoke": include_roots.append(root / "smoke_data")
for value in include_roots:
    paths = [value] if value.is_file() else sorted(path for path in value.rglob("*") if path.is_file())
    for path in paths:
        if "checkpoints" in path.parts: continue
        files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {"manifest_version": f"mera-gnip-{mode}-v1", "files": dict(sorted(files.items()))}
manifest_file = root / "artifact_manifest.json"
manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
marker = {
    "status": "validated", "mode": mode,
    "artifact_manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
    "artifact_count": len(files),
}
marker_file.write_text(json.dumps(marker, indent=2) + "\n")
print("MERA_GNIP_DISTRIBUTED_GATE_OK", mode, len(files))
PY
