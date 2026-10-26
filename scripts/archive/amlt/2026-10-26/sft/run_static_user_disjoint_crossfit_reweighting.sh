#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: run_static_user_disjoint_crossfit_reweighting.sh <smoke|oof|final> [qwen35|llama31]}"
MODEL_FAMILY="${2:-qwen35}"
case "$MODE" in
  smoke|oof|final) ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    exit 2
    ;;
esac

case "$MODEL_FAMILY" in
  qwen35)
    MODEL_NAME="Qwen/Qwen3.5-9B"
    MODEL_SLUG="qwen35"
    ;;
  llama31)
    MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
    MODEL_SLUG="llama31"
    ;;
  *)
    echo "Unsupported model family: $MODEL_FAMILY" >&2
    exit 2
    ;;
esac

NUM_GPUS="${STATIC_SUD_CROSSFIT_NUM_GPUS:-4}"
case "$NUM_GPUS" in
  4|8) ;;
  *)
    echo "Cross-fit training requires 4 or 8 GPUs; got $NUM_GPUS" >&2
    exit 2
    ;;
esac
GRADIENT_ACCUMULATION_STEPS=$((32 / NUM_GPUS))
CHECKPOINT_SAVE_STEPS="${CROSSFIT_CHECKPOINT_SAVE_STEPS:-100}"
if [[ ! "$CHECKPOINT_SAVE_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "CROSSFIT_CHECKPOINT_SAVE_STEPS must be a positive integer" >&2
  exit 2
fi

case "$AMLT_JOB_NAME" in
  *behavior*)
    VIEW="behavior"
    TASK_TYPE="behavior_only"
    ;;
  *interest*)
    VIEW="interest"
    TASK_TYPE="interest_only"
    ;;
  *)
    echo "Cannot infer input view from AMLT_JOB_NAME=$AMLT_JOB_NAME" >&2
    exit 2
    ;;
esac

case "$AMLT_JOB_NAME" in
  *weighted*) CONDITION="weighted" ;;
  *) CONDITION="vanilla" ;;
esac

FOLD=""
for CANDIDATE in 0 1 2 3 4; do
  if [[ "$AMLT_JOB_NAME" == *"fold-$CANDIDATE"* ]]; then
    FOLD="$CANDIDATE"
  fi
done
if [[ "$MODE" == "oof" && -z "$FOLD" ]]; then
  echo "OOF job name must contain fold-0 through fold-4" >&2
  exit 2
fi
if [[ "$MODE" == "smoke" ]]; then
  FOLD=0
  CONDITION="weighted"
fi

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export WANDB_DISABLED=true
export HF_HOME="${HF_HOME:-/mnt/default/huggingface-cache}"
mkdir -p "$HF_HOME"

if [[ "$MODEL_FAMILY" == "qwen35" ]]; then
  python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download("Qwen/Qwen3.5-9B")
print("QWEN35_SNAPSHOT_READY", path)
PY
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
else
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  python - <<'PY'
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

name = "meta-llama/Llama-3.1-8B-Instruct"
snapshot = snapshot_download(name, local_files_only=True)
config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
print(
    "LLAMA31_SHARED_CACHE_READY",
    snapshot,
    type(config).__name__,
    type(tokenizer).__name__,
)
PY
fi

DATA_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_crossfit_reweighting_qwen35_v1"
WEIGHT_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_crossfit_reweighting_${MODEL_SLUG}_v1"
SOURCE_SFT_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_balanced_200_h50_v1"
LABELS_FILE="$AMLT_CODE_DIR/data/training/generative_new_interest_static_user_disjoint_balanced_200_h50_v1/retained_interests.parquet"
FOLD_MANIFEST="$DATA_ROOT/fold_manifest.parquet"
CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_crossfit_${MODEL_SLUG}_${VIEW}_${CONDITION}.yaml"
BEAM_CONFIG="$AMLT_CODE_DIR/configs/beam/static_sud_${MODEL_SLUG}_${VIEW}_full.json"
SFT_DIR="$AMLT_OUTPUT_DIR/sft"
LOCAL_CHECKPOINT_ROOT="/scratch/static-sud-crossfit-${MODEL_SLUG}-${VIEW}-${CONDITION}-${MODE}${FOLD:+-fold${FOLD}}-checkpoints"
PERSISTENT_CHECKPOINT_ROOT="$SFT_DIR/checkpoints"
mkdir -p "$AMLT_OUTPUT_DIR"

python - "$AMLT_OUTPUT_DIR/environment.json" <<'PY'
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

packages = (
    "accelerate",
    "peft",
    "pyarrow",
    "torch",
    "transformers",
    "wandb",
)
payload = {
    "python": sys.version,
    "platform": platform.platform(),
    "packages": {
        package: importlib.metadata.version(package) for package in packages
    },
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
PY

write_artifact_manifest() {
  python - "$AMLT_OUTPUT_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest_path = root / "artifact_manifest.json"
files = sorted(
    path for path in root.rglob("*")
    if path.is_file() and path != manifest_path
)
hashes = {}
for path in files:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    hashes[path.relative_to(root).as_posix()] = digest.hexdigest()
manifest_path.write_text(
    json.dumps({"files": hashes}, indent=2) + "\n", encoding="utf-8"
)
print("ARTIFACT_MANIFEST_FILES", len(hashes))
PY
}

if [[ "$MODE" == "final" ]]; then
  TRAIN_FILE="$SOURCE_SFT_ROOT/train/$TASK_TYPE/21600.parquet"
else
  TRAIN_FILE="$DATA_ROOT/teacher/$TASK_TYPE/fold_${FOLD}/train.parquet"
  HELDOUT_FILE="$DATA_ROOT/teacher/$TASK_TYPE/fold_${FOLD}/heldout.parquet"
fi

for path in "$CONFIG" "$TRAIN_FILE" "$FOLD_MANIFEST"; do
  if [[ ! -f "$path" ]]; then
    echo "Required input is missing: $path" >&2
    exit 1
  fi
done

TRAIN_ARGS=(
  --config "$CONFIG"
  --train-file "$TRAIN_FILE"
  --output-dir "$SFT_DIR"
  --run-name "static-sud-crossfit-${MODEL_SLUG}-${VIEW}-${CONDITION}-${MODE}${FOLD:+-fold${FOLD}}"
  --per-device-train-batch-size 1
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --checkpoint-dir "$LOCAL_CHECKPOINT_ROOT"
  --persistent-checkpoint-dir "$PERSISTENT_CHECKPOINT_ROOT"
  --resume-from-checkpoint auto
  --save-steps "$CHECKPOINT_SAVE_STEPS"
  --wandb-mode disabled
)

if [[ "$MODE" == "final" && "$CONDITION" == "weighted" ]]; then
  MODEL_WEIGHT_FILE="$WEIGHT_ROOT/sample_weights/${VIEW}.parquet"
  if [[ ! -f "$MODEL_WEIGHT_FILE" ]]; then
    echo "Required model-specific sample weights are missing: $MODEL_WEIGHT_FILE" >&2
    exit 1
  fi
  TRAIN_ARGS+=(--sample-weight-file "$MODEL_WEIGHT_FILE")
fi

if [[ "$MODE" == "smoke" ]]; then
  SMOKE_WEIGHTS="/scratch/static-sud-crossfit-${VIEW}-smoke-weights.parquet"
  python - "$TRAIN_FILE" "$SMOKE_WEIGHTS" <<'PY'
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

source = Path(sys.argv[1])
output = Path(sys.argv[2])
ids = pq.read_table(source, columns=["source_sample_id"])[
    "source_sample_id"
].to_pylist()
pq.write_table(
    pa.Table.from_pylist(
        [
            {"source_sample_id": str(source_id), "sample_weight": 1.0}
            for source_id in ids
        ]
    ),
    output,
)
print("SMOKE_WEIGHT_ROWS", len(ids))
PY
  TRAIN_ARGS+=(
    --sample-weight-file "$SMOKE_WEIGHTS"
    --max-samples 32
    --epochs 2
    --save-steps 1
  )
else
  TRAIN_ARGS+=(--epochs 3)
fi

run_training() {
  python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    -m new_interest.training.train_sft \
    "${TRAIN_ARGS[@]}"
}

run_training
if [[ "$MODE" == "smoke" ]]; then
  # A second fresh torchrun must hydrate checkpoint-2 from persistent storage.
  run_training
fi

python - "$SFT_DIR/run_summary.json" "$MODE" "$NUM_GPUS" <<'PY'
import json
import math
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
mode = sys.argv[2]
world_size = int(sys.argv[3])
if int(summary["world_size"]) != world_size:
    raise RuntimeError("Unexpected DDP world size")
if int(summary["effective_global_batch_size"]) != 32:
    raise RuntimeError("Effective global batch must equal 32")
if int(summary["token_audit"]["over_max_seq_length"]) != 0:
    raise RuntimeError("Token audit found an over-length sample")
if summary["model_audit"]["forbidden_trainable_parameters"]:
    raise RuntimeError("Visual parameters became trainable")
if not math.isfinite(float(summary["train_metrics"]["train_loss"])):
    raise RuntimeError("Training loss is not finite")
expected_rows = 32 if mode == "smoke" else 21600 if mode == "final" else 17280
expected_steps = 2 if mode == "smoke" else 2025 if mode == "final" else 1620
if int(summary["train_samples"]) != expected_rows:
    raise RuntimeError(
        f"Expected {expected_rows} Train rows, got {summary['train_samples']}"
    )
if int(summary["expected_optimizer_steps"]) != expected_steps:
    raise RuntimeError(
        f"Expected {expected_steps} steps, got {summary['expected_optimizer_steps']}"
    )
print("CROSSFIT_TRAIN_AUDIT_OK", mode, expected_rows, expected_steps)
PY

if [[ "$MODE" == "smoke" ]]; then
  python - "$SFT_DIR/run_summary.json" "$PERSISTENT_CHECKPOINT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

from new_interest.training.checkpoint_persistence import (
    select_latest_complete_checkpoint,
)

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
selected, diagnostics = select_latest_complete_checkpoint(
    root,
    summary["checkpoint_identity"],
)
if diagnostics:
    raise RuntimeError(f"Checkpoint smoke diagnostics: {diagnostics}")
if selected is None or selected.name != "checkpoint-2":
    raise RuntimeError(f"Expected complete checkpoint-2, got {selected}")
resumed = summary.get("resumed_from_checkpoint")
if resumed is None or Path(resumed).name != "checkpoint-2":
    raise RuntimeError(f"Fresh-process auto-resume failed: {resumed}")
print("CROSSFIT_CHECKPOINT_RECOVERY_SMOKE_OK", selected, resumed)
PY
fi

if [[ "$MODE" == "smoke" || "$MODE" == "oof" ]]; then
  SCORE_ARGS=(
    --model-name "$MODEL_NAME"
    --adapter-path "$SFT_DIR/final_adapter"
    --heldout-file "$HELDOUT_FILE"
    --fold-manifest "$FOLD_MANIFEST"
    --output-dir "$AMLT_OUTPUT_DIR/oof"
    --expected-fold "$FOLD"
  )
  if [[ "$MODE" == "smoke" ]]; then
    SCORE_ARGS+=(--max-samples 32)
  fi
  python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    -m new_interest.experiments.score_crossfit_sft \
    "${SCORE_ARGS[@]}"
  python - "$AMLT_OUTPUT_DIR/oof/score_summary.json" "$MODE" <<'PY'
import json
import math
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = 32 if sys.argv[2] == "smoke" else 4320
if int(summary["rows"]) != expected:
    raise RuntimeError(f"Expected {expected} OOF rows, got {summary['rows']}")
if not math.isfinite(float(summary["category_token_nll_mean"])):
    raise RuntimeError("OOF mean NLL is not finite")
if int(summary["category_token_count_min"]) <= 0:
    raise RuntimeError("OOF scorer did not score category-name tokens")
print("CROSSFIT_OOF_AUDIT_OK", expected)
PY
  write_artifact_manifest
  exit 0
fi

if [[ ! -f "$BEAM_CONFIG" || ! -f "$LABELS_FILE" ]]; then
  echo "Beam config or label vocabulary is missing" >&2
  exit 1
fi

EVAL_NAMES_TEXT="${CROSSFIT_EVAL_NAMES:-real shuffled_20260818 shuffled_20260819 shuffled_20260820}"
read -r -a EVAL_NAMES <<< "$EVAL_NAMES_TEXT"
if [[ "$MODEL_FAMILY" == "llama31" && "${EVAL_NAMES[*]}" != "real" ]]; then
  echo "Llama cross-fit final evaluation is real-only" >&2
  exit 2
fi
for EVAL_NAME in "${EVAL_NAMES[@]}"; do
  case "$EVAL_NAME" in
    real|shuffled_20260818|shuffled_20260819|shuffled_20260820) ;;
    *)
      echo "Unsupported evaluation name: $EVAL_NAME" >&2
      exit 2
      ;;
  esac
  EVAL_FILE="$DATA_ROOT/eval/$TASK_TYPE/$EVAL_NAME.parquet"
  BEAM_ROOT="$AMLT_OUTPUT_DIR/eval/$EVAL_NAME"
  mkdir -p "$BEAM_ROOT"
  PIDS=()
  SHARD_DIRS=()
  for SHARD_INDEX in $(seq 0 $((NUM_GPUS - 1))); do
    SHARD_NAME="$(printf 'shard-%02d' "$SHARD_INDEX")"
    SHARD_DIR="$BEAM_ROOT/$SHARD_NAME"
    SHARD_DIRS+=("$SHARD_DIR")
    (
      export CUDA_VISIBLE_DEVICES="$SHARD_INDEX"
      python -m new_interest.evaluation.evaluate_beam \
        --config "$BEAM_CONFIG" \
        --adapter-path "$SFT_DIR/final_adapter" \
        --eval-file "$EVAL_FILE" \
        --output-dir "$SHARD_DIR" \
        --run-name "static-sud-crossfit-${VIEW}-${CONDITION}-${EVAL_NAME}-${SHARD_NAME}" \
        --num-shards "$NUM_GPUS" \
        --shard-index "$SHARD_INDEX" \
        --wandb-mode disabled \
        >"$BEAM_ROOT/${SHARD_NAME}.log" 2>&1
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
    echo "At least one Beam shard failed for $EVAL_NAME" >&2
    exit 1
  fi
  python -m new_interest.evaluation.merge_beam_shards \
    --shard-dirs "${SHARD_DIRS[@]}" \
    --eval-file "$EVAL_FILE" \
    --retained-interests-file "$LABELS_FILE" \
    --output-dir "$BEAM_ROOT/merged" \
    --expected-records 31392
done

python - "$AMLT_OUTPUT_DIR/eval" "${EVAL_NAMES[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for name in sys.argv[2:]:
    metrics = json.loads(
        (root / name / "merged" / "metrics.json").read_text(encoding="utf-8")
    )
    if int(metrics["samples"]) != 31392:
        raise RuntimeError(f"{name}: incorrect prediction count")
    if int(metrics["supported_interest_count"]) != 104:
        raise RuntimeError(f"{name}: incorrect Interest-Macro denominator")
    if int(metrics["top_k"]) != 20:
        raise RuntimeError(f"{name}: output is not Top-20")
print("CROSSFIT_FINAL_EVAL_AUDIT_OK")
PY
write_artifact_manifest
