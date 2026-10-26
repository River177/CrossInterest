#!/usr/bin/env bash
set -euo pipefail

MODEL_FAMILY="${1:?Usage: run_static_user_disjoint_semantic_id_token_sft.sh <qwen35|llama31> <smoke|full>}"
MODE="${2:?Usage: run_static_user_disjoint_semantic_id_token_sft.sh <qwen35|llama31> <smoke|recovery-smoke|full>}"
NUM_GPUS="${SEMANTIC_ID_NUM_GPUS:-4}"
TRAIN_SIZE="${SEMANTIC_ID_TRAIN_SIZE:-21600}"
GRADIENT_AUDIT_BATCH_SIZE="${SEMANTIC_ID_GRADIENT_AUDIT_BATCH_SIZE:-2}"
FULL_SAVE_STEPS="${SEMANTIC_ID_FULL_SAVE_STEPS:-25}"

case "$TRAIN_SIZE" in
  5400) EXPECTED_FULL_STEPS=507 ;;
  10800) EXPECTED_FULL_STEPS=1014 ;;
  16200) EXPECTED_FULL_STEPS=1521 ;;
  21600) EXPECTED_FULL_STEPS=2025 ;;
  *)
    echo "Unsupported Semantic-ID Train size: $TRAIN_SIZE" >&2
    exit 2
    ;;
esac

if [[ "$NUM_GPUS" != "4" ]]; then
  echo "semantic_id_token_v1 is fixed to four GPUs; got $NUM_GPUS" >&2
  exit 2
fi

case "${AMLT_JOB_NAME:?AMLT_JOB_NAME is required}" in
  *behavior-interest*)
    TASK_TYPE="behavior_interest"
    SHORT_TASK="behavior_interest"
    ;;
  *behavior*)
    TASK_TYPE="behavior_only"
    SHORT_TASK="behavior"
    ;;
  *interest*)
    TASK_TYPE="interest_only"
    SHORT_TASK="interest"
    ;;
  *)
    echo "Cannot infer task type from AMLT job name: $AMLT_JOB_NAME" >&2
    exit 2
    ;;
esac

case "$MODEL_FAMILY" in
  qwen35)
    MODEL_NAME="Qwen/Qwen3.5-9B"
    SHORT_MODEL="qwen35"
    ;;
  llama31)
    MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
    SHORT_MODEL="llama31"
    ;;
  *)
    echo "Unsupported model family: $MODEL_FAMILY" >&2
    exit 2
    ;;
esac

case "$MODE" in
  smoke|recovery-smoke|full) ;;
  *)
    echo "Unsupported run mode: $MODE" >&2
    exit 2
    ;;
esac

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export WANDB_DISABLED=true

prepare_model_offline() {
  # Keep the cache on the persistent Feeds mount when the AMLT config
  # provides one.  A job can be rescheduled onto a fresh node while the
  # model snapshot is downloading; a /scratch-only cache would then restart
  # the full download on every allocation.
  export HF_HOME="${HF_HOME:-/mnt/default/huggingface-cache}"
  mkdir -p "$HF_HOME"
  python - "$MODEL_NAME" <<'PY'
import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

name = sys.argv[1]
token = os.environ.get("HF_TOKEN")


def validate_snapshot(snapshot_value: str) -> Path:
    snapshot_path = Path(snapshot_value)
    weight_files: set[str] = set()
    for index_path in (
        snapshot_path / "model.safetensors.index.json",
        snapshot_path / "pytorch_model.bin.index.json",
    ):
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_files.update(payload.get("weight_map", {}).values())
    if weight_files:
        missing = [
            value
            for value in sorted(weight_files)
            if not (snapshot_path / value).is_file()
            or (snapshot_path / value).stat().st_size == 0
        ]
        if missing:
            raise RuntimeError(
                "Cached model snapshot has missing weight shards: "
                + ", ".join(missing[:5])
            )
    elif not any(
        path.is_file() and path.stat().st_size > 0
        for path in (
            snapshot_path / "model.safetensors",
            snapshot_path / "pytorch_model.bin",
        )
    ):
        raise RuntimeError("Cached model snapshot has no complete model weights")
    AutoConfig.from_pretrained(snapshot_path, local_files_only=True)
    AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
    return snapshot_path


try:
    snapshot = validate_snapshot(
        snapshot_download(name, local_files_only=True)
    )
    source = "persistent-cache"
except Exception as cached_error:
    if name.startswith("meta-llama/") and not token:
        raise RuntimeError(
            "Complete persistent Llama snapshot is unavailable and HF_TOKEN "
            "was not injected"
        ) from cached_error
    snapshot = validate_snapshot(snapshot_download(name, token=token))
    source = "downloaded-or-repaired"
config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
print(
    "SEMANTIC_ID_MODEL_SNAPSHOT_READY",
    name,
    snapshot,
    source,
    type(config).__name__,
    type(tokenizer).__name__,
)
PY
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
}

prepare_model_offline

if [[ "$TASK_TYPE" == "behavior_interest" ]]; then
  CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_semantic_id_${SHORT_MODEL}_behavior_interest_${TRAIN_SIZE}.yaml"
  EVAL_CONFIG="$AMLT_CODE_DIR/configs/id_token/static_sud_${SHORT_MODEL}_behavior_interest_full.json"
  DATA_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_scaling_v1"
else
  if [[ "$TRAIN_SIZE" != "21600" ]]; then
    echo "Legacy Behavior-only/Interest-only Semantic-ID runs require 21,600 rows" >&2
    exit 2
  fi
  CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_semantic_id_${SHORT_MODEL}_${SHORT_TASK}_21600.yaml"
  EVAL_CONFIG="$AMLT_CODE_DIR/configs/id_token/static_sud_${SHORT_MODEL}_${SHORT_TASK}_full.json"
  DATA_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_semantic_id_token_200_h50_v1"
fi
TRAIN_FILE="$DATA_ROOT/train/$TASK_TYPE/${TRAIN_SIZE}.parquet"
EVAL_FILE="$DATA_ROOT/eval/$TASK_TYPE/dev_full_grouped.parquet"
MAPPING_FILE="$DATA_ROOT/interest_token_mapping.parquet"
RUN_NAME="static-sud-semantic-id-${SHORT_MODEL}-${SHORT_TASK}-${TRAIN_SIZE}-seed42-${MODE}-g4"

for path in "$CONFIG" "$EVAL_CONFIG" "$TRAIN_FILE" "$EVAL_FILE" "$MAPPING_FILE"; do
  if [[ ! -f "$path" ]]; then
    echo "Required Semantic ID experiment input not found: $path" >&2
    exit 1
  fi
done

validate_adapter_state() {
  local sft_dir="$1"
  python - "$sft_dir" <<'PY'
import json
import sys
from pathlib import Path

from safetensors import safe_open

root = Path(sys.argv[1])
summary = json.loads((root / "run_summary.json").read_text(encoding="utf-8"))
adapter = root / "final_adapter" / "adapter_model.safetensors"
if not adapter.is_file() or adapter.stat().st_size == 0:
    raise RuntimeError("Missing or empty final adapter")
if not (root / "final_adapter" / "interest_token_mapping.parquet").is_file():
    raise RuntimeError("Final adapter misses the interest-token mapping")
with safe_open(adapter, framework="pt", device="cpu") as handle:
    keys = list(handle.keys())
    shapes = {key: list(handle.get_slice(key).get_shape()) for key in keys}
token_keys = [key for key in keys if "trainable_tokens_delta" in key]
expected_token_tensors = (
    1
    if summary["semantic_token_initialization"]["tie_word_embeddings"]
    else 2
)
if len(token_keys) != expected_token_tensors:
    raise RuntimeError(
        "Unexpected trainable-token tensor count: "
        f"expected {expected_token_tensors}, got {len(token_keys)}"
    )
if any(shapes[key][0] != 108 for key in token_keys):
    raise RuntimeError(f"Trainable-token tensor does not contain 108 rows: {shapes}")
resized_vocab = int(
    summary["semantic_token_initialization"]["resized_vocabulary_size"]
)
full_embedding_keys = [
    key
    for key, shape in shapes.items()
    if len(shape) == 2 and shape[0] == resized_vocab and shape[1] > 108
]
if full_embedding_keys:
    raise RuntimeError(
        f"Full embedding matrices were saved unexpectedly: {full_embedding_keys}"
    )
if not any("lora_" in key for key in keys):
    raise RuntimeError("Adapter contains no LoRA tensors")
audit = {
    "adapter_bytes": adapter.stat().st_size,
    "tensor_count": len(keys),
    "trainable_token_tensor_count": len(token_keys),
    "trainable_token_shapes": {key: shapes[key] for key in token_keys},
    "full_embedding_tensors": full_embedding_keys,
}
(root / "semantic_adapter_state_audit.json").write_text(
    json.dumps(audit, indent=2) + "\n",
    encoding="utf-8",
)
print("SEMANTIC_ID_ADAPTER_STATE_OK", audit)
PY
}

validate_training_summary() {
  local summary_path="$1"
  local gate_mode="$2"
  local expected_source_samples="$3"
  local expected_full_steps="$4"
  python - \
    "$summary_path" "$gate_mode" \
    "$expected_source_samples" "$expected_full_steps" <<'PY'
import json
import math
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
mode = sys.argv[2]
expected_source_samples = int(sys.argv[3])
expected_full_steps = int(sys.argv[4])
if int(summary["token_audit"]["samples"]) != expected_source_samples:
    raise RuntimeError("Token audit did not cover the full configured Train file")
if int(summary["token_audit"]["over_max_seq_length"]) != 0:
    raise RuntimeError("Token audit found over-length examples")
if summary["sample_weight_file"] is not None:
    raise RuntimeError("Vanilla Semantic-ID training consumed sample weights")
if summary["model_audit"]["forbidden_trainable_parameters"]:
    raise RuntimeError("Forbidden visual parameters are trainable")
semantic = summary["model_audit"]["semantic_id_tokens"]
if not semantic["enabled"] or int(semantic["count"]) != 108:
    raise RuntimeError("Model audit does not contain exactly 108 Semantic IDs")
if semantic["save_embedding_layers"] is not False:
    raise RuntimeError("Full embedding saving was not disabled")
initialization = summary["semantic_token_initialization"]
if int(initialization["interest_token_count"]) != 108:
    raise RuntimeError("Initialization audit does not cover 108 tokens")
drift = summary["semantic_token_drift"]
changed_input_rows = int(drift["changed_input_rows"])
changed_output_rows = int(drift["changed_output_rows"])
if not 0 < changed_input_rows <= 108:
    raise RuntimeError("No input ID row changed")
if not 0 < changed_output_rows <= 108:
    raise RuntimeError("No output ID row changed")
loss = float(summary["train_metrics"]["train_loss"])
if not math.isfinite(loss):
    raise RuntimeError(f"Non-finite train loss: {loss}")

if mode == "single-smoke":
    if int(summary["world_size"]) != 1 or int(summary["train_samples"]) != 32:
        raise RuntimeError("Single-GPU smoke identity mismatch")
    if float(summary["configured_epochs"]) != 5.0:
        raise RuntimeError("Single-GPU smoke must run five epochs")
    if int(summary["expected_optimizer_steps"]) != 20:
        raise RuntimeError("Single-GPU smoke must run 20 optimizer steps")
    history = summary["loss_history"]
    if len(history) < 2 or not float(history[-1]["loss"]) < float(history[0]["loss"]):
        raise RuntimeError(f"Single-GPU smoke loss did not decrease: {history}")
elif mode == "ddp-smoke":
    if int(summary["world_size"]) != 4 or int(summary["train_samples"]) != 32:
        raise RuntimeError("4-GPU DDP smoke identity mismatch")
    if int(summary["effective_global_batch_size"]) != 32:
        raise RuntimeError("DDP smoke effective global batch is not 32")
    if int(summary["expected_optimizer_steps"]) != 2:
        raise RuntimeError("DDP smoke did not run exactly two optimizer steps")
    if summary["gradient_checkpointing_kwargs"] != {"use_reentrant": False}:
        raise RuntimeError("DDP smoke did not use non-reentrant checkpointing")
elif mode == "full":
    if int(summary["world_size"]) != 4:
        raise RuntimeError("Full run did not use four GPUs")
    if int(summary["train_samples"]) != expected_source_samples:
        raise RuntimeError("Full run Train row count mismatch")
    if float(summary["configured_epochs"]) != 3.0:
        raise RuntimeError("Full run did not complete exactly three epochs")
    if int(summary["effective_global_batch_size"]) != 32:
        raise RuntimeError("Full effective global batch is not 32")
    if int(summary["expected_optimizer_steps"]) != expected_full_steps:
        raise RuntimeError("Full run optimizer-step count mismatch")
    if summary["gradient_checkpointing_kwargs"] != {"use_reentrant": False}:
        raise RuntimeError("Full run did not use non-reentrant checkpointing")
    if not drift["all_input_rows_changed"]:
        raise RuntimeError("Full run left at least one input ID row unchanged")
    if not drift["all_output_rows_changed"]:
        raise RuntimeError("Full run left at least one output ID row unchanged")
else:
    raise RuntimeError(f"Unknown gate mode: {mode}")
print(
    "SEMANTIC_ID_TRAIN_GATE_OK",
    mode,
    loss,
    changed_input_rows,
    changed_output_rows,
)
PY
}

verify_reload() {
  local sft_dir="$1"
  local output_name="$2"
  CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.verify_semantic_id_token_adapter \
    --model-name "$MODEL_NAME" \
    --adapter-path "$sft_dir/final_adapter" \
    --mapping-file "$MAPPING_FILE" \
    --reference-file "$sft_dir/semantic_reload_reference.json" \
    --output "$sft_dir/$output_name"
}

latest_complete_checkpoint() {
  local checkpoint_root="$1"
  local checkpoint
  while IFS= read -r checkpoint; do
    if [[ -s "$checkpoint/trainer_state.json" \
      && -s "$checkpoint/optimizer.pt" \
      && -s "$checkpoint/scheduler.pt" \
      && -s "$checkpoint/adapter_model.safetensors" \
      && -s "$checkpoint/adapter_config.json" ]]; then
      printf '%s\n' "$checkpoint"
      return 0
    fi
  done < <(
    find "$checkpoint_root" -mindepth 1 -maxdepth 1 \
      -type d -name 'checkpoint-*' -print 2>/dev/null | sort -Vr
  )
  return 1
}

if [[ "$MODE" == "smoke" ]]; then
  SMOKE_ROOT="$AMLT_OUTPUT_DIR/smoke"
  GRADIENT_DIR="$SMOKE_ROOT/gradient"
  SINGLE_DIR="$SMOKE_ROOT/single"
  DDP_DIR="$SMOKE_ROOT/ddp"
  mkdir -p "$GRADIENT_DIR"

  CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.audit_semantic_id_token_gradients \
    --config "$CONFIG" \
    --train-file "$TRAIN_FILE" \
    --output "$GRADIENT_DIR/gradient_audit.json" \
    --batch-size "$GRADIENT_AUDIT_BATCH_SIZE"

  CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.train_sft \
    --config "$CONFIG" \
    --train-file "$TRAIN_FILE" \
    --output-dir "$SINGLE_DIR" \
    --run-name "${RUN_NAME}-single" \
    --max-samples 32 \
    --epochs 5 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --wandb-mode disabled
  validate_training_summary \
    "$SINGLE_DIR/run_summary.json" single-smoke \
    "$TRAIN_SIZE" "$EXPECTED_FULL_STEPS"
  validate_adapter_state "$SINGLE_DIR"
  verify_reload "$SINGLE_DIR" semantic_adapter_reload_audit.json

  python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    -m new_interest.training.train_sft \
    --config "$CONFIG" \
    --train-file "$TRAIN_FILE" \
    --output-dir "$DDP_DIR" \
    --run-name "${RUN_NAME}-ddp" \
    --max-samples 32 \
    --epochs 2 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --wandb-mode disabled
  validate_training_summary \
    "$DDP_DIR/run_summary.json" ddp-smoke \
    "$TRAIN_SIZE" "$EXPECTED_FULL_STEPS"
  validate_adapter_state "$DDP_DIR"
  verify_reload "$DDP_DIR" semantic_adapter_reload_audit.json
  echo "SEMANTIC_ID_FOUR_GPU_SMOKE_GATE_OK $SHORT_MODEL $SHORT_TASK"
  exit 0
fi

if [[ "$MODE" == "recovery-smoke" ]]; then
  RECOVERY_ROOT="$AMLT_OUTPUT_DIR/recovery-smoke"
  DDP_DIR="$RECOVERY_ROOT/ddp"

  python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    -m new_interest.training.train_sft \
    --config "$CONFIG" \
    --train-file "$TRAIN_FILE" \
    --output-dir "$DDP_DIR" \
    --run-name "${RUN_NAME}-recovery-part1" \
    --max-samples 32 \
    --epochs 1 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --save-steps 1 \
    --wandb-mode disabled

  FIRST_CHECKPOINT="$(latest_complete_checkpoint "$DDP_DIR/checkpoints")"
  if [[ "$(basename "$FIRST_CHECKPOINT")" != "checkpoint-1" ]]; then
    echo "Recovery smoke did not create checkpoint-1: $FIRST_CHECKPOINT" >&2
    exit 1
  fi
  echo "SEMANTIC_ID_RECOVERY_CHECKPOINT_READY $FIRST_CHECKPOINT"

  python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    -m new_interest.training.train_sft \
    --config "$CONFIG" \
    --train-file "$TRAIN_FILE" \
    --output-dir "$DDP_DIR" \
    --run-name "${RUN_NAME}-recovery-part2" \
    --max-samples 32 \
    --epochs 2 \
    --per-device-train-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --save-steps 1 \
    --resume-from-checkpoint "$FIRST_CHECKPOINT" \
    --wandb-mode disabled
  validate_training_summary \
    "$DDP_DIR/run_summary.json" ddp-smoke \
    "$TRAIN_SIZE" "$EXPECTED_FULL_STEPS"
  validate_adapter_state "$DDP_DIR"
  verify_reload "$DDP_DIR" semantic_adapter_reload_audit.json
  SECOND_CHECKPOINT="$(latest_complete_checkpoint "$DDP_DIR/checkpoints")"
  if [[ "$(basename "$SECOND_CHECKPOINT")" != "checkpoint-2" ]]; then
    echo "Recovery smoke did not resume through checkpoint-2: $SECOND_CHECKPOINT" >&2
    exit 1
  fi
  echo "SEMANTIC_ID_FOUR_GPU_RECOVERY_SMOKE_GATE_OK $SHORT_MODEL $SHORT_TASK"
  exit 0
fi

SFT_DIR="$AMLT_OUTPUT_DIR/sft"
EVAL_ROOT="$AMLT_OUTPUT_DIR/eval"
RESUME_ARGS=()
if RESUME_CHECKPOINT="$(latest_complete_checkpoint "$SFT_DIR/checkpoints")"; then
  RESUME_ARGS=(--resume-from-checkpoint "$RESUME_CHECKPOINT")
  echo "SEMANTIC_ID_RESUME_FROM_CHECKPOINT $RESUME_CHECKPOINT"
else
  echo "SEMANTIC_ID_START_FROM_BASE_MODEL"
fi
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NUM_GPUS" \
  -m new_interest.training.train_sft \
  --config "$CONFIG" \
  --train-file "$TRAIN_FILE" \
  --output-dir "$SFT_DIR" \
  --run-name "$RUN_NAME" \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --save-steps "$FULL_SAVE_STEPS" \
  "${RESUME_ARGS[@]}" \
  --wandb-mode disabled
validate_training_summary \
  "$SFT_DIR/run_summary.json" full "$TRAIN_SIZE" "$EXPECTED_FULL_STEPS"
validate_adapter_state "$SFT_DIR"
verify_reload "$SFT_DIR" semantic_adapter_reload_audit.json

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
      --run-name "${RUN_NAME}-${SHARD_NAME}" \
      --num-shards 4 \
      --shard-index "$SHARD_INDEX" \
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
  echo "At least one ID-token evaluation shard failed" >&2
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
  --output-dir "$EVAL_ROOT/merged" \
  --expected-records 31392

python - "$EVAL_ROOT/merged/metrics.json" <<'PY'
import json
import math
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if int(metrics["samples"]) != 31392:
    raise RuntimeError("Merged ID-token prediction count is not 31,392")
if int(metrics["supported_interest_count"]) != 104:
    raise RuntimeError("Interest Macro denominator is not 104")
if int(metrics["label_vocabulary_size"]) != 108:
    raise RuntimeError("Restricted label vocabulary is not 108")
if int(metrics["top_k"]) != 20:
    raise RuntimeError("Evaluation did not preserve Top-20")
if metrics["history_prediction_filtering"] is not False:
    raise RuntimeError("History filtering was enabled unexpectedly")
for key in ("interest_macro_recall_at_10", "interest_macro_ndcg_at_10"):
    value = float(metrics[key])
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite metric {key}: {value}")
print(
    "SEMANTIC_ID_FULL_EVAL_GATE_OK",
    metrics["interest_macro_recall_at_10"],
    metrics["interest_macro_ndcg_at_10"],
)
PY
