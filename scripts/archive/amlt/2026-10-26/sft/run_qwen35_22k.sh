#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: run_qwen35_22k.sh <smoke|pilot|full>}"

prefetch_qwen35_model() {
  python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download("Qwen/Qwen3.5-9B")
print("QWEN35_SNAPSHOT_READY", path)
PY
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
}

case "$AMLT_JOB_NAME" in
  *interest*)
    TASK_TYPE="interest_only"
    SHORT_TASK="interest"
    ;;
  *behavior*)
    TASK_TYPE="behavior_only"
    SHORT_TASK="behavior"
    ;;
  *)
    echo "Cannot infer task type from AMLT job name: $AMLT_JOB_NAME" >&2
    exit 2
    ;;
esac

case "$MODE" in
  smoke)
    TRAIN_SIZE="22k"
    OUTPUT_NAME="sft"
    MAX_SAMPLE_ARGS=(--max-samples 32)
    EPOCH_ARGS=(--epochs 20)
    EVAL_MAX=32
    WANDB_MODE="disabled"
    RUN_SUFFIX="smoke32"
    TRAIN_COMMAND=(python -m new_interest.training.train_sft)
    DISTRIBUTED_ARGS=()
    ;;
  pilot)
    TRAIN_SIZE="1k"
    OUTPUT_NAME="sft"
    MAX_SAMPLE_ARGS=()
    EPOCH_ARGS=()
    EVAL_MAX=1000
    WANDB_MODE="online"
    RUN_SUFFIX="pilot1k"
    TRAIN_COMMAND=(python -m new_interest.training.train_sft)
    DISTRIBUTED_ARGS=()
    ;;
  full)
    TRAIN_SIZE="22k"
    OUTPUT_NAME="sft"
    MAX_SAMPLE_ARGS=()
    EPOCH_ARGS=()
    EVAL_MAX=0
    WANDB_MODE="online"
    RUN_SUFFIX="full22k"
    export HF_HOME="/scratch/huggingface-cache"
    mkdir -p "$HF_HOME"
    prefetch_qwen35_model
    TRAIN_COMMAND=(
      python -m torch.distributed.run
      --standalone
      --nproc_per_node=8
      -m new_interest.training.train_sft
    )
    DISTRIBUTED_ARGS=(
      --per-device-train-batch-size 1
      --gradient-accumulation-steps 4
    )
    ;;
  *)
    echo "Unsupported Qwen3.5 run mode: $MODE" >&2
    exit 2
    ;;
esac

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/qwen35_${SHORT_TASK}_22k.yaml"
BEAM_CONFIG="$AMLT_CODE_DIR/configs/beam/qwen35_${SHORT_TASK}_22k.json"
TRAIN_FILE="$AMLT_CODE_DIR/data/training/sft_experiments/train/$TASK_TYPE/$TRAIN_SIZE.parquet"
OUTPUT_DIR="$AMLT_OUTPUT_DIR/$OUTPUT_NAME"
RUN_NAME="qwen35-9b-${SHORT_TASK}-${RUN_SUFFIX}-seed42"

for path in "$CONFIG" "$BEAM_CONFIG" "$TRAIN_FILE"; do
  if [[ ! -f "$path" ]]; then
    echo "Required Qwen3.5 input not found: $path" >&2
    exit 1
  fi
done

python - <<'PY'
from transformers import AutoConfig, AutoTokenizer

name = "Qwen/Qwen3.5-9B"
config = AutoConfig.from_pretrained(name, trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(
    name,
    trust_remote_code=False,
)
print(
    "QWEN35_ACCESS_OK",
    name,
    type(config).__name__,
    config.model_type,
    config.architectures,
    type(tokenizer).__name__,
)
PY

if [[ ! -f "$OUTPUT_DIR/final_adapter/adapter_model.safetensors" ]]; then
  RESUME_ARGS=()
  LATEST_CHECKPOINT="$(
    (find "$OUTPUT_DIR/checkpoints" -maxdepth 1 -type d \
      -name 'checkpoint-*' 2>/dev/null || true) |
      sort -V |
      tail -n 1
  )"
  if [[ -n "$LATEST_CHECKPOINT" ]]; then
    RESUME_ARGS=(--resume-from-checkpoint "$LATEST_CHECKPOINT")
  fi

  "${TRAIN_COMMAND[@]}" \
    --config "$CONFIG" \
    --train-file "$TRAIN_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$RUN_NAME" \
    --wandb-mode "$WANDB_MODE" \
    "${MAX_SAMPLE_ARGS[@]}" \
    "${EPOCH_ARGS[@]}" \
    "${DISTRIBUTED_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
fi

python - "$OUTPUT_DIR/run_summary.json" "$MODE" <<'PY'
import json
import math
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
mode = sys.argv[2]
summary = json.loads(summary_path.read_text(encoding="utf-8"))
audit = summary["model_audit"]
if audit["forbidden_trainable_parameters"]:
    raise RuntimeError(audit["forbidden_trainable_parameters"])
if summary["token_audit"]["over_max_seq_length"]:
    raise RuntimeError("Token audit found over-length examples")
train_loss = float(summary["train_metrics"]["train_loss"])
if not math.isfinite(train_loss):
    raise RuntimeError(f"Non-finite train loss: {train_loss}")
if mode == "smoke":
    history = summary["loss_history"]
    if len(history) < 2:
        raise RuntimeError(f"Insufficient smoke loss history: {history}")
    if not float(history[-1]["loss"]) < float(history[0]["loss"]):
        raise RuntimeError(f"Smoke loss did not decrease: {history}")
print("QWEN35_TRAIN_GATE_OK", mode, train_loss)
PY

if [[ "$EVAL_MAX" -gt 0 ]]; then
  BEAM_DIR="$AMLT_OUTPUT_DIR/beam"
  python -m new_interest.evaluation.evaluate_beam \
    --config "$BEAM_CONFIG" \
    --adapter-path "$OUTPUT_DIR/final_adapter" \
    --output-dir "$BEAM_DIR" \
    --run-name "${RUN_NAME}-beam${EVAL_MAX}" \
    --max-examples "$EVAL_MAX" \
    --wandb-mode "$WANDB_MODE"

  if [[ "$MODE" == "pilot" ]]; then
    python - "$BEAM_DIR/metrics.json" <<'PY'
import json
import math
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ndcg = float(metrics["ndcg_at_20"])
max_share = float(metrics["max_top1_label_share"])
entropy = float(metrics["normalized_top1_label_entropy"])
unique = int(metrics["unique_top1_labels"])
if ndcg <= 0.0669:
    raise RuntimeError(f"Pilot NDCG@20 did not beat random: {ndcg}")
if (
    unique < 10
    or max_share >= 0.35
    or not math.isfinite(entropy)
    or entropy < 0.40
):
    raise RuntimeError(
        "Pilot Top-1 distribution collapsed: "
        f"unique={unique}, max_share={max_share}, entropy={entropy}"
    )
print("QWEN35_PILOT_GATE_OK", ndcg, unique, max_share, entropy)
PY
  fi
fi
