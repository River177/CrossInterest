#!/usr/bin/env bash
set -euo pipefail

MODEL_FAMILY="${1:?Usage: run_static_user_disjoint_text_label_ablation.sh <qwen35|llama31> <t0e|t1> <smoke|full>}"
ABLATION="${2:?Usage: run_static_user_disjoint_text_label_ablation.sh <qwen35|llama31> <t0e|t1> <smoke|full>}"
MODE="${3:?Usage: run_static_user_disjoint_text_label_ablation.sh <qwen35|llama31> <t0e|t1> <smoke|full|eval-recovery>}"
NUM_GPUS="${TEXT_LABEL_ABLATION_NUM_GPUS:-4}"
case "${AMLT_JOB_NAME:?AMLT_JOB_NAME is required}" in
  *behavior*) TASK_TYPE="behavior_only"; SHORT_TASK="behavior" ;;
  *interest*) TASK_TYPE="interest_only"; SHORT_TASK="interest" ;;
  *) echo "Cannot infer task type from AMLT job name: $AMLT_JOB_NAME" >&2; exit 2 ;;
esac
case "$MODEL_FAMILY" in
  qwen35) MODEL_NAME="Qwen/Qwen3.5-9B"; SHORT_MODEL="qwen35" ;;
  llama31) MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"; SHORT_MODEL="llama31" ;;
  *) echo "Unsupported model family: $MODEL_FAMILY" >&2; exit 2 ;;
esac
case "$ABLATION" in t0e|t1) ;; *) echo "Unsupported ablation: $ABLATION" >&2; exit 2 ;; esac
case "$MODE" in smoke|full|eval-recovery) ;; *) echo "Unsupported mode: $MODE" >&2; exit 2 ;; esac
if [[ "$ABLATION" == "t0e" && "$MODE" == "eval-recovery" ]]; then
  echo "eval-recovery is only valid for T1" >&2
  exit 2
fi
if [[ "$ABLATION:$MODE:$NUM_GPUS" != "t0e:smoke:1" && "$NUM_GPUS" != "4" ]]; then
  echo "Only T0-E smoke may use one GPU; other runs require four" >&2
  exit 2
fi

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export WANDB_DISABLED=true

prepare_model_offline() {
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


def validate_snapshot(value: str) -> Path:
    path = Path(value)
    shards: set[str] = set()
    for index_path in (
        path / "model.safetensors.index.json",
        path / "pytorch_model.bin.index.json",
    ):
        if index_path.is_file():
            shards.update(
                json.loads(index_path.read_text(encoding="utf-8"))
                .get("weight_map", {})
                .values()
            )
    if shards:
        missing = [
            name
            for name in sorted(shards)
            if not (path / name).is_file() or (path / name).stat().st_size == 0
        ]
        if missing:
            raise RuntimeError(f"Incomplete model snapshot: {missing[:5]}")
    elif not any(
        candidate.is_file() and candidate.stat().st_size > 0
        for candidate in (
            path / "model.safetensors",
            path / "pytorch_model.bin",
        )
    ):
        raise RuntimeError("Model snapshot contains no complete weights")
    AutoConfig.from_pretrained(path, local_files_only=True)
    AutoTokenizer.from_pretrained(path, local_files_only=True)
    return path


try:
    snapshot = validate_snapshot(snapshot_download(name, local_files_only=True))
    source = "persistent-cache"
except Exception as cached_error:
    if name.startswith("meta-llama/") and not token:
        raise RuntimeError(
            "Complete persistent Llama snapshot is unavailable and HF_TOKEN "
            "was not injected"
        ) from cached_error
    snapshot = validate_snapshot(snapshot_download(name, token=token))
    source = "downloaded-or-repaired"
print("TEXT_LABEL_MODEL_SNAPSHOT_READY", name, snapshot, source)
PY
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
}

prepare_model_offline

DATA_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_balanced_200_h50_v1"
TRAIN_FILE="$DATA_ROOT/train/$TASK_TYPE/21600.parquet"
EVAL_FILE="$DATA_ROOT/eval/$TASK_TYPE/dev_full_grouped.parquet"
LABELS_FILE="$AMLT_CODE_DIR/data/training/generative_new_interest_static_user_disjoint_balanced_200_h50_v1/retained_interests.parquet"
LIKELIHOOD_CONFIG="$AMLT_CODE_DIR/configs/text_label_ranking/static_sud_${SHORT_MODEL}_${SHORT_TASK}_full.json"
BEAM_CONFIG="$AMLT_CODE_DIR/configs/beam/static_sud_${SHORT_MODEL}_${SHORT_TASK}_full.json"
TRAIN_CONFIG="$AMLT_CODE_DIR/configs/sft/experiments/static_sud_t1_${SHORT_MODEL}_${SHORT_TASK}_21600.yaml"
RUN_NAME="static-sud-${ABLATION}-${SHORT_MODEL}-${SHORT_TASK}-seed42-${MODE}-g4"

for path in "$EVAL_FILE" "$LABELS_FILE" "$LIKELIHOOD_CONFIG"; do
  if [[ ! -f "$path" ]]; then
    echo "Required experiment input not found: $path" >&2
    exit 1
  fi
done

validate_adapter() {
  local sft_dir="$1"
  python - "$sft_dir" "$ABLATION" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
ablation = sys.argv[2]
adapter = root / "final_adapter" / "adapter_model.safetensors"
summary_path = root / "run_summary.json"
if not adapter.is_file() or adapter.stat().st_size == 0:
    raise RuntimeError(f"Missing or empty adapter: {adapter}")
if not summary_path.is_file():
    raise RuntimeError(f"Missing run summary: {summary_path}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
loss_reduction = str(summary.get("loss_reduction", "token_mean"))
expected = "per_example_mean" if ablation == "t1" else "token_mean"
if loss_reduction != expected:
    raise RuntimeError(
        f"Unexpected loss reduction: expected {expected}, got {loss_reduction}"
    )
if int(summary["train_samples"]) != 21600 and ablation == "t0e":
    raise RuntimeError("T0-E source adapter was not trained on 21,600 samples")
if not math.isfinite(float(summary["train_metrics"]["train_loss"])):
    raise RuntimeError("Adapter run summary has non-finite train loss")
print("TEXT_LABEL_ADAPTER_GATE_OK", ablation, adapter.stat().st_size)
PY
}

run_likelihood() {
  local sft_dir="$1"
  local eval_root="$2"
  local eval_mode="$3"
  mkdir -p "$eval_root"
  if [[ "$eval_mode" == "smoke" ]]; then
    CUDA_VISIBLE_DEVICES=0 python -m new_interest.evaluation.evaluate_text_label_ranking \
      --config "$LIKELIHOOD_CONFIG" \
      --adapter-path "$sft_dir/final_adapter" \
      --eval-file "$EVAL_FILE" \
      --retained-interests-file "$LABELS_FILE" \
      --output-dir "$eval_root/smoke" \
      --run-name "${RUN_NAME}-likelihood-smoke" \
      --max-examples 4 \
      --num-shards 1 \
      --shard-index 0
    python - "$eval_root/smoke/metrics.json" <<'PY'
import json
import math
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if int(metrics["samples"]) != 4 or int(metrics["label_vocabulary_size"]) != 108:
    raise RuntimeError("Likelihood smoke size mismatch")
if int(metrics["top_k"]) != 20:
    raise RuntimeError("Likelihood smoke did not emit Top-20")
if metrics["history_prediction_filtering"] is not False:
    raise RuntimeError("History filtering was enabled unexpectedly")
if not math.isfinite(float(metrics["mean_score_softmax_entropy"])):
    raise RuntimeError("Likelihood smoke entropy is non-finite")
print("TEXT_LABEL_LIKELIHOOD_SMOKE_GATE_OK")
PY
    return
  fi

  local pids=()
  local shard_dirs=()
  for shard_index in 0 1 2 3; do
    local shard_name
    shard_name="$(printf 'shard-%02d' "$shard_index")"
    local shard_dir="$eval_root/$shard_name"
    shard_dirs+=("$shard_dir")
    (
      export CUDA_VISIBLE_DEVICES="$shard_index"
      python -m new_interest.evaluation.evaluate_text_label_ranking \
        --config "$LIKELIHOOD_CONFIG" \
        --adapter-path "$sft_dir/final_adapter" \
        --eval-file "$EVAL_FILE" \
        --retained-interests-file "$LABELS_FILE" \
        --output-dir "$shard_dir" \
        --run-name "${RUN_NAME}-likelihood-${shard_name}" \
        --num-shards 4 \
        --shard-index "$shard_index" \
        >"$eval_root/${shard_name}.log" 2>&1
    ) &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if [[ "$failed" -ne 0 ]]; then
    for log in "$eval_root"/shard-*.log; do
      echo "===== $log =====" >&2
      tail -n 80 "$log" >&2 || true
    done
    exit 1
  fi
  python -m new_interest.evaluation.merge_text_label_ranking_shards \
    --shard-dirs "${shard_dirs[@]}" \
    --eval-file "$EVAL_FILE" \
    --retained-interests-file "$LABELS_FILE" \
    --output-dir "$eval_root/merged" \
    --expected-records 31392
  python - "$eval_root/merged/metrics.json" <<'PY'
import json
import math
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
checks = {
    "samples": int(metrics["samples"]),
    "supported_interest_count": int(metrics["supported_interest_count"]),
    "label_vocabulary_size": int(metrics["label_vocabulary_size"]),
    "top_k": int(metrics["top_k"]),
}
expected = {
    "samples": 31392,
    "supported_interest_count": 104,
    "label_vocabulary_size": 108,
    "top_k": 20,
}
if checks != expected:
    raise RuntimeError(f"Likelihood full gate mismatch: {checks}")
if metrics["history_prediction_filtering"] is not False:
    raise RuntimeError("History filtering was enabled unexpectedly")
for key in ("interest_macro_recall_at_10", "interest_macro_ndcg_at_10"):
    if not math.isfinite(float(metrics[key])):
        raise RuntimeError(f"Non-finite metric: {key}")
print(
    "TEXT_LABEL_LIKELIHOOD_FULL_GATE_OK",
    metrics["interest_macro_recall_at_10"],
    metrics["interest_macro_ndcg_at_10"],
)
PY
}

if [[ "$ABLATION" == "t0e" ]]; then
  SFT_DIR="${TEXT_LABEL_SOURCE_SFT_DIR:-${AMLT_MAP_INPUT_DIR:?AMLT_MAP_INPUT_DIR is required for T0-E}/sft}"
  validate_adapter "$SFT_DIR"
  run_likelihood "$SFT_DIR" "$AMLT_OUTPUT_DIR/likelihood" "$MODE"
  exit 0
fi

for path in "$TRAIN_FILE" "$TRAIN_CONFIG" "$BEAM_CONFIG"; do
  if [[ ! -f "$path" ]]; then
    echo "Required T1 input not found: $path" >&2
    exit 1
  fi
done

if [[ "$MODE" == "eval-recovery" ]]; then
  SFT_DIR="${AMLT_MAP_INPUT_DIR:?AMLT_MAP_INPUT_DIR is required for T1 eval recovery}/sft"
else
  SFT_DIR="$AMLT_OUTPUT_DIR/sft"
fi
if [[ ! -s "$SFT_DIR/final_adapter/adapter_model.safetensors" ]]; then
  mkdir -p "$SFT_DIR"
  if [[ "$MODE" == "smoke" ]]; then
    python -m torch.distributed.run \
      --standalone \
      --nproc_per_node="$NUM_GPUS" \
      -m new_interest.training.train_sft \
      --config "$TRAIN_CONFIG" \
      --train-file "$TRAIN_FILE" \
      --output-dir "$SFT_DIR" \
      --run-name "$RUN_NAME" \
      --loss-reduction per_example_mean \
      --max-samples 32 \
      --epochs 2 \
      --per-device-train-batch-size 1 \
      --gradient-accumulation-steps 8 \
      --wandb-mode disabled
  else
    LOCAL_CHECKPOINT_DIR="/scratch/${AMLT_JOB_ID:-text-label-t1}/checkpoints"
    PERSISTENT_CHECKPOINT_DIR="$AMLT_OUTPUT_DIR/persistent_checkpoints"
    mkdir -p "$LOCAL_CHECKPOINT_DIR" "$PERSISTENT_CHECKPOINT_DIR"
    python -m torch.distributed.run \
      --standalone \
      --nproc_per_node="$NUM_GPUS" \
      -m new_interest.training.train_sft \
      --config "$TRAIN_CONFIG" \
      --train-file "$TRAIN_FILE" \
      --output-dir "$SFT_DIR" \
      --run-name "$RUN_NAME" \
      --loss-reduction per_example_mean \
      --per-device-train-batch-size 1 \
      --gradient-accumulation-steps 8 \
      --checkpoint-dir "$LOCAL_CHECKPOINT_DIR" \
      --persistent-checkpoint-dir "$PERSISTENT_CHECKPOINT_DIR" \
      --resume-from-checkpoint auto \
      --save-steps 25 \
      --wandb-mode disabled
  fi
fi
validate_adapter "$SFT_DIR"

python - "$SFT_DIR/run_summary.json" "$MODE" <<'PY'
import json
import math
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
mode = sys.argv[2]
if summary.get("loss_reduction") != "per_example_mean":
    raise RuntimeError("T1 did not use per-example mean loss")
if summary.get("sample_weight_file") is not None:
    raise RuntimeError("T1 unexpectedly used a sample-weight file")
if int(summary["world_size"]) != 4:
    raise RuntimeError("T1 world size is not four")
if int(summary["effective_global_batch_size"]) != 32:
    raise RuntimeError("T1 effective global batch is not 32")
if int(summary["token_audit"]["over_max_seq_length"]) != 0:
    raise RuntimeError("T1 token audit found over-length rows")
if summary["model_audit"]["forbidden_trainable_parameters"]:
    raise RuntimeError("T1 has forbidden visual trainable parameters")
if not math.isfinite(float(summary["train_metrics"]["train_loss"])):
    raise RuntimeError("T1 train loss is non-finite")
if mode == "smoke":
    expected = {"train_samples": 32, "steps": 2, "epochs": 2.0}
else:
    expected = {"train_samples": 21600, "steps": 2025, "epochs": 3.0}
actual = {
    "train_samples": int(summary["train_samples"]),
    "steps": int(summary["expected_optimizer_steps"]),
    "epochs": float(summary["configured_epochs"]),
}
if actual != expected:
    raise RuntimeError(f"T1 training contract mismatch: {actual}")
print("TEXT_LABEL_T1_TRAIN_GATE_OK", mode, actual)
PY

if [[ "$MODE" == "smoke" ]]; then
  run_likelihood "$SFT_DIR" "$AMLT_OUTPUT_DIR/likelihood" smoke
  echo "TEXT_LABEL_T1_DDP_AND_EVAL_SMOKE_GATE_OK $SHORT_MODEL $SHORT_TASK"
  exit 0
fi

run_likelihood "$SFT_DIR" "$AMLT_OUTPUT_DIR/likelihood" full

BEAM_ROOT="$AMLT_OUTPUT_DIR/beam"
mkdir -p "$BEAM_ROOT"
PIDS=()
SHARD_DIRS=()
for SHARD_INDEX in 0 1 2 3; do
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
      --run-name "${RUN_NAME}-beam-${SHARD_NAME}" \
      --num-shards 4 \
      --shard-index "$SHARD_INDEX" \
      --wandb-mode disabled \
      >"$BEAM_ROOT/${SHARD_NAME}.log" 2>&1
  ) &
  PIDS+=("$!")
done
EVAL_FAILURE=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then EVAL_FAILURE=1; fi
done
if [[ "$EVAL_FAILURE" -ne 0 ]]; then
  for log in "$BEAM_ROOT"/shard-*.log; do
    echo "===== $log =====" >&2
    tail -n 80 "$log" >&2 || true
  done
  exit 1
fi
python -m new_interest.evaluation.merge_beam_shards \
  --shard-dirs "${SHARD_DIRS[@]}" \
  --eval-file "$EVAL_FILE" \
  --retained-interests-file "$LABELS_FILE" \
  --output-dir "$BEAM_ROOT/merged" \
  --expected-records 31392

python - "$BEAM_ROOT/merged/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
checks = {
    "samples": int(metrics["samples"]),
    "supported_interest_count": int(metrics["supported_interest_count"]),
    "label_vocabulary_size": int(metrics["label_vocabulary_size"]),
    "top_k": int(metrics["top_k"]),
}
expected = {
    "samples": 31392,
    "supported_interest_count": 104,
    "label_vocabulary_size": 108,
    "top_k": 20,
}
if checks != expected:
    raise RuntimeError(f"T1 Beam full gate mismatch: {checks}")
if metrics.get("history_prediction_filtering", False):
    raise RuntimeError("History filtering was enabled unexpectedly")
print(
    "TEXT_LABEL_T1_BEAM_FULL_GATE_OK",
    metrics["interest_macro_recall_at_10"],
    metrics["interest_macro_ndcg_at_10"],
)
PY
