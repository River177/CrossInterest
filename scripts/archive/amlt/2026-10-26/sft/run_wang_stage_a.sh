#!/usr/bin/env bash
set -euo pipefail

MODEL_FAMILY=${1:?usage: run_wang_stage_a.sh <qwen35|qwen2> <smoke|full-validation|zero-shot-validation> <arts|grocery|mind> <balanced|random>}
MODE=${2:?missing mode}
DATASET=${3:?missing dataset}
CONDITION=${4:-balanced}

case "$MODEL_FAMILY" in
  qwen35|qwen2) ;;
  *) echo "unsupported model family: $MODEL_FAMILY" >&2; exit 2 ;;
esac
case "$MODE" in
  smoke|full-validation|zero-shot-validation) ;;
  *) echo "unsupported mode: $MODE" >&2; exit 2 ;;
esac
case "$DATASET" in
  arts|grocery|mind) ;;
  *) echo "unsupported dataset: $DATASET" >&2; exit 2 ;;
esac
case "$CONDITION" in
  balanced|random) ;;
  *) echo "unsupported condition: $CONDITION" >&2; exit 2 ;;
esac

: "${AMLT_OUTPUT_DIR:?AMLT_OUTPUT_DIR is required}"
: "${HF_HOME:=/mnt/default/huggingface-cache}"
: "${PIP_CACHE_DIR:=/mnt/default/pip-cache-wang-stage-a-${MODEL_FAMILY}}"
REUSED_SFT_DIR=${WANG_REUSE_SFT_DIR:-}
: "${WANG_NUM_GPUS:=1}"
if [[ ! "$WANG_NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "WANG_NUM_GPUS must be a positive integer" >&2
  exit 2
fi
export WANG_NUM_GPUS
export HF_HOME PIP_CACHE_DIR
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export NEW_INTEREST_PARQUET_STAGE_DIR="/scratch/wang-stage-a-parquet-${MODEL_FAMILY}-${DATASET}-${CONDITION}"

MATRIX=configs/wang_stage_a/experiment_matrix_v1.yaml
RUN_ID="${MODEL_FAMILY}-${DATASET}-${CONDITION}-${MODE}"
CONFIG_ROOT="$AMLT_OUTPUT_DIR/configs"
SFT_DIR="$AMLT_OUTPUT_DIR/sft"
LOCAL_CHECKPOINT_DIR="/scratch/wang-stage-a-checkpoints-${RUN_ID}"
PERSISTENT_CHECKPOINT_DIR="$AMLT_OUTPUT_DIR/persistent_checkpoints"
SUCCESS_MARKER="$AMLT_OUTPUT_DIR/${MODE}_success.json"
mkdir -p "$AMLT_OUTPUT_DIR" "$CONFIG_ROOT" "$HF_HOME" "$PIP_CACHE_DIR" "$NEW_INTEREST_PARQUET_STAGE_DIR"

python -m new_interest.experiments.wang_stage_a --matrix "$MATRIX" audit \
  > "$AMLT_OUTPUT_DIR/input_audit.json"

python - "$MATRIX" "$MODEL_FAMILY" "$DATASET" "$CONDITION" "$MODE" "$REUSED_SFT_DIR" "$AMLT_OUTPUT_DIR/run_contract.json" <<'PY'
import json
import sys
from pathlib import Path

import yaml

matrix_path, model_key, dataset_key, condition, mode, reused_sft_dir, output_path = sys.argv[1:]
matrix = yaml.safe_load(Path(matrix_path).read_text())
model = matrix["models"][model_key]
dataset = matrix["datasets"][dataset_key]
payload = {
    "contract_version": "wang-stage-a-run-contract-v1",
    "scope": matrix["method"]["scope"],
    "model_family": model_key,
    "model_name": model["name"],
    "model_revision": model["revision"],
    "dataset": dataset_key,
    "condition": condition,
    "mode": mode,
    "train_rows": dataset["train_rows"],
    "train_sha256": dataset["train"][condition]["sha256"],
    "supported_labels": dataset["supported_labels"],
    "validation_rows": dataset["validation_supported"]["rows"],
    "validation_sha256": dataset["validation_supported"]["sha256"],
    "labels_sha256": dataset["labels_supported"]["sha256"],
    "response_prefix": matrix["method"]["response_prefix"],
    "reused_sft_dir": reused_sft_dir or None,
}
if "test_supported" in dataset:
    payload["test_rows"] = dataset["test_supported"]["rows"]
    payload["test_sha256"] = dataset["test_supported"]["sha256"]
Path(output_path).write_text(json.dumps(payload, indent=2) + "\n")
print("WANG_STAGE_A_RUN_CONTRACT_OK", json.dumps(payload, sort_keys=True))
PY

if [[ -f "$SUCCESS_MARKER" ]]; then
  python - "$SUCCESS_MARKER" "$AMLT_OUTPUT_DIR/run_contract.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

marker_path, contract_path = map(Path, sys.argv[1:])
marker = json.loads(marker_path.read_text())
contract = json.loads(contract_path.read_text())
if marker.get("contract") != contract:
    raise RuntimeError("Existing Wang Stage-A success marker has a different identity")
for relative, expected in marker["artifact_sha256"].items():
    path = marker_path.parent / relative
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Committed artifact hash mismatch: {relative}")
print("WANG_STAGE_A_REUSING_VALIDATED_RESULT", marker_path)
PY
  exit 0
fi

readarray -t MODEL_FIELDS < <(
  python - "$MATRIX" "$MODEL_FAMILY" <<'PY'
import sys
from pathlib import Path
import yaml
model = yaml.safe_load(Path(sys.argv[1]).read_text())["models"][sys.argv[2]]
print(model["name"])
print(model["revision"])
PY
)
MODEL_NAME=${MODEL_FIELDS[0]}
MODEL_REVISION=${MODEL_FIELDS[1]}

python - "$MODEL_NAME" "$MODEL_REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download

name, revision = sys.argv[1:]
snapshot = snapshot_download(name, revision=revision)
print("WANG_STAGE_A_MODEL_SNAPSHOT_READY", name, revision, snapshot)
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python - "$AMLT_OUTPUT_DIR/environment.json" "$MODEL_FAMILY" "$WANG_NUM_GPUS" <<'PY'
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import torch

output_path = Path(sys.argv[1])
family = sys.argv[2]
expected_devices = int(sys.argv[3])
expected = {
    "qwen35": {
        "transformers": "5.14.1",
        "peft": "0.18.0",
        "accelerate": "1.12.0",
    },
    "qwen2": {
        "transformers": "4.53.3",
        "peft": "0.16.0",
        "accelerate": "1.8.1",
    },
}[family]
actual = {name: importlib.metadata.version(name) for name in expected}
if actual != expected:
    raise RuntimeError(f"Worker dependency mismatch: {actual} != {expected}")
payload = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cuda_devices": torch.cuda.device_count(),
    "packages": actual,
}
if payload["cuda_devices"] != expected_devices:
    raise RuntimeError(
        f"Expected {expected_devices} GPUs, got {payload['cuda_devices']}"
    )
output_path.write_text(json.dumps(payload, indent=2) + "\n")
print("WANG_STAGE_A_ENVIRONMENT_OK", json.dumps(payload, sort_keys=True))
PY

ADAPTER_ARGS=()
if [[ "$MODE" != "zero-shot-validation" ]]; then
  TRAIN_CONFIG="$CONFIG_ROOT/train.yaml"
  python -m new_interest.experiments.wang_stage_a \
    --matrix "$MATRIX" write-training \
    --model "$MODEL_FAMILY" \
    --dataset "$DATASET" \
    --condition "$CONDITION" \
    --output-dir "$SFT_DIR" \
    --destination "$TRAIN_CONFIG"

  if [[ -n "$REUSED_SFT_DIR" ]]; then
    if [[ ! -f "$REUSED_SFT_DIR/final_adapter/adapter_model.safetensors" || ! -f "$REUSED_SFT_DIR/text_reload_reference.json" ]]; then
      echo "reused SFT directory is incomplete: $REUSED_SFT_DIR" >&2
      exit 2
    fi
    SFT_DIR="$REUSED_SFT_DIR"
  else
    TRAIN_ARGS=(
      --config "$TRAIN_CONFIG"
      --checkpoint-dir "$LOCAL_CHECKPOINT_DIR"
      --persistent-checkpoint-dir "$PERSISTENT_CHECKPOINT_DIR"
      --resume-from-checkpoint auto
    )
    if [[ "$MODE" == "smoke" ]]; then
      TRAIN_ARGS+=(--max-samples 128)
    fi
    CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.train_sft "${TRAIN_ARGS[@]}"
  fi

  TRAIN_FILE=$(python - "$TRAIN_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml
print(yaml.safe_load(Path(sys.argv[1]).read_text())["train_file"])
PY
)
  CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.text_adapter_reload \
    --model-name "$MODEL_NAME" \
    --model-revision "$MODEL_REVISION" \
    --adapter-path "$SFT_DIR/final_adapter" \
    --train-file "$TRAIN_FILE" \
    --reference-json "$SFT_DIR/text_reload_reference.json" \
    --output "$AMLT_OUTPUT_DIR/text_reload_verification.json"
  CUDA_VISIBLE_DEVICES=0 python -m new_interest.training.verify_lora_adapter \
    --model-name "$MODEL_NAME" \
    --model-revision "$MODEL_REVISION" \
    --adapter-path "$SFT_DIR/final_adapter" \
    --output "$AMLT_OUTPUT_DIR/adapter_state_verification.json"
  ADAPTER_ARGS=(--adapter-path "$SFT_DIR/final_adapter")
fi

run_view() {
  local split=$1
  local view=$2
  local suffix=$3
  local generation_config="$CONFIG_ROOT/generation-${suffix}.json"
  local ranking_config="$CONFIG_ROOT/ranking-${suffix}.json"
  local generation_output="$AMLT_OUTPUT_DIR/eval/generation-${suffix}"
  local ranking_output="$AMLT_OUTPUT_DIR/eval/ranking-${suffix}"
  python -m new_interest.experiments.wang_stage_a \
    --matrix "$MATRIX" write-generation \
    --model "$MODEL_FAMILY" \
    --dataset "$DATASET" \
    --split "$split" \
    --view "$view" \
    "${ADAPTER_ARGS[@]}" \
    --output-dir "$generation_output" \
    --destination "$generation_config"
  python -m new_interest.experiments.wang_stage_a \
    --matrix "$MATRIX" write-ranking \
    --model "$MODEL_FAMILY" \
    --dataset "$DATASET" \
    --split "$split" \
    --view "$view" \
    "${ADAPTER_ARGS[@]}" \
    --output-dir "$ranking_output" \
    --destination "$ranking_config"

  local limit_args=()
  if [[ "$MODE" == "smoke" ]]; then
    limit_args=(--max-examples 64)
  fi
  if [[ "$WANG_NUM_GPUS" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES=0 python -m new_interest.evaluation.evaluate_wang_stage_a_generation \
      --config "$generation_config" "${limit_args[@]}"
  else
    local generation_shard_root="$generation_output/logical_shards"
    mkdir -p "$generation_shard_root"
    local -a generation_pids=()
    local -a generation_shard_dirs=()
    local generation_shard_index
    for ((generation_shard_index = 0; generation_shard_index < WANG_NUM_GPUS; generation_shard_index++)); do
      local generation_shard_dir="$generation_shard_root/shard-${generation_shard_index}"
      generation_shard_dirs+=("$generation_shard_dir")
      CUDA_VISIBLE_DEVICES="$generation_shard_index" \
        python -m new_interest.evaluation.evaluate_wang_stage_a_generation \
        --config "$generation_config" \
        --num-shards "$WANG_NUM_GPUS" \
        --shard-index "$generation_shard_index" \
        --output-dir "$generation_shard_dir" \
        > "$generation_shard_root/shard-${generation_shard_index}.log" 2>&1 &
      generation_pids+=("$!")
    done
    local generation_failure=0
    local generation_pid
    for generation_pid in "${generation_pids[@]}"; do
      if ! wait "$generation_pid"; then
        generation_failure=1
      fi
    done
    if [[ "$generation_failure" -ne 0 ]]; then
      echo "one or more generation shards failed" >&2
      exit 1
    fi
    readarray -t generation_merge_fields < <(
      python - "$generation_config" <<'PY'
import json
import sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())
print(config["eval_file"])
print(config["labels_file"])
PY
    )
    python -m new_interest.evaluation.merge_wang_stage_a_generation_shards \
      --shard-dirs "${generation_shard_dirs[@]}" \
      --eval-file "${generation_merge_fields[0]}" \
      --labels-file "${generation_merge_fields[1]}" \
      --output-dir "$generation_output"
  fi
  if [[ "$WANG_NUM_GPUS" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES=0 python -m new_interest.evaluation.evaluate_text_label_ranking \
      --config "$ranking_config" "${limit_args[@]}"
  else
    local shard_root="$ranking_output/logical_shards"
    mkdir -p "$shard_root"
    local -a pids=()
    local -a shard_dirs=()
    local shard_index
    for ((shard_index = 0; shard_index < WANG_NUM_GPUS; shard_index++)); do
      local shard_dir="$shard_root/shard-${shard_index}"
      shard_dirs+=("$shard_dir")
      CUDA_VISIBLE_DEVICES="$shard_index" \
        python -m new_interest.evaluation.evaluate_text_label_ranking \
        --config "$ranking_config" \
        --num-shards "$WANG_NUM_GPUS" \
        --shard-index "$shard_index" \
        --output-dir "$shard_dir" \
        > "$shard_root/shard-${shard_index}.log" 2>&1 &
      pids+=("$!")
    done
    local any_failure=0
    local pid
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        any_failure=1
      fi
    done
    if [[ "$any_failure" -ne 0 ]]; then
      echo "one or more ranking shards failed" >&2
      exit 1
    fi
    readarray -t merge_fields < <(
      python - "$ranking_config" <<'PY'
import json
import sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())
print(config["eval_file"])
print(config["retained_interests_file"])
PY
    )
    python -m new_interest.evaluation.merge_text_label_ranking_shards \
      --shard-dirs "${shard_dirs[@]}" \
      --eval-file "${merge_fields[0]}" \
      --retained-interests-file "${merge_fields[1]}" \
      --output-dir "$ranking_output"
  fi
}

run_view validation supported supported
if [[ "$MODE" != "smoke" && "$DATASET" != "mind" ]]; then
  run_view validation full full
fi

if [[ "$MODE" == "full-validation" && "$CONDITION" == "balanced" && "$DATASET" != "mind" ]]; then
  python - "$AMLT_OUTPUT_DIR/eval/generation-supported/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text())
if metrics["match_rate"] < 0.90:
    raise RuntimeError(
        f"Validation controlled-generation gate failed: {metrics['match_rate']:.4f} < 0.90"
    )
print("WANG_STAGE_A_VALIDATION_GATE_OK", metrics["match_rate"])
PY
  run_view test supported test-supported
  run_view test full test-full
fi

python - "$AMLT_OUTPUT_DIR" "$MODE" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]
contract = json.loads((root / "run_contract.json").read_text())
generation = json.loads(
    (root / "eval/generation-supported/metrics.json").read_text()
)
ranking = json.loads((root / "eval/ranking-supported/metrics.json").read_text())
expected_eval_rows = 64 if mode == "smoke" else int(contract["validation_rows"])
if int(generation["samples"]) != expected_eval_rows:
    raise RuntimeError("Generation row-count gate failed")
if int(ranking["samples"]) != expected_eval_rows:
    raise RuntimeError("Ranking row-count gate failed")
for payload in (generation, ranking):
    for key, value in payload.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"Non-finite metric {key}")
if mode == "smoke" and generation["match_rate"] <= 0:
    raise RuntimeError("Smoke produced no valid predefined cluster")
if mode == "full-validation" and contract["condition"] == "balanced":
    if generation["match_rate"] < 0.90:
        raise RuntimeError(
            f"Controlled-generation gate failed: {generation['match_rate']:.4f} < 0.90"
        )

required = [
    "input_audit.json",
    "run_contract.json",
    "environment.json",
    "eval/generation-supported/metrics.json",
    "eval/generation-supported/generation_predictions.parquet",
    "eval/ranking-supported/metrics.json",
    "eval/ranking-supported/text_label_predictions.parquet",
]
if mode == "full-validation" and contract["condition"] == "balanced" and contract["dataset"] != "mind":
    required.extend(
        [
            "eval/generation-test-supported/metrics.json",
            "eval/generation-test-supported/generation_predictions.parquet",
            "eval/ranking-test-supported/metrics.json",
            "eval/ranking-test-supported/text_label_predictions.parquet",
            "eval/generation-test-full/metrics.json",
            "eval/generation-test-full/generation_predictions.parquet",
            "eval/ranking-test-full/metrics.json",
            "eval/ranking-test-full/text_label_predictions.parquet",
        ]
    )
if mode != "zero-shot-validation":
    required.extend(["text_reload_verification.json", "adapter_state_verification.json"])
    reused_sft = contract.get("reused_sft_dir")
    sft_root = Path(reused_sft) if reused_sft else root / "sft"
    if not reused_sft:
        required.extend(
            [
                "sft/run_summary.json",
                "sft/resolved_config.json",
                "sft/final_adapter/adapter_config.json",
                "sft/final_adapter/adapter_model.safetensors",
                "sft/text_reload_reference.json",
                "sft/text_reload_reference.safetensors",
            ]
        )
    summary = json.loads((sft_root / "run_summary.json").read_text())
    expected_train = 128 if mode == "smoke" else int(contract["train_rows"])
    if int(summary["train_samples"]) != expected_train:
        raise RuntimeError("Training row-count gate failed")
    losses = [float(row["loss"]) for row in summary["loss_history"]]
    if len(losses) < 2 or not all(math.isfinite(value) for value in losses):
        raise RuntimeError("Training loss-history gate failed")
    window = min(5, len(losses) // 2)
    if sum(losses[-window:]) / window >= sum(losses[:window]) / window:
        raise RuntimeError("Training loss did not decrease")
    reload_audit = json.loads((root / "text_reload_verification.json").read_text())
    if not reload_audit["passed"]:
        raise RuntimeError("Fresh adapter reload gate failed")

artifact_hashes = {}
for relative in required:
    path = root / relative
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Missing required artifact: {relative}")
    artifact_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
marker = {
    "marker_version": "wang-stage-a-success-v1",
    "contract": contract,
    "generation": {
        key: generation[key]
        for key in (
            "match_rate",
            "set_hit_at_1",
            "recall_at_1",
            "interest_macro_recall_at_1",
            "novelty_compliance_rate",
        )
    },
    "ranking": {
        key: ranking[key]
        for key in (
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "ndcg_at_10",
            "interest_macro_recall_at_10",
            "interest_macro_ndcg_at_10",
        )
    },
    "artifact_sha256": artifact_hashes,
}
if contract.get("reused_sft_dir"):
    reused_root = Path(contract["reused_sft_dir"])
    marker["reused_sft_sha256"] = {
        relative: hashlib.sha256((reused_root / relative).read_bytes()).hexdigest()
        for relative in (
            "run_summary.json",
            "final_adapter/adapter_model.safetensors",
            "text_reload_reference.json",
            "text_reload_reference.safetensors",
        )
    }
test_generation_path = root / "eval/generation-test-supported/metrics.json"
test_ranking_path = root / "eval/ranking-test-supported/metrics.json"
if test_generation_path.is_file() and test_ranking_path.is_file():
    test_generation = json.loads(test_generation_path.read_text())
    test_ranking = json.loads(test_ranking_path.read_text())
    if int(test_generation["samples"]) != int(contract["test_rows"]):
        raise RuntimeError("Test generation row-count gate failed")
    if int(test_ranking["samples"]) != int(contract["test_rows"]):
        raise RuntimeError("Test ranking row-count gate failed")
    marker["test_generation"] = {
        key: test_generation[key]
        for key in (
            "match_rate",
            "set_hit_at_1",
            "recall_at_1",
            "interest_macro_recall_at_1",
            "novelty_compliance_rate",
        )
    }
    marker["test_ranking"] = {
        key: test_ranking[key]
        for key in (
            "recall_at_1",
            "recall_at_5",
            "recall_at_10",
            "ndcg_at_10",
            "interest_macro_recall_at_10",
            "interest_macro_ndcg_at_10",
        )
    }
marker_path = root / f"{mode}_success.json"
temporary = marker_path.with_suffix(".tmp")
temporary.write_text(json.dumps(marker, indent=2) + "\n")
temporary.replace(marker_path)
print("WANG_STAGE_A_RUN_OK", json.dumps(marker, sort_keys=True))
PY
