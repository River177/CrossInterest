#!/usr/bin/env bash
set -euo pipefail

MODEL_FAMILY=${1:?usage: run_wang_stage_a_top20.sh <qwen35|qwen2> <arts|grocery|mind>}
DATASET=${2:?missing dataset}
case "$MODEL_FAMILY" in
  qwen35|qwen2) ;;
  *) echo "unsupported model family: $MODEL_FAMILY" >&2; exit 2 ;;
esac
case "$DATASET" in
  arts|grocery|mind) ;;
  *) echo "unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

: "${AMLT_OUTPUT_DIR:?AMLT_OUTPUT_DIR is required}"
: "${WANG_REUSE_SFT_DIR:?WANG_REUSE_SFT_DIR is required}"
: "${WANG_EXPECTED_ADAPTER_SHA256:?WANG_EXPECTED_ADAPTER_SHA256 is required}"
: "${WANG_NUM_GPUS:=1}"
: "${WANG_MAX_EXAMPLES:=}"
: "${HF_HOME:=/mnt/default/huggingface-cache}"
: "${PIP_CACHE_DIR:=/mnt/default/pip-cache-wang-stage-a-${MODEL_FAMILY}}"
if [[ ! "$WANG_NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "WANG_NUM_GPUS must be a positive integer" >&2
  exit 2
fi
if [[ -n "$WANG_MAX_EXAMPLES" && ! "$WANG_MAX_EXAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "WANG_MAX_EXAMPLES must be empty or a positive integer" >&2
  exit 2
fi

export HF_HOME PIP_CACHE_DIR
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export NEW_INTEREST_PARQUET_STAGE_DIR="/scratch/wang-stage-a-top20-${MODEL_FAMILY}-${DATASET}"

MATRIX=configs/wang_stage_a/experiment_matrix_v1.yaml
ADAPTER_PATH="$WANG_REUSE_SFT_DIR/final_adapter"
CONFIG_DIR="$AMLT_OUTPUT_DIR/configs"
RANKING_DIR="$AMLT_OUTPUT_DIR/eval/ranking-primary-top20"
SUCCESS_MARKER="$AMLT_OUTPUT_DIR/top20_success.json"
mkdir -p "$AMLT_OUTPUT_DIR" "$CONFIG_DIR" "$RANKING_DIR" "$HF_HOME" "$PIP_CACHE_DIR" "$NEW_INTEREST_PARQUET_STAGE_DIR"

python -m new_interest.experiments.wang_stage_a --matrix "$MATRIX" audit \
  > "$AMLT_OUTPUT_DIR/input_audit.json"

if [[ ! -f "$ADAPTER_PATH/adapter_model.safetensors" || ! -f "$ADAPTER_PATH/adapter_config.json" ]]; then
  echo "reused adapter is incomplete: $ADAPTER_PATH" >&2
  exit 2
fi
ACTUAL_ADAPTER_SHA256=$(python - "$ADAPTER_PATH" <<'PY'
import sys
from pathlib import Path
from new_interest.evaluation.evaluate_beam import sha256_adapter_semantics
print(sha256_adapter_semantics(Path(sys.argv[1])))
PY
)
if [[ "$ACTUAL_ADAPTER_SHA256" != "$WANG_EXPECTED_ADAPTER_SHA256" ]]; then
  echo "reused adapter semantic hash mismatch" >&2
  exit 2
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

if [[ "$DATASET" == "mind" ]]; then
  SPLIT=validation
else
  SPLIT=test
fi

RANKING_CONFIG="$CONFIG_DIR/ranking-primary-top20.json"
python -m new_interest.experiments.wang_stage_a \
  --matrix "$MATRIX" write-ranking \
  --model "$MODEL_FAMILY" \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --view supported \
  --adapter-path "$ADAPTER_PATH" \
  --top-k 20 \
  --cutoffs 1 5 10 20 \
  --output-dir "$RANKING_DIR" \
  --destination "$RANKING_CONFIG"

python - "$RANKING_CONFIG" "$MODEL_FAMILY" "$DATASET" "$ACTUAL_ADAPTER_SHA256" "$WANG_MAX_EXAMPLES" "$AMLT_OUTPUT_DIR/run_contract.json" <<'PY'
import json
import sys
from pathlib import Path

config_path, family, dataset, adapter_sha256, max_examples, output_path = sys.argv[1:]
config = json.loads(Path(config_path).read_text())
payload = {
    "contract_version": "wang-stage-a-top20-eval-v1",
    "scope": "Stage A primary held-out ranking only",
    "model_family": family,
    "model_name": config["model_name"],
    "model_revision": config["model_revision"],
    "dataset": dataset,
    "split": "validation" if dataset == "mind" else "test",
    "view": "supported",
    "adapter_sha256": adapter_sha256,
    "eval_file": config["eval_file"],
    "labels_file": config["retained_interests_file"],
    "top_k": config["top_k"],
    "cutoffs": config["cutoffs"],
    "max_examples": int(max_examples) if max_examples else None,
}
Path(output_path).write_text(json.dumps(payload, indent=2) + "\n")
print("WANG_STAGE_A_TOP20_CONTRACT_OK", json.dumps(payload, sort_keys=True))
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
if marker["contract"] != contract:
    raise RuntimeError("Existing Top-20 marker has a different identity")
for relative, expected in marker["artifact_sha256"].items():
    actual = hashlib.sha256((marker_path.parent / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Committed Top-20 artifact hash mismatch: {relative}")
print("WANG_STAGE_A_TOP20_REUSING_VALIDATED_RESULT", marker_path)
PY
  exit 0
fi

python - "$MODEL_NAME" "$MODEL_REVISION" <<'PY'
import sys
from huggingface_hub import snapshot_download

name, revision = sys.argv[1:]
snapshot = snapshot_download(name, revision=revision)
print("WANG_STAGE_A_TOP20_MODEL_READY", name, revision, snapshot)
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
    "qwen35": {"transformers": "5.14.1", "peft": "0.18.0", "accelerate": "1.12.0"},
    "qwen2": {"transformers": "4.53.3", "peft": "0.16.0", "accelerate": "1.8.1"},
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
    raise RuntimeError(f"Expected {expected_devices} GPUs, got {payload['cuda_devices']}")
output_path.write_text(json.dumps(payload, indent=2) + "\n")
print("WANG_STAGE_A_TOP20_ENVIRONMENT_OK", json.dumps(payload, sort_keys=True))
PY

SHARD_ROOT="$RANKING_DIR/logical_shards"
mkdir -p "$SHARD_ROOT"
declare -a PIDS=()
declare -a SHARD_DIRS=()
for ((SHARD_INDEX = 0; SHARD_INDEX < WANG_NUM_GPUS; SHARD_INDEX++)); do
  SHARD_DIR="$SHARD_ROOT/shard-${SHARD_INDEX}"
  SHARD_DIRS+=("$SHARD_DIR")
  EXTRA_ARGS=()
  if [[ -n "$WANG_MAX_EXAMPLES" ]]; then
    EXTRA_ARGS+=(--max-examples "$WANG_MAX_EXAMPLES")
  fi
  CUDA_VISIBLE_DEVICES="$SHARD_INDEX" \
    python -m new_interest.evaluation.evaluate_text_label_ranking \
    --config "$RANKING_CONFIG" \
    --num-shards "$WANG_NUM_GPUS" \
    --shard-index "$SHARD_INDEX" \
    --output-dir "$SHARD_DIR" \
    "${EXTRA_ARGS[@]}" \
    > "$SHARD_ROOT/shard-${SHARD_INDEX}.log" 2>&1 &
  PIDS+=("$!")
done
ANY_FAILURE=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    ANY_FAILURE=1
  fi
done
if [[ "$ANY_FAILURE" -ne 0 ]]; then
  echo "one or more Top-20 ranking shards failed" >&2
  exit 1
fi

readarray -t MERGE_FIELDS < <(
  python - "$RANKING_CONFIG" <<'PY'
import json
import sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())
print(config["eval_file"])
print(config["retained_interests_file"])
PY
)
MERGE_ARGS=()
if [[ -n "$WANG_MAX_EXAMPLES" ]]; then
  MERGE_ARGS+=(--expected-records "$WANG_MAX_EXAMPLES")
fi
python -m new_interest.evaluation.merge_text_label_ranking_shards \
  --shard-dirs "${SHARD_DIRS[@]}" \
  --eval-file "${MERGE_FIELDS[0]}" \
  --retained-interests-file "${MERGE_FIELDS[1]}" \
  --output-dir "$RANKING_DIR" \
  "${MERGE_ARGS[@]}"

python - "$AMLT_OUTPUT_DIR" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
contract = json.loads((root / "run_contract.json").read_text())
metrics = json.loads((root / "eval/ranking-primary-top20/metrics.json").read_text())
expected_samples = contract["max_examples"] or metrics["full_record_count"]
if metrics["samples"] != expected_samples:
    raise RuntimeError("Top-20 row count mismatch")
if metrics["top_k"] != 20 or metrics["cutoffs"] != [1, 5, 10, 20]:
    raise RuntimeError("Top-20 metric contract mismatch")
for name in ("interest_macro_recall_at_10", "interest_macro_ndcg_at_10", "interest_macro_recall_at_20", "interest_macro_ndcg_at_20"):
    if not math.isfinite(float(metrics[name])):
        raise RuntimeError(f"Non-finite metric {name}")
required = [
    "input_audit.json",
    "run_contract.json",
    "environment.json",
    "configs/ranking-primary-top20.json",
    "eval/ranking-primary-top20/metrics.json",
    "eval/ranking-primary-top20/text_label_predictions.parquet",
    "eval/ranking-primary-top20/per_interest_metrics.parquet",
    "eval/ranking-primary-top20/merge_manifest.json",
]
artifact_hashes = {}
for relative in required:
    path = root / relative
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Missing Top-20 artifact: {relative}")
    artifact_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
marker = {
    "marker_version": "wang-stage-a-top20-success-v1",
    "contract": contract,
    "metrics": {
        key: metrics[key]
        for key in (
            "samples",
            "recall_at_10",
            "ndcg_at_10",
            "recall_at_20",
            "ndcg_at_20",
            "interest_macro_recall_at_10",
            "interest_macro_ndcg_at_10",
            "interest_macro_recall_at_20",
            "interest_macro_ndcg_at_20",
        )
    },
    "artifact_sha256": artifact_hashes,
}
temporary = root / "top20_success.tmp"
temporary.write_text(json.dumps(marker, indent=2) + "\n")
temporary.replace(root / "top20_success.json")
print("WANG_STAGE_A_TOP20_RUN_OK", json.dumps(marker, sort_keys=True))
PY
