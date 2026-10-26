#!/usr/bin/env bash
set -euo pipefail

MODEL_FAMILY="${1:?Usage: run_semantic_id_token_reload_diagnostic.sh <qwen35|llama31>}"
DIAGNOSTIC_MODE="${2:-full}"

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

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HOME="/scratch/semantic-id-${SHORT_MODEL}-diagnostic-hf-cache"
mkdir -p "$HF_HOME"

python - "$MODEL_NAME" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

name = sys.argv[1]
snapshot = snapshot_download(name, token=os.environ.get("HF_TOKEN"))
print("SEMANTIC_ID_DIAGNOSTIC_MODEL_READY", name, snapshot)
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ -n "${SEMANTIC_ID_SOURCE_SINGLE:-}" ]]; then
  SOURCE_SINGLE="$SEMANTIC_ID_SOURCE_SINGLE"
else
  SOURCE_SINGLE="${AMLT_MAP_INPUT_DIR:?AMLT_MAP_INPUT_DIR is required}/smoke/single"
fi
ADAPTER_PATH="$SOURCE_SINGLE/final_adapter"
MAPPING_FILE="$ADAPTER_PATH/interest_token_mapping.parquet"
PROMPT_REFERENCE="$SOURCE_SINGLE/semantic_reload_reference.json"
OUTPUT_DIR="$AMLT_OUTPUT_DIR/reload_diagnostic"
mkdir -p "$OUTPUT_DIR"

for path in \
  "$ADAPTER_PATH/adapter_model.safetensors" \
  "$MAPPING_FILE" \
  "$PROMPT_REFERENCE"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing mapped diagnostic input: $path" >&2
    exit 1
  fi
done

if [[ "$DIAGNOSTIC_MODE" == "training-reference-minimal" ]]; then
  python -m new_interest.training.diagnose_semantic_id_token_reload capture \
    --model-name "$MODEL_NAME" \
    --adapter-path "$ADAPTER_PATH" \
    --mapping-file "$MAPPING_FILE" \
    --prompt-reference-file "$PROMPT_REFERENCE" \
    --inference-dtype bfloat16 \
    --max-samples 1 \
    --output "$OUTPUT_DIR/bfloat16_reload.json"
  python -m new_interest.training.diagnose_semantic_id_token_reload \
    compare-reference \
    --reference "$PROMPT_REFERENCE" \
    --capture "$OUTPUT_DIR/bfloat16_reload.json" \
    --output "$OUTPUT_DIR/training_reference_comparison.json"
  python - "$OUTPUT_DIR/training_reference_comparison.json" <<'PY'
import json
import sys
from pathlib import Path

comparison = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not comparison["all_close"]:
    print(
        "SEMANTIC_ID_TRAINING_REFERENCE_REPRODUCED",
        json.dumps(comparison["samples"], sort_keys=True),
    )
    raise RuntimeError("Fresh reload does not reproduce training reference")
print("SEMANTIC_ID_TRAINING_REFERENCE_MATCH_OK")
PY
  exit 0
fi

if [[ "$DIAGNOSTIC_MODE" != "full" ]]; then
  echo "Unsupported diagnostic mode: $DIAGNOSTIC_MODE" >&2
  exit 2
fi

for dtype in bfloat16 float32; do
  for replicate in a b; do
    python -m new_interest.training.diagnose_semantic_id_token_reload capture \
      --model-name "$MODEL_NAME" \
      --adapter-path "$ADAPTER_PATH" \
      --mapping-file "$MAPPING_FILE" \
      --prompt-reference-file "$PROMPT_REFERENCE" \
      --inference-dtype "$dtype" \
      --output "$OUTPUT_DIR/${dtype}_${replicate}.json"
  done
  python -m new_interest.training.diagnose_semantic_id_token_reload compare \
    --left "$OUTPUT_DIR/${dtype}_a.json" \
    --right "$OUTPUT_DIR/${dtype}_b.json" \
    --output "$OUTPUT_DIR/${dtype}_comparison.json"
done

python - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
bf16 = json.loads((root / "bfloat16_comparison.json").read_text())
fp32 = json.loads((root / "float32_comparison.json").read_text())
if not bf16["all_close"]:
    raise RuntimeError("Independent BF16 reloads are not reproducible")
if not fp32["all_close"]:
    raise RuntimeError("FP32 differential still mismatches")
print(
    "SEMANTIC_ID_RELOAD_DIAGNOSTIC_OK",
    "bf16_all_close=true",
    "fp32_all_close=true",
)
PY
