#!/usr/bin/env bash
set -euo pipefail

if [[ "${AMLT_JOB_NAME:?AMLT_JOB_NAME is required}" != "qwen35-bi-reasoning-stability-preflight" ]]; then
  echo "Unexpected job name: $AMLT_JOB_NAME" >&2
  exit 2
fi

export PYTHONPATH="$AMLT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-/mnt/default/huggingface-cache}"
mkdir -p "$HF_HOME" "$AMLT_OUTPUT_DIR"

MODEL_NAME="Qwen/Qwen3.5-9B"
MAX_NEW_TOKENS="${REASONING_STABILITY_MAX_NEW_TOKENS:-192}"
if [[ "$MAX_NEW_TOKENS" != "192" && "$MAX_NEW_TOKENS" != "384" ]]; then
  echo "Unsupported reasoning-stability generation length: $MAX_NEW_TOKENS" >&2
  exit 2
fi
ADAPTER_PATH="/mnt/output/projects/sft4safety/amlt-results/7212029499.70786-e4538499-b9a2/sft/final_adapter"
EXPECTED_ADAPTER_SHA256="b1d698724a90d929f28973433ba41ca8b95f2a5dc15523b4aead627e3c901f2f"
DATA_ROOT="$AMLT_CODE_DIR/data/training/sft_static_user_disjoint_semantic_id_token_200_h50_behavior_interest_scaling_v1"
EVAL_FILE="$DATA_ROOT/eval/behavior_interest/dev_full_grouped.parquet"
MAPPING_FILE="$DATA_ROOT/interest_token_mapping.parquet"

python - "$ADAPTER_PATH" "$EXPECTED_ADAPTER_SHA256" "$EVAL_FILE" "$MAPPING_FILE" "$MAX_NEW_TOKENS" "$AMLT_OUTPUT_DIR/input_audit.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

adapter = Path(sys.argv[1])
expected_adapter = sys.argv[2]
eval_file = Path(sys.argv[3])
mapping_file = Path(sys.argv[4])
max_new_tokens = int(sys.argv[5])
output_file = Path(sys.argv[6])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

required = [
    adapter / "adapter_model.safetensors",
    adapter / "adapter_config.json",
    adapter / "interest_token_mapping.parquet",
    adapter / "tokenizer.json",
    eval_file,
    mapping_file,
]
for path in required:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing reasoning-stability input: {path}")
if sha256(adapter / "adapter_model.safetensors") != expected_adapter:
    raise RuntimeError("Vanilla adapter hash mismatch")
if sha256(mapping_file) != "45397451ac2ae2e64d8156b2b47be3f454cc309a8d30a83bc23c2d3267653a0a":
    raise RuntimeError("Mapping hash mismatch")
if sha256(eval_file) != "c05b20033ce0d1a0c2d4b7bb97ca0cd6b7cffb7f198a7265cce9e54046b665fd":
    raise RuntimeError("Dev hash mismatch")
if pq.ParquetFile(eval_file).metadata.num_rows != 31392:
    raise RuntimeError("Dev row count mismatch")
payload = {
    "audit_version": "reasoning-stability-input-v1",
    "model_name": "Qwen/Qwen3.5-9B",
    "adapter_path": str(adapter),
    "adapter_sha256": expected_adapter,
    "eval_sha256": sha256(eval_file),
    "mapping_sha256": sha256(mapping_file),
    "eval_rows": 31392,
    "selection": "16 deterministic single-positive user-disjoint Dev rows",
    "rollouts_per_row": 8,
    "max_new_tokens": max_new_tokens,
}
output_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("REASONING_STABILITY_INPUT_OK", json.dumps(payload, sort_keys=True))
PY

python - "$MODEL_NAME" <<'PY'
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

root = Path(snapshot_download(sys.argv[1]))
index = root / "model.safetensors.index.json"
if not index.is_file():
    raise RuntimeError("Qwen snapshot has no index")
files = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
if any(not (root / name).is_file() for name in files):
    raise RuntimeError("Qwen snapshot is incomplete")
AutoConfig.from_pretrained(root, local_files_only=True)
AutoTokenizer.from_pretrained(root, local_files_only=True)
print("REASONING_STABILITY_MODEL_READY", root)
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=0 python -m new_interest.experiments.evaluate_reasoning_stability \
  --model-name "$MODEL_NAME" \
  --adapter-path "$ADAPTER_PATH" \
  --eval-file "$EVAL_FILE" \
  --mapping-file "$MAPPING_FILE" \
  --output-dir "$AMLT_OUTPUT_DIR/stability" \
  --max-examples 16 \
  --rollouts 8 \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature 0.8 \
  --top-p 0.95 \
  --seed 42

python - "$AMLT_OUTPUT_DIR" "$MAX_NEW_TOKENS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

root = Path(sys.argv[1])
expected_max_new_tokens = int(sys.argv[2])
summary_file = root / "stability/summary.json"
predictions_file = root / "stability/reasoning_stability.parquet"
summary = json.loads(summary_file.read_text(encoding="utf-8"))
rows = pq.read_table(predictions_file).to_pylist()
if summary["max_examples"] != 16 or summary["rollouts_per_example"] != 8:
    raise RuntimeError("Reasoning-stability scope mismatch")
if summary["sampling"]["max_new_tokens"] != expected_max_new_tokens:
    raise RuntimeError("Reasoning-stability generation-length mismatch")
if len(rows) != 16 or summary["total_rollouts"] != 128:
    raise RuntimeError("Reasoning-stability output count mismatch")
gate = {
    "valid_rate_at_least_075": summary["valid_rate"] >= 0.75,
    "at_least_one_correct_rollout": summary["correct_rollouts"] >= 1,
    "at_least_one_mixed_group": summary["mixed_correctness_groups"] >= 1,
}
passed = all(gate.values())

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest = {
    "manifest_version": "reasoning-stability-preflight-v1",
    "identity": {
        "model": summary["model_name"],
        "adapter_sha256": summary["adapter_sha256"],
        "eval_sha256": summary["eval_sha256"],
        "rows": 16,
        "rollouts_per_row": 8,
        "sampling": summary["sampling"],
    },
    "gate": gate,
    "passed": passed,
    "files": {
        "input_audit.json": sha256(root / "input_audit.json"),
        "stability/summary.json": sha256(summary_file),
        "stability/reasoning_stability.parquet": sha256(predictions_file),
    },
}
manifest_file = root / "artifact_manifest.json"
manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
(root / "preflight_result.json").write_text(
    json.dumps(
        {
            "passed": passed,
            "gate": gate,
            "artifact_manifest_sha256": sha256(manifest_file),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print("REASONING_STABILITY_PREFLIGHT_RESULT", json.dumps(manifest["gate"], sort_keys=True), passed)
if not passed:
    raise SystemExit(2)
PY
