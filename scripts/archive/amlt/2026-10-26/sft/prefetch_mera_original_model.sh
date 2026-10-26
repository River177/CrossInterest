#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/mnt/default/huggingface-cache}"
mkdir -p "$HF_HOME" "$AMLT_OUTPUT_DIR"
python -m pip install --user --no-warn-script-location \
  "huggingface-hub==0.22.2" "transformers==4.37.0" \
  "tokenizers==0.15.2" "sentencepiece==0.2.0" "safetensors==0.4.3"
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

name = "Qwen/Qwen2-1.5B"
revision = "8a16abf2848eda07cc5253dec660bf1ce007ad7a"
snapshot = Path(snapshot_download(name, revision=revision, cache_dir=os.environ["HF_HOME"]))
config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
index_file = snapshot / "model.safetensors.index.json"
if index_file.is_file():
    files = sorted(set(json.loads(index_file.read_text())["weight_map"].values()))
else:
    files = ["model.safetensors"]
for relative in files:
    path = snapshot / relative
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Incomplete model snapshot: {relative}")
payload = {
    "audit_version": "mera-original-model-prefetch-v1",
    "model": name,
    "revision": revision,
    "snapshot": str(snapshot),
    "model_type": config.model_type,
    "model_vocab_size": config.vocab_size,
    "tokenizer_size": len(tokenizer),
    "weight_files": files,
}
output = Path(os.environ["AMLT_OUTPUT_DIR"]) / "model_prefetch.json"
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("MERA_ORIGINAL_MODEL_PREFETCH_OK", json.dumps(payload, sort_keys=True))
PY
