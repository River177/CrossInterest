#!/usr/bin/env bash
set -euo pipefail

python -m pip install \
  --user \
  --no-warn-script-location \
  --retries 20 \
  --timeout 120 \
  -r requirements-semantic-id-token-llama-worker.txt

python - <<'PY'
from importlib.metadata import version

expected = {
    "accelerate": "1.8.1",
    "huggingface_hub": "0.36.2",
    "peft": "0.18.0",
    "pyarrow": "20.0.0",
    "PyYAML": "6.0.2",
    "safetensors": "0.5.3",
    "sentencepiece": "0.2.0",
    "tokenizers": "0.21.4",
    "transformers": "4.53.3",
    "wandb": "0.20.1",
}
actual = {package: version(package) for package in expected}
mismatches = {
    package: {"expected": expected[package], "actual": actual[package]}
    for package in expected
    if actual[package] != expected[package]
}
if mismatches:
    raise RuntimeError(f"Dependency version mismatch: {mismatches}")

import accelerate
import peft
import pyarrow
import torch
import transformers
import wandb

print(
    "Verified Semantic ID Llama worker dependencies:",
    torch.__version__,
    transformers.__version__,
    peft.__version__,
    pyarrow.__version__,
    accelerate.__version__,
    wandb.__version__,
)
PY
