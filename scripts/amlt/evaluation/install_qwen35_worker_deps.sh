#!/usr/bin/env bash
set -euo pipefail

python -m pip install \
  --user \
  --no-warn-script-location \
  --retries 20 \
  --timeout 120 \
  -r requirements-qwen35-worker.txt

python - <<'PY'
from importlib.metadata import version

expected = {
    "accelerate": "1.12.0",
    "peft": "0.18.0",
    "pyarrow": "20.0.0",
    "PyYAML": "6.0.2",
    "sentencepiece": "0.2.0",
    "transformers": "5.14.1",
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
    "Verified Qwen3.5 worker dependencies:",
    torch.__version__,
    transformers.__version__,
    peft.__version__,
    pyarrow.__version__,
    accelerate.__version__,
    wandb.__version__,
)
PY
