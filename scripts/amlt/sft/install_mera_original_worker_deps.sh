#!/usr/bin/env bash
set -euo pipefail

python -m pip install --user --upgrade pip
python -m pip install --user --no-warn-script-location \
  --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.3.0"
python -m pip install --user --no-warn-script-location \
  -r requirements-mera-original-worker.txt

if ! python - <<'PY'
import bitsandbytes as bnb
import torch
if bnb.__version__ != "0.37.2":
    raise RuntimeError(bnb.__version__)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")
layer = bnb.nn.Linear8bitLt(8, 8).cuda()
_ = layer(torch.zeros(1, 8, device="cuda", dtype=torch.float16))
print("MERA_ORIGINAL_BNB_0372_OK", bnb.__version__)
PY
then
  python -m pip install --user --no-warn-script-location --upgrade \
    "bitsandbytes==0.43.1"
  export MERA_BNB_FALLBACK=1
fi

python - <<'PY'
import accelerate
import bitsandbytes
import datasets
import peft
import torch
import transformers

expected = {
    "torch": "2.3.0+cu121",
    "transformers": "4.37.0",
    "peft": "0.9.0",
    "accelerate": "0.29.3",
    "datasets": "2.19.0",
}
actual = {
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "accelerate": accelerate.__version__,
    "datasets": datasets.__version__,
}
if actual != expected:
    raise RuntimeError(f"MeRA worker version mismatch: {actual}")
if bitsandbytes.__version__ not in {"0.37.2", "0.43.1"}:
    raise RuntimeError(f"Unexpected bitsandbytes: {bitsandbytes.__version__}")
print("MERA_ORIGINAL_WORKER_DEPS_OK", actual, bitsandbytes.__version__)
PY
