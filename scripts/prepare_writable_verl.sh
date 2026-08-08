#!/usr/bin/env bash
set -euo pipefail

RUNTIME_DIR="${VERL_RUNTIME_DIR:-$HOME/verl-runtime}"
ARCHIVE="${VERL_ARCHIVE:-${PROJECT_DIR:?PROJECT_DIR is required}/vendor/verl-safegrpo-0.7.0.dev.zip}"

if [[ ! -f "$RUNTIME_DIR/verl/version/version" ]]; then
  rm -rf "$RUNTIME_DIR"
  mkdir -p "$RUNTIME_DIR"
  python - "$ARCHIVE" "$RUNTIME_DIR" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as archive:
    archive.extractall(sys.argv[2])
PY
fi

python - "$RUNTIME_DIR/verl/workers/fsdp_workers.py" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
source = (
    'local_path, trust_remote_code=trust_remote_code, '
    'attn_implementation="flash_attention_2"'
)
replacement = (
    'local_path, trust_remote_code=trust_remote_code, '
    'attn_implementation=override_model_config.get('
    '"attn_implementation", "flash_attention_2")'
)
text = path.read_text(encoding="utf-8")
if source in text:
    path.write_text(text.replace(source, replacement, 1), encoding="utf-8")
elif replacement not in text:
    raise RuntimeError("Unexpected veRL FSDP attention-loading code")
PY

export PYTHONPATH="$RUNTIME_DIR${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
import verl

print("Vendored veRL runtime:", verl.__version__, verl.__file__)
PY
