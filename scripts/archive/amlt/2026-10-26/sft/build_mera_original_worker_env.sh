#!/usr/bin/env bash
set -euo pipefail

ENV_DIR="/scratch/mera-original-worker-venv"
STAGING_REQUIREMENTS="/scratch/mera-original-worker-requirements.txt"
PUBLISH_ROOT="/mnt/default/mera-original-worker-env/torch230-cu121-v1"
ARCHIVE="$PUBLISH_ROOT/worker-env.tar"
MARKER="$PUBLISH_ROOT/worker-env.json"

mkdir -p "$PUBLISH_ROOT" "$AMLT_OUTPUT_DIR"
if [[ -f "$ARCHIVE" && -f "$MARKER" ]]; then
  python - "$ARCHIVE" "$MARKER" <<'PY'
import hashlib, json, sys
from pathlib import Path
archive, marker = map(Path, sys.argv[1:])
payload = json.loads(marker.read_text())
digest = hashlib.sha256()
with archive.open("rb") as handle:
    for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != payload["sha256"]:
    raise RuntimeError("Committed worker environment hash mismatch")
print("MERA_ORIGINAL_WORKER_ENV_REUSE_OK", payload["sha256"])
PY
  cp "$MARKER" "$AMLT_OUTPUT_DIR/worker-env.json"
  exit 0
fi

python -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.3.0"
grep -v '^bitsandbytes==' requirements-mera-original-worker.txt > "$STAGING_REQUIREMENTS"
python -m pip install -r "$STAGING_REQUIREMENTS"
python -m pip install "bitsandbytes==0.43.1"
python - <<'PY'
import importlib.metadata
import accelerate, datasets, peft, torch, transformers
expected = {
    "torch": "2.3.0+cu121",
    "transformers": "4.37.0",
    "peft": "0.9.0",
    "accelerate": "0.29.3",
    "datasets": "2.19.0",
    "bitsandbytes": "0.43.1",
}
actual = {
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "accelerate": accelerate.__version__,
    "datasets": datasets.__version__,
    "bitsandbytes": importlib.metadata.version("bitsandbytes"),
}
if actual != expected:
    raise RuntimeError(actual)
print("MERA_ORIGINAL_WORKER_ENV_BUILD_OK", actual)
PY

LOCAL_ARCHIVE="/scratch/worker-env.tar"
tar -C /scratch -cf "$LOCAL_ARCHIVE" mera-original-worker-venv
LOCAL_SHA="$(sha256sum "$LOCAL_ARCHIVE" | awk '{print $1}')"
LOCAL_SIZE="$(stat -c %s "$LOCAL_ARCHIVE")"
TEMP_ARCHIVE="$PUBLISH_ROOT/.worker-env.tar.tmp"
dd if="$LOCAL_ARCHIVE" of="$TEMP_ARCHIVE" bs=16M status=progress
mv "$TEMP_ARCHIVE" "$ARCHIVE"
python - "$ARCHIVE" "$MARKER" "$LOCAL_SHA" "$LOCAL_SIZE" <<'PY'
import json, sys
from pathlib import Path
archive, marker = map(Path, sys.argv[1:3])
payload = {
    "environment_version": "mera-original-worker-env-v1",
    "archive": str(archive),
    "sha256": sys.argv[3],
    "size_bytes": int(sys.argv[4]),
    "torch": "2.3.0+cu121",
    "transformers": "4.37.0",
    "peft": "0.9.0",
    "accelerate": "0.29.3",
    "datasets": "2.19.0",
    "bitsandbytes": "0.43.1",
}
marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("MERA_ORIGINAL_WORKER_ENV_PUBLISH_OK", json.dumps(payload, sort_keys=True))
PY
cp "$MARKER" "$AMLT_OUTPUT_DIR/worker-env.json"
