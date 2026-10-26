"""Check Git candidates for large outputs, credentials, and broken archive roots.

Read-only: this does not stage files, install a hook, submit jobs, or fetch data.
Run from any directory with ``python /path/to/scripts/check_repository.py``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOTS = {"data", "artifacts", "amlt_inputs", "outputs", "tmp", ".codex_tmp", ".vscode", ".venv"}
MODEL_OR_DATA_SUFFIXES = {".parquet", ".safetensors", ".pt", ".pth", ".bin"}
SECRET = re.compile(
    rb"(?:hf_[A-Za-z0-9]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{32,}|"
    rb"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|"
    rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|"
    rb"[?&]sig=[A-Za-z0-9%/+]{20,})"
)


def check_files() -> tuple[int, list[str]]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
    )
    paths = sorted({Path(name.decode()) for name in output.split(b"\0") if name})
    errors = []
    count = 0
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            continue  # A staged/unstaged deletion is not an upload candidate.
        count += 1
        if relative.parts[0] in LOCAL_ROOTS or path.suffix in MODEL_OR_DATA_SUFFIXES:
            errors.append(f"Local artifact would enter Git: {relative}")
        if path.stat().st_size > 10 * 1024**2:
            errors.append(f"File exceeds 10 MiB: {relative}")
            continue
        data = path.read_bytes()
        if b"\0" not in data and SECRET.search(data):
            errors.append(f"Possible credential or signed URL: {relative}")
    return count, errors


def check_archive() -> list[str]:
    index = json.loads((ROOT / "manifests/repository_archive_20261026.json").read_text())
    errors = []
    for entry in index["moves"]:
        old, new = ROOT / entry["old"], ROOT / entry["new"]
        if old.exists() or not new.is_file():
            errors.append(f"Archive move is inconsistent: {entry['old']}")
            continue
        if hashlib.sha256(new.read_bytes()).hexdigest() != entry["after_sha256"]:
            errors.append(f"Archived file changed without updating its receipt: {entry['new']}")
        if entry["kind"] == "amlt":
            # The recorded local_dir is checked against the YAML bytes without
            # requiring a training environment or a YAML parser.
            if entry["local_dir_after"] not in new.read_text():
                errors.append(f"Archived code.local_dir changed: {entry['new']}")
            suffix = entry["local_dir_after"].removeprefix("$CONFIG_DIR/")
            if (new.parent / suffix).resolve() != (ROOT / entry["code_root"]).resolve():
                errors.append(f"Archived AMLT code root moved: {entry['new']}")
    return errors


def main() -> int:
    count, errors = check_files()
    errors += check_archive()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"PASS: {count} Git candidates; no large data/model outputs or credential patterns; archive roots preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
