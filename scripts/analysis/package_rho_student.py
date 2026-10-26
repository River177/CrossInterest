"""Seal the approved student study without packaging remote weights or scores."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(output, gate_acceptance=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    previous = REPO / "artifacts/rho-twofold-reference-20260915/bundle-stable-rescore-v1"
    for source in sorted((REPO / "src").rglob("*.py")):
        destination = output / source.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    names = ["scripts/amlt/sft/run_rho_student.sh", "scripts/amlt/common/install_qwen35_fast_kernels.sh",
             "scripts/amlt/common/install_qwen35_fast_kernels.py", "configs/experiments/qwen35_fast_kernel_install_v1.json"]
    for name in names:
        source = REPO / name
        if not source.exists():
            source = previous / name
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for name in ("requirements-worker.txt", "source_snapshot.json"):
        shutil.copyfile(previous / name, output / name)
    shutil.copyfile(REPO / "configs/experiments/mind_rho_student_v1.json", output / "protocol.json")
    shutil.copytree(REPO / "artifacts/rho-student-20260916/inputs-v1", output / "inputs")
    if gate_acceptance is not None:
        shutil.copyfile(gate_acceptance, output / "gate_acceptance.json")
    inventory = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    if any(name.endswith((".safetensors", ".pt", ".bin")) for name in inventory):
        raise ValueError("Large model files must remain remote")
    runtime = {k: v for k, v in inventory.items() if k.startswith(("src/", "scripts/", "configs/")) or k == "requirements-worker.txt"}
    runtime_sha = hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest()
    if gate_acceptance is not None:
        gate = json.loads(Path(gate_acceptance).read_text())
        if (gate.get("passed") is not True or gate["identity"]["runtime_sha256"] != runtime_sha
                or gate["identity"]["protocol_sha256"] != inventory["protocol.json"]):
            raise ValueError("Formal bundle changed from its accepted gate")
    manifest = {"runtime_sha256": runtime_sha, "files": inventory,
                "source_file_count": len(inventory), "source_bytes": sum((output / n).stat().st_size for n in inventory)}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {k: v for k, v in manifest.items() if k != "files"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-acceptance", type=Path)
    args = parser.parse_args()
    print(json.dumps(package(args.output, args.gate_acceptance)))
