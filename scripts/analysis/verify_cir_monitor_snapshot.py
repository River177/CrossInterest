"""Verify a bounded native-log snapshot and downloaded compact checkpoint metadata.

Read-only with respect to remote jobs and all frozen experiment artifacts. The
only writes are a new observer receipt and this experiment's live status pointer.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
from email.utils import parsedate_to_datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import math


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(project: Path, stamp: str, stage: str, checkpoint: int, *, manifest_file: Path | None = None):
    manifest_path = project / (manifest_file or "manifests/mind_cir_108_20260916_v1.json")
    manifest = read(manifest_path)
    active = manifest["active_run"]
    artifact_root = project / manifest.get("artifact_root", "artifacts/mind-cir-20260916")
    base = artifact_root / "progress" / stamp
    local = artifact_root / "readback" / stamp / active["experiment"] / active["job"]
    bundle = project / active["bundle"]
    if sha(bundle / "manifests/cir_bundle.json") != active["bundle_sha256"]:
        raise ValueError("Active bundle identity differs")
    contract = read(bundle / "configs/experiments/mind_cir_108_v1.json")
    schedule = contract.get("training_schedule", {"cir_epochs": 1, "ni_epochs": 3})
    native = read(base / "receipt.json")
    if native["backend"] != active["backend_id"] or native["native_status"] != "Running":
        raise ValueError("Snapshot belongs to another backend or an inactive lifecycle; inspect recovery separately")
    stream = max((x for x in native["logs"] if x["name"].startswith("user_logs/")),
                 key=lambda x: parsedate_to_datetime(x["last_modified"]))
    content = (project / stream["file"]).read_text()
    samples = contract["data"]["cir_rows"] if stage == "cir" else 21600
    epochs = schedule[f"{stage}_epochs"]
    expected_steps = math.ceil(samples / 32) * epochs
    # Keep the requested stage separate from diagnostic and prior-stage logs.
    boundary = "CIR_GATES_ACCEPTED_STARTING_FORMAL" if stage == "cir" else "CIR_STAGE_ACCEPTED cir "
    content = content.split(boundary)[-1]
    if f"CIR_STAGE_ACCEPTED {stage} " in content:
        raise ValueError(f"{stage.upper()} finished; inspect the next stage instead of reporting training as running")
    last_start = max(content.rfind("CHECKPOINT_AUTO_RESUME_READY"), content.rfind("CHECKPOINT_AUTO_RESUME_NONE"))
    if content.rfind("Caught signal 15") > last_start:
        raise ValueError("Interrupted stdout is not current training progress")
    steps = [int(x) for x in re.findall(rf"\b(\d+)/{expected_steps}\b", content)]
    if not steps or max(steps) > expected_steps or "Traceback" in content:
        raise ValueError("No valid current-stage progress or an exception is present")
    losses = [ast.literal_eval(line) for line in content.splitlines() if line.startswith("{'loss':")]
    if not losses or not all(math.isfinite(float(value)) for row in losses for value in row.values()):
        raise ValueError("Missing or nonfinite training diagnostics")
    cp = local / f"formal/checkpoints-{stage}/checkpoint-{checkpoint}"
    marker = read(cp / "checkpoint_complete.json")
    if marker["global_step"] != checkpoint:
        raise ValueError("Checkpoint step differs")
    for name in ("trainer_state.json", "training_exposure_state.json", "old_new_execution_cursor.json"):
        info = marker["files"][name]
        if sha(cp / name) != info["sha256"] or (cp / name).stat().st_size != info["size"]:
            raise ValueError(f"Compact checkpoint file differs: {name}")
    model = contract["model"]
    expected_identity = {"model_name": model["model_name"], "model_revision": model["model_revision"],
        "epochs": epochs, "world_size": 4, "effective_global_batch_size": 32,
        "per_device_train_batch_size": 2, "gradient_accumulation_steps": 4,
        "seed": 42, "loss_reduction": "token_mean", "semantic_id_loss_scope": "full_vocabulary"}
    if any(marker["identity"].get(key) != value for key, value in expected_identity.items()):
        raise ValueError("Checkpoint model, objective or execution identity differs")
    if stage == "cir" and marker["identity"]["input_adapter_sha256"] is not None:
        raise ValueError("Formal CIR unexpectedly inherited another adapter")
    if stage == "ni":
        parent = manifest.get("stage_acceptance", {}).get("cir", {}).get("adapter_sha256")
        if not parent or marker["identity"]["input_adapter_sha256"] != parent:
            raise ValueError("NI checkpoint is not bound to the accepted CIR adapter")
    spec = importlib.util.spec_from_file_location("cir_frozen_cursor", bundle / "src/new_interest/training/old_new_execution.py")
    cursor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cursor_module)
    cursor_module.STAGES.add(f"cir/{stage}")
    data = bundle / f"data/training/mind_cir_108_v1/{stage}.parquet"
    cursor = cursor_module.make_cursor_contract(data, stage=f"cir/{stage}", world_size=4,
        per_device_batch_size=2, source_checkpoint=cp, epochs=epochs)
    if cursor["samples"] != samples or cursor["total_steps"] != expected_steps:
        raise ValueError("Frozen training scope differs")
    if marker["identity"]["train_file_sha256"] != sha(data):
        raise ValueError("Checkpoint is from another dataset")
    previous = active.get("latest_logged_optimizer_step")
    unchanged = active.get("phase") == f"formal_{stage}_training" and previous == max(steps)
    report = {"status": f"formal_{stage}_no_new_logged_progress" if unchanged else f"formal_{stage}_advancing", "observed_at_utc": native["observed_utc"],
        "native_status": native["native_status"], "active_stdout": stream["name"],
        "last_modified": stream["last_modified"], "latest_logged_optimizer_step": max(steps),
        "expected_optimizer_steps": expected_steps, "latest_loss": float(losses[-1]["loss"]),
        "all_logged_loss_grad_lr_finite": True, "latest_committed_checkpoint_step": checkpoint,
        "exact_committed_sample_prefix": cursor["consumed_total"],
        "checkpoint_identity_sha256": marker["identity_sha256"], "checkpoint_marker_sha256": sha(cp / "checkpoint_complete.json"),
        "formal_ni_started": stage == "ni", "large_payloads": "Remote only; local compact metadata and exact sample-prefix replay",
        "verifier_source_sha256": sha(Path(__file__)),
        "log_age_seconds_at_query": max(0, (datetime.fromisoformat(native["observed_utc"]) - parsedate_to_datetime(stream["last_modified"])).total_seconds())}
    output = base / "verified_observation.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2) + "\n")
    active.update(status="running", phase=f"formal_{stage}_training", latest_observation=str(output.relative_to(project)),
                  latest_logged_optimizer_step=max(steps), last_committed_checkpoint_step=checkpoint)
    manifest.update(status=report["status"], observed_at_utc=native["observed_utc"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--stage", choices=("cir", "ni"), required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.project.resolve(), args.stamp, args.stage, args.checkpoint, manifest_file=args.manifest), indent=2))
