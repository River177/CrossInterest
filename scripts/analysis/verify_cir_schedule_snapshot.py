"""Verify one four-condition stage's logged progress and committed sample cursor."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
from email.utils import parsedate_to_datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(project, manifest_path, mode, stage, stamp, checkpoint):
    tracking = read(project / manifest_path)
    job = tracking["jobs"][mode]
    bundle = project / tracking["bundle"]
    if sha(bundle / "manifests/cir_bundle.json") != tracking["bundle_sha256"]:
        raise ValueError("Bundle identity differs")
    preflight = read(bundle / "manifests/four_condition_preflight.json")
    expected = preflight["stages"][stage]
    allowed = {"staged": {"cir5", "ni3", "ni5"}, "mixed3": {"mixed3"}, "mixed5": {"mixed5"}}
    if stage not in allowed[mode]:
        raise ValueError("Stage belongs to another job")
    base = project / tracking["artifact_root"]
    native = read(base / "progress" / stamp / mode / "receipt.json")
    if native["backend"] != job["backend_id"] or native["native_status"] != "Running":
        raise ValueError("Backend is different or not actively running")
    stream = max((x for x in native["logs"] if x["name"].startswith("user_logs/")), key=lambda x: parsedate_to_datetime(x["last_modified"]))
    text = (project / stream["file"]).read_text()
    if f"CIR_STAGE_ACCEPTED {stage} " in text:
        raise ValueError("Stage already accepted; inspect the following phase")
    start = max(text.rfind("CHECKPOINT_AUTO_RESUME_READY"), text.rfind("CHECKPOINT_AUTO_RESUME_NONE"), text.rfind("CIR_SCHEDULE_FORMAL_START"))
    if text.rfind("Caught signal 15") > start:
        raise ValueError("Interrupted attempt is not live progress")
    steps = [int(x) for x in re.findall(rf"\b(\d+)/{expected['steps']}\b", text)]
    if not steps or "Traceback (most recent call last)" in text or max(steps) > expected["steps"]:
        raise ValueError("No clean current-stage progress")
    diagnostics = [ast.literal_eval(x) for x in text.splitlines() if x.startswith("{'loss':")]
    if not diagnostics or not all(math.isfinite(float(v)) for row in diagnostics for v in row.values()):
        raise ValueError("Missing or nonfinite training diagnostics")
    local = base / "readback" / stamp / job.get("experiment", tracking["experiment"]) / job["job"]
    cp = local / "formal" / f"checkpoints-{stage}" / f"checkpoint-{checkpoint}"
    marker = read(cp / "checkpoint_complete.json")
    if marker["global_step"] != checkpoint:
        raise ValueError("Wrong committed checkpoint")
    for name in ("trainer_state.json", "training_exposure_state.json", "old_new_execution_cursor.json"):
        if sha(cp / name) != marker["files"][name]["sha256"] or (cp / name).stat().st_size != marker["files"][name]["size"]:
            raise ValueError("Checkpoint metadata digest differs")
    contract = read(bundle / "configs/experiments/mind_cir_108_v1.json")
    identity = {**{k: contract["model"][k] for k in ("model_name", "model_revision")},
        "epochs": expected["epochs"], "world_size": 4, "effective_global_batch_size": 32,
        "per_device_train_batch_size": 2, "gradient_accumulation_steps": 4, "seed": 42,
        "loss_reduction": "token_mean", "semantic_id_loss_scope": "full_vocabulary",
        "train_file_sha256": expected["train_sha256"]}
    if any(marker["identity"].get(k) != v for k, v in identity.items()):
        raise ValueError("Stage model/data/execution identity differs")
    parent = expected["parent"]
    parent_hash = job.get("stage_acceptance", {}).get(parent, {}).get("adapter_sha256") if parent else None
    if parent and not parent_hash:
        raise ValueError("Verify the accepted CIR5 parent before its NI branch")
    if marker["identity"]["input_adapter_sha256"] != parent_hash:
        raise ValueError("Wrong parent adapter")
    spec = importlib.util.spec_from_file_location("cir_four_cursor", bundle / "src/new_interest/training/old_new_execution.py")
    cursor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cursor)
    cursor.STAGES.add(f"cir/{stage}")
    data = bundle / ("data/training/mind_cir_schedule_108_v1/mixed.parquet" if stage.startswith("mixed") else
                     f"data/training/mind_cir_108_v1/{'cir' if stage == 'cir5' else 'ni'}.parquet")
    verified = cursor.make_cursor_contract(data, stage=f"cir/{stage}", world_size=4, per_device_batch_size=2,
        source_checkpoint=cp, epochs=expected["epochs"], global_batch_size=32)
    if verified["total_steps"] != expected["steps"] or verified["samples"] != expected["samples"]:
        raise ValueError("Stage horizon differs")
    result = {"status": "advancing", "observed_at_utc": native["observed_utc"], "mode": mode, "stage": stage,
        "backend": job["backend_id"], "native_status": "Running", "active_stdout": stream["name"],
        "latest_logged_step": max(steps), "expected_steps": expected["steps"], "latest_loss": float(diagnostics[-1]["loss"]),
        "finite_diagnostics": True, "committed_checkpoint_step": checkpoint, "verified_presentations": verified["consumed_total"],
        "checkpoint_marker_sha256": sha(cp / "checkpoint_complete.json"), "parent_adapter_sha256": parent_hash,
        "log_age_seconds": max(0, (datetime.fromisoformat(native["observed_utc"]) - parsedate_to_datetime(stream["last_modified"])).total_seconds()),
        "large_payloads": "remote_only"}
    output = base / "progress" / stamp / mode / "verified_observation.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    job.update(status="running", phase=f"formal_{stage}_training", latest_logged_step=max(steps),
               last_committed_checkpoint_step=checkpoint, latest_observation=str(output.relative_to(project)))
    tracking["observed_at_utc"] = native["observed_utc"]
    (project / manifest_path).write_text(json.dumps(tracking, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/mind_cir_four_108_20260918_v1.json"))
    parser.add_argument("--mode", choices=("staged", "mixed3", "mixed5"), required=True)
    parser.add_argument("--stage", choices=("cir5", "ni3", "ni5", "mixed3", "mixed5"), required=True)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--checkpoint", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(Path.cwd(), args.manifest, args.mode, args.stage, args.stamp, args.checkpoint), indent=2))
