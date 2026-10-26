import importlib.util
import json
from pathlib import Path

import pytest

script = Path(__file__).resolve().parents[1] / "scripts/analysis/verify_cir_monitor_snapshot.py"
spec = importlib.util.spec_from_file_location("cir_monitor_snapshot", script)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def fixture(tmp_path, text, epochs=1):
    bundle = tmp_path / "bundle"
    (bundle / "manifests").mkdir(parents=True)
    (bundle / "manifests/cir_bundle.json").write_text("{}")
    (bundle / "configs/experiments").mkdir(parents=True)
    (bundle / "configs/experiments/mind_cir_108_v1.json").write_text(json.dumps({
        "data": {"cir_rows": 17271}, "training_schedule": {"cir_epochs": epochs, "ni_epochs": 3}}))
    manifest = tmp_path / "manifests/mind_cir_108_20260916_v1.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"active_run":{"backend_id":"test", "experiment":"exp", "job":"job",
        "bundle":"bundle", "bundle_sha256":monitor.sha(bundle / "manifests/cir_bundle.json")}}))
    root = tmp_path / "artifacts/mind-cir-20260916/progress/snapshot"
    root.mkdir(parents=True)
    log = root / "stdout.txt"
    log.write_text(text)
    (root / "receipt.json").write_text(json.dumps({"backend":"test", "native_status":"Running", "logs":[
        {"name":"user_logs/retry_002/std_log.txt", "file":str(log), "last_modified":"Wed, 16 Sep 2026 17:00:00 GMT"}]}))


def test_sigterm_grace_period_steps_are_not_live_progress(tmp_path):
    fixture(tmp_path, "CIR_GATES_ACCEPTED_STARTING_FORMAL\nCHECKPOINT_AUTO_RESUME_READY checkpoint-100\n162/540\nCaught signal 15\n165/540\n{'loss':'1.382'}")
    with pytest.raises(ValueError, match="Interrupted stdout"):
        monitor.verify(tmp_path, "snapshot", "cir", 100)


def test_finished_cir_cannot_be_reported_as_active_during_ni(tmp_path):
    fixture(tmp_path, "CIR_GATES_ACCEPTED_STARTING_FORMAL\n540/540\n{'loss':'1.2'}\nCIR_STAGE_ACCEPTED cir {}\n10/2025\n{'loss':'1.1'}")
    with pytest.raises(ValueError, match="CIR finished"):
        monitor.verify(tmp_path, "snapshot", "cir", 540)


def test_finished_ni_cannot_be_reported_as_active_during_evaluation(tmp_path):
    fixture(tmp_path, "CIR_STAGE_ACCEPTED cir {}\n2025/2025\n{'loss':'1.4'}\nCIR_STAGE_ACCEPTED ni {}\nLoading weights")
    with pytest.raises(ValueError, match="NI finished"):
        monitor.verify(tmp_path, "snapshot", "ni", 2025)


def test_three_epoch_monitor_uses_separate_manifest_and_artifact_root(tmp_path):
    fixture(tmp_path, "CIR_GATES_ACCEPTED_STARTING_FORMAL\n1620/1620\n{'loss':'1.2'}\nCIR_STAGE_ACCEPTED cir {}", epochs=3)
    old_manifest = tmp_path / "manifests/mind_cir_108_20260916_v1.json"
    before = old_manifest.read_bytes()
    value = json.loads(before)
    value["artifact_root"] = "artifacts/cir3"
    (tmp_path / "artifacts/mind-cir-20260916").rename(tmp_path / "artifacts/cir3")
    receipt_path = tmp_path / "artifacts/cir3/progress/snapshot/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["logs"][0]["file"] = str(receipt_path.parent / "stdout.txt")
    receipt_path.write_text(json.dumps(receipt))
    new_manifest = tmp_path / "manifests/cir3.json"
    new_manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="CIR finished"):
        monitor.verify(tmp_path, "snapshot", "cir", 1620, manifest_file=new_manifest)
    assert old_manifest.read_bytes() == before
