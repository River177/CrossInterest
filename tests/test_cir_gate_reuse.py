import json
from pathlib import Path

import pytest

from new_interest.data.build_cir_dataset import sha256
from new_interest.experiments.cir_warmup import CONTRACT, reuse_verified_gates


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(project):
    source = project / "reference"
    contract = {k: {"fixed": True} for k in ("model", "models", "recipe", "worker_packages", "execution")}
    contract.update(kernel_profile="torch-fallback-explicit", data_metadata_sha256="data-hash")
    put(project / CONTRACT, contract)
    put(source / "source_contract.json", contract)
    trainer = project / "src/trainer.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("unchanged native trainer")
    put(source / "source_bundle.json", {"files": {
        str(CONTRACT): sha256(source / "source_contract.json"), "src/trainer.py": sha256(trainer)}})
    bundle_sha = sha256(source / "source_bundle.json")
    put(source / "complete.json", {"status": "accepted", "bundle_sha256": bundle_sha,
        "stages": [{"epochs": 1}, {"epochs": 3}]})
    names = []
    for key, steps in (("single_cir", 8), ("single_ni", 8), ("ddp_cir", 2), ("ddp_ni", 2)):
        name = f"gates/{key}/stage_accepted.json"
        put(source / name, {"optimizer_steps": steps, "reload": {
            "all_tensors_exact": True, "logits_exact": True, "tensor_count": 373}})
        names.append(name)
    name = "gates/evaluation_readback.json"
    put(source / name, {"records": 64})
    names.append(name)
    put(source / "acceptance.json", {"status": "accepted", "bundle_sha256": bundle_sha,
        "complete_json_sha256": sha256(source / "complete.json"),
        "files": {n: sha256(source / n) for n in names}})
    return source


def test_duration_can_reuse_unchanged_accepted_gates(tmp_path):
    fixture(tmp_path)
    contract = json.loads((tmp_path / CONTRACT).read_text())
    contract["training_schedule"] = {"cir_epochs": 3, "ni_epochs": 3}
    put(tmp_path / CONTRACT, contract)
    result = reuse_verified_gates(tmp_path, Path("reference"))
    assert result["status"] == "reused_verified" and len(result["stages"]) == 4


@pytest.mark.parametrize("change", ["trainer", "execution", "gate", "incomplete"])
def test_gate_reuse_rejects_drift_or_invalid_source(tmp_path, change):
    source = fixture(tmp_path)
    if change == "trainer":
        (tmp_path / "src/trainer.py").write_text("different loss")
    elif change == "execution":
        contract = json.loads((tmp_path / CONTRACT).read_text())
        contract["execution"] = {"different": True}
        put(tmp_path / CONTRACT, contract)
    elif change == "gate":
        put(source / "gates/ddp_cir/stage_accepted.json", {"optimizer_steps": 0})
    else:
        acceptance = json.loads((source / "acceptance.json").read_text())
        acceptance["status"] = "pending"
        put(source / "acceptance.json", acceptance)
    with pytest.raises(ValueError):
        reuse_verified_gates(tmp_path, Path("reference"))
