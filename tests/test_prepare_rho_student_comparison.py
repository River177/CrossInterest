from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil

import pytest
import yaml


def module(tmp_path):
    path = Path(__file__).resolve().parents[1]/"scripts/analysis/prepare_rho_student_comparison.py"
    spec = importlib.util.spec_from_file_location("prepare_rho_comparison_fixture", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    result.REPO = tmp_path.resolve()
    return result


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value, sort_keys=True)+"\n")


def fixture(m):
    root = m.REPO/"artifacts/rho-student-20260916"
    frozen = root/"bundle-gate-v1"
    files = {"src/new_interest/__init__.py": "", "src/new_interest/experiments/__init__.py": "",
             "src/new_interest/experiments/dependency.py": "SOURCE = 'frozen gate source'\n",
             "inputs/validation.parquet": "frozen validation bytes", "inputs/dev.parquet": "frozen dev bytes",
             "inputs/interest_token_mapping.parquet": "frozen mapping bytes", "inputs/train.parquet": "must not upload train"}
    for name, value in files.items():
        put(frozen/name, value)
    protocol = {"reference_acceptance_sha256": "fixed-reference", "evaluation": {
        "primary": "macro_ndcg_at_20", "paired_bootstrap_unit": "user", "paired_bootstrap_replicates": 2000,
        "seed": 42, "validation_main_min_class_support": 20, "validation_rows": 5354, "dev_rows": 31392}}
    put(frozen/"protocol.json", protocol)
    inventory = {p.relative_to(frozen).as_posix(): m.sha(p) for p in frozen.rglob("*") if p.is_file()}
    manifest = {"runtime_sha256": "frozen-training-runtime", "files": inventory}
    put(frozen/"source_manifest.json", manifest)
    put(m.REPO/m.COMPARISON_MODULE, "raise RuntimeError('comparison must not import during bootstrap validation')\n")
    put(m.REPO/"src/new_interest/experiments/dependency.py", "SOURCE = 'wrong mutable source'\n")
    put(m.REPO/"src/new_interest/experiments/unfrozen.py", "MUST_NOT_COPY = True\n")
    plan = {**m.PLAN_FIELDS, "comparison_code_sha256": m.sha(m.REPO/m.COMPARISON_MODULE),
            "training_protocol_sha256": m.sha(frozen/"protocol.json")}
    put(root/"comparison_plan.json", plan)
    workers = {}
    for arm in m.ARMS:
        identity = {"arm": arm, "phase": "full", "runtime_sha256": manifest["runtime_sha256"],
            "protocol_sha256": plan["training_protocol_sha256"],
            "inputs_sha256": m.canonical({k: v for k, v in inventory.items() if k.startswith("inputs/")}),
            "reference_acceptance_sha256": protocol["reference_acceptance_sha256"]}
        evaluations = {split: {"accepted": True, "split": split, "limit": None,
            "rows": protocol["evaluation"][split+"_rows"], "source_sha256": inventory[f"inputs/{split}.parquet"],
            "protocol_sha256": plan["training_protocol_sha256"], "kernel_configs_consistent": True,
            "prediction_file": f"/mnt/output/accepted-{arm}/{split}/predictions.parquet"} for split in ("validation", "dev")}
        workers[arm] = {"worker": {"passed": True, "identity": identity, "evaluations": evaluations,
            "training": {"full": {"passed": True, "rows": 21600, "steps": 2025, "epochs": 3, "world_size": 4,
                "initial_state_sha256": "same-initial-state", "student_summary": {"reference_file_sha256": "same-reference-scores"}}}}}
    accepted = {"passed": True, "arms": workers, "same_initial_state": True, "same_reference_scores": True}
    accepted_path = root/"three_arm_acceptance.json"
    put(accepted_path, accepted)
    return root, frozen, accepted_path, accepted, plan


def test_missing_acceptance_does_not_prepare_any_output(tmp_path):
    m = module(tmp_path)
    bundle, config = tmp_path/"never-created"/"bundle", tmp_path/"never-config"/"job.yaml"
    with pytest.raises(FileNotFoundError):
        m.prepare_comparison(tmp_path/"missing.json", bundle, config)
    assert not bundle.parent.exists() and not config.parent.exists()


def test_frozen_cpu_bundle_and_exact_job_contract(tmp_path):
    m = module(tmp_path)
    root, frozen, acceptance, _, plan = fixture(m)
    before = {p.relative_to(frozen).as_posix(): m.sha(p) for p in frozen.rglob("*") if p.is_file()}
    bundle, output_yaml = root/"comparison bundle", tmp_path/"configs/amlt/comparison.yaml"
    ready = m.prepare_comparison(acceptance, bundle, output_yaml)
    assert ready["ready"] and ready["submitted"] is False and ready["gpu_count"] == 0
    assert ready["acceptance_sha256"] == m.sha(acceptance)
    assert ready["source_manifest_sha256"] == m.sha(bundle/"source_manifest.json")
    assert (bundle/"src/new_interest/experiments/dependency.py").read_text() == "SOURCE = 'frozen gate source'\n"
    assert not (bundle/"src/new_interest/experiments/unfrozen.py").exists()
    assert not (bundle/"inputs/train.parquet").exists()
    assert m.sha(bundle/m.COMPARISON_MODULE) == plan["comparison_code_sha256"]
    assert {p.relative_to(frozen).as_posix(): m.sha(p) for p in frozen.rglob("*") if p.is_file()} == before
    manifest = m.read(bundle/"source_manifest.json")
    assert all(m.sha(bundle/name) == digest for name, digest in manifest["files"].items())
    assert manifest["packages"] == {"numpy": "2.2.6", "pyarrow": "20.0.0", "scipy": "1.15.3"}
    config = yaml.safe_load(output_yaml.read_text())
    assert config["target"]["name"] == "Feeds"
    job = config["jobs"][0]
    assert job["sku"] == "5C3" and job["sla_tier"] == "Basic"
    assert job["submit_args"]["env"]["_AZUREML_SINGULARITY_JOB_UAI"] == m.UAI
    assert config["environment"]["image"] == "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04"
    assert config["environment"]["setup"] == []
    assert Path(config["code"]["local_dir"].replace("$CONFIG_DIR", str(output_yaml.parent))).resolve() == bundle
    commands = job["command"]
    assert commands[0] == "set -euo pipefail"
    pip = [command for command in commands if "pip install" in command]
    assert len(pip) == 1 and "--user --no-deps" in pip[0]
    assert pip[0].split()[-3:] == ["numpy==2.2.6", "pyarrow==20.0.0", "scipy==1.15.3"]
    assert all("nvidia-smi" not in command and "torch==" not in command and "snapshot_download" not in command for command in commands)
    assert commands[-1].startswith("python -m new_interest.experiments.compare_rho_students ")
    assert "--acceptance-sha256 "+ready["acceptance_sha256"] in commands[-1]
    assert '--output "$$AMLT_OUTPUT_DIR/comparison.json"' in commands[-1]
    assert commands[-2].startswith("python bootstrap_verify.py") and "--check-runtime" in commands[-2]
    with pytest.raises(FileExistsError):
        m.prepare_comparison(acceptance, bundle, output_yaml)


def test_bootstrap_rejects_changed_bytes_before_comparison_import(tmp_path):
    m = module(tmp_path)
    root, _, acceptance, *_ = fixture(m)
    bundle = root/"comparison"
    ready = m.prepare_comparison(acceptance, bundle, tmp_path/"comparison.yaml")
    namespace = {"__name__": "bootstrap_fixture", "__file__": str(bundle/"bootstrap_verify.py")}
    exec(compile((bundle/"bootstrap_verify.py").read_text(), str(bundle/"bootstrap_verify.py"), "exec"), namespace)
    assert namespace["verify"](bundle, ready["source_manifest_sha256"])["verified"]
    (bundle/m.COMPARISON_MODULE).write_text("changed after freeze")
    with pytest.raises(ValueError, match="Frozen comparison file changed"):
        namespace["verify"](bundle, ready["source_manifest_sha256"])


@pytest.mark.parametrize("case", ["not_passed", "missing_arm", "worker_not_passed", "unfinished_training", "different_initial",
    "different_reference", "wrong_runtime", "partial_evaluation", "wrong_primary", "changed_code", "changed_frozen_source"])
def test_incomplete_or_unbound_inputs_never_stage_comparison(tmp_path, case):
    m = module(tmp_path)
    root, frozen, acceptance, accepted, plan = fixture(m)
    if case == "not_passed": accepted["passed"] = False
    elif case == "missing_arm": accepted["arms"].pop("static")
    elif case == "worker_not_passed": accepted["arms"]["static"]["worker"]["passed"] = False
    elif case == "unfinished_training": accepted["arms"]["static"]["worker"]["training"]["full"]["steps"] = 2024
    elif case == "different_initial": accepted["arms"]["static"]["worker"]["training"]["full"]["initial_state_sha256"] = "different"
    elif case == "different_reference": accepted["arms"]["static"]["worker"]["training"]["full"]["student_summary"]["reference_file_sha256"] = "different"
    elif case == "wrong_runtime": accepted["arms"]["static"]["worker"]["identity"]["runtime_sha256"] = "different"
    elif case == "partial_evaluation": accepted["arms"]["static"]["worker"]["evaluations"]["dev"]["rows"] -= 1
    elif case == "wrong_primary":
        plan["primary_metric"] = "macro_recall_at_10"
        put(root/"comparison_plan.json", plan)
    elif case == "changed_code": put(m.REPO/m.COMPARISON_MODULE, "changed comparison")
    elif case == "changed_frozen_source": put(frozen/"src/new_interest/experiments/dependency.py", "changed frozen dependency")
    put(acceptance, accepted)
    bundle, config = root/"comparison", tmp_path/"comparison.yaml"
    with pytest.raises(ValueError):
        m.prepare_comparison(acceptance, bundle, config)
    assert not bundle.exists() and not config.exists()


def test_preparation_cannot_write_inside_frozen_training_bundle(tmp_path):
    m = module(tmp_path)
    _, frozen, acceptance, *_ = fixture(m)
    with pytest.raises(ValueError, match="must not alter frozen training"):
        m.prepare_comparison(acceptance, frozen/"comparison", tmp_path/"comparison.yaml")
    assert not (frozen/"comparison").exists()


def test_explicit_gate_bundle_must_match_accepted_runtime_before_copying(tmp_path):
    m = module(tmp_path)
    root, frozen, acceptance, accepted, _ = fixture(m)
    revised = root/"bundle-gate-v2"
    shutil.copytree(frozen, revised)
    manifest = m.read(revised/"source_manifest.json")
    manifest["runtime_sha256"] = "new-frozen-runtime"
    put(revised/"source_manifest.json", manifest)
    bundle, config = root/"comparison-v2", tmp_path/"comparison-v2.yaml"
    with pytest.raises(ValueError, match="differs from the frozen training experiment"):
        m.prepare_comparison(acceptance, bundle, config, frozen_bundle=revised)
    assert not bundle.exists() and not config.exists()
    for item in accepted["arms"].values():
        item["worker"]["identity"]["runtime_sha256"] = "new-frozen-runtime"
    put(acceptance, accepted)
    ready = m.prepare_comparison(acceptance, bundle, config, frozen_bundle=revised)
    assert ready["ready"]
    assert m.read(bundle/"source_manifest.json")["frozen_training_runtime_sha256"] == "new-frozen-runtime"
    assert m.read(bundle/"input_manifest.json")["original_sources"]["frozen_training_bundle"].endswith("bundle-gate-v2")
