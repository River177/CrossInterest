"""Accept the student gate, launch the frozen three arms once, and collect receipts."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

from run_rho_twofold_reference_pipeline import AMLT, REPO, command

ROOT = REPO / "artifacts/rho-student-20260916"
GATE = "mind-rho-student-gate-v1-0916"
GATE_JOB = "qwen35-2b-rho-student-gate-dynamic"
FULL = "mind-rho-student-full-v1-0916"
ARMS = ("uniform", "static", "dynamic")
JOBS = [f"qwen35-2b-rho-student-full-{arm}" for arm in ARMS]
COMPARISON = "mind-rho-student-comparison-v1-0916"
COMPARISON_JOB = "rho-student-three-arm-comparison"
METRICS = ("macro_recall_at_10", "macro_recall_at_20", "macro_ndcg_at_10", "macro_ndcg_at_20")
CONTRASTS = ("static_minus_uniform", "dynamic_minus_uniform", "dynamic_minus_static")
TRAINING_NUMERIC_POLICY = "torch-deterministic-algorithms-v1"
PRODUCTION_PARITY_TRIALS = (
    "original_cold_native_1", "original_weighted_1", "original_native_1_5", "original_weighted_1_5",
    "warm_native_1", "warm_native_2", "warm_weighted_1", "warm_weighted_2",
)


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def statuses(experiment, jobs):
    # The older reference controller's helper binds its own artifact ROOT.
    # Keep this controller independent if those old local artifacts are pruned.
    try:
        output = command([AMLT, "status", experiment, "--hide-urls"], ROOT / f"{experiment}-status.txt")
    except (RuntimeError, TimeoutError) as error:
        raise TimeoutError("Retryable student status transport: " + str(error)) from error
    found = {}
    for job in jobs:
        lines = [line for line in output.splitlines() if re.match(r"^\s*(?::\d+\s+)?:?" + re.escape(job) + r"\s", line)]
        if len(lines) != 1:
            raise ValueError("Missing/ambiguous exact student job in status")
        match = re.search(r"\b(queued|preparing|running|pass|completed|failed|killed|canceled)\b", lines[0])
        if not match:
            raise ValueError("Unrecognized student job state")
        found[job] = match[1]
    return found


def collect(experiment, job, label, names):
    if (not names or any(not isinstance(name, str) or any(c in name for c in "/\\*?[]") or not name.endswith(".json") for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("Only explicitly named compact JSON files may be collected")
    destination = ROOT / "compact" / label
    destination.mkdir(parents=True, exist_ok=True)
    include = [item for name in names for item in ("--include", name)]
    try:
        command([AMLT, "results", "download", experiment, ":" + job, *include, "--output", str(destination), "--n-workers", "4"],
                ROOT / f"collect-{label}.log", timeout=300)
    except (RuntimeError, TimeoutError) as error:
        raise TimeoutError("Compact receipt transport failed: " + str(error)) from error
    files = {}
    for name in names:
        matches = list(destination.rglob(name))
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous compact receipt: " + name)
        files[name] = matches[0]
    return files


def check_training(value, rule, arm, *, require_parity):
    if (value.get("passed") is not True or value["rows"] != rule["rows"] or value["steps"] != rule["optimizer_steps"]
            or value["epochs"] != rule["epochs"] or value["world_size"] != rule["world_size"]
            or value.get("fresh_reload_exact") is not True or value["adapter_tensor_count"] != 373):
        raise ValueError("Student training endpoint/native reload differs")
    student = value["student_summary"]
    if (student.get("passed") is not True or student["arm"] != arm or student["rows"] != rule["rows"]
            or student["classes"] != rule["classes"] or student["rows_per_class"] != rule["rows_per_class"]
            or student["optimizer_steps"] != rule["optimizer_steps"] or student["epochs"] != rule["epochs"]
            or student["initial_state_sha256"] != value["initial_state_sha256"]
            or student["train_file_sha256"] != value["train_file_sha256"]
            or student.get("class_weight_mass_preserved") is not True
            or student.get("all_weights_positive") is not True
            or student["weight_bounds"] != [.5, 1.5]
            or student.get("current_student_scoring_all_arms") is not True
            or student.get("optimizer_reinitialized_at_refresh") is not False
            or student.get("training_kernel_selection") != "stable_fla_v1"
            or student.get("training_deterministic_algorithms") is not True
            or student.get("training_deterministic_warn_only") is not False):
        raise ValueError("Student weighting/training contract differs")
    per_epoch = rule["optimizer_steps"] // rule["epochs"]
    if student["refresh_steps"] != [i * per_epoch for i in range(rule["epochs"])]:
        raise ValueError("Student refresh schedule differs")
    if len(student["epoch_receipts"]) != rule["epochs"] or len(student["rank_order_history"]) != rule["world_size"]:
        raise ValueError("Incomplete epoch weights or rank order history")
    for epoch, receipt in enumerate(student["epoch_receipts"]):
        if (receipt["epoch"] != epoch or receipt["epoch_start_step"] != epoch * per_epoch
                or receipt["new_optimizer_updates_during_refresh"] != 0 or receipt["optimizer_reinitialized"] is not False):
            raise ValueError("Refresh changed optimizer state or epoch identity")
    probe = value["fresh_fp32_probes"]
    if probe.get("accepted") is not True or probe.get("production_batch4_fp32_exact") is not True:
        raise ValueError("Fresh production-batch FP32 probes failed")
    if require_parity:
        parity = student["native_gradient_parity"]
        if len(parity) != rule["world_size"]:
            raise ValueError("Gate lacks a native gradient control on every rank")
        for rank, item in enumerate(parity):
            if (item["rank"] != rank or not item["parameters_unchanged"] or item["optimizer_updates"] != 0
                    or [x["scale"] for x in item["comparisons"]] != [1., 1.5]
                    or not all(x["all_gradients_exact"] and x["tensor_count"] == 373 for x in item["comparisons"])):
                raise ValueError("Gate native gradient control failed")
            trials = item.get("trials", [])
            if not isinstance(trials, list) or any(not isinstance(trial, dict) or not isinstance(trial.get("label"), str) for trial in trials):
                raise ValueError("Gate native parity trial schema differs")
            production = [trial for trial in trials if trial["label"].startswith(("original_", "warm_"))]
            if [trial["label"] for trial in production] != list(PRODUCTION_PARITY_TRIALS):
                raise ValueError("Gate native parity lacks the complete original/warm production controls")
            for trial in production:
                backend = trial.get("numeric_backend", {})
                if (not isinstance(backend, dict) or backend.get("deterministic_algorithms") is not True
                        or backend.get("deterministic_warn_only") is not False
                        or trial.get("gradient_tensors") != 373
                        or trial.get("gradients_complete_and_finite") is not True):
                    raise ValueError("Gate production parity did not use strict deterministic training with all 373 gradients")


def check_storage_retention_override(worker, source, protocol, phase, arm, expected):
    """Admit only the separately recorded, storage-only recovery exception."""
    report = worker.get("storage_retention_override")
    if report is None and expected is None:
        return
    if not isinstance(report, dict) or not isinstance(expected, dict):
        raise ValueError("Missing or unplanned storage retention override")
    if phase != "full" or arm not in ("static", "dynamic") or expected.get("arm") != arm:
        raise ValueError("Unexpected arm or phase for storage retention override")
    if (report.get("version") != "rho-storage-retention-recovery-v1"
            or report.get("arm") != arm or report.get("stage") != "full"
            or report.get("base_runtime_sha256") != source["runtime_sha256"]
            or report.get("base_protocol_sha256") != source["files"]["protocol.json"]
            or report.get("native_training_entrypoint_unchanged") is not True
            or report.get("changed_fields") != ["persistent_checkpoint_limit"]
            or type(report.get("old_value")) is not int or report["old_value"] != 3
            or type(report.get("new_value")) is not int or report["new_value"] != 25
            or report.get("planned_optimizer_steps") != 2025
            or report.get("planned_max_checkpoint_count") != 21):
        raise ValueError("Storage retention recovery contract differs")
    for key in ("base_source_manifest_sha256", "recovery_manifest_sha256", "launcher_sha256"):
        if not expected.get(key) or report.get(key) != expected[key]:
            raise ValueError("Unbound storage recovery source: " + key)
    bundle = Path(expected["bundle"])
    for name, key in (("recovery_manifest.json", "recovery_manifest_sha256"),
                      ("recovery_worker.py", "launcher_sha256"),
                      ("base/source_manifest.json", "base_source_manifest_sha256")):
        if sha(bundle / name) != expected[key]:
            raise ValueError("Frozen storage recovery package changed: " + name)
    before, after = report.get("config_before"), report.get("config_after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("Missing complete storage recovery configuration audit")
    if (before.get("persistent_checkpoint_limit") != 3
            or after != {**before, "persistent_checkpoint_limit": 25}
            or before.get("rho_arm") != arm
            or before.get("rho_training_numeric_policy") != TRAINING_NUMERIC_POLICY):
        raise ValueError("Storage recovery changed more than remote checkpoint retention")
    original_sft = protocol.get("sft", {})
    if (not original_sft or any(before.get(key) != value for key, value in original_sft.items())
            or before.get("save_total_limit") != 3 or before.get("save_steps") != 100):
        raise ValueError("Storage recovery differs from the frozen training configuration")


def check_worker(worker, source, protocol, phase, arm, *, expected_storage_override=None):
    check_storage_retention_override(worker, source, protocol, phase, arm, expected_storage_override)
    if protocol.get("training_numeric_policy") != TRAINING_NUMERIC_POLICY:
        raise ValueError("Student protocol does not require the shared deterministic training policy")
    identity = worker["identity"]
    if (worker.get("passed") is not True or identity["phase"] != phase or identity["arm"] != arm
            or identity["runtime_sha256"] != source["runtime_sha256"]
            or identity["protocol_sha256"] != source["files"]["protocol.json"]
            or identity.get("inputs_sha256") != canonical({k: v for k, v in source["files"].items() if k.startswith("inputs/")})
            or identity["reference_acceptance_sha256"] != protocol["reference_acceptance_sha256"]):
        raise ValueError("Worker is not the frozen student experiment")
    stages = ("single", "ddp") if phase == "gate" else ("full",)
    if set(worker["training"]) != set(stages):
        raise ValueError("Wrong training stages")
    for stage in stages:
        rule = protocol["gate"][stage] if phase == "gate" else {"rows": 21600, "classes": 108,
            "rows_per_class": 200, "epochs": 3, "world_size": 4, "optimizer_steps": 2025}
        check_training(worker["training"][stage], rule, arm, require_parity=phase == "gate")
        expected_train = source["files"]["inputs/" + ("train.parquet" if phase == "full" else f"{stage}_train.parquet")]
        if worker["training"][stage]["train_file_sha256"] != expected_train:
            raise ValueError("Student trained on a different input file")
    splits = ("dev",) if phase == "gate" else ("validation", "dev")
    if set(worker["evaluations"]) != set(splits):
        raise ValueError("Wrong heldout evaluation splits")
    for split in splits:
        result = worker["evaluations"][split]
        expected = 64 if phase == "gate" else protocol["evaluation"][f"{split}_rows"]
        if (result.get("accepted") is not True or result["rows"] != expected
                or result["source_sha256"] != source["files"][f"inputs/{split}.parquet"]
                or result["protocol_sha256"] != source["files"]["protocol.json"]
                or result["reader_sha256"] != source["files"]["src/new_interest/experiments/rho_student_experiment.py"]
                or result.get("kernel_configs_consistent") is not True or result["kernel_cache_keys_checked"] <= 0):
            raise ValueError("Heldout readback identity or coverage differs")
        if phase == "full":
            count = result["metrics"]["main_107"]["supported_classes"] if split == "validation" else result["metrics"]["supported_classes"]
            if count != (107 if split == "validation" else 104):
                raise ValueError("Heldout supported-class denominator differs")


def prepare_full(gate_file, original=None, target=None):
    original = Path(original) if original is not None else ROOT / "bundle-gate-v1"
    target = Path(target) if target is not None else ROOT / "bundle-full-v1"
    source = read(original / "source_manifest.json")
    for name, digest in source["files"].items():
        if sha(original / name) != digest:
            raise ValueError("Frozen accepted gate bundle changed")
    if target.exists():
        ready = read(target / "source_manifest.json")
        if ready["runtime_sha256"] != source["runtime_sha256"] or sha(target / "gate_acceptance.json") != sha(gate_file):
            raise ValueError("Existing formal bundle differs")
        for name, digest in ready["files"].items():
            if sha(target / name) != digest:
                raise ValueError("Existing formal bundle has changed bytes")
        return
    shutil.copytree(original, target)
    shutil.copyfile(gate_file, target / "gate_acceptance.json")
    source["files"]["gate_acceptance.json"] = sha(target / "gate_acceptance.json")
    source["source_file_count"] = len(source["files"])
    source["source_bytes"] += (target / "gate_acceptance.json").stat().st_size
    write(target / "source_manifest.json", source)


def check_comparison_ready(ready, source, gate_bundle):
    """Authenticate prepared bytes and the exact CPU-only launch interface."""
    import yaml
    from prepare_rho_student_comparison import PACKAGES, UAI, VERSION as bundle_version
    bundle, config_file = Path(ready["bundle"]), Path(ready["config"])
    if (ready.get("ready") is not True or ready.get("version") != bundle_version
            or ready.get("gpu_count") != 0 or sha(config_file) != ready["config_sha256"]
            or sha(bundle / "source_manifest.json") != ready["source_manifest_sha256"]
            or ready["acceptance_sha256"] != sha(ROOT / "three_arm_acceptance.json")
            or ready["comparison_plan_sha256"] != sha(ROOT / "comparison_plan.json")):
        raise ValueError("Frozen comparison readiness receipt or inputs changed")
    manifest, plan = read(bundle / "source_manifest.json"), read(ROOT / "comparison_plan.json")
    if (manifest.get("version") != bundle_version or manifest["packages"] != PACKAGES
            or manifest["acceptance_sha256"] != ready["acceptance_sha256"]
            or manifest["comparison_code_sha256"] != plan["comparison_code_sha256"]
            or ready["comparison_code_sha256"] != plan["comparison_code_sha256"]
            or manifest["comparison_plan_sha256"] != ready["comparison_plan_sha256"]
            or manifest["frozen_training_runtime_sha256"] != source["runtime_sha256"]
            or manifest["frozen_training_source_manifest_sha256"] != sha(Path(gate_bundle) / "source_manifest.json")
            or plan["training_protocol_sha256"] != source["files"]["protocol.json"]):
        raise ValueError("Comparison bundle differs from its frozen training/analysis source")
    for name, digest in manifest["files"].items():
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                or (bundle / relative).is_symlink() or sha(bundle / relative) != digest):
            raise ValueError("Frozen comparison file changed: " + name)
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()
              and p != bundle / "source_manifest.json" and "__pycache__" not in p.relative_to(bundle).parts and p.suffix != ".pyc"}
    if actual != set(manifest["files"]):
        raise ValueError("Comparison upload bundle contains missing or untracked files")
    if (sha(bundle / "inputs/three_arm_acceptance.json") != ready["acceptance_sha256"]
            or sha(bundle / "comparison_plan.json") != ready["comparison_plan_sha256"]):
        raise ValueError("Comparison bundle contains different accepted inputs")
    commands = ["set -euo pipefail", 'cd "$$AMLT_CODE_DIR"',
        'export PYTHONPATH="$$AMLT_CODE_DIR/src" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        f"python bootstrap_verify.py --manifest-sha256 {ready['source_manifest_sha256']}",
        "python -m pip install --user --no-deps --no-warn-script-location --retries 4 --timeout 90 numpy==2.2.6 pyarrow==20.0.0 scipy==1.15.3",
        f"python bootstrap_verify.py --manifest-sha256 {ready['source_manifest_sha256']} --check-runtime",
        "python -m new_interest.experiments.compare_rho_students --acceptance inputs/three_arm_acceptance.json "
        f"--acceptance-sha256 {ready['acceptance_sha256']} --validation-file inputs/validation.parquet --dev-file inputs/dev.parquet "
        '--mapping-file inputs/interest_token_mapping.parquet --output "$$AMLT_OUTPUT_DIR/comparison.json"']
    config = yaml.safe_load(config_file.read_text())
    expected_job = {"name": COMPARISON_JOB, "sku": "5C3", "sla_tier": "Basic",
        "submit_args": {"env": {"AMLT_DOCKERFILE_TEMPLATE": "DEFAULT", "_AZUREML_SINGULARITY_JOB_UAI": UAI}},
        "command": commands}
    code_dir = config["code"]["local_dir"]
    if (not code_dir.startswith("$CONFIG_DIR/")
            or Path(code_dir.replace("$CONFIG_DIR", str(config_file.parent))).resolve() != bundle.resolve()
            or set(config["code"]) != {"local_dir", "ignore"} or config["code"]["ignore"] != ["__pycache__/", "*.pyc"]
            or config["jobs"] != [expected_job]
            or config["target"] != {"service": "sing", "name": "Feeds", "workspace_name": "CS-NewsAndFeeds-Singularity@rg-cs-newsandfeeds-singularity"}
            or config["environment"] != {"image": "amlt-sing/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04", "setup": []}):
        raise ValueError("Comparison launch is not the frozen CPU-only command contract")
    return ready


def prepare_or_recover_comparison(source, gate_bundle):
    from prepare_rho_student_comparison import prepare_comparison, VERSION as bundle_version
    ready_file = ROOT / "comparison_ready.json"
    bundle, config = ROOT / "bundle-comparison-v1", REPO / "configs/amlt/mind_rho_student_comparison_0916_v1.yaml"
    if ready_file.exists():
        ready = read(ready_file)
    elif bundle.exists() and config.exists():
        # A local crash may occur after immutable bundle/config publication but
        # before the ready receipt. Recover only after full contract validation.
        manifest = read(bundle / "source_manifest.json")
        ready = {"ready": True, "version": bundle_version, "bundle": str(bundle), "config": str(config),
            "acceptance_sha256": manifest["acceptance_sha256"], "source_manifest_sha256": sha(bundle / "source_manifest.json"),
            "comparison_code_sha256": manifest["comparison_code_sha256"], "comparison_plan_sha256": manifest["comparison_plan_sha256"],
            "config_sha256": sha(config), "submitted": False, "gpu_count": 0}
    elif bundle.exists() or config.exists():
        raise ValueError("Partial comparison preparation; preserve existing files for inspection")
    else:
        ready = prepare_comparison(ROOT / "three_arm_acceptance.json", bundle, config, frozen_bundle=gate_bundle)
    check_comparison_ready(ready, source, gate_bundle)
    if not ready_file.exists():
        write(ready_file, ready)
    return ready


def check_comparison(report, source):
    plan = read(ROOT / "comparison_plan.json")
    if (report.get("passed") is not True or report["acceptance_sha256"] != sha(ROOT / "three_arm_acceptance.json")
            or report["comparison_code_sha256"] != plan["comparison_code_sha256"]
            or report["mapping_sha256"] != source["files"]["inputs/interest_token_mapping.parquet"]
            or report["primary_comparison"] != {"split": "validation", "contrast": "dynamic_minus_static", "metric": "macro_ndcg_at_20",
                "dev_role": "consistency check under the same fixed contrast and metric"}
            or report.get("version") != "rho-three-student-paired-user-bootstrap-v1"
            or report.get("metric_implementation_sha256") != source["files"]["src/new_interest/experiments/sid_relation_bias.py"]
            or report.get("numpy_version") != "2.2.6" or report.get("scipy_version") != "1.15.3"):
        raise ValueError("Paired comparison identity/primary definition differs")
    for split, rows, users, classes in (("validation", 5354, 5354, 107), ("dev", 31392, 27082, 104)):
        result = report["results"][split]
        if (result["records"] != rows or result["users"] != users or result["main_macro_classes"] != classes
                or result.get("target_occurrences") != (5354 if split == "validation" else 37728)
                or result.get("cross_arm_kernel_configs_consistent") is not True
                or result["kernel_cache_keys_checked"] <= 0 or result["bootstrap"]["draws"] != 2000
                or result["bootstrap"]["seed_per_split"] != 42
                or result["bootstrap"].get("unit") != "user"
                or result["bootstrap"].get("observed_users") != users
                or result["bootstrap"].get("sampled_users_per_draw") != users
                or not 0 < result["bootstrap"].get("usable_draws", 0) <= 2000
                or report["provenance"][split]["source_sha256"] != source["files"][f"inputs/{split}.parquet"]):
            raise ValueError("Paired comparison population or kernel/bootstrap settings differ")
        if set(result["means"]) != set(ARMS) or set(result["comparisons"]) != set(CONTRASTS):
            raise ValueError("Paired comparison arm/contrast family differs")
        for arm in ARMS:
            if set(result["means"][arm]) != set(METRICS) or any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in result["means"][arm].values()):
                raise ValueError("Comparison contains invalid point metrics")
        for contrast in CONTRASTS:
            newer, older = contrast.split("_minus_")
            item = result["comparisons"][contrast]
            for metric in METRICS:
                expected = result["means"][newer][metric] - result["means"][older][metric]
                if not math.isclose(item["delta"][metric], expected, rel_tol=0, abs_tol=1e-12) or not math.isclose(item["delta_percentage_points"][metric], 100 * expected, rel_tol=0, abs_tol=1e-9):
                    raise ValueError("Comparison point difference or percentage-point conversion differs")
                interval, points = item["ci95"][metric], item["ci95_percentage_points"][metric]
                if (len(interval) != 2 or len(points) != 2 or not all(math.isfinite(v) for v in interval + points)
                        or interval[0] > interval[1] or any(not math.isclose(p, 100 * v, rel_tol=0, abs_tol=1e-9) for v, p in zip(interval, points))):
                    raise ValueError("Comparison interval or percentage-point conversion differs")
    rare = report["results"]["validation"]["rare_validation_class"]
    if (rare.get("class") != "Artificial Intelligence & Robotics" or rare.get("support") != 4 or rare.get("users") != 4
            or rare.get("included_in_primary_macro") is not False or rare.get("intervals_computed") is not False
            or set(rare.get("point_metrics", {})) != set(ARMS)):
        raise ValueError("Rare validation class scope differs")


def write_report(report, comparison_file):
    """Publish only values from independently accepted compact reports."""
    metrics = METRICS
    labels = {"uniform": "Uniform", "static": "Static OOF", "dynamic": "Dynamic"}
    text = ["# 动态可约减损失权重：三组学生实验", "",
        "三组各完成 21,600 个训练用户、3 轮和 2,025 次更新；完整验证集与 Dev 预测经远端逐条核验，配对用户 bootstrap 已完成。", "",
        "## 研究问题与训练设计", "",
        "比较动态学习状态是否在静态 OOF 难度之外提供增量收益。三组从同一 Qwen3.5-2B 预训练权重与 seed42 LoRA 开始，保留原 History Behavior+Interest → SID 数据、原生完整词表 completion loss 和采样顺序。", "",
        "Uniform 使用权重 1；Static 在类别内按 −OOF loss 排序；Dynamic 按当前学生 loss−OOF loss 排序。两种加权都使用平均并列排名映射到 0.5–1.5，并保持每类总权重相同。所有样本保留；权重每轮开始时更新，不重置优化器。三个条件都执行相同的当前学生评分轮次。", "",
        "## 评价设计", "",
        "固定评估最后 checkpoint。主检验为验证集 Dynamic−Static 的 Macro NDCG@20；Dev 作为同口径一致性检查。每个数据集的三个差值共享 2,000 次 seed42 用户簇重采样，同一用户的所有 impressions 和目标共同重采样。区间是单个训练 seed 条件下的用户抽样不确定性。", "",
        "验证集主指标覆盖支持充分的 107 类，AI 类 4 条单列；Dev 保持 31,392 impressions、27,082 用户、104 类。排名覆盖全部 108 个 SID，不过滤历史类别。下表点估计用百分数，差值和区间用百分点。", ""]
    acceptance_path = ROOT / "three_arm_acceptance.json"
    if acceptance_path.exists():
        accepted = read(acceptance_path)
        recovered = [labels[arm] for arm in ARMS
            if accepted.get("arms", {}).get(arm, {}).get("worker", {}).get("storage_retention_override")]
        if recovered:
            text.extend(["恢复期间，" + "、".join(recovered)
                + " 的远端 checkpoint 保留上限由 3 增至 25；模型、训练数据、loss 和优化器配置保持原方案。这个存储设置覆盖已通过独立回执核验。", ""])
    for split, title in (("validation", "验证集 107 类"), ("dev", "官方 Dev 104 类")):
        result = report["results"][split]
        text.extend([f"## {title}", "", "| 条件 | Macro R@10 | Macro R@20 | Macro NDCG@10 | Macro NDCG@20 |",
                     "|---|---:|---:|---:|---:|"])
        for arm in ARMS:
            text.append("| " + labels[arm] + " | " + " | ".join(f"{100 * result['means'][arm][metric]:.4f}" for metric in metrics) + " |")
        text.extend(["", "| 比较 | NDCG@20 差值 | 用户配对 95% CI |", "|---|---:|---:|"])
        for contrast, label in (("static_minus_uniform", "Static−Uniform"), ("dynamic_minus_uniform", "Dynamic−Uniform"), ("dynamic_minus_static", "Dynamic−Static")):
            value = result["comparisons"][contrast]
            delta = value["delta_percentage_points"]["macro_ndcg_at_20"]
            low, high = value["ci95_percentage_points"]["macro_ndcg_at_20"]
            text.append(f"| {label} | {delta:+.4f} | [{low:+.4f}, {high:+.4f}] |")
        text.extend(["", f"重采样中至少缺失一个原定类别的次数：{result['bootstrap']['draws_missing_any_eligible_class']} / 2,000；这类抽样在三组间使用相同的非空类别集合。", ""])
    rare = report["results"]["validation"]["rare_validation_class"]
    text.extend(["## 验证集稀有 AI 类", "", "4 个用户样本，不计入 107 类主指标；仅报告点估计。", "",
                 "| 条件 | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 |", "|---|---:|---:|---:|---:|"])
    for arm in ARMS:
        text.append("| " + labels[arm] + " | " + " | ".join(f"{100 * rare['point_metrics'][arm][key]:.4f}" for key in ("r10", "r20", "n10", "n20")) + " |")
    text.append("")
    primary = report["results"]["validation"]["comparisons"]["dynamic_minus_static"]
    low, high = primary["ci95_percentage_points"]["macro_ndcg_at_20"]
    if low > 0:
        statement = "验证集主对照的区间整体高于零，支持本次固定训练 seed 下动态项有增量收益；仍需结合 Dev 一致性及后续独立训练 seed 检验稳定性。"
    elif high < 0:
        statement = "验证集主对照的区间整体低于零，本次设置下动态项不如静态 OOF 权重。"
    else:
        statement = "验证集主对照的区间覆盖零，本次结果不足以证明动态项优于静态 OOF 权重。"
    text.extend([statement, "", "参考损失表示跨用户预测难度的估计；本实验没有人工噪声真值，不能将高损失样本认定为错标签。其余比较是边际区间，未进行多重比较校正。", "",
                 "## 数据与产物", "",
                 f"- [训练设计]({REPO / 'docs/research/2026-09-16_rho_student_execution.md'})",
                 f"- [三组训练与评估验收]({ROOT / 'three_arm_acceptance.json'})",
                 f"- [完整配对指标、AI 四例及 hash 来源]({comparison_file})",
                 f"- [运行状态]({ROOT / 'pipeline_state.json'})",
                 f"- [恢复清单]({REPO / 'manifests/rho_student_20260916.json'})", "",
                 "模型、checkpoint、完整评分、权重和预测均保留在远端；本机仅保存紧凑回执和报告。", ""])
    path = REPO / "docs/experiments/2026-09-16/rho_student_three_arm.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(text)
    if not path.exists() or path.read_text() != content:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(content)
        temporary.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    lock = (ROOT / "pipeline.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Student pipeline already running")
        return
    path = ROOT / "pipeline_state.json"
    state = read(path)
    if state["phase"] in ("complete", "needs_attention"):
        print(json.dumps(state), flush=True)
        return
    gate_experiment = state.get("gate_experiment", GATE)
    gate_job = state.get("gate_job", GATE_JOB)
    gate_bundle = ROOT / state.get("gate_bundle", "bundle-gate-v1")
    full_bundle = ROOT / state.get("full_bundle", "bundle-full-v1")
    full_config = REPO / state.get("full_config", "configs/amlt/mind_rho_student_full_0916_v1.yaml")
    source = read(gate_bundle / "source_manifest.json")
    protocol = read(gate_bundle / "protocol.json")
    write(ROOT / "pipeline_process.json", {"pid": os.getpid(), "started_unix": time.time()})
    while state["phase"] not in ("complete", "needs_attention"):
        try:
            if state["phase"] == "gate_submitted":
                observed = statuses(gate_experiment, [gate_job])[gate_job]
                state["gate_status"] = observed
                if observed in ("failed", "killed", "canceled"):
                    raise ValueError("Student gate failed; inspect its existing artifacts")
                if observed in ("pass", "completed"):
                    state["phase"] = "gate_readback"
            if state["phase"] == "gate_readback":
                files = collect(gate_experiment, gate_job, "gate", ["gate_acceptance.json", "training_acceptance.json", "dev_acceptance.json", "environment.json"])
                gate = read(files["gate_acceptance.json"])
                if gate["training"] != read(files["training_acceptance.json"]) or gate["evaluations"]["dev"] != read(files["dev_acceptance.json"]):
                    raise ValueError("Gate compact receipts disagree")
                check_worker(gate, source, protocol, "gate", "dynamic")
                state.update(gate_acceptance=str(files["gate_acceptance.json"]), gate_acceptance_sha256=sha(files["gate_acceptance.json"]), phase="full_prepare")
            if state["phase"] == "full_prepare":
                if sha(full_config) != state["full_config_sha256"]:
                    raise ValueError("Frozen three-arm launch configuration changed")
                prepare_full(state["gate_acceptance"], gate_bundle, full_bundle)
                command([AMLT, "run", str(full_config), FULL, "--dump"],
                        ROOT / "full-parsed-config.txt", timeout=120)
                state["phase"] = "full_submit"
                write(path, state)
            if state["phase"] == "full_submit":
                if sha(full_config) != state["full_config_sha256"]:
                    raise ValueError("Frozen three-arm launch configuration changed")
                if state.get("full_submission_attempted"):
                    raise ValueError("Prior formal submission outcome ambiguous; never duplicate")
                state["full_submission_attempted"] = True
                write(path, state)
                command([AMLT, "run", str(full_config), FULL, "-d",
                         "User-authorized uniform/static/dynamic RHO soft weighting. Same accepted runtime, fresh Qwen3.5-2B,21600 users,3epochs,2025updates,4A100 each; stable FLA, exact reload and full heldout remote readback.", "-y"],
                        ROOT / "submit-full.log", timeout=900)
                state["phase"] = "full_wait"
            if state["phase"] == "full_wait":
                observed = statuses(FULL, JOBS)
                state["full_status"] = observed
                if any(x in ("failed", "killed", "canceled") for x in observed.values()):
                    raise ValueError("A student arm failed; preserve successful peers and inspect")
                if all(x in ("pass", "completed") for x in observed.values()):
                    state["phase"] = "full_readback"
            if state["phase"] == "full_readback":
                accepted = {}
                for arm, job in zip(ARMS, JOBS):
                    storage_override = state.get("storage_retention_overrides", {}).get(arm)
                    names = ["worker_success.json", "training_acceptance.json", "validation_acceptance.json", "dev_acceptance.json", "environment.json"]
                    if storage_override is not None:
                        names.append("storage_retention_override.json")
                    files = collect(FULL, job, arm, names)
                    result = read(files["worker_success.json"])
                    if result["training"] != read(files["training_acceptance.json"]):
                        raise ValueError("Formal compact training receipts disagree")
                    if any(result["evaluations"][split] != read(files[f"{split}_acceptance.json"]) for split in ("validation", "dev")):
                        raise ValueError("Formal compact evaluation receipts disagree")
                    if storage_override is not None and result.get("storage_retention_override") != read(files["storage_retention_override.json"]):
                        raise ValueError("Compact storage recovery receipts disagree")
                    check_worker(result, source, protocol, "full", arm, expected_storage_override=storage_override)
                    accepted[arm] = {"worker": result, "compact_files": {name: {"path": str(p), "sha256": sha(p)} for name, p in files.items()}}
                initials = {x["worker"]["training"]["full"]["initial_state_sha256"] for x in accepted.values()}
                references = {x["worker"]["training"]["full"]["student_summary"]["reference_file_sha256"] for x in accepted.values()}
                if len(initials) != 1 or len(references) != 1:
                    raise ValueError("Student arms used different initial models or reference scores")
                write(ROOT / "three_arm_acceptance.json", {"passed": True, "arms": accepted,
                    "frozen_training_bundle": str(gate_bundle.relative_to(REPO)),
                    "same_initial_state": True, "same_reference_scores": True,
                    "student_rows_per_arm": 21600, "optimizer_updates_per_arm": 2025,
                    "large_artifact_policy": "remote_only", "paired_comparison_pending": True})
                state.update(phase="training_evaluation_complete", acceptance=str(ROOT / "three_arm_acceptance.json"), paired_comparison_pending=True)
            if state["phase"] == "training_evaluation_complete":
                ready = prepare_or_recover_comparison(source, gate_bundle)
                state.update(phase="comparison_submit", comparison_config=ready["config"])
                write(path, state)
            if state["phase"] == "comparison_submit":
                if state.get("comparison_submission_attempted"):
                    raise ValueError("Prior comparison submission ambiguous; never duplicate")
                ready = read(ROOT / "comparison_ready.json")
                check_comparison_ready(ready, source, gate_bundle)
                command([AMLT, "run", ready["config"], COMPARISON, "--dump"], ROOT / "comparison-parsed-config.txt", timeout=120)
                state["comparison_submission_attempted"] = True
                write(path, state)
                command([AMLT, "run", ready["config"], COMPARISON, "-d",
                    "CPU-only independent readback and frozen2000-draw paired user bootstrap for the three completed RHO student arms. No model inference or artifact downloads.", "-y"],
                    ROOT / "submit-comparison.log", timeout=900)
                state["phase"] = "comparison_wait"
            if state["phase"] == "comparison_wait":
                observed = statuses(COMPARISON, [COMPARISON_JOB])[COMPARISON_JOB]
                state["comparison_status"] = observed
                if observed in ("failed", "killed", "canceled"):
                    raise ValueError("CPU comparison failed; preserve all completed training arms")
                if observed in ("pass", "completed"):
                    state["phase"] = "comparison_readback"
            if state["phase"] == "comparison_readback":
                check_comparison_ready(read(ROOT / "comparison_ready.json"), source, gate_bundle)
                files = collect(COMPARISON, COMPARISON_JOB, "comparison", ["comparison.json"])
                report = read(files["comparison.json"])
                check_comparison(report, source)
                state.update(phase="comparison_report", comparison=str(files["comparison.json"]),
                             comparison_sha256=sha(files["comparison.json"]))
                write(path, state)
            if state["phase"] == "comparison_report":
                comparison_file = Path(state["comparison"])
                if sha(comparison_file) != state["comparison_sha256"]:
                    raise ValueError("Accepted compact comparison changed before report generation")
                report = read(comparison_file)
                check_comparison(report, source)
                report_path = write_report(report, comparison_file)
                state.update(phase="complete", paired_comparison_pending=False, comparison=str(comparison_file),
                             comparison_sha256=sha(comparison_file), report=str(report_path), report_sha256=sha(report_path))
                state.pop("reason", None)
                state.pop("report_error", None)
            state.pop("transport_error", None)
        except TimeoutError as error:
            if (state["phase"] == "full_submit" and state.get("full_submission_attempted")) or (state["phase"] == "comparison_submit" and state.get("comparison_submission_attempted")):
                state.update(phase="needs_attention", reason="Formal submission outcome ambiguous; inspect exact experiment")
            else:
                state["transport_error"] = str(error)
        except OSError as error:
            if state["phase"] == "comparison_report":
                state["report_error"] = str(error)
            else:
                state.update(phase="needs_attention", reason=type(error).__name__ + ": " + str(error))
        except Exception as error:
            state.update(phase="needs_attention", reason=type(error).__name__ + ": " + str(error))
        state["observed_unix"] = time.time()
        write(path, state)
        print(json.dumps(state), flush=True)
        if args.once or state["phase"] in ("complete", "needs_attention"):
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
