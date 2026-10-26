"""Host-only launcher for one unchanged arm after validated AMLT cancellation.

No Torch/PyArrow import, image pull, environment fallback, job cancellation, or
training implementation belongs here. Without --launch, validation is read-only
and Docker/NVIDIA are never called. All execution provenance is separate from
the immutable source, dispatch, and legacy recovery manifests.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
from typing import Any, Callable


VERSION = "oi-ni-meta-gcr-container-operation-v1"
HOST = "GCRAZGDL1681"
TASK_ROOT = Path("/home/v-wuhao1/experiments/oi-ni-meta-20260913")
SOURCE_ID = "38931d42872c03eecae1eca6355c1ca78d60a77f52b4b3793bda96662ce66b02"
SOURCE_MANIFEST_SHA = "c74eb61d5e581a5fdf9135ea34f22a79fce8e668ada338a5a8fd9b83b5e5e9d5"
PROTOCOL_SHA = "d8b79b1b49ca9a85c1a62618b3f3cb85976ad9438e972fe15be6f8421b1b31be"
RECOVERY_SHA = "c73aac1b8eb9cc5408b11fb24b86d62e5f4f11d67e8dc8723214a9217d15ce03"
WRAPPER_SHA = "666be3a9be73a227d024a6774738f61fd2c6bc27f4a1ce9291037a4ca39fe951"
DISPATCHER_SHA = "70814c0ab44fe9cbac94e6e622e53e0cde7e75ab6555251a9a757370d340bd88"
DISPATCH_MANIFEST_SHA = "5b954711572cc45e0a7a4c3431332313a2e62d9378bdcf32db333bc6b620b624"
REQUIREMENTS_SHA = "f326c6b8c5b677fcd38936ebf29ec20428f817539b137578381bf78ca037c116"
BASE_REFERENCE = "singularitybase.azurecr.io/base/job/pytorch/acpt-torch2.8.x-py3.10-cuda12.6-ubuntu22.04:20260818T030356015"
AMLT_BACKENDS = {1: "eager_candle_nxj1811m52", 2: "strong_dream_470kq82mzq"}
RUN_IDS = {
    1: "c72786e0457ca7d4e0202b48dbcf65d752bf6b09eb21b5c5e8f8d15a3ab17a07",
    2: "acb003caf66a463d96f1e45e6031543ffb87041b9a1331311f1f09d073c375f8",
}
MODEL = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
MODEL_CONFIG_SHA = "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4"
ACCEPTED_SNAPSHOT_SHA = "b1611d7f58232f9ecb18ffee20eabc7078774e86be24cff5bac64385db517f0a"
PROTOCOL = "configs/experiments/mind_oi_ni_meta_pilot_v1.json"
WRAPPER = "scripts/amlt/sft/run_oi_ni_meta_calibration_arm.sh"
DISPATCHER = "src/new_interest/experiments/oi_ni_meta_calibration_arm.py"
LEGACY_FILES = {
    "oi_ni_meta.py": "src/new_interest/experiments/oi_ni_meta.py",
    "../training/oi_ni_meta_backend.py": "src/new_interest/training/oi_ni_meta_backend.py",
    "../training/oi_ni_meta_objective.py": "src/new_interest/training/oi_ni_meta_objective.py",
}
PACKAGES = {
    "torch": "2.8.0+cu126", "torchvision": "0.23.0+cu126", "torchaudio": "2.8.0+cu126",
    "transformers": "5.14.1", "peft": "0.18.0", "accelerate": "1.12.0",
    "pyarrow": "20.0.0", "PyYAML": "6.0.2", "sentencepiece": "0.2.0",
    "wandb": "0.20.1", "tokenizers": "0.22.2", "safetensors": "0.8.0",
    "huggingface-hub": "1.14.0", "pytest": "8.3.5",
}
HF_DEST = "/mnt/default/huggingface-cache-mind-old-new-0911"
PIP_DEST = "/mnt/default/pip-cache-mind-old-new-0911"
SCRATCH_DEST = "/scratch/oi-ni-meta-checkpoints"
LABEL = "org.crossinterest.legacy-run-identity"
LIFECYCLE_LOG = "logs/azureml/virtualcluster_activity_log.txt"
# Only this replacement has a reviewed full result-storage scope for v2. Adding
# another arm requires its actual scope; deriving one from a job name is unsafe.
CANCELLATION_V2_RESULT_SCOPES = {
    1: {
        "experiment": "mind-oi-ni-meta-calibrate-recovery-v1-0913",
        "storage_account": "csnewsandfeeds4150361735",
        "container": "amulet",
        "prefix": "projects/sft4safety/amlt-results/7210714430.10800-f9d51b48-1a32/",
    },
}
CANCELLATION_V2_EVIDENCE_SHAS = {
    1: {
        "pre_cancel_native_evidence": "fef09ae92dcd38919b89585e53504bdf6e3a9d084fed3432c9c49ab81d52b25d",
        "native_evidence": "005b7826b0be37f98280d7e9b0b577f28a5908de9c68ee3d84eb552c7807cf16",
        "results_inventory_evidence": "8c7531c8ed18cf603d47a3d7aa7a4a1d80d6c9a98150e929f3e0ae8b6fc1bbe4",
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    require(isinstance(result, dict), f"Expected a JSON object: {path}")
    return result


def absolute_path(value: str) -> Path:
    path = Path(value)
    require(path.is_absolute(), f"An absolute path is required: {value}")
    require(path == path.resolve(), f"Host path must be canonical, without symlink aliases: {value}")
    require(not any(char in str(path) for char in (",", "\n", "\r")), "Unsafe bind-mount path")
    return path


def contained(root: Path, relative: str) -> Path:
    require(not Path(relative).is_absolute(), f"Expected a relative inventory path: {relative}")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f"Inventory path escapes its root: {relative}")
    return path


def checked_file(path: Path, expected: str) -> None:
    require(bool(re.fullmatch(r"[0-9a-f]{64}", expected or "")), "Invalid SHA-256 value")
    require(path.is_file() and sha256(path) == expected, f"File hash mismatch: {path}")


def artifact(reference: dict[str, str]) -> tuple[Path, dict[str, Any]]:
    path = absolute_path(reference["path"])
    checked_file(path, reference["sha256"])
    return path, load_json(path)


def validate_bundle(bundle: Path, operation: dict[str, Any], index: int) -> dict[str, Any]:
    checked_file(bundle / "source_bundle_manifest.json", SOURCE_MANIFEST_SHA)
    checked_file(bundle / PROTOCOL, PROTOCOL_SHA)
    checked_file(bundle / "recovery_parent.json", RECOVERY_SHA)
    checked_file(bundle / WRAPPER, WRAPPER_SHA)
    checked_file(bundle / DISPATCHER, DISPATCHER_SHA)
    checked_file(bundle / "requirements-oi-ni-meta-worker.txt", REQUIREMENTS_SHA)
    require(operation["dispatch_manifest_sha256"] == DISPATCH_MANIFEST_SHA, "Dispatch manifest binding changed")
    checked_file(bundle / "dispatch_manifest.json", DISPATCH_MANIFEST_SHA)
    source = load_json(bundle / "source_bundle_manifest.json")
    dispatch = load_json(bundle / "dispatch_manifest.json")
    require(operation["source_manifest_sha256"] == SOURCE_MANIFEST_SHA, "Source manifest binding changed")
    require(dispatch.get("version") == "oi-ni-meta-calibration-arm-dispatch-v1"
            and dispatch.get("legacy_source_identity") == SOURCE_ID
            and dispatch.get("legacy_source_manifest_sha256") == SOURCE_MANIFEST_SHA
            and dispatch.get("protocol_sha256") == PROTOCOL_SHA
            and dispatch.get("allowed_alpha_indices") == [1, 2]
            and dispatch.get("scientific_protocol_unchanged") is True
            and dispatch.get("parent_recovery_sha256") == RECOVERY_SHA
            and dispatch.get("expected_run_identities") == {str(k): v for k, v in RUN_IDS.items()},
            "Frozen dispatcher provenance changed")
    require(dispatch.get("files", {}).get(WRAPPER) == WRAPPER_SHA
            and dispatch["files"].get(DISPATCHER) == DISPATCHER_SHA, "Dispatcher/launcher not bound")
    for inventory in (source["files"], dispatch["files"]):
        for relative, digest in inventory.items():
            checked_file(contained(bundle, relative), digest)
    config = load_json(bundle / PROTOCOL)
    data = contained(bundle, config["data_dir"])
    checked_file(data / "metadata.json", config["data_metadata_sha256"])
    metadata = load_json(data / "metadata.json")
    hashes = {"metadata.json": sha256(data / "metadata.json")}
    for relative, item in metadata["artifacts"].items():
        checked_file(contained(data, relative), item["sha256"])
        hashes[relative] = item["sha256"]
    hashes.update(protocol=PROTOCOL_SHA, worker_bundle=SOURCE_MANIFEST_SHA)
    hashes.update({key: sha256(contained(bundle, relative)) for key, relative in LEGACY_FILES.items()})
    require(hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest() == SOURCE_ID,
            "Legacy source identity changed")
    settings = {"config": config, "mode": "meta", "alpha": config["inner_alpha_candidates"][index],
                "steps": config["calibration_steps"], "phase": "calibrate", "source_identity": SOURCE_ID,
                "eval_dev": False, "selection_sha256": None, "budget_sha256": None}
    require(hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest() == RUN_IDS[index],
            "Legacy calibration run identity changed")
    return config


def validate_cancellation(operation: dict[str, Any], index: int) -> dict[str, Any]:
    _, receipt = artifact(operation["cancellation"])
    amlt = operation["amlt_source"]
    expected_job = f"qwen35-2b-exact-meta-calibrate-alpha{index}"
    require(amlt["backend_run_id"] == AMLT_BACKENDS[index],
            "Cancellation must bind the current replacement backend, not the old serial run")
    require(amlt["job_name"] == expected_job, "AMLT source must be the corresponding replacement arm")
    require(receipt.get("version") in ("oi-ni-meta-terminal-cancel-v1", "oi-ni-meta-terminal-cancel-v2")
            and receipt.get("backend_run_id") == amlt["backend_run_id"]
            and receipt.get("job_name") == expected_job
            and type(receipt.get("alpha_index")) is int and receipt["alpha_index"] == index
            and receipt.get("legacy_run_identity") == RUN_IDS[index], "Cancellation belongs to another arm")
    require(receipt.get("state") in ("Canceled", "Cancelled"), "AMLT cancellation must be terminal")
    if receipt["version"] == "oi-ni-meta-terminal-cancel-v2":
        validate_cancellation_v2(receipt, index)
        return receipt
    # The original v1 contract stays strict: a filled native start time cannot
    # be waived with a receipt boolean or other v2-style evidence fields.
    require(receipt.get("started") is False, "Only an unstarted replacement arm may migrate")
    ended = datetime.fromisoformat(receipt.get("ended_at", "").replace("Z", "+00:00"))
    require(ended.tzinfo is not None and ended.timestamp() > 0
            and ended <= datetime.now(timezone.utc), "Cancellation needs a real terminal end time")
    _, native = artifact(receipt["native_evidence"])
    require(native.get("backend") == AMLT_BACKENDS[index]
            and native.get("native_status") == receipt["state"]
            and "start_time_utc" in native and native["start_time_utc"] is None,
            "Native cancellation evidence is not terminal/unstarted for this backend")
    native_end = datetime.fromisoformat((native.get("end_time_utc") or "").replace("Z", "+00:00"))
    require(native_end == ended, "Native terminal end time differs from cancellation receipt")
    return receipt


def cancellation_time(value: Any, label: str) -> datetime:
    require(isinstance(value, str) and bool(value), f"Missing cancellation time: {label}")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Invalid cancellation time: {label}") from error
    require(result.tzinfo is not None and result.utcoffset().total_seconds() == 0
            and 0 < result.timestamp() <= datetime.now(timezone.utc).timestamp(),
            f"Cancellation time must be a real past UTC timestamp: {label}")
    return result


def cancellation_reference(receipt: dict[str, Any], name: str) -> dict[str, str]:
    reference = receipt.get(name)
    require(isinstance(reference, dict) and isinstance(reference.get("path"), str)
            and isinstance(reference.get("sha256"), str), f"Missing cancellation evidence: {name}")
    return reference


def cancellation_lifecycle(native: dict[str, Any], reference: dict[str, str]) -> bytes:
    require(not any(key in native for key in ("observation_error_type", "observation_error", "error")),
            "Cancellation native observation contains a collection error")
    require(native.get("available_log_names") == [LIFECYCLE_LOG]
            and isinstance(native.get("logs"), dict) and set(native["logs"]) == {LIFECYCLE_LOG},
            "Cancellation evidence must enumerate only the lifecycle log, without worker logs")
    metadata = native["logs"][LIFECYCLE_LOG]
    require(isinstance(metadata, dict) and metadata.get("sha256") == reference["sha256"]
            and type(metadata.get("offset")) is int and metadata["offset"] == 0
            and type(metadata.get("remote_bytes")) is int and metadata["remote_bytes"] > 0,
            "Cancellation lifecycle must bind the original complete native log")
    # Native receipts retain their original local_file field unchanged. The
    # separately hashed reference permits the same bytes on the destination.
    path = absolute_path(reference["path"])
    checked_file(path, reference["sha256"])
    data = path.read_bytes()
    require(len(data) == metadata["remote_bytes"] and data.endswith(b"\n"),
            "Cancellation lifecycle bytes are incomplete")
    return data


def cancellation_events(data: bytes) -> list[tuple[datetime, str, str]]:
    events = []
    for line in data.decode("utf-8").splitlines():
        match = re.fullmatch(r"\[(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2} \+00:00)\] ([A-Za-z]+): (.*)", line)
        require(match is not None, "Unrecognized or incomplete lifecycle event")
        timestamp = datetime.strptime(match[1], "%m/%d/%Y %H:%M:%S %z")
        require(not events or timestamp >= events[-1][0], "Lifecycle event time goes backwards")
        events.append((timestamp, match[2], match[3]))
    require(bool(events), "Cancellation lifecycle is empty")
    return events


def validate_queued_lifecycle(events: list[tuple[datetime, str, str]], observed: datetime) -> None:
    require(len(events) >= 2 and events[0][1:] == ("Created", "Reason: Created job")
            and events[1][1] == "Queued" and events[-1][0] <= observed,
            "Pre-cancel lifecycle must begin with Created then Queued")
    state = "queued"
    for _, event, reason in events[2:]:
        if event == "ScheduleNotPossible" and state == "queued":
            continue
        if event == "ScheduleSelected" and state == "queued":
            state = "selected"
        elif event in ("Suspended", "Requeued") and state == {"Suspended": "selected", "Requeued": "suspended"}[event]:
            require(reason.startswith("Reason: Quota reservation request") and "not approved" in reason,
                    "Pre-cancel scheduling must be closed by a recorded quota rejection")
            state = "suspended" if event == "Suspended" else "queued"
        else:
            raise ValueError("Pre-cancel lifecycle contains execution or non-queue events")
    require(state == "queued", "Pre-cancel lifecycle has an unresolved resource selection")


def validate_cancellation_v2(receipt: dict[str, Any], index: int) -> None:
    """Verify an explicit observation-based migration boundary, not zero work.

    A platform start time appearing around cancellation is retained verbatim.
    Complete queued/cancel-only logs and empty full-scope results support the
    named observation; they do not establish zero GPU use or optimizer updates.
    """
    require(index in CANCELLATION_V2_RESULT_SCOPES, "No reviewed v2 result scope for this arm")
    require("started" not in receipt and receipt.get("observation") == "no_training_artifacts_observed",
            "V2 must retain the observation boundary, without a started assertion")
    # The collector's full Blob traversal is a reviewed fixed receipt, not a
    # claim that an arbitrary empty JSON list independently proves enumeration.
    for name, expected in CANCELLATION_V2_EVIDENCE_SHAS[index].items():
        require(cancellation_reference(receipt, name)["sha256"] == expected,
                f"V2 requires the reviewed original evidence: {name}")
    _, before = artifact(cancellation_reference(receipt, "pre_cancel_native_evidence"))
    _, terminal = artifact(cancellation_reference(receipt, "native_evidence"))
    require(before.get("backend") == AMLT_BACKENDS[index] and before.get("native_status") == "Queued"
            and "start_time_utc" in before and before["start_time_utc"] is None
            and "end_time_utc" in before and before["end_time_utc"] is None,
            "Pre-cancel native evidence must be queued and unstarted for this backend")
    require(terminal.get("backend") == AMLT_BACKENDS[index]
            and terminal.get("native_status") == receipt["state"],
            "Native cancellation evidence is not terminal for this backend")
    require(receipt.get("native_start_time_utc") == terminal.get("start_time_utc")
            and receipt.get("ended_at") == terminal.get("end_time_utc"),
            "V2 must preserve exact native start and end values")
    observed_before = cancellation_time(before.get("observed_at_utc"), "pre-cancel observation")
    started = cancellation_time(terminal.get("start_time_utc"), "actual native start")
    ended = cancellation_time(terminal.get("end_time_utc"), "native terminal end")
    observed_terminal = cancellation_time(terminal.get("observed_at_utc"), "terminal observation")
    before_bytes = cancellation_lifecycle(before, cancellation_reference(receipt, "pre_cancel_lifecycle_evidence"))
    terminal_bytes = cancellation_lifecycle(terminal, cancellation_reference(receipt, "terminal_lifecycle_evidence"))
    before_events = cancellation_events(before_bytes)
    validate_queued_lifecycle(before_events, observed_before)
    require(terminal_bytes.startswith(before_bytes), "Terminal lifecycle changed the pre-cancel bytes")
    additions = cancellation_events(terminal_bytes[len(before_bytes):])
    require([(event, reason) for _, event, reason in additions] == [
        ("Canceling", "Reason: Canceling job due to user request"),
        ("Canceled", "Reason: Deleted/canceled due to a user request."),
    ], "Terminal lifecycle must add only the two user-cancellation events")
    require(observed_before < started <= additions[0][0] <= additions[1][0]
            <= ended <= observed_terminal, "Cancellation evidence timeline is inconsistent")
    _, inventory = artifact(cancellation_reference(receipt, "results_inventory_evidence"))
    require(inventory.get("version") == "oi-ni-meta-result-inventory-v1"
            and inventory.get("backend_run_id") == AMLT_BACKENDS[index]
            and inventory.get("job_name") == receipt["job_name"]
            and all(inventory.get(key) == value for key, value in CANCELLATION_V2_RESULT_SCOPES[index].items()),
            "Cancellation results inventory has a different backend, job, or full storage scope")
    require(inventory.get("enumeration_complete") is True
            and isinstance(inventory.get("files"), list) and inventory["files"] == [],
            "Cancellation requires a completed full-scope empty results inventory")
    inventory_started = cancellation_time(inventory.get("observed_at_utc"), "results enumeration start")
    inventory_ended = cancellation_time(inventory.get("completed_at_utc"), "results enumeration completion")
    require(observed_terminal <= inventory_started <= inventory_ended,
            "Results inventory predates terminal cancellation or is incomplete")


def validate_preparation(operation: dict[str, Any]) -> dict[str, Any]:
    _, prep = artifact(operation["preparation_audit"])
    require(prep.get("version") == "oi-ni-meta-container-preparation-v1"
            and prep.get("base_image_id") == operation["image"]["base_id"]
            and prep.get("prepared_image_id") == operation["image"]["prepared_id"]
            and prep.get("cuda_available") is False
            and type(prep.get("gpu_visible_count")) is int and prep["gpu_visible_count"] == 0
            and prep.get("pip_check_passed") is True, "CPU-only preparation audit is incomplete")
    require(all(prep.get("packages", {}).get(name) == version for name, version in PACKAGES.items()),
            "Prepared numerical package versions differ")
    require(prep.get("original_wrapper_sha256") == WRAPPER_SHA
            and prep.get("requirements_sha256") == REQUIREMENTS_SHA, "Preparation used different requirements")
    for key in ("python_executable", "user_site", "home"):
        require(Path(prep.get("runtime", {}).get(key, "")).is_absolute(), f"Preparation lacks runtime {key}")
    require(isinstance(prep["runtime"].get("image_config_user"), str), "Preparation lacks image user")
    require(prep["runtime"].get("user_site_enabled") is True, "Audited --user packages are not enabled")
    for key in ("uid", "gid"):
        require(type(prep["runtime"].get(key)) is int and prep["runtime"][key] >= 0,
                f"Preparation lacks runtime {key}")
    for name in ("torch", "transformers", "peft"):
        require(Path(prep["runtime"].get("import_paths", {}).get(name, "")).is_absolute(),
                f"Preparation lacks actual import path: {name}")
    require(bool(re.fullmatch(r"[a-f0-9]{64}", prep.get("image_config_sha256", ""))),
            "Preparation lacks final image configuration hash")
    return prep


def validate_cache(operation: dict[str, Any], cache: Path) -> None:
    _, audit = artifact(operation["cache_audit"])
    require(operation["accepted_snapshot_audit"]["sha256"] == ACCEPTED_SNAPSHOT_SHA,
            "Original accepted snapshot audit binding changed")
    _, accepted = artifact(operation["accepted_snapshot_audit"])
    require(accepted.get("model_name") == MODEL and accepted.get("revision") == REVISION,
            "Original accepted snapshot has a different identity")
    require(audit.get("version") == "oi-ni-meta-model-cache-v1" and audit.get("cache_root") == str(cache)
            and audit.get("model_name") == MODEL and audit.get("revision") == REVISION
            and audit.get("complete") is True and audit.get("tokenizer_local_load_passed") is True,
            "Pinned model cache audit differs or is incomplete")
    prefix = f"hub/models--Qwen--Qwen3.5-2B/snapshots/{REVISION}"
    snapshot = cache / prefix
    require(snapshot.is_dir(), "Missing pinned model snapshot")
    inventory = audit.get("files", {})
    require(all(inventory.get(f"{prefix}/{name}") == digest for name, digest in accepted["files"].items()),
            "Model/tokenizer bytes differ from the original accepted snapshot")
    actual = {f"{prefix}/{p.name}" for p in snapshot.iterdir() if p.is_file()}
    require(actual and actual == set(inventory), "Cache audit must cover every snapshot file exactly")
    for relative, digest in inventory.items():
        checked_file(contained(cache, relative), digest)
    checked_file(snapshot / "config.json", MODEL_CONFIG_SHA)
    require((snapshot / "tokenizer.json").is_file() and (snapshot / "tokenizer_config.json").is_file(),
            "Tokenizer snapshot is incomplete")
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        names = set(load_json(index)["weight_map"].values())
        require(names and all(Path(name).name == name and f"{prefix}/{name}" in inventory
                              and contained(cache, f"{prefix}/{name}").is_file() for name in names),
                "Pinned model weight index is incomplete")
    else:
        require((snapshot / "model.safetensors").is_file(), "Pinned model weights are incomplete")


def ensure_fresh(paths: dict[str, Path]) -> None:
    for name in ("output_root", "scratch", "evidence_dir"):
        path = paths[name]
        require(not path.exists() or (path.is_dir() and not any(path.iterdir())),
                f"Refusing a previously started or nonempty {name}: {path}")


def validate_operation(manifest: Path, alpha_index: int) -> dict[str, Any]:
    require(type(alpha_index) is int and alpha_index in RUN_IDS, "Only alpha indices 1 and 2 are allowed")
    manifest = absolute_path(str(manifest))
    operation = load_json(manifest)
    require(operation.get("version") == VERSION, "Unknown operation manifest version")
    require(type(operation.get("alpha_index")) is int and operation["alpha_index"] == alpha_index
            and operation.get("legacy_source_identity") == SOURCE_ID
            and operation.get("legacy_run_identity") == RUN_IDS[alpha_index], "Operation identity differs")
    require(operation.get("host") == HOST, "Operation host differs")
    require(bool(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+", operation.get("container_name", ""))),
            "Invalid container name")
    require(bool(re.fullmatch(r"GPU-[a-fA-F0-9-]+", operation.get("gpu_uuid", ""))), "Bind an explicit GPU UUID")
    for key in ("prepared_id", "base_id"):
        require(bool(re.fullmatch(r"sha256:[a-f0-9]{64}", operation.get("image", {}).get(key, ""))),
                "Use immutable local image IDs, never a tag")
    require(operation["image"].get("base_reference") == BASE_REFERENCE, "Original AMLT base reference changed")
    bundle = absolute_path(operation["bundle_dir"])
    paths = {key: absolute_path(operation["paths"][key]) for key in
             ("task_root", "output_root", "hf_cache", "scratch", "pip_cache", "evidence_dir")}
    require(paths["task_root"] == TASK_ROOT and TASK_ROOT.is_dir(), "Unexpected task-owned root")
    require(not (TASK_ROOT / "control" / "gcr-calibration-launch-lock.json").exists(),
            "A previous task-global launch intent already exists; inspect its receipts")
    mounted = [bundle] + [value for key, value in paths.items() if key != "task_root"]
    for path in mounted:
        require(path != TASK_ROOT and path.is_relative_to(TASK_ROOT), "All mounts/evidence must be task-owned")
    for i, path in enumerate(mounted):
        for other in mounted[i + 1:]:
            require(not path.is_relative_to(other) and not other.is_relative_to(path),
                    "Read-only inputs and writable operation paths must not overlap")
    require(bundle.is_dir() and paths["hf_cache"].is_dir(), "Bundle/cache paths must already exist")
    config = validate_bundle(bundle, operation, alpha_index)
    cancellation = validate_cancellation(operation, alpha_index)
    preparation = validate_preparation(operation)
    validate_cache(operation, paths["hf_cache"])
    for filename in ("run_oi_ni_meta_calibration_container.sh", "oi_ni_meta_calibration_container.py"):
        checked_file(Path(__file__).resolve().with_name(filename), operation["host_tools"][filename])
    ensure_fresh(paths)
    return {"manifest": manifest, "manifest_sha256": sha256(manifest), "operation": operation,
            "paths": paths, "bundle": bundle, "config": config, "preparation": preparation,
            "cancellation": cancellation}


def docker_command(plan: dict[str, Any]) -> list[str]:
    operation, paths = plan["operation"], plan["paths"]
    command = ["docker", "run", "--detach", "--init", "--pull=never", "--network", "none",
               "--name", operation["container_name"], "--label", f"{LABEL}={operation['legacy_run_identity']}",
               "--label", f"org.crossinterest.operation-sha256={plan['manifest_sha256']}",
               "--gpus", f"device={operation['gpu_uuid']}", "--shm-size", "8g",
               "--security-opt", "no-new-privileges", "--workdir", "/workspace"]
    for source, target, readonly in ((plan["bundle"], "/workspace", True),
                                     (paths["hf_cache"], HF_DEST, True),
                                     (paths["output_root"], "/output", False),
                                     (paths["scratch"], SCRATCH_DEST, False),
                                     (paths["pip_cache"], PIP_DEST, False)):
        command.extend(["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")])
    for name, value in {"AMLT_CODE_DIR": "/workspace", "AMLT_OUTPUT_DIR": "/output",
                        "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                        "PYTHONDONTWRITEBYTECODE": "1"}.items():
        command.extend(["--env", f"{name}={value}"])
    # Bash deliberately replaces only the image entrypoint; image User/HOME and
    # installed --user packages are preserved. The original wrapper is unchanged.
    command.extend(["--entrypoint", "/bin/bash", operation["image"]["prepared_id"], WRAPPER,
                    str(operation["alpha_index"])])
    return command


def run_command(command: list[str]) -> str:
    return subprocess.run(command, check=True, text=True, capture_output=True, timeout=120).stdout.strip()


def host_preflight(plan: dict[str, Any], runner: Callable[[list[str]], str] = run_command) -> dict[str, Any]:
    operation = plan["operation"]
    require(socket.gethostname().split(".")[0].lower() == HOST.lower(), "This is not the reserved GCR host")
    require(not os.environ.get("DOCKER_HOST") and os.environ.get("DOCKER_CONTEXT", "default") in ("", "default"),
            "A remote or alternate Docker endpoint is not permitted")
    info = json.loads(runner(["docker", "info", "--format", "{{json .}}"])); images = {}
    require(info.get("Name", "").split(".")[0].lower() == HOST.lower(), "Docker daemon belongs to another host")
    require("nvidia" in info.get("Runtimes", {}), "Docker NVIDIA runtime is unavailable")
    for kind in ("prepared_id", "base_id"):
        result = json.loads(runner(["docker", "image", "inspect", operation["image"][kind]]))
        require(len(result) == 1 and result[0].get("Id") == operation["image"][kind], "Local image ID mismatch")
        images[kind] = result[0]
    base_layers = images["base_id"].get("RootFS", {}).get("Layers", [])
    prepared_layers = images["prepared_id"].get("RootFS", {}).get("Layers", [])
    require(base_layers and prepared_layers[:len(base_layers)] == base_layers
            and len(prepared_layers) > len(base_layers), "Prepared image does not extend the recorded base")
    require(BASE_REFERENCE in images["base_id"].get("RepoTags", []), "Recorded original base tag does not match local image")
    image_config = images["prepared_id"].get("Config", {})
    require(hashlib.sha256(json.dumps(image_config, sort_keys=True).encode()).hexdigest()
            == plan["preparation"]["image_config_sha256"], "Prepared image configuration differs from its CPU audit")
    require(image_config.get("User", "") == plan["preparation"]["runtime"]["image_config_user"],
            "Prepared image user differs from the preparation audit")
    image_env = dict(value.split("=", 1) for value in image_config.get("Env", []) if "=" in value)
    require("HOME" not in image_env or image_env["HOME"] == plan["preparation"]["runtime"]["home"],
            "Prepared image HOME differs from the audited --user installation")
    gpu_lines = runner(["nvidia-smi", "--query-gpu=uuid,name,memory.total", "--format=csv,noheader,nounits"]).splitlines()
    require(len(gpu_lines) == 1, "Exactly one host GPU is required")
    uuid, name, memory = [value.strip() for value in gpu_lines[0].split(",")]
    require(uuid == operation["gpu_uuid"] and "A100" in name and "80GB" in name
            and 80000 <= float(memory) <= 83000, "Expected reserved A100 80GB identity/capacity")
    compute = runner(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"])
    require(not compute.strip(), "GPU already has an active compute process")
    for selector in (f"label={LABEL}={operation['legacy_run_identity']}", f"name=^/{operation['container_name']}$"):
        existing = runner(["docker", "ps", "-a", "--filter", selector, "--format", "{{.ID}}"])
        require(not existing.strip(), "A conflicting or previously launched container already exists")
    ensure_fresh(plan["paths"])
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(),
            "docker_server_version": info.get("ServerVersion"), "gpu": {"uuid": uuid, "name": name, "memory_mib": float(memory)},
            "images": images, "active_compute_processes": [], "conflicting_containers": []}


def write_new_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def launch(plan: dict[str, Any], runner: Callable[[list[str]], str] = run_command) -> dict[str, Any]:
    # A task-global exclusive intent prevents competing arm/name manifests racing
    # between the idle-GPU check and Docker creation. Keep it on every outcome.
    lock = plan["paths"]["task_root"] / "control" / "gcr-calibration-launch-lock.json"
    lock.parent.mkdir(parents=True, exist_ok=True)
    write_new_json(lock, {"operation_manifest_sha256": plan["manifest_sha256"],
                          "operation_manifest_path": str(plan["manifest"]),
                          "created_at": datetime.now(timezone.utc).isoformat()})
    evidence = plan["paths"]["evidence_dir"]
    container_id, inspection = None, None
    try:
        preflight = host_preflight(plan, runner)
        evidence.mkdir(parents=True, exist_ok=True)
        write_new_json(evidence / "launch_intent.json", {"version": VERSION,
                       "operation_manifest_sha256": plan["manifest_sha256"], "operation": plan["operation"],
                       "preflight": preflight, "command": docker_command(plan)})
        # Keep portable byte-for-byte copies alongside the Docker receipts.
        references = {"operation_manifest.json": {"path": str(plan["manifest"]), "sha256": plan["manifest_sha256"]},
                      **{f"{name}.json": plan["operation"][name] for name in
                         ("preparation_audit", "cache_audit", "accepted_snapshot_audit", "cancellation")},
                      "native_cancellation.json": plan["cancellation"]["native_evidence"]}
        if plan["cancellation"]["version"] == "oi-ni-meta-terminal-cancel-v2":
            references.update({name: plan["cancellation"][key] for name, key in {
                "pre_cancel_native.json": "pre_cancel_native_evidence",
                "pre_cancel_lifecycle.txt": "pre_cancel_lifecycle_evidence",
                "terminal_lifecycle.txt": "terminal_lifecycle_evidence",
                "terminal_results_inventory.json": "results_inventory_evidence",
            }.items()})
        for name, reference in references.items():
            source = Path(reference["path"])
            checked_file(source, reference["sha256"])
            with (evidence / name).open("xb") as stream:
                stream.write(source.read_bytes())
        if plan["cancellation"]["version"] == "oi-ni-meta-terminal-cancel-v2":
            write_new_json(evidence / "evidence_copies.json", {
                "version": "oi-ni-meta-launch-evidence-copies-v1", "files": references,
            })
        for key in ("output_root", "scratch", "pip_cache"):
            plan["paths"][key].mkdir(parents=True, exist_ok=True)
            stat = plan["paths"][key].stat()
            uid, gid = (plan["preparation"]["runtime"][key] for key in ("uid", "gid"))
            access_bits = (stat.st_mode >> 6) if stat.st_uid == uid else ((stat.st_mode >> 3) if stat.st_gid == gid else stat.st_mode)
            require(uid == 0 or access_bits & 3 == 3, f"Prepared image user cannot write/traverse {key}")
        container_id = runner(docker_command(plan))
        require(bool(re.fullmatch(r"[0-9a-f]{64}", container_id)), "Docker did not return an immutable container ID")
        inspection = json.loads(runner(["docker", "container", "inspect", container_id]))
        require(len(inspection) == 1 and inspection[0].get("Id") == container_id
                and inspection[0].get("Image") == plan["operation"]["image"]["prepared_id"],
                "Created container/image differs from the requested identity")
        state = inspection[0].get("State", {})
        require(state.get("Status") in ("created", "running", "exited") and not state.get("Error")
                and not state.get("OOMKilled")
                and not (state.get("Status") == "exited" and state.get("ExitCode") != 0),
                f"Container already failed during launch: {state}")
        result = {"container_id": container_id, "launched_at": datetime.now(timezone.utc).isoformat(),
                  "operation_manifest_sha256": plan["manifest_sha256"], "container_inspect": inspection,
                  "logs_command": ["docker", "logs", "--timestamps", container_id]}
        write_new_json(evidence / "launch_receipt.json", result)
        return result
    except Exception as error:
        evidence.mkdir(parents=True, exist_ok=True)
        write_new_json(evidence / "launch_failure.json", {"error_type": type(error).__name__, "message": str(error),
                       "container_id": container_id, "container_inspect": inspection,
                       "recorded_at": datetime.now(timezone.utc).isoformat()})
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--alpha-index", type=int, choices=(1, 2), required=True)
    parser.add_argument("--launch", action="store_true", help="Run read-only host checks, then launch once")
    args = parser.parse_args(argv)
    plan = validate_operation(args.manifest, args.alpha_index)
    if args.launch:
        print(json.dumps(launch(plan), indent=2))
    else:
        print(json.dumps({"validated": True, "launched": False, "operation_manifest_sha256": plan["manifest_sha256"],
                          "command": docker_command(plan), "shell_display": shlex.join(docker_command(plan))}, indent=2))


if __name__ == "__main__":
    main()
