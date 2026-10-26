"""Read all native log attempts and bounded compact receipts for the matched history-token experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import re

import requests
from azure.identity import AzureCliCredential
from azure.storage.blob import BlobServiceClient
from azure.core import MatchConditions

WORKSPACE = ("subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895/"
             "resourceGroups/rg-cs-newsandfeeds-singularity/"
             "providers/Microsoft.MachineLearningServices/workspaces/CS-NewsAndFeeds-Singularity")


def redact(text):
    text = re.sub(r"eyJ[A-Za-z0-9_.-]{50,}", "[REDACTED_JWT]", text)
    text = re.sub(r'(?i)(\b(?:accessToken|access_token|authorization|AZUREML_RUN_TOKEN|WANDB_API_KEY|HF_TOKEN)'
                  r'\b["\s]*[:=]\s*)[^\n]+', r"\1[REDACTED]", text)
    return re.sub(r'https?://[^\s"<>]+', lambda m: m.group().split("?")[0] + "?[REDACTED_QUERY]"
                  if re.search(r"[?&](?:sig|token|access_token|sv)=", m.group(), re.I) else m.group(), text)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def observe(project, manifest_path, stamp=None, *, skip_logs=False):
    tracking = json.loads(manifest_path.read_text())
    job = tracking["active_run"]
    backend = job["backend_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", backend):
        raise ValueError("Invalid backend ID")
    now = datetime.now(timezone.utc)
    out = project / tracking["artifact_root"] / "progress" / (stamp or now.strftime("%Y%m%dT%H%M%SZ"))
    out.mkdir(parents=True, exist_ok=False)
    credential = AzureCliCredential(process_timeout=30)
    headers = {"Authorization": "Bearer " + credential.get_token("https://management.azure.com/.default").token}
    base = "https://ml.azure.com/api/eastus"

    def api(path, **kwargs):
        r = requests.get(base + path, headers=headers, timeout=(15, 40), **kwargs)
        if r.status_code != 200:
            raise RuntimeError(f"Azure API HTTP {r.status_code}")
        result = r.json()
        if not isinstance(result, dict):
            raise ValueError("Azure API response is not an object")
        return result

    details = api(f"/history/v1.0/{WORKSPACE}/runs/{backend}/details")
    names = list(details.get("logFiles") or {})
    selected = [n for n in names if re.fullmatch(r"user_logs/(?:retry_\d+/)?std_log\.txt", n)]
    selected += [n for n in names if n.endswith("virtualcluster_activity_log.txt")]
    if skip_logs:
        selected = []
    receipt = {"logs_skipped": skip_logs, "observed_utc": now.isoformat(), "backend": backend, "native_status": details.get("status"),
               "start_time": details.get("startTimeUtc"), "end_time": details.get("endTimeUtc"),
               "available_log_names": names, "logs": []}
    for i, name in enumerate(selected):
        info = api(f"/artifact/v2.0/{WORKSPACE}/artifacts/ExperimentRun/dcid.{backend}/contentinfo", params={"path": name})
        uri = info.get("contentUri")
        if not isinstance(uri, str) or not uri.startswith("https://"):
            raise ValueError("Missing HTTPS log content location")
        head = requests.head(uri, timeout=(15, 40))
        if head.status_code != 200:
            raise RuntimeError(f"Log metadata HTTP {head.status_code}")
        length = int(head.headers.get("Content-Length", 0))
        if not length:
            continue
        offset = max(0, length - 65536)
        data = requests.get(uri, headers={"Range": f"bytes={offset}-"}, timeout=(15, 40))
        if data.status_code not in (200, 206) or len(data.content) > 262144:
            raise ValueError("Invalid bounded native log response")
        text = redact(data.content.decode(errors="replace")).replace("\r", "\n")
        path = out / f"native-{i:02d}.txt"
        path.write_text(text)
        steps = {label: [int(x) for x in re.findall(rf"\b(\d+)/{total}\b", text)]
                 for label, total in tracking.get("monitor_step_totals", {"cir":2700,"ni":2025}).items()}
        receipt["logs"].append({"name": name, "file": str(path.relative_to(project)),
            "last_modified": data.headers.get("Last-Modified"), "remote_bytes": length, "range_start": offset,
            "redacted_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "logged_steps": {k: v[-1] for k, v in steps.items() if v},
            "has_traceback": "Traceback (most recent call last)" in text,
            "signal15": "Caught signal 15" in text})
    client = BlobServiceClient("https://csnewsandfeeds4150361735.blob.core.windows.net",
        credential=credential, connection_timeout=15, read_timeout=30, retry_total=1).get_container_client("amulet")
    prefix = job["results_prefix"].rstrip("/") + "/"
    blobs = list(client.list_blobs(name_starts_with=prefix))
    inventory = [{"name": b.name[len(prefix):], "size": b.size, "etag": str(b.etag),
                  "last_modified": b.last_modified.isoformat()} for b in blobs]
    write(out / "blob_inventory.json", inventory)
    checkpoints = {}
    for b in inventory:
        match = re.fullmatch(r"(formal|gates)/checkpoints-([^/]+)/checkpoint-(\d+)/checkpoint_complete.json", b["name"])
        if match and b["size"]:
            key = match[1] + "/checkpoints-" + match[2]
            checkpoints[key] = max(checkpoints.get(key, 0), int(match[3]))
    checkpoint_names = {"checkpoint_complete.json", "trainer_state.json", "training_exposure_state.json", "old_new_execution_cursor.json"}
    allowed = {"identity.json", "environment.json", "snapshot_audit.json", "token_audit_cir.json", "token_audit_ni.json", "joint_length_audit.json",
               "complete.json", "comparison.json", "analysis_receipt.json", "gradient_audit.json", "dev_token_audit.json", "semantic_token_initialization.json", "accepted.json", "stage_accepted.json", "run_summary.json", "initial_adapter_state.json",
               "training_start_audit.json", "semantic_adapter_reload_audit.json", "cursor_contract.json", "launch_config.json",
               "evaluation_readback.json", "metrics.json", "merge_manifest.json", "shard_manifest.json", "resolved_config.json"}
    downloads = {}
    compact = out / "compact"
    for b in blobs:
        rel = b.name[len(prefix):]
        path = Path(rel)
        if ".." in path.parts or path.is_absolute() or not b.size:
            continue
        is_checkpoint = "/checkpoint-" in rel
        if is_checkpoint:
            if path.name not in checkpoint_names or not any(rel.startswith(f"{k}/checkpoint-{v}/") for k, v in checkpoints.items()):
                continue
        elif path.name not in allowed:
            continue
        if path.suffix != ".json" or b.size > 3_000_000:
            raise ValueError("Unexpected compact file size/type: " + rel)
        payload = client.get_blob_client(b.name).download_blob(etag=b.etag, match_condition=MatchConditions.IfNotModified).readall()
        json.loads(payload)
        target = compact / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        downloads[rel] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload), "etag": str(b.etag)}
    receipt.update(checkpoints=checkpoints, compact_files=downloads, results_prefix=job["results_prefix"],
                   large_payloads="remote_only", blob_count=len(blobs))
    write(out / "receipt.json", receipt)
    user_logs = [r for r in receipt["logs"] if r["name"].startswith("user_logs/") and r["last_modified"]]
    latest = max(user_logs, key=lambda r: parsedate_to_datetime(r["last_modified"])) if user_logs else None
    return {"native_status": receipt["native_status"], "backend": backend, "receipt": str((out / "receipt.json").relative_to(project)),
            "latest_stdout": latest, "checkpoints": checkpoints, "compact_files": len(downloads)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/mind_history_token_token_20260930_v1.json"))
    parser.add_argument("--stamp")
    parser.add_argument("--skip-logs", action="store_true", help="Read native status and compact artifacts without signed log transport")
    args = parser.parse_args()
    try:
        print(json.dumps(observe(Path.cwd(), args.manifest, args.stamp, skip_logs=args.skip_logs), indent=2))
    except requests.RequestException as error:
        raise SystemExit(f"Native log request failed ({type(error).__name__}); URL omitted") from None
