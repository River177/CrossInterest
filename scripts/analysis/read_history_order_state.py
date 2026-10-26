"""Read compact durable evidence; never download model tensors or predictions."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from azure.core import MatchConditions
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import AzureCliCredential
from azure.storage.blob import BlobServiceClient


def validate_complete(payloads, receipt):
    complete = payloads["complete.json"][0]
    manifest, manifest_digest = payloads["artifact_manifest.json"]
    comparison, comparison_digest = payloads["comparison.json"]
    if complete["bundle_sha256"] != receipt["bundle_sha256"]:
        raise ValueError("Completion belongs to another frozen bundle")
    if complete["artifact_manifest_sha256"] != manifest_digest or complete["comparison_sha256"] != comparison_digest:
        raise ValueError("Completion content hashes differ")
    if complete["status"] != "remote_training_reload_full_dev_and_comparison_verified" or comparison["status"] != "passed":
        raise ValueError("Incomplete acceptance state")
    for name, (_, digest) in payloads.items():
        if name in manifest["files"] and manifest["files"][name]["sha256"] != digest:
            raise ValueError(f"Compact artifact differs from committed manifest: {name}")
    full = comparison["full_dev"]
    if (full["records"], full["users"], full["target_occurrences"], full["supported_classes"]) != (31392, 27082, 37728, 104):
        raise ValueError("Accepted coverage differs")
    for arm in ("A", "B", "C", "D"):
        expected = comparison["source_identity"][arm]
        if arm == "A":
            if expected["predictions_sha256"] != "c07251bb82843c9caf35a0d6d8d58a57295bd8f4ddb4006e397a6ec1e22dbac6":
                raise ValueError("Baseline predictions changed")
            continue
        name = f"evaluations/{arm}/real_eval/merged/metrics.json"
        metrics, digest = payloads[name]
        if digest != expected["metrics_sha256"]:
            raise ValueError("Comparison metrics source differs")
        for kind in ("recall", "ndcg"):
            for k in (1, 5, 10, 20):
                values = metrics[f"per_class_{kind}"][str(k)]
                if len(values) != 104:
                    raise ValueError("Class support differs")
                key = f"interest_macro_{kind}_at_{k}"
                value = math.fsum(values.values()) / len(values)
                if abs(value - metrics[key]) > 1e-12 or abs(value - full["metrics"][arm][key]) > 1e-12:
                    raise ValueError("Compact independent macro recomputation failed")
        acceptance = payloads[f"evaluations/{arm}/accepted.json"][0]
        if acceptance["bundle_sha256"] != receipt["bundle_sha256"] or acceptance["evaluation"]["prediction_sha256"] != expected["predictions_sha256"]:
            raise ValueError("Evaluation acceptance identity differs")
    b = payloads["evaluations/B/accepted.json"][0]
    c = payloads["evaluations/C/accepted.json"][0]
    d = payloads["evaluations/D/accepted.json"][0]
    if b["eval_sha256"] != d["eval_sha256"] or c["adapter_file_sha256"] != d["adapter_file_sha256"]:
        raise ValueError("Paired data/model sharing differs")
    if b["adapter_file_sha256"] != "0a41effcaac398a0ca9d914142b7975dffaad98f435379e436e6132b59c3abf0":
        raise ValueError("B did not reuse the original model")
    if c["eval_sha256"] != "c05b20033ce0d1a0c2d4b7bb97ca0cd6b7cffb7f198a7265cce9e54046b665fd":
        raise ValueError("C did not use original Dev")
    training = complete["training"]
    if c["adapter_file_sha256"] != training["adapter"]["adapter_sha256"]:
        raise ValueError("C/D do not use the accepted training endpoint")
    if b["eval_sha256"] != "e53abcf33c9e67a6b8603740ee802671c466ced19586f1cda5b7cf3567ac6354":
        raise ValueError("B/D do not use the frozen shuffled Dev")
    if (training["steps"], training["samples"], training["epochs"], training["cursor"]["consumed_global_samples"]) != (2025, 21600, 3, 64800):
        raise ValueError("Training budget differs")
    if not training["reload"]["logits_exact"] or not training["reload"]["all_tensors_exact"]:
        raise ValueError("Final reload not exact")
    return {"status": "compact_independent_readback_passed", "full_dev_metrics": full["metrics"],
            "comparison_sha256": comparison_digest, "artifact_manifest_sha256": manifest_digest,
            "scope": "Cloud ETag-bound compact artifact hashes and macro recomputation; full tensors and per-row predictions verified remotely"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=Path("manifests/mind_history_order_20260923_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads(args.receipt.read_text())
    prefix = receipt["jobs"]["order"]["results_prefix"] + "/"
    client = BlobServiceClient("https://csnewsandfeeds4150361735.blob.core.windows.net",
        credential=AzureCliCredential(process_timeout=30), connection_timeout=15, read_timeout=40, retry_total=2).get_container_client("amulet")
    names = ["identity.json", "input_audit.json", "environment.json", "snapshot_audit.json", "token_audit.json", "baseline_fresh_reload.json",
             "startup_ready.json", "single/accepted.json", "single/sft/training_start_audit.json", "training_accepted.json",
             "final/sft/run_summary.json", "final/sft/training_sample_cursor.json", "final/sft/semantic_adapter_reload_audit.json",
             "complete.json", "artifact_manifest.json", "comparison.json"]
    for arm in ("B", "C", "D"):
        names += [f"evaluations/{arm}/accepted.json", f"evaluations/{arm}/real_eval/merged/metrics.json"]

    def fetch(name):
        blob = client.get_blob_client(prefix + name)
        try:
            props = blob.get_blob_properties()
        except ResourceNotFoundError:
            return name, None
        if props.size > 2_000_000:
            raise ValueError(f"Unexpectedly large compact artifact: {name}")
        raw = blob.download_blob(etag=props.etag, match_condition=MatchConditions.IfNotModified).readall()
        parsed = json.loads(raw)
        dest = args.output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        return name, (parsed, hashlib.sha256(raw).hexdigest(), props.etag, len(raw))

    with ThreadPoolExecutor(max_workers=6) as pool:
        fetched = dict(pool.map(fetch, names))
    existing = {name: value for name, value in fetched.items() if value is not None}
    report = {"observed_at_utc": datetime.now(timezone.utc).isoformat(), "source_prefix": prefix,
              "present": {name: {"sha256": value[1], "etag": value[2], "bytes": value[3]} for name, value in existing.items()},
              "not_yet_present": [name for name, value in fetched.items() if value is None]}
    if "complete.json" in existing:
        report["acceptance"] = validate_complete({name: value[:2] for name, value in existing.items()}, receipt)
    elif args.require_complete:
        raise ValueError("No complete marker yet")
    (args.output / "observation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"observed_at_utc": report["observed_at_utc"], "present": list(existing), "acceptance": report.get("acceptance")}))


if __name__ == "__main__":
    main()
