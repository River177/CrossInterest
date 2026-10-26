"""Read bounded original/retry stdout for the three frozen history ablation jobs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import requests
from azure.identity import AzureCliCredential

WORKSPACE = ("subscriptions/b6dc87f3-c479-49c8-8cb5-7896da3ff895/"
             "resourceGroups/rg-cs-newsandfeeds-singularity/providers/"
             "Microsoft.MachineLearningServices/workspaces/CS-NewsAndFeeds-Singularity")


def redact(text):
    text = re.sub(r"eyJ[A-Za-z0-9_.-]{50,}", "[REDACTED_JWT]", text)
    text = re.sub(r'(?i)(\b(?:accessToken|access_token|authorization|AZUREML_RUN_TOKEN|WANDB_API_KEY|HF_TOKEN)\b["\s]*[:=]\s*)[^\n]+',
                  r"\1[REDACTED]", text)
    return re.sub(r'https?://[^\s"<>]+', lambda m: m.group().split("?")[0] + "?[REDACTED_QUERY]"
                  if re.search(r"[?&](?:sig|token|access_token|sv)=", m.group(), re.I) else m.group(), text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jobs = json.loads(args.receipt.read_text())["jobs"]
    args.output.mkdir(parents=True, exist_ok=False)
    bearer = AzureCliCredential().get_token("https://management.azure.com/.default").token
    header = {"Authorization": "Bearer " + bearer}
    base = "https://ml.azure.com/api/eastus"

    def get(url, **kwargs):
        response = requests.get(url, headers=header, timeout=(15, 40), **kwargs)
        if response.status_code != 200:
            raise RuntimeError(f"Azure metadata HTTP {response.status_code}")
        return response.json()

    results = {"observed_at_utc": datetime.now(timezone.utc).isoformat(), "jobs": {}}
    for arm, job in jobs.items():
        backend = job["backend_id"]
        assert re.fullmatch(r"[A-Za-z0-9_-]+", backend)
        details = get(f"{base}/history/v1.0/{WORKSPACE}/runs/{backend}/details")
        names = list(details.get("logFiles") or {})
        selected = [name for name in names if re.fullmatch(r"user_logs/(retry_\d+/)?std_log.txt", name)]
        result = {"backend_id": backend, "native_status": details.get("status"), "streams": {}}
        for name in selected:
            info = get(f"{base}/artifact/v2.0/{WORKSPACE}/artifacts/ExperimentRun/dcid.{backend}/contentinfo", params={"path": name})
            uri = info["contentUri"]
            assert uri.startswith("https://")
            head = requests.head(uri, timeout=(15, 40))
            if head.status_code == 404:
                result["streams"][name] = {"pending": True, "reason": "metadata_visible_before_log_blob"}
                continue
            if head.status_code != 200:
                raise RuntimeError(f"Log HEAD HTTP {head.status_code}")
            size = int(head.headers.get("Content-Length", 0))
            if not size:
                result["streams"][name] = {"bytes": 0}
                continue
            response = requests.get(uri, headers={"Range": f"bytes={max(0, size-65536)}-"}, timeout=(15, 40))
            if response.status_code not in (200, 206):
                raise RuntimeError(f"Log GET HTTP {response.status_code}")
            clean = redact(response.content.decode(errors="replace")).replace("\r", "\n")
            filename = arm + "-" + name.replace("/", "-")
            (args.output / filename).write_text(clean)
            relevant = [line[-1200:] for line in clean.splitlines() if any(x in line for x in
                        ("HISTORY_", "EXECUTION_", "CHECKPOINT_", "Traceback", "Error", "loss", "/2025", "Successfully installed"))]
            result["streams"][name] = {"bytes": size, "last_modified": response.headers.get("Last-Modified"),
                                       "raw_tail_sha256": hashlib.sha256(response.content).hexdigest(),
                                       "file": filename, "relevant_tail": relevant[-6:]}
        results["jobs"][arm] = result
    (args.output / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
