from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.candidate_metrics import evaluate_candidate_outputs


def _pilot_records(
    records: list[dict[str, object]],
    pilot_size: int | None,
    seed: int,
) -> list[dict[str, object]]:
    if pilot_size is None:
        return records
    if pilot_size <= 0 or pilot_size > len(records):
        raise ValueError("pilot_size must be within the dataset size")

    def priority(row: dict[str, object]) -> str:
        material = (
            f"{seed}:gpt-pilot:{row['selection_id']}".encode("utf-8")
        )
        return hashlib.sha256(material).hexdigest()

    return sorted(
        records,
        key=lambda row: (priority(row), str(row["selection_id"])),
    )[:pilot_size]


def _is_retryable(error: Exception) -> tuple[bool, int | None]:
    status_code = getattr(error, "status_code", None)
    if status_code in {401, 403}:
        return False, status_code
    if status_code in {408, 409, 429}:
        return True, status_code
    if isinstance(status_code, int) and status_code >= 500:
        return True, status_code
    name = type(error).__name__.lower()
    if "timeout" in name or "connection" in name:
        return True, status_code
    return False, status_code


def _call_one(
    client: Any,
    record: dict[str, object],
    *,
    deployment: str,
    reasoning_effort: str,
    max_completion_tokens: int,
    max_attempts: int,
) -> dict[str, object]:
    started_at = time.time()
    last_error: Exception | None = None
    last_status: int | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {
                        "role": "user",
                        "content": str(record["prompt"]),
                    }
                ],
                max_completion_tokens=max_completion_tokens,
                reasoning_effort=reasoning_effort,
            )
            usage = response.usage
            details = getattr(
                usage,
                "completion_tokens_details",
                None,
            )
            return {
                "selection_id": str(record["selection_id"]),
                "success": True,
                "output": response.choices[0].message.content or "",
                "attempts": attempt,
                "error_type": None,
                "error_message": None,
                "status_code": None,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(
                    usage,
                    "completion_tokens",
                    None,
                ),
                "reasoning_tokens": getattr(
                    details,
                    "reasoning_tokens",
                    None,
                ),
                "latency_seconds": time.time() - started_at,
            }
        except Exception as error:
            last_error = error
            retryable, last_status = _is_retryable(error)
            if last_status in {401, 403}:
                raise
            if not retryable or attempt == max_attempts:
                break
            time.sleep(min(30.0, 2 ** (attempt - 1) + attempt * 0.1))
    assert last_error is not None
    return {
        "selection_id": str(record["selection_id"]),
        "success": False,
        "output": "",
        "attempts": max_attempts,
        "error_type": type(last_error).__name__,
        "error_message": str(last_error)[:500],
        "status_code": last_status,
        "prompt_tokens": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "latency_seconds": time.time() - started_at,
    }


def _create_client(instance: str, api_version: str) -> Any:
    from azure.identity import (
        AzureCliCredential,
        ChainedTokenCredential,
        ManagedIdentityCredential,
        get_bearer_token_provider,
    )
    from openai import AzureOpenAI

    token_provider = get_bearer_token_provider(
        ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(),
        ),
        "api://trapi/.default",
    )
    return AzureOpenAI(
        azure_endpoint=(
            f"https://trapi.research.microsoft.com/{instance}"
        ),
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )


def evaluate_trapi(config: dict[str, Any]) -> dict[str, object]:
    data_file = Path(config["data_file"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    instance = str(config.get("instance", "redmond/interactive"))
    deployment = str(config["deployment"])
    api_version = str(config.get("api_version", "2024-10-21"))
    reasoning_effort = str(config.get("reasoning_effort", "medium"))
    max_completion_tokens = int(
        config.get("max_completion_tokens", 256)
    )
    concurrency = int(config.get("concurrency", 16))
    checkpoint_size = int(config.get("checkpoint_size", 64))
    max_attempts = int(config.get("max_attempts", 5))
    retry_failed_on_resume = bool(
        config.get("retry_failed_on_resume", True)
    )
    pilot_size = config.get("pilot_size")
    if pilot_size is not None:
        pilot_size = int(pilot_size)
    seed = int(config.get("seed", 20260806))
    if concurrency <= 0 or checkpoint_size <= 0 or max_attempts <= 0:
        raise ValueError(
            "concurrency, checkpoint_size, and max_attempts must be positive"
        )

    all_records = pq.read_table(data_file).to_pylist()
    records = _pilot_records(all_records, pilot_size, seed)
    client = _create_client(instance, api_version)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    request_rows: list[dict[str, object]] = []
    generated_samples = 0
    resumed_samples = 0
    started_at = time.time()

    for start in range(0, len(records), checkpoint_size):
        stop = min(start + checkpoint_size, len(records))
        shard_path = shards_dir / f"{start:09d}-{stop:09d}.parquet"
        if shard_path.is_file():
            shard_rows = pq.read_table(shard_path).to_pylist()
            expected_ids = [
                str(row["selection_id"]) for row in records[start:stop]
            ]
            actual_ids = [
                str(row["selection_id"]) for row in shard_rows
            ]
            if actual_ids != expected_ids:
                raise ValueError(
                    f"{shard_path}: selection IDs do not match data"
                )
            failed_offsets = [
                offset
                for offset, row in enumerate(shard_rows)
                if not row["success"]
            ]
            retried_count = 0
            if retry_failed_on_resume and failed_offsets:
                results_by_offset: dict[int, dict[str, object]] = {}
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=concurrency
                ) as executor:
                    futures = {
                        executor.submit(
                            _call_one,
                            client,
                            records[start + offset],
                            deployment=deployment,
                            reasoning_effort=reasoning_effort,
                            max_completion_tokens=max_completion_tokens,
                            max_attempts=max_attempts,
                        ): offset
                        for offset in failed_offsets
                    }
                    for future in concurrent.futures.as_completed(
                        futures
                    ):
                        offset = futures[future]
                        results_by_offset[offset] = future.result()
                for offset, result in results_by_offset.items():
                    shard_rows[offset] = result
                with tempfile.NamedTemporaryFile(
                    suffix=".parquet",
                    prefix=f".{shard_path.name}.",
                    dir=shards_dir,
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                try:
                    pq.write_table(
                        pa.Table.from_pylist(shard_rows),
                        temporary,
                        compression="zstd",
                        use_dictionary=True,
                    )
                    os.replace(temporary, shard_path)
                finally:
                    temporary.unlink(missing_ok=True)
                generated_samples += len(failed_offsets)
                retried_count = len(failed_offsets)
            request_rows.extend(shard_rows)
            resumed_samples += len(shard_rows) - retried_count
            continue

        chunk = records[start:stop]
        results_by_id: dict[str, dict[str, object]] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency
        ) as executor:
            futures = {
                executor.submit(
                    _call_one,
                    client,
                    row,
                    deployment=deployment,
                    reasoning_effort=reasoning_effort,
                    max_completion_tokens=max_completion_tokens,
                    max_attempts=max_attempts,
                ): str(row["selection_id"])
                for row in chunk
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results_by_id[str(result["selection_id"])] = result
        shard_rows = [
            results_by_id[str(row["selection_id"])] for row in chunk
        ]
        with tempfile.NamedTemporaryFile(
            suffix=".parquet",
            prefix=f".{shard_path.name}.",
            dir=shards_dir,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            pq.write_table(
                pa.Table.from_pylist(shard_rows),
                temporary,
                compression="zstd",
                use_dictionary=True,
            )
            os.replace(temporary, shard_path)
        finally:
            temporary.unlink(missing_ok=True)
        request_rows.extend(shard_rows)
        generated_samples += len(shard_rows)

    outputs = [str(row["output"]) for row in request_rows]
    metrics, details = evaluate_candidate_outputs(records, outputs)
    successful = sum(bool(row["success"]) for row in request_rows)
    runtime_seconds = time.time() - started_at
    latencies = [
        float(row["latency_seconds"])
        for row in request_rows
        if row["success"]
    ]
    metrics.update(
        {
            "instance": instance,
            "deployment": deployment,
            "api_version": api_version,
            "reasoning_effort": reasoning_effort,
            "samples_requested": len(records),
            "request_successes": successful,
            "request_failures": len(records) - successful,
            "request_success_rate": successful / len(records),
            "total_prompt_tokens": sum(
                int(row["prompt_tokens"] or 0) for row in request_rows
            ),
            "total_completion_tokens": sum(
                int(row["completion_tokens"] or 0)
                for row in request_rows
            ),
            "total_reasoning_tokens": sum(
                int(row["reasoning_tokens"] or 0)
                for row in request_rows
            ),
            "mean_latency_seconds": (
                sum(latencies) / len(latencies) if latencies else None
            ),
            "runtime_seconds": runtime_seconds,
            "generated_samples_this_run": generated_samples,
            "resumed_samples": resumed_samples,
            "pilot_size": pilot_size,
        }
    )
    prediction_rows = [
        {
            **detail,
            **{
                key: request[key]
                for key in (
                    "success",
                    "attempts",
                    "error_type",
                    "error_message",
                    "status_code",
                    "prompt_tokens",
                    "completion_tokens",
                    "reasoning_tokens",
                    "latency_seconds",
                )
            },
        }
        for detail, request in zip(details, request_rows)
    ]
    pq.write_table(
        pa.Table.from_pylist(prediction_rows),
        output_dir / "predictions.parquet",
        compression="zstd",
        use_dictionary=True,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    safe_config = {
        key: value
        for key, value in config.items()
        if "token" not in key.lower() and "credential" not in key.lower()
    }
    (output_dir / "resolved_config.json").write_text(
        json.dumps(safe_config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    minimum_success_rate = float(
        config.get("minimum_success_rate", 0.99)
    )
    if metrics["request_success_rate"] < minimum_success_rate:
        raise RuntimeError(
            f"TRAPI success rate {metrics['request_success_rate']:.4f} "
            f"is below {minimum_success_rate:.4f}"
        )
    return metrics


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate candidate selection through TRAPI."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pilot-size", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--max-attempts", type=int)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = _load_json(args.config)
    for key in (
        "data_file",
        "output_dir",
        "pilot_size",
        "concurrency",
        "max_attempts",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    print(json.dumps(evaluate_trapi(config), indent=2))


if __name__ == "__main__":
    main()
