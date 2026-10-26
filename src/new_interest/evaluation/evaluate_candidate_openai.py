from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.evaluation.candidate_metrics import evaluate_candidate_outputs

DEFAULT_BASE_URL = "http://127.0.0.1:4141/v1"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_MODEL = "gemini-3.6-flash"
_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|authorization|credential|password|secret)",
    re.IGNORECASE,
)


def _pilot_records(
    records: list[dict[str, object]],
    pilot_size: int | None,
    seed: int,
) -> list[dict[str, object]]:
    if pilot_size is None:
        return records
    if pilot_size <= 0 or pilot_size > len(records):
        raise ValueError("pilot_size must be within the dataset size")

    def priority(row: dict[str, object]) -> tuple[str, str]:
        selection_id = str(row["selection_id"])
        material = f"{seed}:openai-pilot:{selection_id}".encode("utf-8")
        return hashlib.sha256(material).hexdigest(), selection_id

    return sorted(records, key=priority)[:pilot_size]


def _is_retryable(error: Exception) -> tuple[bool, int | None]:
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
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


def _require_api_key(config: dict[str, Any]) -> tuple[str, str]:
    env_name = str(config.get("api_key_env", DEFAULT_API_KEY_ENV))
    api_key = os.environ.get(env_name)
    if not api_key:
        raise RuntimeError(
            f"Required API key environment variable {env_name!r} is not set"
        )
    return env_name, api_key


def _sanitize_config(
    config: dict[str, Any],
    *,
    secrets: tuple[str, ...] = (),
) -> dict[str, Any]:
    def is_sensitive_key(key: str) -> bool:
        normalized = key.lower().replace("-", "_")
        if normalized.endswith("_env"):
            return False
        return bool(_SENSITIVE_KEY.search(key)) or normalized in {
            "token",
            "access_token",
            "auth_token",
            "bearer_token",
            "azure_ad_token",
        }

    def sanitize(value: Any, key: str | None = None) -> Any:
        if key and is_sensitive_key(key):
            return None
        if isinstance(value, dict):
            return {
                str(child_key): sanitized
                for child_key, child_value in value.items()
                if (
                    sanitized := sanitize(
                        child_value,
                        str(child_key),
                    )
                )
                is not None
            }
        if isinstance(value, (list, tuple)):
            return [sanitize(item) for item in value]
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, str):
            for secret in secrets:
                if secret:
                    value = value.replace(secret, "[REDACTED]")
        return value

    return sanitize(config)


def _safe_error_message(error: Exception, api_key: str) -> str:
    message = str(error)
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    message = re.sub(
        r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+",
        r"\1[REDACTED]",
        message,
    )
    return message[:500]


def _completion_kwargs(
    config: dict[str, Any],
    prompt: str,
) -> dict[str, object]:
    if (
        config.get("max_completion_tokens") is not None
        and config.get("max_tokens") is not None
    ):
        raise ValueError(
            "Configure only one of max_completion_tokens or max_tokens"
        )
    kwargs: dict[str, object] = {
        "model": str(config.get("model", DEFAULT_MODEL)),
        "messages": [{"role": "user", "content": prompt}],
    }
    for key in (
        "max_completion_tokens",
        "max_tokens",
        "temperature",
        "top_p",
        "seed",
        "reasoning_effort",
    ):
        if key in config and config[key] is not None:
            kwargs[key] = config[key]
    return kwargs


def _raw_response(response: Any) -> str:
    if hasattr(response, "model_dump_json"):
        return str(response.model_dump_json())
    if hasattr(response, "model_dump"):
        return json.dumps(response.model_dump(), default=str)
    if isinstance(response, dict):
        return json.dumps(response, default=str)
    return json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": getattr(
                            getattr(choice, "message", None),
                            "content",
                            None,
                        )
                    },
                    "finish_reason": getattr(
                        choice,
                        "finish_reason",
                        None,
                    ),
                }
                for choice in getattr(response, "choices", [])
            ],
            "usage": {
                "prompt_tokens": getattr(
                    getattr(response, "usage", None),
                    "prompt_tokens",
                    None,
                ),
                "completion_tokens": getattr(
                    getattr(response, "usage", None),
                    "completion_tokens",
                    None,
                ),
                "total_tokens": getattr(
                    getattr(response, "usage", None),
                    "total_tokens",
                    None,
                ),
            },
        },
        default=str,
    )


def _call_one(
    client: Any,
    record: dict[str, object],
    *,
    config: dict[str, Any],
    api_key: str,
    max_attempts: int,
) -> dict[str, object]:
    started_at = time.monotonic()
    last_error: Exception | None = None
    last_status: int | None = None
    attempts = 0
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        try:
            response = client.chat.completions.create(
                **_completion_kwargs(config, str(record["prompt"]))
            )
            choice = response.choices[0]
            message = choice.message
            usage = getattr(response, "usage", None)
            return {
                "selection_id": str(record["selection_id"]),
                "success": True,
                "output": getattr(message, "content", None) or "",
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
                "total_tokens": getattr(usage, "total_tokens", None),
                "latency_seconds": time.monotonic() - started_at,
                "finish_reason": getattr(choice, "finish_reason", None),
                "reasoning_text": getattr(
                    message,
                    "reasoning_text",
                    None,
                ),
                "raw_response": _raw_response(response),
            }
        except Exception as error:
            last_error = error
            retryable, last_status = _is_retryable(error)
            if not retryable or attempt == max_attempts:
                break
            time.sleep(min(30.0, 2 ** (attempt - 1) + attempt * 0.1))
    assert last_error is not None
    return {
        "selection_id": str(record["selection_id"]),
        "success": False,
        "output": "",
        "attempts": attempts,
        "error_type": type(last_error).__name__,
        "error_message": _safe_error_message(last_error, api_key),
        "status_code": last_status,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "latency_seconds": time.monotonic() - started_at,
        "finish_reason": None,
        "reasoning_text": None,
        "raw_response": None,
    }


def _create_client(config: dict[str, Any], api_key: str) -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key=api_key,
        base_url=str(config.get("base_url", DEFAULT_BASE_URL)),
        timeout=float(config.get("request_timeout_seconds", 120.0)),
        max_retries=0,
    )


def _model_ids(response: Any) -> list[str]:
    data = response.get("data", []) if isinstance(response, dict) else response.data
    ids = []
    for item in data:
        model_id = item.get("id") if isinstance(item, dict) else item.id
        ids.append(str(model_id))
    return ids


def _run_preflight(client: Any, config: dict[str, Any]) -> dict[str, object]:
    model = str(config.get("model", DEFAULT_MODEL))
    try:
        model_ids = _model_ids(client.models.list())
    except Exception as error:
        status = getattr(error, "status_code", None)
        raise RuntimeError(
            "OpenAI preflight model listing failed "
            f"({type(error).__name__}, status={status})"
        ) from None
    if model not in model_ids:
        raise RuntimeError(
            f"OpenAI preflight model {model!r} was not returned by /models"
        )
    try:
        response = client.chat.completions.create(
            **_completion_kwargs(
                config,
                "Reply with exactly OK.",
            )
        )
        if not getattr(response, "choices", None):
            raise ValueError("completion returned no choices")
    except Exception as error:
        status = getattr(error, "status_code", None)
        raise RuntimeError(
            "OpenAI preflight completion failed "
            f"({type(error).__name__}, status={status})"
        ) from None
    return {
        "base_url": str(config.get("base_url", DEFAULT_BASE_URL)),
        "model": model,
        "model_count": len(model_ids),
        "completion_succeeded": True,
    }


def preflight_openai(
    config: dict[str, Any],
    *,
    client: Any | None = None,
) -> dict[str, object]:
    _, api_key = _require_api_key(config)
    active_client = client or _create_client(config, api_key)
    return _run_preflight(active_client, config)


def _write_parquet_atomic(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    with tempfile.NamedTemporaryFile(
        suffix=".parquet",
        prefix=f".{path.name}.",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(
            pa.Table.from_pylist(rows),
            temporary,
            compression="zstd",
            use_dictionary=True,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_selection_ids(records: list[dict[str, object]]) -> None:
    selection_ids = [str(row["selection_id"]) for row in records]
    if len(selection_ids) != len(set(selection_ids)):
        raise ValueError("Candidate data contains duplicate selection IDs")


def evaluate_openai(
    config: dict[str, Any],
    *,
    client: Any | None = None,
) -> dict[str, object]:
    resolved = dict(config)
    resolved.setdefault("base_url", DEFAULT_BASE_URL)
    resolved.setdefault("api_key_env", DEFAULT_API_KEY_ENV)
    resolved.setdefault("model", DEFAULT_MODEL)
    resolved.setdefault("concurrency", 8)
    resolved.setdefault("checkpoint_size", 128)
    resolved.setdefault("max_attempts", 5)
    resolved.setdefault("minimum_success_rate", 0.99)
    resolved.setdefault("retry_failed_on_resume", True)
    resolved.setdefault("request_timeout_seconds", 120.0)
    resolved.setdefault("preflight", True)
    resolved.setdefault("pilot_seed", 20260808)

    data_file = Path(resolved["data_file"])
    output_dir = Path(resolved["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    concurrency = int(resolved["concurrency"])
    checkpoint_size = int(resolved["checkpoint_size"])
    max_attempts = int(resolved["max_attempts"])
    if concurrency <= 0 or checkpoint_size <= 0 or max_attempts <= 0:
        raise ValueError(
            "concurrency, checkpoint_size, and max_attempts must be positive"
        )
    pilot_size = resolved.get("pilot_size")
    if pilot_size is not None:
        pilot_size = int(pilot_size)
    _completion_kwargs(resolved, "validation")

    _, api_key = _require_api_key(resolved)
    active_client = client or _create_client(resolved, api_key)
    if bool(resolved["preflight"]):
        _run_preflight(active_client, resolved)

    all_records = pq.read_table(data_file).to_pylist()
    expected_samples = resolved.get("expected_samples")
    if (
        expected_samples is not None
        and len(all_records) != int(expected_samples)
    ):
        raise ValueError(
            f"Expected {int(expected_samples)} candidate rows, "
            f"found {len(all_records)}"
        )
    _validate_selection_ids(all_records)
    records = _pilot_records(
        all_records,
        pilot_size,
        int(resolved["pilot_seed"]),
    )

    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    request_rows: list[dict[str, object]] = []
    generated_samples = 0
    resumed_samples = 0
    started_at = time.monotonic()

    for start in range(0, len(records), checkpoint_size):
        stop = min(start + checkpoint_size, len(records))
        chunk = records[start:stop]
        shard_path = shards_dir / f"{start:09d}-{stop:09d}.parquet"
        if shard_path.is_file():
            shard_rows = pq.read_table(shard_path).to_pylist()
            expected_ids = [
                str(row["selection_id"]) for row in chunk
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
                if not bool(row["success"])
            ]
            retried_count = 0
            if bool(resolved["retry_failed_on_resume"]) and failed_offsets:
                results_by_offset: dict[int, dict[str, object]] = {}
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=concurrency
                ) as executor:
                    futures = {
                        executor.submit(
                            _call_one,
                            active_client,
                            chunk[offset],
                            config=resolved,
                            api_key=api_key,
                            max_attempts=max_attempts,
                        ): offset
                        for offset in failed_offsets
                    }
                    for future in concurrent.futures.as_completed(futures):
                        offset = futures[future]
                        results_by_offset[offset] = future.result()
                for offset, result in results_by_offset.items():
                    shard_rows[offset] = result
                _write_parquet_atomic(shard_rows, shard_path)
                generated_samples += len(failed_offsets)
                retried_count = len(failed_offsets)
            request_rows.extend(shard_rows)
            resumed_samples += len(shard_rows) - retried_count
            continue

        results_by_id: dict[str, dict[str, object]] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency
        ) as executor:
            futures = {
                executor.submit(
                    _call_one,
                    active_client,
                    row,
                    config=resolved,
                    api_key=api_key,
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
        _write_parquet_atomic(shard_rows, shard_path)
        request_rows.extend(shard_rows)
        generated_samples += len(shard_rows)

    outputs = [str(row["output"]) for row in request_rows]
    metrics, details = evaluate_candidate_outputs(records, outputs)
    successful = sum(bool(row["success"]) for row in request_rows)
    latencies = [
        float(row["latency_seconds"])
        for row in request_rows
        if bool(row["success"])
    ]
    runtime_seconds = time.monotonic() - started_at
    metrics.update(
        {
            "base_url": str(resolved["base_url"]),
            "model": str(resolved["model"]),
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
            "total_tokens": sum(
                int(row["total_tokens"] or 0) for row in request_rows
            ),
            "mean_latency_seconds": (
                sum(latencies) / len(latencies) if latencies else None
            ),
            "runtime_seconds": runtime_seconds,
            "generated_samples_this_run": generated_samples,
            "resumed_samples": resumed_samples,
            "pilot_size": pilot_size,
            "reasoning_effort": resolved.get("reasoning_effort"),
            "reasoning_output_count": sum(
                bool(row["reasoning_text"]) for row in request_rows
            ),
            "total_reasoning_characters": sum(
                len(str(row["reasoning_text"] or ""))
                for row in request_rows
            ),
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
                    "total_tokens",
                    "latency_seconds",
                    "finish_reason",
                    "reasoning_text",
                    "raw_response",
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
    safe_config = _sanitize_config(resolved, secrets=(api_key,))
    (output_dir / "resolved_config.json").write_text(
        json.dumps(safe_config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    minimum_success_rate = float(resolved["minimum_success_rate"])
    if metrics["request_success_rate"] < minimum_success_rate:
        raise RuntimeError(
            f"OpenAI success rate {metrics['request_success_rate']:.4f} "
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
        description="Evaluate candidate selection through an OpenAI API."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pilot-size", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--preflight-only", action="store_true")
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
    result = (
        preflight_openai(config)
        if args.preflight_only
        else evaluate_openai(config)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
