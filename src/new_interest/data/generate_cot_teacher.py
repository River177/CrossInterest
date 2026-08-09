from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.infrastructure.paths import portable_path

TRAPI_INSTANCE = "redmond/interactive"
TRAPI_DEPLOYMENT = "gpt-5.6-sol_2026-07-09"
TRAPI_API_VERSION = "2024-10-21"
TRAPI_SCOPE = "api://trapi/.default"
MAX_COMPLETION_TOKENS = 256
MAX_ATTEMPTS = 3
TASK_TYPES = ("behavior_only", "interest_only")
RULE_VERSION = "gpt56-sol-cot-teacher-v1"

_THINK_TAG = re.compile(r"<\s*/?\s*think\b", re.IGNORECASE)
_NEW_TOPIC = re.compile(r"\bnew\s+topic\b", re.IGNORECASE)

TEACHER_RECORD_SCHEMA = pa.schema(
    [
        pa.field("request_hash", pa.string(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("source_sample_id", pa.string(), nullable=False),
        pa.field("impression_id", pa.string(), nullable=False),
        pa.field("task_type", pa.string(), nullable=False),
        pa.field("original_prompt", pa.string(), nullable=False),
        pa.field("selected_target", pa.string(), nullable=False),
        pa.field(
            "history_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field("status", pa.string(), nullable=False),
        pa.field("filter_reason", pa.string()),
        pa.field("reasoning", pa.string(), nullable=False),
        pa.field("raw_response", pa.string(), nullable=False),
        pa.field("attempts", pa.int16(), nullable=False),
        pa.field("error_type", pa.string()),
        pa.field("error_message", pa.string()),
        pa.field("status_code", pa.int32()),
        pa.field("prompt_tokens", pa.int64()),
        pa.field("completion_tokens", pa.int64()),
        pa.field("reasoning_tokens", pa.int64()),
        pa.field("latency_seconds", pa.float64(), nullable=False),
    ]
)


@dataclass(frozen=True)
class GenerationResult:
    output_dir: Path
    requested: int
    successes: int
    generated_this_run: int
    resumed: int


class InvalidTeacherResponse(ValueError):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _size_name(size: int) -> str:
    return f"{size // 1_000}k" if size % 1_000 == 0 else str(size)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_parquet_write(
    rows: list[dict[str, object]],
    path: Path,
    schema: pa.Schema,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.pending"
    try:
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schema),
            temporary,
            compression="zstd",
            use_dictionary=True,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_write(value: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.pending"
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def extract_interest_history(prompt: str) -> list[str]:
    try:
        history_block = prompt.split("\n\nBased on ", 1)[0]
        numbered_history = history_block.split("\n\n", 1)[1]
    except (IndexError, ValueError) as error:
        raise ValueError("Cannot parse Interest-only history") from error
    history: list[str] = []
    for line in numbered_history.splitlines():
        _, separator, value = line.partition(". ")
        if not separator or not value.strip():
            raise ValueError(f"Invalid numbered history line: {line!r}")
        history.append(value.strip())
    if not history:
        raise ValueError("Interest-only history is empty")
    return history


def validate_reasoning(reasoning: str) -> str:
    value = reasoning.strip()
    if not value:
        raise InvalidTeacherResponse(
            "empty_reasoning",
            "Teacher reasoning must be nonempty",
        )
    if _THINK_TAG.search(value):
        raise InvalidTeacherResponse(
            "nested_think_tag",
            "Teacher reasoning contains a think tag",
        )
    if _NEW_TOPIC.search(value):
        raise InvalidTeacherResponse(
            "new_topic_in_reasoning",
            "Teacher reasoning contains 'New topic'",
        )
    return value


def parse_teacher_response(raw_response: str) -> str:
    try:
        value = json.loads(raw_response)
    except json.JSONDecodeError as error:
        raise InvalidTeacherResponse(
            "invalid_json",
            f"Teacher response is not valid JSON: {error}",
        ) from error
    if not isinstance(value, dict):
        raise InvalidTeacherResponse(
            "not_json_object",
            "Teacher response must be a JSON object",
        )
    if set(value) != {"reasoning"}:
        raise InvalidTeacherResponse(
            "unexpected_json_keys",
            "Teacher response must contain only the reasoning field",
        )
    if not isinstance(value["reasoning"], str):
        raise InvalidTeacherResponse(
            "reasoning_not_string",
            "Teacher reasoning must be a string",
        )
    return validate_reasoning(value["reasoning"])


def build_teacher_prompt(original_prompt: str, selected_target: str) -> str:
    return (
        "Generate a rationale that connects the user's history to "
        "the selected target. Return exactly one JSON object with exactly "
        'one string field named "reasoning". The reasoning must be nonempty. '
        "Do not emit <think> tags, an answer, or the phrase \"New topic\".\n\n"
        "Original student prompt:\n"
        f"{original_prompt}\n\n"
        "Selected target:\n"
        f"{json.dumps(selected_target, ensure_ascii=False)}"
    )


def _request_hash(
    row: dict[str, object],
    *,
    deployment: str,
    api_version: str,
) -> str:
    material = {
        "rule_version": RULE_VERSION,
        "deployment": deployment,
        "api_version": api_version,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "sample_id": row["sample_id"],
        "source_sample_id": row["source_sample_id"],
        "task_type": row["task_type"],
        "original_prompt": row["original_prompt"],
        "selected_target": row["selected_target"],
        "teacher_prompt": row["teacher_prompt"],
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_source_requests(
    source_dir: Path,
    source_size: int,
    *,
    deployment: str,
    api_version: str,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    size_name = _size_name(source_size)
    source_paths = {
        task_type: source_dir
        / "train"
        / task_type
        / f"{size_name}.parquet"
        for task_type in TASK_TYPES
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    columns = [
        "sample_id",
        "source_sample_id",
        "impression_id",
        "task_type",
        "prompt",
        "response",
        "target_subcategory",
    ]
    rows_by_task = {
        task_type: pq.read_table(path, columns=columns).to_pylist()
        for task_type, path in source_paths.items()
    }
    for task_type, rows in rows_by_task.items():
        if len(rows) != source_size:
            raise ValueError(
                f"{source_paths[task_type]}: expected {source_size} rows, "
                f"got {len(rows)}"
            )

    interest_by_id: dict[str, dict[str, object]] = {}
    histories: dict[str, list[str]] = {}
    for row in rows_by_task["interest_only"]:
        source_id = str(row["source_sample_id"])
        if source_id in interest_by_id:
            raise ValueError(f"Duplicate Interest source ID: {source_id}")
        if row["task_type"] != "interest_only":
            raise ValueError(f"{source_id}: Interest task_type mismatch")
        target = str(row["target_subcategory"])
        if row["response"] != f"New topic: {target}":
            raise ValueError(f"{source_id}: response and target differ")
        history = extract_interest_history(str(row["prompt"]))
        if target.casefold() in {item.casefold() for item in history}:
            raise ValueError(f"{source_id}: target appears in history")
        interest_by_id[source_id] = row
        histories[source_id] = history

    requests_by_task: dict[str, list[dict[str, object]]] = {}
    for task_type in TASK_TYPES:
        task_requests: list[dict[str, object]] = []
        seen_ids: set[str] = set()
        for row in rows_by_task[task_type]:
            source_id = str(row["source_sample_id"])
            if source_id in seen_ids:
                raise ValueError(
                    f"{task_type}: duplicate source ID {source_id}"
                )
            seen_ids.add(source_id)
            if row["task_type"] != task_type:
                raise ValueError(f"{source_id}: task_type mismatch")
            paired = interest_by_id.get(source_id)
            if paired is None:
                raise ValueError(
                    f"{task_type}: source ID {source_id} has no Interest row"
                )
            target = str(row["target_subcategory"])
            if target != str(paired["target_subcategory"]):
                raise ValueError(f"{source_id}: targets do not match")
            if str(row["impression_id"]) != str(paired["impression_id"]):
                raise ValueError(f"{source_id}: impression IDs do not match")
            if row["response"] != f"New topic: {target}":
                raise ValueError(f"{source_id}: response and target differ")
            original_prompt = str(row["prompt"])
            teacher_prompt = build_teacher_prompt(
                original_prompt,
                target,
            )
            request = {
                "sample_id": str(row["sample_id"]),
                "source_sample_id": source_id,
                "impression_id": str(row["impression_id"]),
                "task_type": task_type,
                "original_prompt": original_prompt,
                "selected_target": target,
                "history_subcategories": histories[source_id],
                "teacher_prompt": teacher_prompt,
            }
            request["request_hash"] = _request_hash(
                request,
                deployment=deployment,
                api_version=api_version,
            )
            task_requests.append(request)
        requests_by_task[task_type] = task_requests

    requests = [
        request
        for index in range(source_size)
        for task_type in TASK_TYPES
        for request in [requests_by_task[task_type][index]]
    ]

    source_hashes = {
        str(path.relative_to(source_dir)): _sha256(path)
        for path in source_paths.values()
    }
    return requests, source_hashes


def _pilot_requests(
    requests: list[dict[str, object]],
    pilot_size: int | None,
    seed: int,
) -> list[dict[str, object]]:
    if pilot_size is not None and (
        pilot_size <= 0 or pilot_size > len(requests)
    ):
        raise ValueError("pilot_size must be within the request count")

    del seed
    return requests if pilot_size is None else requests[:pilot_size]


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


def _base_result(record: dict[str, object]) -> dict[str, object]:
    return {
        key: record[key]
        for key in (
            "request_hash",
            "sample_id",
            "source_sample_id",
            "impression_id",
            "task_type",
            "original_prompt",
            "selected_target",
            "history_subcategories",
        )
    }


def _call_teacher(
    client: Any,
    record: dict[str, object],
    *,
    deployment: str,
    max_completion_tokens: int = MAX_COMPLETION_TOKENS,
    max_attempts: int = MAX_ATTEMPTS,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    if max_completion_tokens != MAX_COMPLETION_TOKENS:
        raise ValueError("max_completion_tokens must be 256")
    if max_attempts <= 0 or max_attempts > MAX_ATTEMPTS:
        raise ValueError("max_attempts must be between 1 and 3")

    started_at = time.time()
    last_raw = ""
    last_error: Exception | None = None
    last_filter_reason: str | None = None
    last_status_code: int | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return only the requested JSON object. "
                            "Do not add Markdown fences or commentary."
                        ),
                    },
                    {
                        "role": "user",
                        "content": str(record["teacher_prompt"]),
                    },
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=max_completion_tokens,
            )
            last_raw = response.choices[0].message.content or ""
            reasoning = parse_teacher_response(last_raw)
            usage = getattr(response, "usage", None)
            details = getattr(usage, "completion_tokens_details", None)
            return {
                **_base_result(record),
                "status": "success",
                "filter_reason": None,
                "reasoning": reasoning,
                "raw_response": last_raw,
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
        except InvalidTeacherResponse as error:
            last_error = error
            last_filter_reason = error.reason
            if attempt < max_attempts:
                sleep_fn(float(5 * (2 ** (attempt - 1))))
                continue
            break
        except Exception as error:
            last_error = error
            retryable, last_status_code = _is_retryable(error)
            if retryable and attempt < max_attempts:
                sleep_fn(float(5 * (2 ** (attempt - 1))))
                continue
            break

    assert last_error is not None
    invalid_response = isinstance(last_error, InvalidTeacherResponse)
    return {
        **_base_result(record),
        "status": "filtered" if invalid_response else "api_error",
        "filter_reason": (
            last_filter_reason if invalid_response else "api_error"
        ),
        "reasoning": "",
        "raw_response": last_raw,
        "attempts": attempt,
        "error_type": type(last_error).__name__,
        "error_message": str(last_error)[:500],
        "status_code": last_status_code,
        "prompt_tokens": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "latency_seconds": time.time() - started_at,
    }


def _create_client(
    instance: str,
    api_version: str,
    scope: str,
) -> Any:
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
        scope,
    )
    return AzureOpenAI(
        azure_endpoint=f"https://trapi.research.microsoft.com/{instance}",
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )


def _validated_config(config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    resolved.setdefault("instance", TRAPI_INSTANCE)
    resolved.setdefault("deployment", TRAPI_DEPLOYMENT)
    resolved.setdefault("api_version", TRAPI_API_VERSION)
    resolved.setdefault("scope", TRAPI_SCOPE)
    resolved.setdefault("max_completion_tokens", MAX_COMPLETION_TOKENS)
    resolved.setdefault("max_attempts", MAX_ATTEMPTS)
    resolved.setdefault("source_size", 5_000)
    resolved.setdefault("concurrency", 8)
    resolved.setdefault("shard_size", 64)
    resolved.setdefault("pilot_seed", 20260809)
    resolved.setdefault("retry_failed_on_resume", True)

    required_values = {
        "instance": TRAPI_INSTANCE,
        "deployment": TRAPI_DEPLOYMENT,
        "api_version": TRAPI_API_VERSION,
        "scope": TRAPI_SCOPE,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
    }
    for key, expected in required_values.items():
        if resolved[key] != expected:
            raise ValueError(f"{key} must be {expected!r}")
    if "temperature" in resolved:
        raise ValueError("temperature must not be configured")
    max_attempts = int(resolved["max_attempts"])
    if max_attempts <= 0 or max_attempts > MAX_ATTEMPTS:
        raise ValueError("max_attempts must be between 1 and 3")
    for key in ("source_size", "concurrency", "shard_size"):
        if int(resolved[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    pilot_size = resolved.get("pilot_size")
    if pilot_size is not None and int(pilot_size) <= 0:
        raise ValueError("pilot_size must be positive or null")
    return resolved


def generate_teacher_reasoning(
    config: dict[str, Any],
    *,
    client: Any | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> GenerationResult:
    resolved = _validated_config(config)
    source_dir = Path(resolved["source_dir"])
    output_dir = Path(resolved["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    requests, source_hashes = _load_source_requests(
        source_dir,
        int(resolved["source_size"]),
        deployment=str(resolved["deployment"]),
        api_version=str(resolved["api_version"]),
    )
    pilot_size = resolved.get("pilot_size")
    selected = _pilot_requests(
        requests,
        int(pilot_size) if pilot_size is not None else None,
        int(resolved["pilot_seed"]),
    )

    shard_size = int(resolved["shard_size"])
    concurrency = int(resolved["concurrency"])
    max_attempts = int(resolved["max_attempts"])
    retry_failed = bool(resolved["retry_failed_on_resume"])
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, object]] = []
    generated_this_run = 0
    resumed = 0
    api_client = client
    expected_shards: set[Path] = set()

    def ensure_client() -> Any:
        nonlocal api_client
        if api_client is None:
            api_client = _create_client(
                str(resolved["instance"]),
                str(resolved["api_version"]),
                str(resolved["scope"]),
            )
        return api_client

    for start in range(0, len(selected), shard_size):
        stop = min(start + shard_size, len(selected))
        chunk = selected[start:stop]
        shard_path = shards_dir / f"{start:09d}-{stop:09d}.parquet"
        expected_shards.add(shard_path)
        shard_rows: list[dict[str, object]] | None = None
        retry_offsets: list[int] = []
        if shard_path.is_file():
            shard_rows = pq.read_table(shard_path).to_pylist()
            expected_identity = [
                (
                    str(row["request_hash"]),
                    str(row["sample_id"]),
                    str(row["task_type"]),
                )
                for row in chunk
            ]
            actual_identity = [
                (
                    str(row["request_hash"]),
                    str(row["sample_id"]),
                    str(row["task_type"]),
                )
                for row in shard_rows
            ]
            if actual_identity != expected_identity:
                raise ValueError(
                    f"{shard_path}: request identities do not match"
                )
            if retry_failed:
                retry_offsets = [
                    offset
                    for offset, row in enumerate(shard_rows)
                    if row["status"] != "success"
                ]
        else:
            retry_offsets = list(range(len(chunk)))
            shard_rows = [{} for _ in chunk]

        if retry_offsets:
            active_client = ensure_client()
            results_by_offset: dict[int, dict[str, object]] = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=concurrency
            ) as executor:
                futures = {
                    executor.submit(
                        _call_teacher,
                        active_client,
                        chunk[offset],
                        deployment=str(resolved["deployment"]),
                        max_completion_tokens=MAX_COMPLETION_TOKENS,
                        max_attempts=max_attempts,
                        sleep_fn=sleep_fn,
                    ): offset
                    for offset in retry_offsets
                }
                for future in concurrent.futures.as_completed(futures):
                    offset = futures[future]
                    results_by_offset[offset] = future.result()
            for offset, result in results_by_offset.items():
                shard_rows[offset] = result
            _atomic_parquet_write(
                shard_rows,
                shard_path,
                TEACHER_RECORD_SCHEMA,
            )
            generated_this_run += len(retry_offsets)
        resumed += len(chunk) - len(retry_offsets)
        all_results.extend(shard_rows)

    stale_shards = set(shards_dir.glob("*.parquet")) - expected_shards
    if stale_shards:
        names = ", ".join(sorted(path.name for path in stale_shards))
        raise ValueError(
            "Output contains shards outside this request selection: "
            f"{names}"
        )

    records_path = output_dir / "teacher_records.parquet"
    _atomic_parquet_write(
        all_results,
        records_path,
        TEACHER_RECORD_SCHEMA,
    )
    status_counts = Counter(str(row["status"]) for row in all_results)
    filter_counts = Counter(
        str(row["filter_reason"])
        for row in all_results
        if row["filter_reason"] is not None
    )
    shard_hashes = {
        str(path.relative_to(output_dir)): _sha256(path)
        for path in sorted(shards_dir.glob("*.parquet"))
    }
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rule_version": RULE_VERSION,
        "source_dir": portable_path(source_dir),
        "source_size": int(resolved["source_size"]),
        "instance": resolved["instance"],
        "deployment": resolved["deployment"],
        "api_version": resolved["api_version"],
        "scope": resolved["scope"],
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "temperature": None,
        "max_attempts": max_attempts,
        "pilot_size": (
            int(pilot_size) if pilot_size is not None else None
        ),
        "pilot_seed": int(resolved["pilot_seed"]),
        "request_selection": "paired-source-order-prefix",
        "requested": len(selected),
        "status_counts": dict(sorted(status_counts.items())),
        "filter_reason_counts": dict(sorted(filter_counts.items())),
        "generated_this_run": generated_this_run,
        "resumed": resumed,
        "source_sha256": source_hashes,
        "artifact_sha256": {
            "teacher_records.parquet": _sha256(records_path),
            **shard_hashes,
        },
    }
    _atomic_json_write(metadata, output_dir / "metadata.json")
    return GenerationResult(
        output_dir=output_dir,
        requested=len(selected),
        successes=status_counts["success"],
        generated_this_run=generated_this_run,
        resumed=resumed,
    )


def _load_config(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML object")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate GPT-5.6 Sol teacher reasoning through TRAPI."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cot_teacher/gpt56_sol.yaml"),
    )
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-size", type=int)
    pilot = parser.add_mutually_exclusive_group()
    pilot.add_argument("--pilot-size", type=int)
    pilot.add_argument("--full", action="store_true")
    parser.add_argument("--concurrency", type=int)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = _load_config(args.config)
    if args.source_dir is not None:
        config["source_dir"] = str(args.source_dir)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.source_size is not None:
        config["source_size"] = args.source_size
    if args.pilot_size is not None:
        config["pilot_size"] = args.pilot_size
    if args.full:
        config["pilot_size"] = None
    if args.concurrency is not None:
        config["concurrency"] = args.concurrency
    result = generate_teacher_reasoning(config)
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "requested": result.requested,
                "successes": result.successes,
                "generated_this_run": result.generated_this_run,
                "resumed": result.resumed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
