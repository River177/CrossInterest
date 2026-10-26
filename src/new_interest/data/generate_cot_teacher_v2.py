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
CONCURRENCY = 4
RULE_VERSION = "gpt56-sol-cot-teacher-v2-native-thinking-prompt"

_DIRECT_ONLY_INSTRUCTION = (
    'Begin your response with "New topic: " and provide only the category '
    "name after it."
)
_THINK_TAG = re.compile(r"<\s*/?\s*think\b", re.IGNORECASE)
_NEW_TOPIC = re.compile(r"\bnew\s+topic\b", re.IGNORECASE)
_META_TARGET = re.compile(
    r"\b(?:private target|reference (?:item|target|category)|"
    r"given target|ground truth|correct answer)\b",
    re.IGNORECASE,
)

TEACHER_RECORD_SCHEMA = pa.schema(
    [
        pa.field("request_hash", pa.string(), nullable=False),
        pa.field("selection_rank", pa.int16(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("source_sample_id", pa.string(), nullable=False),
        pa.field("impression_id", pa.string(), nullable=False),
        pa.field("task_type", pa.string(), nullable=False),
        pa.field("original_history", pa.string(), nullable=False),
        pa.field("original_prompt", pa.string(), nullable=False),
        pa.field("new_student_prompt", pa.string(), nullable=False),
        pa.field("selected_target", pa.string(), nullable=False),
        pa.field("history_length", pa.int32(), nullable=False),
        pa.field(
            "history_subcategories",
            pa.list_(pa.string()),
            nullable=False,
        ),
        pa.field("original_prompt_length", pa.int32(), nullable=False),
        pa.field("student_prompt_length", pa.int32(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("filter_reason", pa.string()),
        pa.field("reasoning", pa.string(), nullable=False),
        pa.field("completion", pa.string(), nullable=False),
        pa.field("reasoning_length", pa.int32(), nullable=False),
        pa.field("reasoning_word_count", pa.int32(), nullable=False),
        pa.field("completion_length", pa.int32(), nullable=False),
        pa.field("strict_valid", pa.bool_(), nullable=False),
        pa.field("target_name_violation", pa.bool_(), nullable=False),
        pa.field("think_tag_violation", pa.bool_(), nullable=False),
        pa.field("new_topic_violation", pa.bool_(), nullable=False),
        pa.field("empty_reasoning", pa.bool_(), nullable=False),
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


class InvalidTeacherResponse(ValueError):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class GenerationResult:
    output_dir: Path
    requested: int
    successes: int
    generated_this_run: int
    resumed: int


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def build_student_prompt(original_prompt: str) -> str:
    _, separator, final_paragraph = original_prompt.rpartition("\n\n")
    if not separator or _DIRECT_ONLY_INSTRUCTION not in final_paragraph:
        raise ValueError(
            "Original prompt does not end with the direct-only instruction"
        )
    return original_prompt


def build_teacher_prompt(
    history_representation: str,
    selected_target: str,
    *,
    task_type: str,
) -> str:
    if task_type not in {"behavior_only", "interest_only"}:
        raise ValueError(f"Unsupported task_type: {task_type!r}")
    if not history_representation.strip():
        raise ValueError("History representation must be nonempty")
    if not selected_target.strip():
        raise ValueError("Selected target must be nonempty")
    return (
        "Write a rationale explaining why the private fixed target "
        "could plausibly follow the user's history. Base the rationale only "
        "on the history representation. Do not name, quote, spell, or "
        "otherwise reveal the private target. Return exactly one JSON "
        'object with exactly one string field named "reasoning". Do not '
        "refer to a private target, reference target, ground truth, or "
        "correct answer. Do not "
        "return an answer, Markdown, <think> tags, or the phrase "
        '"New topic".\n\n'
        f"History representation ({task_type}):\n"
        f"{history_representation}\n\n"
        "Private fixed target:\n"
        f"{json.dumps(selected_target.strip(), ensure_ascii=False)}"
    )


def validate_reasoning(reasoning: str, selected_target: str) -> str:
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
    if _META_TARGET.search(value):
        raise InvalidTeacherResponse(
            "meta_target_reference",
            "Teacher reasoning refers to a private/reference target",
        )
    normalized_target = _normalize_whitespace(selected_target)
    if not normalized_target:
        raise ValueError("Selected target must be nonempty")
    if (
        normalized_target.casefold()
        in _normalize_whitespace(value).casefold()
    ):
        raise InvalidTeacherResponse(
            "target_name_in_reasoning",
            "Teacher reasoning contains the canonical target name",
        )
    return value


def parse_teacher_response(
    raw_response: str,
    *,
    selected_target: str,
) -> str:
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
    return validate_reasoning(value["reasoning"], selected_target)


def format_completion(reasoning: str, selected_target: str) -> str:
    clean_reasoning = validate_reasoning(reasoning, selected_target)
    clean_target = selected_target.strip()
    if not clean_target:
        raise ValueError("Selected target must be nonempty")
    if "\n" in clean_target or "\r" in clean_target:
        raise ValueError("Selected target must be a single line")
    return (
        f"<think>\n{clean_reasoning}\n</think>\n"
        f"New topic: {clean_target}"
    )


def _base_result(record: dict[str, object]) -> dict[str, object]:
    return {
        key: record[key]
        for key in (
            "request_hash",
            "selection_rank",
            "sample_id",
            "source_sample_id",
            "impression_id",
            "task_type",
            "original_history",
            "original_prompt",
            "new_student_prompt",
            "selected_target",
            "history_length",
            "history_subcategories",
            "original_prompt_length",
            "student_prompt_length",
        )
    }


def _validation_flags(
    filter_reason: str | None,
) -> dict[str, bool]:
    return {
        "strict_valid": filter_reason is None,
        "target_name_violation": (
            filter_reason == "target_name_in_reasoning"
        ),
        "think_tag_violation": filter_reason == "nested_think_tag",
        "new_topic_violation": (
            filter_reason == "new_topic_in_reasoning"
        ),
        "empty_reasoning": filter_reason == "empty_reasoning",
    }


def _is_retryable(error: Exception) -> tuple[bool, int | None]:
    status_code = getattr(error, "status_code", None)
    if status_code in {401, 403}:
        return False, status_code
    if status_code in {408, 409, 429}:
        return True, status_code
    if isinstance(status_code, int) and status_code >= 500:
        return True, status_code
    error_name = type(error).__name__.casefold()
    if "timeout" in error_name or "connection" in error_name:
        return True, status_code
    return False, status_code


def _call_teacher(
    client: Any,
    record: dict[str, object],
    *,
    deployment: str = TRAPI_DEPLOYMENT,
    max_completion_tokens: int = MAX_COMPLETION_TOKENS,
    max_attempts: int = MAX_ATTEMPTS,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    if deployment != TRAPI_DEPLOYMENT:
        raise ValueError(f"deployment must be {TRAPI_DEPLOYMENT!r}")
    if max_completion_tokens != MAX_COMPLETION_TOKENS:
        raise ValueError("max_completion_tokens must be 256")
    if max_attempts != MAX_ATTEMPTS:
        raise ValueError("max_attempts must be 3")

    started_at = time.time()
    last_raw = ""
    last_error: Exception | None = None
    last_filter_reason: str | None = None
    last_status_code: int | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return only the requested JSON object, without "
                            "Markdown fences or commentary."
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
            reasoning = parse_teacher_response(
                last_raw,
                selected_target=str(record["selected_target"]),
            )
            completion = format_completion(
                reasoning,
                str(record["selected_target"]),
            )
            usage = getattr(response, "usage", None)
            details = getattr(usage, "completion_tokens_details", None)
            return {
                **_base_result(record),
                "status": "success",
                "filter_reason": None,
                "reasoning": reasoning,
                "completion": completion,
                "reasoning_length": len(reasoning),
                "reasoning_word_count": len(reasoning.split()),
                "completion_length": len(completion),
                **_validation_flags(None),
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
        except Exception as error:
            last_error = error
            retryable, last_status_code = _is_retryable(error)
            if not retryable:
                break
        if attempt < MAX_ATTEMPTS:
            sleep_fn(float(5 * (2 ** (attempt - 1))))

    assert last_error is not None
    invalid_response = isinstance(last_error, InvalidTeacherResponse)
    return {
        **_base_result(record),
        "status": "filtered" if invalid_response else "api_error",
        "filter_reason": (
            last_filter_reason if invalid_response else "api_error"
        ),
        "reasoning": "",
        "completion": "",
        "reasoning_length": 0,
        "reasoning_word_count": 0,
        "completion_length": 0,
        **_validation_flags(
            last_filter_reason if invalid_response else "api_error"
        ),
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_parquet_write(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.parent / f".{path.name}.{uuid.uuid4().hex}.pending"
    try:
        pq.write_table(
            pa.Table.from_pylist(rows, schema=TEACHER_RECORD_SCHEMA),
            pending,
            compression="zstd",
            use_dictionary=True,
        )
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _atomic_json_write(
    value: dict[str, object],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.parent / f".{path.name}.{uuid.uuid4().hex}.pending"
    try:
        pending.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _history_representation(prompt: str) -> str:
    marker = "\n\nBased on "
    if marker not in prompt:
        raise ValueError("Original prompt has no final prediction paragraph")
    history = prompt.split(marker, 1)[0].strip()
    if not history:
        raise ValueError("Original history representation is empty")
    return history


def _request_hash(
    record: dict[str, object],
) -> str:
    material = {
        "rule_version": RULE_VERSION,
        "instance": TRAPI_INSTANCE,
        "deployment": TRAPI_DEPLOYMENT,
        "api_version": TRAPI_API_VERSION,
        "scope": TRAPI_SCOPE,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "sample_id": record["sample_id"],
        "source_sample_id": record["source_sample_id"],
        "task_type": record["task_type"],
        "original_history": record["original_history"],
        "new_student_prompt": record["new_student_prompt"],
        "selected_target": record["selected_target"],
        "teacher_prompt": record["teacher_prompt"],
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_selection(
    selection_path: Path,
) -> list[dict[str, object]]:
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    rows = pq.read_table(selection_path).to_pylist()
    if not rows:
        raise ValueError("Selection is empty")
    rows.sort(key=lambda row: int(row["selection_rank"]))
    ranks = [int(row["selection_rank"]) for row in rows]
    if ranks != list(range(len(rows))):
        raise ValueError(
            "Selection ranks must be contiguous and start at zero"
        )

    requests: list[dict[str, object]] = []
    seen_source_ids: set[str] = set()
    seen_ranks: set[int] = set()
    for row in rows:
        source_id = str(row["source_sample_id"])
        rank = int(row["selection_rank"])
        if source_id in seen_source_ids:
            raise ValueError(f"Duplicate selected source ID: {source_id}")
        if rank in seen_ranks:
            raise ValueError(f"Duplicate selection rank: {rank}")
        seen_source_ids.add(source_id)
        seen_ranks.add(rank)
        target = str(row["target_subcategory"]).strip()
        if not target:
            raise ValueError(f"{source_id}: target is empty")
        history_subcategories = [
            str(value) for value in row["history_subcategories"]
        ]
        history_length = int(row["history_length"])
        if len(history_subcategories) != history_length:
            raise ValueError(
                f"{source_id}: history length does not match selection"
            )
        normalized_history = {
            _normalize_whitespace(value).casefold()
            for value in history_subcategories
        }
        if _normalize_whitespace(target).casefold() in normalized_history:
            raise ValueError(f"{source_id}: target appears in history")

        for task_type, sample_field, prompt_field in (
            (
                "behavior_only",
                "behavior_sample_id",
                "behavior_original_prompt",
            ),
            (
                "interest_only",
                "interest_sample_id",
                "interest_original_prompt",
            ),
        ):
            original_prompt = str(row[prompt_field])
            original_history = _history_representation(original_prompt)
            new_student_prompt = build_student_prompt(original_prompt)
            teacher_prompt = build_teacher_prompt(
                original_history,
                target,
                task_type=task_type,
            )
            request: dict[str, object] = {
                "selection_rank": rank,
                "sample_id": str(row[sample_field]),
                "source_sample_id": source_id,
                "impression_id": str(row["impression_id"]),
                "task_type": task_type,
                "original_history": original_history,
                "original_prompt": original_prompt,
                "new_student_prompt": new_student_prompt,
                "selected_target": target,
                "history_length": history_length,
                "history_subcategories": history_subcategories,
                "original_prompt_length": len(original_prompt),
                "student_prompt_length": len(new_student_prompt),
                "teacher_prompt": teacher_prompt,
            }
            request["request_hash"] = _request_hash(request)
            requests.append(request)
    return requests


def _load_seed_records(
    path: Path | None,
) -> dict[tuple[str, str], dict[str, object]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(path)
    records: dict[tuple[str, str], dict[str, object]] = {}
    for row in pq.read_table(path).to_pylist():
        key = (str(row["source_sample_id"]), str(row["task_type"]))
        if key in records:
            raise ValueError(f"Duplicate seed teacher record: {key}")
        records[key] = row
    return records


def _import_seed_result(
    request: dict[str, object],
    seed: dict[str, object],
) -> dict[str, object]:
    identity_fields = (
        "sample_id",
        "source_sample_id",
        "impression_id",
        "task_type",
        "original_prompt",
        "selected_target",
    )
    for field in identity_fields:
        if str(seed[field]) != str(request[field]):
            raise ValueError(
                f"Seed record {request['source_sample_id']}/"
                f"{request['task_type']} has mismatched {field}"
            )
    if str(seed["status"]) != "success":
        raise ValueError("Only successful seed records can be imported")
    reasoning = validate_reasoning(
        str(seed["reasoning"]),
        str(request["selected_target"]),
    )
    completion = format_completion(
        reasoning,
        str(request["selected_target"]),
    )
    return {
        **_base_result(request),
        "status": "success",
        "filter_reason": None,
        "reasoning": reasoning,
        "completion": completion,
        "reasoning_length": len(reasoning),
        "reasoning_word_count": len(reasoning.split()),
        "completion_length": len(completion),
        **_validation_flags(None),
        "raw_response": str(seed.get("raw_response") or ""),
        "attempts": 0,
        "error_type": None,
        "error_message": None,
        "status_code": None,
        "prompt_tokens": seed.get("prompt_tokens"),
        "completion_tokens": seed.get("completion_tokens"),
        "reasoning_tokens": seed.get("reasoning_tokens"),
        "latency_seconds": 0.0,
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
    resolved.setdefault("concurrency", CONCURRENCY)
    resolved.setdefault("shard_size", 20)
    resolved.setdefault("retry_failed_on_resume", True)
    resolved.setdefault("seed_records_path", None)

    required = {
        "instance": TRAPI_INSTANCE,
        "deployment": TRAPI_DEPLOYMENT,
        "api_version": TRAPI_API_VERSION,
        "scope": TRAPI_SCOPE,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "max_attempts": MAX_ATTEMPTS,
        "concurrency": CONCURRENCY,
    }
    for key, expected in required.items():
        if resolved.get(key) != expected:
            raise ValueError(f"{key} must be {expected!r}")
    if "temperature" in resolved:
        raise ValueError("temperature must not be configured")
    if "selection_path" not in resolved:
        raise ValueError("selection_path is required")
    if "output_dir" not in resolved:
        raise ValueError("output_dir is required")
    if int(resolved["shard_size"]) <= 0:
        raise ValueError("shard_size must be positive")
    return resolved


def generate_teacher_reasoning(
    config: dict[str, Any],
    *,
    client: Any | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> GenerationResult:
    resolved = _validated_config(config)
    selection_path = Path(resolved["selection_path"])
    output_dir = Path(resolved["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    requests = _load_selection(selection_path)
    seed_records_path = (
        Path(resolved["seed_records_path"])
        if resolved["seed_records_path"] is not None
        else None
    )
    seed_records = _load_seed_records(seed_records_path)

    shard_size = int(resolved["shard_size"])
    retry_failed = bool(resolved["retry_failed_on_resume"])
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    api_client = client
    all_results: list[dict[str, object]] = []
    generated_this_run = 0
    resumed = 0
    imported = 0
    expected_shards: set[Path] = set()

    def ensure_client() -> Any:
        nonlocal api_client
        if api_client is None:
            api_client = _create_client(
                TRAPI_INSTANCE,
                TRAPI_API_VERSION,
                TRAPI_SCOPE,
            )
        return api_client

    for start in range(0, len(requests), shard_size):
        stop = min(start + shard_size, len(requests))
        chunk = requests[start:stop]
        shard_path = shards_dir / f"{start:09d}-{stop:09d}.parquet"
        expected_shards.add(shard_path)
        retry_offsets: list[int]
        if shard_path.is_file():
            shard_rows = pq.read_table(shard_path).to_pylist()
            expected_identity = [
                (
                    str(row["sample_id"]),
                    str(row["source_sample_id"]),
                    str(row["task_type"]),
                    str(row["original_prompt"]),
                    str(row["selected_target"]),
                )
                for row in chunk
            ]
            actual_identity = [
                (
                    str(row["sample_id"]),
                    str(row["source_sample_id"]),
                    str(row["task_type"]),
                    str(row["original_prompt"]),
                    str(row["selected_target"]),
                )
                for row in shard_rows
            ]
            if actual_identity != expected_identity:
                raise ValueError(
                    f"{shard_path}: request identities do not match"
                )
            retry_offsets = []
            for index, (row, request) in enumerate(
                zip(shard_rows, chunk)
            ):
                shard_rows[index] = {
                    **row,
                    **_base_result(request),
                }
                retry = row["status"] != "success" and retry_failed
                if row["status"] == "success":
                    try:
                        validate_reasoning(
                            str(row["reasoning"]),
                            str(request["selected_target"]),
                        )
                    except InvalidTeacherResponse:
                        retry = True
                if retry:
                    retry_offsets.append(index)
        else:
            shard_rows = []
            retry_offsets = []
            for offset, request in enumerate(chunk):
                key = (
                    str(request["source_sample_id"]),
                    str(request["task_type"]),
                )
                seed = seed_records.get(key)
                if seed is None:
                    shard_rows.append({})
                    retry_offsets.append(offset)
                else:
                    try:
                        shard_rows.append(
                            _import_seed_result(request, seed)
                        )
                        imported += 1
                    except InvalidTeacherResponse:
                        shard_rows.append({})
                        retry_offsets.append(offset)

        if retry_offsets:
            active_client = ensure_client()
            completed: dict[int, dict[str, object]] = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=CONCURRENCY
            ) as executor:
                futures = {
                    executor.submit(
                        _call_teacher,
                        active_client,
                        chunk[offset],
                        deployment=TRAPI_DEPLOYMENT,
                        max_completion_tokens=MAX_COMPLETION_TOKENS,
                        max_attempts=MAX_ATTEMPTS,
                        sleep_fn=sleep_fn,
                    ): offset
                    for offset in retry_offsets
                }
                for future in concurrent.futures.as_completed(futures):
                    completed[futures[future]] = future.result()
            for offset, result in completed.items():
                shard_rows[offset] = result
            _atomic_parquet_write(shard_rows, shard_path)
            generated_this_run += len(retry_offsets)
        resumed += len(chunk) - len(retry_offsets)
        all_results.extend(shard_rows)

    stale_shards = set(shards_dir.glob("*.parquet")) - expected_shards
    if stale_shards:
        names = ", ".join(sorted(path.name for path in stale_shards))
        raise ValueError(
            "Output contains shards outside this selection: " + names
        )

    records_path = output_dir / "teacher_records.parquet"
    _atomic_parquet_write(all_results, records_path)
    status_counts = Counter(str(row["status"]) for row in all_results)
    filter_counts = Counter(
        str(row["filter_reason"])
        for row in all_results
        if row["filter_reason"] is not None
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rule_version": RULE_VERSION,
        "selection_path": portable_path(selection_path),
        "selection_sha256": _sha256(selection_path),
        "instance": TRAPI_INSTANCE,
        "deployment": TRAPI_DEPLOYMENT,
        "api_version": TRAPI_API_VERSION,
        "scope": TRAPI_SCOPE,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "temperature": None,
        "max_attempts": MAX_ATTEMPTS,
        "concurrency": CONCURRENCY,
        "shard_size": shard_size,
        "requested": len(requests),
        "status_counts": dict(sorted(status_counts.items())),
        "filter_reason_counts": dict(sorted(filter_counts.items())),
        "generated_this_run": generated_this_run,
        "resumed": resumed,
        "imported_seed_records": sum(
            int(row["attempts"]) == 0 for row in all_results
        ),
        "imported_seed_records_this_run": imported,
        "seed_records_path": (
            portable_path(seed_records_path)
            if seed_records_path is not None
            else None
        ),
        "artifact_sha256": {
            "teacher_records.parquet": _sha256(records_path),
            **{
                str(path.relative_to(output_dir)): _sha256(path)
                for path in sorted(shards_dir.glob("*.parquet"))
            },
        },
    }
    metadata_path = Path(
        resolved.get("metadata_path", output_dir / "metadata.json")
    )
    _atomic_json_write(metadata, metadata_path)
    return GenerationResult(
        output_dir=output_dir,
        requested=len(requests),
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
        description=(
            "Generate target-hidden GPT-5.6 Sol CoT v2 teacher records."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/cot_teacher/gpt56_sol_v2_preview.yaml"
        ),
    )
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = _load_config(args.config)
    if args.selection is not None:
        config["selection_path"] = str(args.selection)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
