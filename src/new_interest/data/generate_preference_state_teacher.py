from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "preference-state-transition-teacher-gpt55-v1"
TRAPI_INSTANCE = "redmond/interactive"
TRAPI_DEPLOYMENT = "gpt-5.5_2026-04-24"
TRAPI_API_VERSION = "2024-10-21"
TRAPI_SCOPE = "api://trapi/.default"
REASONING_EFFORT = "none"
SELECTION_SEED = "20260902:preference-state-transition"
TRAIN_ROWS = 21_600
CLASSES = 108
PER_CLASS = 10
MAX_ATTEMPTS = 3

_FINAL_MARKER = "\n\nBased on both"
_HISTORY_CATEGORY = re.compile(r"^\d+\. \[([^\]]+)\] ", re.MULTILINE)
_SID = re.compile(r"<interest_\d{3}>", re.IGNORECASE)
_FORBIDDEN_TAG = re.compile(r"<\s*/?\s*(?:think|answer)\b", re.IGNORECASE)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_value(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        pending.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _atomic_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        pq.write_table(
            pa.Table.from_pylist(rows),
            pending,
            compression="zstd",
            use_dictionary=True,
        )
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def history_section(prompt: str) -> str:
    if _FINAL_MARKER not in prompt:
        raise ValueError("Behavior+Interest prompt has no canonical final marker")
    value = prompt.split(_FINAL_MARKER, 1)[0].strip()
    if not value:
        raise ValueError("History section is empty")
    return value


def history_subcategories(prompt: str) -> list[str]:
    values = [value.strip() for value in _HISTORY_CATEGORY.findall(prompt)]
    if not values:
        raise ValueError("Behavior+Interest prompt has no parseable categories")
    return values


def student_prompt(prompt: str) -> str:
    history = history_section(prompt)
    return (
        f"{history}\n\n"
        "Based on both the headlines and their topic categories, reason about "
        "a new topic category that I am likely to be interested in next. "
        "Inside <think>, briefly summarize the history pattern, infer the "
        "underlying preference state, and describe a plausible unseen interest "
        "expansion without naming its exact taxonomy category or any learned "
        "interest token. Then close </think> and output exactly one learned "
        "interest token inside <answer> and </answer>."
    )


def profile_prompt(history: str) -> str:
    return (
        "You are given only a user's chronological news-viewing history and do "
        "not know the future target. Produce a compact target-blind user state. "
        "Summarize recurring and recent patterns rather than listing every item, "
        "then infer the underlying content preferences or viewing motivations. "
        "Do not predict a next category. Return exactly one JSON object with "
        'exactly two string fields: "history_summary" and "preference_state". '
        "history_summary must be one sentence of at most 32 words; "
        "preference_state must be one sentence of at most 28 words. If the "
        "history is too sparse for a strong inference, say so directly instead "
        "of inventing a stable preference.\n\n"
        f"History:\n{history}"
    )


def transition_prompt(
    *,
    history_summary: str,
    preference_state: str,
    target_subcategory: str,
    target_category: str,
) -> str:
    return (
        "Using the frozen target-blind user state below, describe why the private "
        "future interest could be a plausible unseen expansion. Describe the "
        "semantic bridge, but do not name, quote, spell, or directly paraphrase "
        "the private fine-grained target; do not emit an interest token, answer, "
        "prediction, or XML tag. Do not claim certainty or invent future events. "
        "Return exactly one JSON object with exactly one string field named "
        '"potential_expansion", containing exactly one sentence of at most 36 '
        "words. If the bridge is weak, express that uncertainty directly rather "
        "than overstating the evidence.\n\n"
        f"History summary:\n{history_summary}\n\n"
        f"Preference state:\n{preference_state}\n\n"
        f"Private fine-grained target:\n{json.dumps(target_subcategory)}\n\n"
        f"Its broad taxonomy category:\n{json.dumps(target_category)}"
    )


def _clean_reasoning_field(
    value: object,
    *,
    field: str,
    max_words: int,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    cleaned = " ".join(value.split())
    if _FORBIDDEN_TAG.search(cleaned) or _SID.search(cleaned):
        raise ValueError(f"{field} contains a forbidden tag or Semantic ID")
    if len(cleaned.split()) > max_words:
        raise ValueError(f"{field} exceeds {max_words} words")
    return cleaned


def validate_profile(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {
        "history_summary",
        "preference_state",
    }:
        raise ValueError("Profile must contain exactly two required fields")
    return {
        "history_summary": _clean_reasoning_field(
            value["history_summary"],
            field="history_summary",
            max_words=32,
        ),
        "preference_state": _clean_reasoning_field(
            value["preference_state"],
            field="preference_state",
            max_words=28,
        ),
    }


def validate_transition(
    value: object,
    *,
    target_subcategory: str,
) -> str:
    if not isinstance(value, dict) or set(value) != {"potential_expansion"}:
        raise ValueError("Transition must contain only potential_expansion")
    expansion = _clean_reasoning_field(
        value["potential_expansion"],
        field="potential_expansion",
        max_words=36,
    )
    target = " ".join(target_subcategory.split())
    if target.casefold() in expansion.casefold():
        raise ValueError("Transition reveals the canonical target name")
    return expansion


def format_completion(
    *,
    history_summary: str,
    preference_state: str,
    potential_expansion: str,
    target_token: str,
) -> str:
    fields = {
        "History summary": _clean_reasoning_field(
            history_summary, field="history_summary", max_words=32
        ),
        "Preference state": _clean_reasoning_field(
            preference_state, field="preference_state", max_words=28
        ),
        "Potential expansion": _clean_reasoning_field(
            potential_expansion,
            field="potential_expansion",
            max_words=36,
        ),
    }
    if not re.fullmatch(r"<interest_\d{3}>", target_token):
        raise ValueError("target_token is not a canonical Semantic ID")
    reasoning = "\n".join(f"{name}: {text}" for name, text in fields.items())
    return f"<think>\n{reasoning}\n</think>\n<answer>{target_token}</answer>"


def _parent_mapping(vocabulary_file: Path) -> dict[str, str]:
    rows = pq.read_table(
        vocabulary_file, columns=["category", "subcategory"]
    ).to_pylist()
    result: dict[str, str] = {}
    for row in rows:
        subcategory = str(row["subcategory"])
        category = str(row["category"])
        if subcategory in result and result[subcategory] != category:
            raise ValueError(f"Ambiguous parent category for {subcategory}")
        result[subcategory] = category
    return result


def select_teacher_rows(
    *,
    train_file: Path,
    mapping_file: Path,
    vocabulary_file: Path,
    per_class: int = PER_CLASS,
) -> list[dict[str, object]]:
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    train = pq.read_table(train_file).to_pylist()
    if len(train) != TRAIN_ROWS:
        raise ValueError(f"Expected {TRAIN_ROWS} Train rows, got {len(train)}")
    mappings = pq.read_table(mapping_file).to_pylist()
    token_by_target = {
        str(row["target_subcategory"]): str(row["interest_token"])
        for row in mappings
    }
    if len(token_by_target) != CLASSES:
        raise ValueError("Semantic-ID mapping must contain 108 classes")
    parent_by_target = _parent_mapping(vocabulary_file)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in train:
        target = str(row["target_subcategory"])
        if target not in token_by_target or target not in parent_by_target:
            raise ValueError(f"Missing target mapping for {target}")
        categories = history_subcategories(str(row["prompt"]))
        if target.casefold() in {value.casefold() for value in categories}:
            raise ValueError(f"{row['source_sample_id']}: target appears in history")
        grouped[target].append(
            {
                "sample_id": str(row["sample_id"]),
                "source_sample_id": str(row["source_sample_id"]),
                "impression_id": str(row["impression_id"]),
                "user_id": str(row["user_id"]),
                "history_length": int(row["history_length"]),
                "history_subcategories": categories,
                "original_prompt": str(row["prompt"]),
                "student_prompt": student_prompt(str(row["prompt"])),
                "target_subcategory": target,
                "target_category": parent_by_target[target],
                "target_token": token_by_target[target],
            }
        )
    if set(grouped) != set(token_by_target):
        raise ValueError("Train classes do not match the Semantic-ID mapping")

    selected: list[dict[str, object]] = []
    for target in sorted(grouped):
        ordered = sorted(
            grouped[target],
            key=lambda row: (
                hashlib.sha256(
                    f"{SELECTION_SEED}:{row['source_sample_id']}".encode()
                ).hexdigest(),
                str(row["source_sample_id"]),
            ),
        )
        if len(ordered) < per_class:
            raise ValueError(f"{target}: fewer than {per_class} rows")
        for within_class_rank, row in enumerate(ordered[:per_class]):
            selected.append(
                {
                    **row,
                    "within_class_rank": within_class_rank,
                    "selection_priority": hashlib.sha256(
                        f"{SELECTION_SEED}:{row['source_sample_id']}".encode()
                    ).hexdigest(),
                }
            )
    selected.sort(
        key=lambda row: (
            int(row["within_class_rank"]),
            str(row["target_subcategory"]),
        )
    )
    for selection_rank, row in enumerate(selected):
        row["selection_rank"] = selection_rank
    return selected


def _safe_error(error: Exception) -> str:
    return " ".join(str(error).split())[:1000]


def _status_code(error: Exception) -> int | None:
    value = getattr(error, "status_code", None)
    if value is None:
        value = getattr(getattr(error, "response", None), "status_code", None)
    return int(value) if isinstance(value, int) else None


def _retryable(error: Exception) -> bool:
    if isinstance(error, (ValueError, json.JSONDecodeError)):
        return True
    status = _status_code(error)
    if status in {401, 403}:
        return False
    if status in {408, 409, 429} or (status is not None and status >= 500):
        return True
    name = type(error).__name__.casefold()
    return "timeout" in name or "connection" in name


def _call_json(
    client: Any,
    *,
    system: str,
    prompt: str,
    validator: Callable[[object], object],
    sleep_fn: Callable[[float], None],
) -> dict[str, object]:
    started = time.monotonic()
    last_error: Exception | None = None
    raw = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=TRAPI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                reasoning_effort=REASONING_EFFORT,
            )
            raw = str(response.choices[0].message.content or "")
            parsed = validator(json.loads(raw))
            usage = getattr(response, "usage", None)
            return {
                "status": "success",
                "parsed": parsed,
                "raw": raw,
                "attempts": attempt,
                "error_type": None,
                "error_message": None,
                "status_code": None,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "latency_seconds": time.monotonic() - started,
            }
        except Exception as error:
            last_error = error
            if not _retryable(error) or attempt == MAX_ATTEMPTS:
                break
            sleep_fn(min(30.0, float(2 ** attempt)))
    assert last_error is not None
    return {
        "status": "error",
        "parsed": None,
        "raw": raw,
        "attempts": attempt,
        "error_type": type(last_error).__name__,
        "error_message": _safe_error(last_error),
        "status_code": _status_code(last_error),
        "prompt_tokens": None,
        "completion_tokens": None,
        "latency_seconds": time.monotonic() - started,
    }


def _create_client(timeout_seconds: float) -> Any:
    from azure.identity import (
        AzureCliCredential,
        ChainedTokenCredential,
        ManagedIdentityCredential,
        get_bearer_token_provider,
    )
    from openai import AzureOpenAI

    credential = ChainedTokenCredential(
        AzureCliCredential(), ManagedIdentityCredential()
    )
    token_provider = get_bearer_token_provider(credential, TRAPI_SCOPE)
    return AzureOpenAI(
        azure_endpoint=f"https://trapi.research.microsoft.com/{TRAPI_INSTANCE}",
        azure_ad_token_provider=token_provider,
        api_version=TRAPI_API_VERSION,
        timeout=timeout_seconds,
        max_retries=0,
    )


def _process_row(
    client: Any,
    row: Mapping[str, object],
    *,
    sleep_fn: Callable[[float], None],
) -> dict[str, object]:
    history = history_section(str(row["original_prompt"]))
    profile_request = profile_prompt(history)
    profile = _call_json(
        client,
        system="Return only the requested target-blind JSON object.",
        prompt=profile_request,
        validator=validate_profile,
        sleep_fn=sleep_fn,
    )
    transition_request = ""
    transition: dict[str, object] = {
        "status": "skipped",
        "parsed": None,
        "raw": "",
        "attempts": 0,
        "error_type": None,
        "error_message": None,
        "status_code": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "latency_seconds": 0.0,
    }
    completion = ""
    if profile["status"] == "success":
        parsed_profile = dict(profile["parsed"])
        canonical_target = " ".join(
            str(row["target_subcategory"]).split()
        ).casefold()
        disclosed = any(
            canonical_target in str(parsed_profile[field]).casefold()
            for field in ("history_summary", "preference_state")
        )
        if disclosed:
            profile = {
                **profile,
                "status": "error",
                "error_type": "TargetDisclosure",
                "error_message": (
                    "Target-blind profile coincidentally reveals the canonical target"
                ),
            }
    if profile["status"] == "success":
        parsed_profile = dict(profile["parsed"])
        transition_request = transition_prompt(
            history_summary=str(parsed_profile["history_summary"]),
            preference_state=str(parsed_profile["preference_state"]),
            target_subcategory=str(row["target_subcategory"]),
            target_category=str(row["target_category"]),
        )
        transition = _call_json(
            client,
            system="Return only the requested target-hidden JSON object.",
            prompt=transition_request,
            validator=lambda value: validate_transition(
                value,
                target_subcategory=str(row["target_subcategory"]),
            ),
            sleep_fn=sleep_fn,
        )
        if transition["status"] == "success":
            completion = format_completion(
                history_summary=str(parsed_profile["history_summary"]),
                preference_state=str(parsed_profile["preference_state"]),
                potential_expansion=str(transition["parsed"]),
                target_token=str(row["target_token"]),
            )
    status = (
        "success"
        if profile["status"] == "success"
        and transition["status"] == "success"
        else "error"
    )
    identity = {
        "rule_version": RULE_VERSION,
        "instance": TRAPI_INSTANCE,
        "deployment": TRAPI_DEPLOYMENT,
        "api_version": TRAPI_API_VERSION,
        "reasoning_effort": REASONING_EFFORT,
        "source_sample_id": row["source_sample_id"],
        "student_prompt": row["student_prompt"],
        "target_subcategory": row["target_subcategory"],
        "target_category": row["target_category"],
        "target_token": row["target_token"],
        "profile_prompt": profile_request,
        "transition_prompt": transition_request,
    }
    return {
        **dict(row),
        "request_hash": _hash_value(identity),
        "status": status,
        "history_summary": (
            str(dict(profile["parsed"])["history_summary"])
            if profile["parsed"] is not None
            else ""
        ),
        "preference_state": (
            str(dict(profile["parsed"])["preference_state"])
            if profile["parsed"] is not None
            else ""
        ),
        "potential_expansion": (
            str(transition["parsed"])
            if transition["parsed"] is not None
            else ""
        ),
        "completion": completion,
        "profile_prompt": profile_request,
        "profile_raw": profile["raw"],
        "profile_attempts": profile["attempts"],
        "profile_error_type": profile["error_type"],
        "profile_error_message": profile["error_message"],
        "profile_status_code": profile["status_code"],
        "profile_prompt_tokens": profile["prompt_tokens"],
        "profile_completion_tokens": profile["completion_tokens"],
        "profile_latency_seconds": profile["latency_seconds"],
        "transition_prompt": transition_request,
        "transition_raw": transition["raw"],
        "transition_attempts": transition["attempts"],
        "transition_error_type": transition["error_type"],
        "transition_error_message": transition["error_message"],
        "transition_status_code": transition["status_code"],
        "transition_prompt_tokens": transition["prompt_tokens"],
        "transition_completion_tokens": transition["completion_tokens"],
        "transition_latency_seconds": transition["latency_seconds"],
    }


def generate(
    config: Mapping[str, object],
    *,
    client: Any | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    required = ("train_file", "mapping_file", "vocabulary_file", "output_dir")
    if any(not config.get(key) for key in required):
        raise ValueError(f"Config requires {required}")
    if config.get("deployment", TRAPI_DEPLOYMENT) != TRAPI_DEPLOYMENT:
        raise ValueError("GPT-5.5 deployment identity mismatch")
    if config.get("reasoning_effort", REASONING_EFFORT) != REASONING_EFFORT:
        raise ValueError("reasoning_effort must be none for the frozen preview")
    if "temperature" in config:
        raise ValueError("GPT-5.5 Teacher must not configure temperature")
    per_class = int(config.get("per_class", PER_CLASS))
    maximum = config.get("max_samples")
    maximum = int(maximum) if maximum is not None else None
    concurrency = int(config.get("concurrency", 4))
    checkpoint_size = int(config.get("checkpoint_size", 32))
    if (
        concurrency <= 0
        or checkpoint_size <= 0
        or maximum is not None
        and maximum <= 0
    ):
        raise ValueError("concurrency and max_samples must be positive")

    train_file = Path(str(config["train_file"]))
    mapping_file = Path(str(config["mapping_file"]))
    vocabulary_file = Path(str(config["vocabulary_file"]))
    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = select_teacher_rows(
        train_file=train_file,
        mapping_file=mapping_file,
        vocabulary_file=vocabulary_file,
        per_class=per_class,
    )
    chosen = selection[:maximum] if maximum is not None else selection
    selection_path = output_dir / "selection.parquet"
    if selection_path.is_file():
        previous_selection = pq.read_table(selection_path).to_pylist()
        previous_identity = [
            (
                int(row["selection_rank"]),
                str(row["source_sample_id"]),
                str(row["target_subcategory"]),
                str(row["target_token"]),
            )
            for row in previous_selection
        ]
        current_identity = [
            (
                int(row["selection_rank"]),
                str(row["source_sample_id"]),
                str(row["target_subcategory"]),
                str(row["target_token"]),
            )
            for row in chosen
        ]
        if previous_identity != current_identity:
            raise ValueError("Existing Teacher selection identity differs")
    else:
        _atomic_parquet(selection_path, chosen)
    if client is None:
        client = _create_client(float(config.get("timeout_seconds", 180.0)))
    records_path = output_dir / "teacher_records.parquet"
    reusable: dict[str, dict[str, object]] = {}
    if records_path.is_file():
        for row in pq.read_table(records_path).to_pylist():
            source_id = str(row["source_sample_id"])
            if source_id in reusable:
                raise ValueError(f"Duplicate existing Teacher row: {source_id}")
            if row["status"] == "success":
                reusable[source_id] = row

    results: list[dict[str, object] | None] = [None] * len(chosen)
    generated = 0
    resumed = 0
    for start in range(0, len(chosen), checkpoint_size):
        chunk = chosen[start : start + checkpoint_size]
        pending: list[tuple[int, dict[str, object]]] = []
        for offset, row in enumerate(chunk, start=start):
            existing = reusable.get(str(row["source_sample_id"]))
            if existing is not None:
                for key in (
                    "selection_rank",
                    "target_subcategory",
                    "target_category",
                    "target_token",
                    "original_prompt",
                    "student_prompt",
                ):
                    if existing[key] != row[key]:
                        raise ValueError(
                            f"Existing Teacher row differs in {key}: "
                            f"{row['source_sample_id']}"
                        )
                results[offset] = dict(existing)
                resumed += 1
            else:
                pending.append((offset, row))
        if pending:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=concurrency
            ) as pool:
                futures = {
                    pool.submit(
                        _process_row,
                        client,
                        row,
                        sleep_fn=sleep_fn,
                    ): offset
                    for offset, row in pending
                }
                for future in concurrent.futures.as_completed(futures):
                    results[futures[future]] = future.result()
                    generated += 1
        checkpoint_rows = [
            dict(row) for row in results if row is not None
        ]
        _atomic_parquet(records_path, checkpoint_rows)

    completed = [dict(row) for row in results if row is not None]
    if len(completed) != len(chosen):
        raise RuntimeError("Teacher generation did not fill every selected row")
    statuses = Counter(str(row["status"]) for row in completed)
    metadata = {
        "rule_version": RULE_VERSION,
        "instance": TRAPI_INSTANCE,
        "deployment": TRAPI_DEPLOYMENT,
        "api_version": TRAPI_API_VERSION,
        "reasoning_effort": REASONING_EFFORT,
        "temperature": None,
        "selection_seed": SELECTION_SEED,
        "per_class": per_class,
        "full_selection_rows": len(selection),
        "requested_rows": len(chosen),
        "generated_this_run": generated,
        "resumed_successes": resumed,
        "checkpoint_size": checkpoint_size,
        "status_counts": dict(sorted(statuses.items())),
        "source": {
            "train_file": portable_path(train_file),
            "train_sha256": _sha256(train_file),
            "mapping_file": portable_path(mapping_file),
            "mapping_sha256": _sha256(mapping_file),
            "vocabulary_file": portable_path(vocabulary_file),
            "vocabulary_sha256": _sha256(vocabulary_file),
        },
        "artifact_sha256": {
            "selection.parquet": _sha256(output_dir / "selection.parquet"),
            "teacher_records.parquet": _sha256(
                output_dir / "teacher_records.parquet"
            ),
        },
    }
    _atomic_json(output_dir / "metadata.json", metadata)
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.max_samples is not None:
        config["max_samples"] = args.max_samples
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    print(json.dumps(generate(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
