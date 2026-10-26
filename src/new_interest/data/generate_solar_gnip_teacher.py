from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.infrastructure.paths import portable_path


RULE_VERSION = "solar-gnip-gpt55-profile-rerank-v1"
TRAPI_INSTANCE = "redmond/interactive"
TRAPI_DEPLOYMENT = "gpt-5.5_2026-04-24"
TRAPI_API_VERSION = "2024-10-21"
TRAPI_SCOPE = "api://trapi/.default"
REASONING_EFFORT = "none"
CONCURRENCY = 4
CHECKPOINT_SIZE = 64
MAX_ATTEMPTS = 3
PROFILE_FIELDS = (
    "implicit_preference",
    "explicit_preference",
    "novelty_preference",
    "vague_intention",
    "specific_intention",
    "exploratory_intention",
)
DEFAULT_CONFIG = Path("configs/solar_gnip/gpt55_teacher_smoke_v1.json")
_RETRY_AFTER_SECONDS = re.compile(
    r"(?:retry|try)\s+again\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*seconds?",
    re.IGNORECASE,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _hash_value(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json_write(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.parent / f".{path.name}.{uuid.uuid4().hex}.pending"
    try:
        pending.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _atomic_parquet_write(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet",
        prefix=f".{path.name}.",
        dir=path.parent,
        delete=False,
    ) as handle:
        pending = Path(handle.name)
    try:
        pq.write_table(
            pa.Table.from_pylist([dict(row) for row in rows]),
            pending,
            compression="zstd",
            use_dictionary=True,
            row_group_size=1024,
        )
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _load_config(path_or_config: Path | str | Mapping[str, object]) -> dict[str, object]:
    if isinstance(path_or_config, Mapping):
        value = dict(path_or_config)
    else:
        path = Path(path_or_config)
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("SOLAR-GNIP Teacher config must be a JSON object")
    return value


def _validated_config(config: Mapping[str, object]) -> dict[str, object]:
    resolved = dict(config)
    defaults: dict[str, object] = {
        "rule_version": RULE_VERSION,
        "instance": TRAPI_INSTANCE,
        "deployment": TRAPI_DEPLOYMENT,
        "api_version": TRAPI_API_VERSION,
        "scope": TRAPI_SCOPE,
        "reasoning_effort": REASONING_EFFORT,
        "concurrency": CONCURRENCY,
        "checkpoint_size": CHECKPOINT_SIZE,
        "max_attempts": MAX_ATTEMPTS,
        "icl_examples": 4,
        "seed": 42,
        "retry_failed_on_resume": True,
        "minimum_success_rate": 1.0,
        "request_timeout_seconds": 180.0,
        "max_samples": None,
    }
    for key, value in defaults.items():
        resolved.setdefault(key, value)
    frozen = {
        "rule_version": RULE_VERSION,
        "instance": TRAPI_INSTANCE,
        "deployment": TRAPI_DEPLOYMENT,
        "api_version": TRAPI_API_VERSION,
        "scope": TRAPI_SCOPE,
        "reasoning_effort": REASONING_EFFORT,
        "concurrency": CONCURRENCY,
        "checkpoint_size": CHECKPOINT_SIZE,
        "max_attempts": MAX_ATTEMPTS,
        "icl_examples": 4,
    }
    for key, expected in frozen.items():
        if resolved.get(key) != expected:
            raise ValueError(f"{key} must be frozen to {expected!r}")
    if "temperature" in resolved:
        raise ValueError("temperature must not be configured for GPT-5.5")
    for key in ("candidate_file", "icl_candidate_file", "output_dir"):
        if not resolved.get(key):
            raise ValueError(f"{key} is required")
    if int(resolved["checkpoint_size"]) <= 0:
        raise ValueError("checkpoint_size must be positive")
    maximum = resolved.get("max_samples")
    if maximum is not None and int(maximum) <= 0:
        raise ValueError("max_samples must be positive when configured")
    minimum_success_rate = float(resolved["minimum_success_rate"])
    if not 0.0 <= minimum_success_rate <= 1.0:
        raise ValueError("minimum_success_rate must be in [0, 1]")
    return resolved


def _load_candidates(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise ValueError(f"{path}: no candidate rows")
    seen: set[str] = set()
    for row in rows:
        example_id = str(row["example_id"])
        if example_id in seen:
            raise ValueError(f"duplicate candidate example {example_id}")
        seen.add(example_id)
        ids = [str(value) for value in row["candidate_interest_ids"]]
        names = [str(value) for value in row["candidate_interests"]]
        scores = [float(value) for value in row["candidate_scores"]]
        if len(ids) != 20 or len(set(ids)) != 20:
            raise ValueError(f"{example_id}: candidates must contain 20 unique IDs")
        if len(names) != 20 or len(scores) != 20:
            raise ValueError(f"{example_id}: candidate fields differ in length")
        if not all(math.isfinite(value) for value in scores):
            raise ValueError(f"{example_id}: non-finite candidate score")
        if any(scores[index] < scores[index + 1] for index in range(19)):
            raise ValueError(f"{example_id}: candidate scores are not descending")
        if not row.get("prompt") or not row.get("history_interests"):
            raise ValueError(f"{example_id}: missing History input")
        positives = [str(value) for value in row["positive_interests"]]
        if not positives:
            raise ValueError(f"{example_id}: missing positives")
    return rows


def _select_records(
    rows: Sequence[dict[str, object]], *, maximum: int | None, seed: int
) -> list[dict[str, object]]:
    if maximum is None:
        return list(rows)
    if maximum > len(rows):
        raise ValueError("max_samples exceeds candidate rows")

    def priority(row: Mapping[str, object]) -> tuple[str, str]:
        example_id = str(row["example_id"])
        material = f"{seed}:solar-gnip-teacher:{example_id}".encode("utf-8")
        return hashlib.sha256(material).hexdigest(), example_id

    return sorted(rows, key=priority)[:maximum]


def _history_section(prompt: str) -> str:
    marker = "\n\nBased on both"
    if marker not in prompt:
        raise ValueError("canonical Behavior+Interest prompt marker is missing")
    value = prompt.split(marker, 1)[0].strip()
    if not value:
        raise ValueError("History section is empty")
    return value


def _icl_examples(
    current: Mapping[str, object],
    pool: Sequence[dict[str, object]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, str]]:
    current_id = str(current["example_id"])
    current_user = str(current["user_id"])
    eligible = [
        row
        for row in pool
        if str(row["example_id"]) != current_id
        and str(row["user_id"]) != current_user
    ]
    if len(eligible) < count:
        raise ValueError("not enough independent ICL examples")

    def priority(row: Mapping[str, object]) -> tuple[str, str]:
        other_id = str(row["example_id"])
        material = f"{seed}:{current_id}:{other_id}".encode("utf-8")
        return hashlib.sha256(material).hexdigest(), other_id

    selected = sorted(eligible, key=priority)[:count]
    examples: list[dict[str, str]] = []
    for row in selected:
        histories = list(dict.fromkeys(str(value) for value in row["history_interests"]))
        positives = [str(value) for value in row["positive_interests"]]
        if not histories or not positives:
            raise ValueError("invalid ICL source row")
        negative_index = int(
            hashlib.sha256(
                f"{seed}:{current_id}:{row['example_id']}:negative".encode("utf-8")
            ).hexdigest(),
            16,
        ) % len(histories)
        positive = positives[0]
        negative = histories[negative_index]
        if positive in histories:
            raise ValueError("ICL positive is not new relative to its History")
        examples.append(
            {
                "example_id": str(row["example_id"]),
                "history_summary": ", ".join(histories),
                "observed_new_interest": positive,
                "historical_negative": negative,
            }
        )
    return examples


def _profile_calibration_prompt() -> str:
    return (
        "Calibrate a weak-supervision teacher for Generative New Interest "
        "Prediction (GNIP). GNIP is: given a user's observed interaction "
        "history, predict a topic category the user will click next that is "
        "absent from that history. Define how each of the six profile fields "
        "should summarize evidence without seeing the current target. Return "
        'exactly {"field_guidance": {FIELD: STRING, ...}} with all and only '
        "these fields: "
        + ", ".join(PROFILE_FIELDS)
        + ". Each guidance string must contain at most 50 words."
    )


def _rerank_calibration_prompt() -> str:
    return (
        "Calibrate a candidate reranker for GNIP: given a user History, a "
        "target-blind profile, and exactly 20 candidate topic categories, "
        "rank candidates by likelihood of being the next clicked category "
        "while treating absence from History as required novelty. Relevance "
        "must remain primary; novelty alone is insufficient. Never add an "
        "outside candidate. Return exactly one JSON object with one key "
        '"ranking_principles" whose value is an ordered list of exactly four '
        "distinct nonempty strings, each at most 50 words."
    )


def validate_profile_calibration(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {"field_guidance"}:
        raise InvalidTeacherResponse(
            "invalid_profile_calibration_schema",
            "profile calibration must contain only field_guidance",
        )
    guidance = value["field_guidance"]
    if not isinstance(guidance, dict) or set(guidance) != set(PROFILE_FIELDS):
        raise InvalidTeacherResponse(
            "invalid_profile_calibration_fields",
            "profile calibration fields differ from the frozen schema",
        )
    for field in PROFILE_FIELDS:
        text = guidance[field]
        if not isinstance(text, str) or not text.strip() or len(text.split()) > 50:
            raise InvalidTeacherResponse(
                "invalid_profile_calibration_guidance",
                f"invalid guidance for {field}",
            )
    return {"field_guidance": {field: str(guidance[field]).strip() for field in PROFILE_FIELDS}}


def validate_rerank_calibration(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {"ranking_principles"}:
        raise InvalidTeacherResponse(
            "invalid_rerank_calibration_schema",
            "rerank calibration must contain only ranking_principles",
        )
    principles = value["ranking_principles"]
    if not isinstance(principles, list) or len(principles) != 4:
        raise InvalidTeacherResponse(
            "invalid_rerank_calibration_count",
            "rerank calibration must contain four principles",
        )
    cleaned = []
    for principle in principles:
        if not isinstance(principle, str) or not principle.strip() or len(principle.split()) > 50:
            raise InvalidTeacherResponse(
                "invalid_rerank_calibration_principle",
                "invalid rerank calibration principle",
            )
        cleaned.append(principle.strip())
    if len(set(cleaned)) != len(cleaned):
        raise InvalidTeacherResponse(
            "duplicate_rerank_calibration_principle",
            "rerank calibration principles must be distinct",
        )
    return {"ranking_principles": cleaned}


def validate_profile(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(PROFILE_FIELDS):
        raise InvalidTeacherResponse(
            "invalid_profile_schema",
            "profile must contain exactly the six frozen fields",
        )
    output: dict[str, str] = {}
    for field in PROFILE_FIELDS:
        text = value[field]
        if not isinstance(text, str) or not text.strip():
            raise InvalidTeacherResponse("empty_profile_field", f"{field} is empty")
        if len(text.split()) > 30:
            raise InvalidTeacherResponse(
                "profile_field_too_long", f"{field} exceeds 30 words"
            )
        output[field] = text.strip()
    return output


def validate_ranking(value: object, candidate_ids: Sequence[str]) -> list[str]:
    if not isinstance(value, dict) or set(value) != {"ranking"}:
        raise InvalidTeacherResponse(
            "invalid_ranking_schema", "rerank output must contain only ranking"
        )
    ranking = value["ranking"]
    if not isinstance(ranking, list) or not all(isinstance(item, str) for item in ranking):
        raise InvalidTeacherResponse("invalid_ranking_type", "ranking must be a string list")
    expected = list(candidate_ids)
    actual = [str(value) for value in ranking]
    if len(actual) != len(expected):
        raise InvalidTeacherResponse("incomplete_ranking", "ranking length differs from candidates")
    if len(set(actual)) != len(actual):
        raise InvalidTeacherResponse("duplicate_ranking_id", "ranking contains duplicates")
    if set(actual) != set(expected):
        raise InvalidTeacherResponse("candidate_set_mismatch", "ranking changes the candidate set")
    return actual


def _profile_prompt(
    record: Mapping[str, object], profile_calibration: Mapping[str, object]
) -> str:
    return (
        "Create a target-blind user profile for GNIP. You must use only the "
        "observed History below. Do not predict a topic, do not infer or ask "
        "for the held-out clicked target, and do not mention candidate IDs. "
        "Return exactly one JSON object with all and only these string fields: "
        + ", ".join(PROFILE_FIELDS)
        + ". Each value must be at most 30 words.\n\n"
        "Frozen field guidance:\n"
        + json.dumps(profile_calibration["field_guidance"], ensure_ascii=False, indent=2)
        + "\n\nObserved History:\n"
        + _history_section(str(record["prompt"]))
    )


def _rerank_prompt(
    record: Mapping[str, object],
    *,
    profile: Mapping[str, str],
    rerank_calibration: Mapping[str, object],
    icl_examples: Sequence[Mapping[str, str]],
) -> str:
    candidates = [
        {
            "id": str(candidate_id),
            "topic": str(name),
            "sasrec_score": float(score),
        }
        for candidate_id, name, score in zip(
            record["candidate_interest_ids"],
            record["candidate_interests"],
            record["candidate_scores"],
            strict=True,
        )
    ]
    demonstrations = [
        {
            "history_interests": example["history_summary"],
            "preferred_observed_new_interest": example["observed_new_interest"],
            "demoted_already_historical_interest": example["historical_negative"],
        }
        for example in icl_examples
    ]
    return (
        "Rerank exactly the 20 candidate IDs for GNIP. Use the History, the "
        "target-blind profile, SASRec scores as a prior, the frozen principles, "
        "and four demonstrations of the objective. The current held-out target "
        "is not provided. Prefer a plausible next click that is absent from "
        "History; demote already-seen interests. Do not add, omit, duplicate, "
        "rename, or explain candidates. Return exactly "
        '{"ranking": [ID1, ..., ID20]}.\n\n'
        "Frozen ranking principles:\n"
        + json.dumps(rerank_calibration["ranking_principles"], ensure_ascii=False, indent=2)
        + "\n\nObjective demonstrations from other Train users:\n"
        + json.dumps(demonstrations, ensure_ascii=False, indent=2)
        + "\n\nCurrent observed History:\n"
        + _history_section(str(record["prompt"]))
        + "\n\nTarget-blind profile:\n"
        + json.dumps(profile, ensure_ascii=False, indent=2)
        + "\n\nCandidates in SASRec order:\n"
        + json.dumps(candidates, ensure_ascii=False, indent=2)
    )


def _safe_error(error: Exception) -> str:
    message = str(error)
    message = re.sub(
        r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+",
        r"\1[REDACTED]",
        message,
    )
    return message[:1000]


def _status_code(error: Exception) -> int | None:
    value = getattr(error, "status_code", None)
    if value is None:
        value = getattr(getattr(error, "response", None), "status_code", None)
    return int(value) if isinstance(value, int) else None


def _retry_delay(error: Exception, attempt: int) -> float:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        try:
            if raw is not None:
                return min(60.0, max(0.0, float(raw)))
        except (TypeError, ValueError):
            pass
    match = _RETRY_AFTER_SECONDS.search(str(error))
    if match:
        return min(60.0, max(0.0, float(match.group(1))))
    return min(30.0, float(2 ** (attempt - 1)))


def _retryable(error: Exception) -> bool:
    status = _status_code(error)
    if status in {401, 403}:
        return False
    if status in {408, 409, 429} or (status is not None and status >= 500):
        return True
    name = type(error).__name__.casefold()
    return "timeout" in name or "connection" in name


def _raw_response(response: Any) -> str:
    if hasattr(response, "model_dump_json"):
        return str(response.model_dump_json())
    if hasattr(response, "model_dump"):
        return json.dumps(response.model_dump(), ensure_ascii=False, default=str)
    return json.dumps(response, ensure_ascii=False, default=str)


def _call_json(
    client: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    validator: Callable[[object], Any],
    max_attempts: int = MAX_ATTEMPTS,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    started = time.monotonic()
    history: list[dict[str, object]] = []
    for attempt in range(1, max_attempts + 1):
        raw_content = ""
        raw_response = ""
        try:
            response = client.chat.completions.create(
                model=TRAPI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                reasoning_effort=REASONING_EFFORT,
            )
            if not getattr(response, "choices", None):
                raise InvalidTeacherResponse("no_choices", "response contains no choices")
            choice = response.choices[0]
            finish_reason = str(getattr(choice, "finish_reason", "") or "")
            raw_content = str(getattr(choice.message, "content", "") or "")
            raw_response = _raw_response(response)
            if finish_reason == "content_filter":
                return {
                    "status": "policy_block",
                    "parsed": None,
                    "raw_content": raw_content,
                    "raw_response": raw_response,
                    "attempts": attempt,
                    "attempt_history_json": _canonical_json(history),
                    "error_type": "ContentFilter",
                    "error_message": "response finish_reason was content_filter",
                    "status_code": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "latency_seconds": time.monotonic() - started,
                }
            try:
                value = json.loads(raw_content)
            except json.JSONDecodeError as error:
                raise InvalidTeacherResponse("invalid_json", str(error)) from error
            parsed = validator(value)
            usage = getattr(response, "usage", None)
            return {
                "status": "success",
                "parsed": parsed,
                "raw_content": raw_content,
                "raw_response": raw_response,
                "attempts": attempt,
                "attempt_history_json": _canonical_json(history),
                "error_type": None,
                "error_message": None,
                "status_code": None,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "latency_seconds": time.monotonic() - started,
            }
        except InvalidTeacherResponse as error:
            history.append(
                {
                    "attempt": attempt,
                    "status": "invalid_response",
                    "reason": error.reason,
                    "message": _safe_error(error),
                    "raw_content": raw_content,
                }
            )
            if attempt == max_attempts:
                return {
                    "status": "invalid_response",
                    "parsed": None,
                    "raw_content": raw_content,
                    "raw_response": raw_response,
                    "attempts": attempt,
                    "attempt_history_json": _canonical_json(history),
                    "error_type": type(error).__name__,
                    "error_message": _safe_error(error),
                    "status_code": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "latency_seconds": time.monotonic() - started,
                }
        except Exception as error:
            history.append(
                {
                    "attempt": attempt,
                    "status": "api_error",
                    "error_type": type(error).__name__,
                    "status_code": _status_code(error),
                    "message": _safe_error(error),
                }
            )
            if not _retryable(error) or attempt == max_attempts:
                return {
                    "status": "api_error",
                    "parsed": None,
                    "raw_content": raw_content,
                    "raw_response": raw_response,
                    "attempts": attempt,
                    "attempt_history_json": _canonical_json(history),
                    "error_type": type(error).__name__,
                    "error_message": _safe_error(error),
                    "status_code": _status_code(error),
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "latency_seconds": time.monotonic() - started,
                }
            sleep_fn(_retry_delay(error, attempt))
            continue
        if attempt < max_attempts:
            sleep_fn(min(10.0, float(2 ** (attempt - 1))))
    raise AssertionError("unreachable Teacher retry state")


def _create_client(config: Mapping[str, object]) -> Any:
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
        timeout=float(config["request_timeout_seconds"]),
        max_retries=0,
    )


def _calibration_request_hash(kind: str, prompt: str) -> str:
    return _hash_value(
        {
            "rule_version": RULE_VERSION,
            "instance": TRAPI_INSTANCE,
            "deployment": TRAPI_DEPLOYMENT,
            "api_version": TRAPI_API_VERSION,
            "reasoning_effort": REASONING_EFFORT,
            "kind": kind,
            "prompt": prompt,
        }
    )


def _ensure_calibration(
    output_dir: Path,
    client: Any,
    *,
    sleep_fn: Callable[[float], None],
) -> dict[str, object]:
    path = output_dir / "calibration.json"
    profile_prompt = _profile_calibration_prompt()
    rerank_prompt = _rerank_calibration_prompt()
    expected_hashes = {
        "profile": _calibration_request_hash("profile", profile_prompt),
        "rerank": _calibration_request_hash("rerank", rerank_prompt),
    }
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("request_hashes") != expected_hashes:
            raise ValueError("existing calibration identity differs from current prompts")
        value["profile"]["parsed"] = validate_profile_calibration(
            value["profile"]["parsed"]
        )
        value["rerank"]["parsed"] = validate_rerank_calibration(
            value["rerank"]["parsed"]
        )
        if value["profile"]["status"] != "success" or value["rerank"]["status"] != "success":
            raise ValueError("existing calibration is not successful")
        return value

    profile = _call_json(
        client,
        system_prompt="Return only the requested JSON object without Markdown.",
        user_prompt=profile_prompt,
        validator=validate_profile_calibration,
        sleep_fn=sleep_fn,
    )
    rerank = _call_json(
        client,
        system_prompt="Return only the requested JSON object without Markdown.",
        user_prompt=rerank_prompt,
        validator=validate_rerank_calibration,
        sleep_fn=sleep_fn,
    )
    value = {
        "rule_version": RULE_VERSION,
        "request_hashes": expected_hashes,
        "profile_prompt": profile_prompt,
        "rerank_prompt": rerank_prompt,
        "profile": profile,
        "rerank": rerank,
    }
    _atomic_json_write(value, path)
    if profile["status"] != "success" or rerank["status"] != "success":
        raise RuntimeError("GPT-5.5 calibration did not produce two valid results")
    return value


def _request_spec(
    record: Mapping[str, object],
    *,
    icl_pool: Sequence[dict[str, object]],
    profile_calibration: Mapping[str, object],
    rerank_calibration: Mapping[str, object],
    seed: int,
) -> dict[str, object]:
    icl = _icl_examples(record, icl_pool, count=4, seed=seed)
    profile_prompt = _profile_prompt(record, profile_calibration)
    identity = {
        "rule_version": RULE_VERSION,
        "instance": TRAPI_INSTANCE,
        "deployment": TRAPI_DEPLOYMENT,
        "api_version": TRAPI_API_VERSION,
        "reasoning_effort": REASONING_EFFORT,
        "example_id": str(record["example_id"]),
        "user_id": str(record["user_id"]),
        "prompt": str(record["prompt"]),
        "candidate_interest_ids": list(record["candidate_interest_ids"]),
        "candidate_interests": list(record["candidate_interests"]),
        "candidate_scores": list(record["candidate_scores"]),
        "icl": icl,
        "profile_calibration": profile_calibration,
        "rerank_calibration": rerank_calibration,
    }
    return {
        "record": dict(record),
        "icl": icl,
        "profile_prompt": profile_prompt,
        "request_hash": _hash_value(identity),
    }


def _stage_fields(prefix: str, result: Mapping[str, object]) -> dict[str, object]:
    return {
        f"{prefix}_status": result["status"],
        f"{prefix}_raw_content": result["raw_content"],
        f"{prefix}_raw_response": result["raw_response"],
        f"{prefix}_attempts": result["attempts"],
        f"{prefix}_attempt_history_json": result["attempt_history_json"],
        f"{prefix}_error_type": result["error_type"],
        f"{prefix}_error_message": result["error_message"],
        f"{prefix}_status_code": result["status_code"],
        f"{prefix}_prompt_tokens": result["prompt_tokens"],
        f"{prefix}_completion_tokens": result["completion_tokens"],
        f"{prefix}_total_tokens": result["total_tokens"],
        f"{prefix}_latency_seconds": result["latency_seconds"],
    }


def _process_record(
    client: Any,
    spec: Mapping[str, object],
    *,
    rerank_calibration: Mapping[str, object],
    sleep_fn: Callable[[float], None],
) -> dict[str, object]:
    record = dict(spec["record"])
    candidate_ids = [str(value) for value in record["candidate_interest_ids"]]
    profile_result = _call_json(
        client,
        system_prompt="Return only the requested target-blind profile JSON.",
        user_prompt=str(spec["profile_prompt"]),
        validator=validate_profile,
        sleep_fn=sleep_fn,
    )
    base = {
        "request_hash": str(spec["request_hash"]),
        "example_id": str(record["example_id"]),
        "user_id": str(record["user_id"]),
        "source_split": str(record["split"]),
        "prompt": str(record["prompt"]),
        "history_interests": list(record["history_interests"]),
        "positive_interest_ids": list(record["positive_interest_ids"]),
        "positive_interests": list(record["positive_interests"]),
        "candidate_interest_ids": candidate_ids,
        "candidate_interests": list(record["candidate_interests"]),
        "candidate_scores": list(record["candidate_scores"]),
        "candidate_target_in_top20": bool(record["target_in_candidates"]),
        "candidate_recall_at_20": float(record["candidate_recall_at_20"]),
        "icl_example_ids": [str(value["example_id"]) for value in spec["icl"]],
        "icl_json": _canonical_json(spec["icl"]),
        "profile_prompt": str(spec["profile_prompt"]),
        "profile_json": (
            _canonical_json(profile_result["parsed"])
            if profile_result["parsed"] is not None
            else None
        ),
        **_stage_fields("profile", profile_result),
    }
    if profile_result["status"] != "success":
        return {
            **base,
            "status": str(profile_result["status"]),
            "rerank_prompt": None,
            "ranking_interest_ids": [],
            "ranking_interests": [],
            "teacher_top1_interest_id": None,
            "teacher_top1_interest": None,
            "teacher_target_rank": None,
            **_stage_fields(
                "rerank",
                {
                    "status": "not_run",
                    "raw_content": "",
                    "raw_response": "",
                    "attempts": 0,
                    "attempt_history_json": "[]",
                    "error_type": None,
                    "error_message": None,
                    "status_code": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "latency_seconds": 0.0,
                },
            ),
        }
    profile = dict(profile_result["parsed"])
    rerank_prompt = _rerank_prompt(
        record,
        profile=profile,
        rerank_calibration=rerank_calibration,
        icl_examples=list(spec["icl"]),
    )
    rerank_result = _call_json(
        client,
        system_prompt="Return only the complete candidate permutation JSON.",
        user_prompt=rerank_prompt,
        validator=lambda value: validate_ranking(value, candidate_ids),
        sleep_fn=sleep_fn,
    )
    if rerank_result["status"] != "success":
        return {
            **base,
            "status": str(rerank_result["status"]),
            "rerank_prompt": rerank_prompt,
            "ranking_interest_ids": [],
            "ranking_interests": [],
            "teacher_top1_interest_id": None,
            "teacher_top1_interest": None,
            "teacher_target_rank": None,
            **_stage_fields("rerank", rerank_result),
        }
    ranking = list(rerank_result["parsed"])
    id_to_name = dict(zip(candidate_ids, record["candidate_interests"], strict=True))
    positives = set(str(value) for value in record["positive_interest_ids"])
    target_positions = [index + 1 for index, value in enumerate(ranking) if value in positives]
    return {
        **base,
        "status": "success",
        "rerank_prompt": rerank_prompt,
        "ranking_interest_ids": ranking,
        "ranking_interests": [str(id_to_name[value]) for value in ranking],
        "teacher_top1_interest_id": ranking[0],
        "teacher_top1_interest": str(id_to_name[ranking[0]]),
        "teacher_target_rank": min(target_positions) if target_positions else None,
        **_stage_fields("rerank", rerank_result),
    }


def _discounted_metrics(
    rows: Sequence[Mapping[str, object]], ranking_field: str
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for cutoff in (1, 5, 10, 20):
        recalls: list[float] = []
        ndcgs: list[float] = []
        hits: list[float] = []
        for row in rows:
            positives = set(str(value) for value in row["positive_interest_ids"])
            ranking = [str(value) for value in row[ranking_field]]
            retrieved = positives.intersection(ranking[:cutoff])
            recalls.append(len(retrieved) / len(positives))
            hits.append(float(bool(retrieved)))
            dcg = sum(
                1.0 / math.log2(rank + 1)
                for rank, value in enumerate(ranking[:cutoff], start=1)
                if value in positives
            )
            ideal = sum(
                1.0 / math.log2(rank + 1)
                for rank in range(1, min(cutoff, len(positives)) + 1)
            )
            ndcgs.append(dcg / ideal if ideal else 0.0)
        metrics[f"recall_at_{cutoff}"] = sum(recalls) / len(recalls)
        metrics[f"ndcg_at_{cutoff}"] = sum(ndcgs) / len(ndcgs)
        metrics[f"hit_at_{cutoff}"] = sum(hits) / len(hits)
    return metrics


def _metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    status_counts = Counter(str(row["status"]) for row in rows)
    success = [row for row in rows if row["status"] == "success"]
    profile_status = Counter(str(row["profile_status"]) for row in rows)
    rerank_status = Counter(str(row["rerank_status"]) for row in rows)
    output: dict[str, object] = {
        "requested": len(rows),
        "successes": len(success),
        "success_rate": len(success) / len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "profile_status_counts": dict(sorted(profile_status.items())),
        "rerank_status_counts": dict(sorted(rerank_status.items())),
        "total_prompt_tokens": sum(
            int(row.get(field) or 0)
            for row in rows
            for field in ("profile_prompt_tokens", "rerank_prompt_tokens")
        ),
        "total_completion_tokens": sum(
            int(row.get(field) or 0)
            for row in rows
            for field in ("profile_completion_tokens", "rerank_completion_tokens")
        ),
    }
    if success:
        output["sasrec_on_successful_teacher_rows"] = _discounted_metrics(
            success, "candidate_interest_ids"
        )
        output["gpt55_rerank"] = _discounted_metrics(
            success, "ranking_interest_ids"
        )
    return output


def generate_teacher(
    path_or_config: Path | str | Mapping[str, object] = DEFAULT_CONFIG,
    *,
    client: Any | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> GenerationResult:
    resolved = _validated_config(_load_config(path_or_config))
    candidate_path = Path(str(resolved["candidate_file"]))
    icl_path = Path(str(resolved["icl_candidate_file"]))
    output_dir = Path(str(resolved["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _load_candidates(candidate_path)
    icl_pool = _load_candidates(icl_path)
    maximum = resolved.get("max_samples")
    selected = _select_records(
        candidates,
        maximum=int(maximum) if maximum is not None else None,
        seed=int(resolved["seed"]),
    )
    api_client = client or _create_client(resolved)
    calibration = _ensure_calibration(output_dir, api_client, sleep_fn=sleep_fn)
    profile_calibration = dict(calibration["profile"]["parsed"])
    rerank_calibration = dict(calibration["rerank"]["parsed"])
    specs = [
        _request_spec(
            row,
            icl_pool=icl_pool,
            profile_calibration=profile_calibration,
            rerank_calibration=rerank_calibration,
            seed=int(resolved["seed"]),
        )
        for row in selected
    ]
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    generated = 0
    resumed = 0
    expected_shards: set[Path] = set()
    checkpoint = int(resolved["checkpoint_size"])
    for start in range(0, len(specs), checkpoint):
        stop = min(start + checkpoint, len(specs))
        chunk = specs[start:stop]
        shard_path = shards_dir / f"{start:09d}-{stop:09d}.parquet"
        expected_shards.add(shard_path)
        if shard_path.is_file():
            shard_rows = pq.read_table(shard_path).to_pylist()
            expected_hashes = [str(spec["request_hash"]) for spec in chunk]
            actual_hashes = [str(row["request_hash"]) for row in shard_rows]
            if actual_hashes != expected_hashes:
                raise ValueError(f"{shard_path}: request identities changed")
            retry_offsets = [
                index
                for index, row in enumerate(shard_rows)
                if row["status"] != "success"
                and bool(resolved["retry_failed_on_resume"])
            ]
        else:
            shard_rows = [{} for _ in chunk]
            retry_offsets = list(range(len(chunk)))
        if retry_offsets:
            completed: dict[int, dict[str, object]] = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=int(resolved["concurrency"])
            ) as executor:
                futures = {
                    executor.submit(
                        _process_record,
                        api_client,
                        chunk[offset],
                        rerank_calibration=rerank_calibration,
                        sleep_fn=sleep_fn,
                    ): offset
                    for offset in retry_offsets
                }
                for future in concurrent.futures.as_completed(futures):
                    completed[futures[future]] = future.result()
            for offset, row in completed.items():
                shard_rows[offset] = row
            _atomic_parquet_write(shard_rows, shard_path)
            generated += len(retry_offsets)
        resumed += len(chunk) - len(retry_offsets)
        all_rows.extend(shard_rows)
        print(
            "SOLAR_GNIP_TEACHER "
            + _canonical_json(
                {
                    "completed_sources": stop,
                    "total_sources": len(specs),
                    "successful_sources": sum(row["status"] == "success" for row in all_rows),
                }
            ),
            flush=True,
        )
    stale = set(shards_dir.glob("*.parquet")) - expected_shards
    if stale:
        raise ValueError(
            "stale Teacher shards: " + ", ".join(sorted(path.name for path in stale))
        )
    records_path = output_dir / "teacher_records.parquet"
    _atomic_parquet_write(all_rows, records_path)
    metrics = _metrics(all_rows)
    _atomic_json_write(metrics, output_dir / "metrics.json")
    safe_config = dict(resolved)
    _atomic_json_write(safe_config, output_dir / "resolved_config.json")
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rule_version": RULE_VERSION,
        "candidate_file": portable_path(candidate_path),
        "candidate_sha256": _sha256(candidate_path),
        "icl_candidate_file": portable_path(icl_path),
        "icl_candidate_sha256": _sha256(icl_path),
        "instance": TRAPI_INSTANCE,
        "deployment": TRAPI_DEPLOYMENT,
        "api_version": TRAPI_API_VERSION,
        "scope": TRAPI_SCOPE,
        "reasoning_effort": REASONING_EFFORT,
        "temperature": None,
        "concurrency": CONCURRENCY,
        "checkpoint_size": CHECKPOINT_SIZE,
        "max_attempts": MAX_ATTEMPTS,
        "requested": len(all_rows),
        "successes": int(metrics["successes"]),
        "generated_this_run": generated,
        "resumed": resumed,
        "calibration_logical_calls": 2,
        "per_source_logical_calls": 2,
        "artifact_sha256": {
            "calibration.json": _sha256(output_dir / "calibration.json"),
            "teacher_records.parquet": _sha256(records_path),
            "metrics.json": _sha256(output_dir / "metrics.json"),
            "resolved_config.json": _sha256(output_dir / "resolved_config.json"),
            **{
                str(path.relative_to(output_dir)): _sha256(path)
                for path in sorted(shards_dir.glob("*.parquet"))
            },
        },
    }
    _atomic_json_write(metadata, output_dir / "metadata.json")
    if float(metrics["success_rate"]) < float(resolved["minimum_success_rate"]):
        raise RuntimeError(
            f"Teacher success rate {metrics['success_rate']:.4f} is below "
            f"{float(resolved['minimum_success_rate']):.4f}"
        )
    return GenerationResult(
        output_dir=output_dir,
        requested=len(all_rows),
        successes=int(metrics["successes"]),
        generated_this_run=generated,
        resumed=resumed,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SOLAR-GNIP GPT-5.5 weak supervision")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config = _load_config(args.config)
    for key in ("candidate_file", "output_dir", "max_samples"):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    result = generate_teacher(config)
    print(json.dumps({"output_dir": str(result.output_dir), "requested": result.requested, "successes": result.successes, "generated_this_run": result.generated_this_run, "resumed": result.resumed}, indent=2))


if __name__ == "__main__":
    main()
