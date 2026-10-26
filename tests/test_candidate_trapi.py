from __future__ import annotations

from types import SimpleNamespace

import pytest

from new_interest.evaluation.evaluate_candidate_trapi import (
    EmptyChoicesError,
    _call_one,
    _is_retryable,
    _pilot_records,
    _sanitize_config,
    _validate_zero_reasoning_tokens,
)


class FakeCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(
            prompt_tokens=20,
            completion_tokens=10,
            completion_tokens_details=SimpleNamespace(
                reasoning_tokens=4
            ),
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="New topic: Music"
                    ),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )


def test_trapi_call_records_usage_and_reasoning() -> None:
    completions = FakeCompletions()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    result = _call_one(
        client,
        {"selection_id": "s1", "prompt": "prompt"},
        deployment="gpt-test",
        reasoning_effort="medium",
        max_completion_tokens=256,
        max_attempts=2,
    )
    assert result["success"]
    assert result["output"] == "New topic: Music"
    assert result["prompt_tokens"] == 20
    assert result["completion_tokens"] == 10
    assert result["reasoning_tokens"] == 4
    assert result["finish_reason"] == "stop"
    assert completions.calls[0]["reasoning_effort"] == "medium"
    assert "temperature" not in completions.calls[0]


def test_trapi_call_passes_none_reasoning_effort_exactly() -> None:
    completions = FakeCompletions()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    _call_one(
        client,
        {"selection_id": "s1", "prompt": "prompt"},
        deployment="gpt-test",
        reasoning_effort="none",
        max_completion_tokens=512,
        max_attempts=1,
    )
    assert completions.calls[0]["reasoning_effort"] == "none"


def test_trapi_call_omits_null_reasoning_effort() -> None:
    completions = FakeCompletions()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    _call_one(
        client,
        {"selection_id": "s1", "prompt": "prompt"},
        deployment="gpt-test",
        reasoning_effort=None,
        max_completion_tokens=512,
        max_attempts=1,
    )
    assert "reasoning_effort" not in completions.calls[0]


def test_zero_reasoning_token_guard() -> None:
    _validate_zero_reasoning_tokens(0, True)
    _validate_zero_reasoning_tokens(4, False)
    with pytest.raises(RuntimeError, match="reported 4"):
        _validate_zero_reasoning_tokens(4, True)


def test_config_sanitization_preserves_non_secret_token_fields() -> None:
    safe_config = _sanitize_config(
        {
            "max_completion_tokens": 512,
            "expected_zero_reasoning_tokens": True,
            "access_token": "secret",
            "api_key": "secret",
            "credential": "secret",
            "token": "secret",
        }
    )
    assert safe_config == {
        "max_completion_tokens": 512,
        "expected_zero_reasoning_tokens": True,
    }


def test_pilot_selection_is_deterministic() -> None:
    rows = [
        {"selection_id": f"s{index}"}
        for index in range(100)
    ]
    first = _pilot_records(rows, 10, 17)
    second = _pilot_records(list(reversed(rows)), 10, 17)
    assert [row["selection_id"] for row in first] == [
        row["selection_id"] for row in second
    ]
    assert len(first) == 10


def test_retryable_error_classification() -> None:
    error_429 = RuntimeError("rate")
    error_429.status_code = 429
    assert _is_retryable(error_429) == (True, 429)

    error_401 = RuntimeError("auth")
    error_401.status_code = 401
    assert _is_retryable(error_401) == (False, 401)

    assert _is_retryable(EmptyChoicesError("empty")) == (True, None)


def test_trapi_call_retries_empty_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyThenSuccess(FakeCompletions):
        def create(self, **kwargs):
            if not self.calls:
                self.calls.append(kwargs)
                return SimpleNamespace(choices=[], usage=None)
            return super().create(**kwargs)

    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    completions = EmptyThenSuccess()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    result = _call_one(
        client,
        {"selection_id": "s1", "prompt": "prompt"},
        deployment="grok-test",
        reasoning_effort=None,
        max_completion_tokens=512,
        max_attempts=2,
    )

    assert result["success"] is True
    assert result["attempts"] == 2
    assert len(completions.calls) == 2
