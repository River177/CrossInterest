from __future__ import annotations

from types import SimpleNamespace

from new_interest.evaluate_candidate_trapi import (
    _call_one,
    _is_retryable,
    _pilot_records,
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
                    )
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
    assert completions.calls[0]["reasoning_effort"] == "medium"
    assert "temperature" not in completions.calls[0]


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
