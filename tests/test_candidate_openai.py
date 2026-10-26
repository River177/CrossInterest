from __future__ import annotations

import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.evaluation.evaluate_candidate_openai import (
    _call_one,
    _is_retryable,
    _pilot_records,
    _sanitize_config,
    evaluate_openai,
    preflight_openai,
)


class FakeResponse:
    def __init__(self, content: str = "New topic: Music") -> None:
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    reasoning_text=None,
                ),
                finish_reason="stop",
            )
        ]
        self.usage = SimpleNamespace(
            prompt_tokens=20,
            completion_tokens=5,
            total_tokens=25,
        )

    def model_dump_json(self) -> str:
        return json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": self.choices[0].message.content
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                },
            }
        )


class FakeCompletions:
    def __init__(self, failing_prompts: set[str] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.failing_prompts = failing_prompts or set()

    def create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        if prompt in self.failing_prompts:
            raise ValueError(f"request failed for {prompt}")
        return FakeResponse()


def _client(
    *,
    failing_prompts: set[str] | None = None,
    models: tuple[str, ...] = ("gemini-3.6-flash",),
):
    completions = FakeCompletions(failing_prompts)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        models=SimpleNamespace(
            list=lambda: SimpleNamespace(
                data=[SimpleNamespace(id=model) for model in models]
            )
        ),
    )
    return client, completions


def _record(selection_id: str, prompt: str) -> dict[str, object]:
    return {
        "selection_id": selection_id,
        "impression_id": selection_id,
        "prompt": prompt,
        "answer": "New topic: Music",
        "answer_category": "Music",
        "candidates": ["Music", "Golf"],
        "correct_position": 0,
        "history_length": 3,
        "all_acceptable_subcategories": ["Music"],
    }


def test_retryable_error_classification() -> None:
    for status in (408, 409, 429, 500, 503):
        error = RuntimeError("retry")
        error.status_code = status
        assert _is_retryable(error) == (True, status)

    for status in (400, 401, 403, 404):
        error = RuntimeError("stop")
        error.status_code = status
        assert _is_retryable(error) == (False, status)

    assert _is_retryable(TimeoutError("late"))[0]
    assert _is_retryable(ConnectionError("down"))[0]


def test_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUSTOM_OPENAI_KEY", raising=False)
    client, _ = _client()
    with pytest.raises(RuntimeError, match="CUSTOM_OPENAI_KEY"):
        preflight_openai(
            {"api_key_env": "CUSTOM_OPENAI_KEY"},
            client=client,
        )


def test_sanitized_config_preserves_env_name_not_secret() -> None:
    secret = "do-not-persist-this"
    sanitized = _sanitize_config(
        {
            "api_key_env": "OPENAI_API_KEY",
            "api_key": secret,
            "max_tokens": 64,
            "nested": {
                "authorization": f"Bearer {secret}",
                "note": f"value={secret}",
            },
        },
        secrets=(secret,),
    )
    assert sanitized["api_key_env"] == "OPENAI_API_KEY"
    assert sanitized["max_tokens"] == 64
    assert "api_key" not in sanitized
    assert "authorization" not in sanitized["nested"]
    assert sanitized["nested"]["note"] == "value=[REDACTED]"
    assert secret not in json.dumps(sanitized)


def test_call_passes_only_configured_generation_parameters() -> None:
    client, completions = _client()
    result = _call_one(
        client,
        {"selection_id": "s1", "prompt": "choose"},
        config={
            "model": "gemini-3.6-flash",
            "max_tokens": 64,
            "temperature": 0.7,
            "top_p": 0.8,
            "seed": 20260808,
            "reasoning_effort": "minimal",
        },
        api_key="secret",
        max_attempts=2,
    )
    call = completions.calls[0]
    assert result["success"]
    assert result["finish_reason"] == "stop"
    assert result["prompt_tokens"] == 20
    assert result["completion_tokens"] == 5
    assert json.loads(result["raw_response"])["usage"]["total_tokens"] == 25
    assert call["model"] == "gemini-3.6-flash"
    assert call["max_tokens"] == 64
    assert call["temperature"] == 0.7
    assert call["top_p"] == 0.8
    assert call["seed"] == 20260808
    assert call["reasoning_effort"] == "minimal"
    assert "max_completion_tokens" not in call


def test_preflight_validates_exact_model_and_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    client, completions = _client(
        models=("other-model", "gemini-3.6-flash")
    )
    result = preflight_openai(
        {
            "model": "gemini-3.6-flash",
            "max_tokens": 8,
        },
        client=client,
    )
    assert result["completion_succeeded"]
    assert completions.calls[0]["max_tokens"] == 8

    missing_client, _ = _client(models=("other-model",))
    with pytest.raises(RuntimeError, match="not returned"):
        preflight_openai(
            {"model": "gemini-3.6-flash"},
            client=missing_client,
        )


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


def test_resume_retries_failures_and_preserves_record_order(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    records = [
        _record("s2", "prompt-2"),
        _record("s1", "prompt-1"),
        _record("s3", "prompt-3"),
    ]
    data_file = tmp_path / "candidates.parquet"
    output_dir = tmp_path / "output"
    pq.write_table(pa.Table.from_pylist(records), data_file)
    config = {
        "data_file": str(data_file),
        "output_dir": str(output_dir),
        "model": "gemini-3.6-flash",
        "max_tokens": 32,
        "concurrency": 2,
        "checkpoint_size": 2,
        "max_attempts": 1,
        "minimum_success_rate": 0.0,
        "preflight": False,
    }

    first_client, _ = _client(failing_prompts={"prompt-1"})
    first_metrics = evaluate_openai(config, client=first_client)
    assert first_metrics["request_failures"] == 1

    second_client, second_completions = _client()
    second_metrics = evaluate_openai(config, client=second_client)
    assert second_metrics["request_failures"] == 0
    assert second_metrics["generated_samples_this_run"] == 1
    assert second_metrics["resumed_samples"] == 2
    assert [
        call["messages"][0]["content"]
        for call in second_completions.calls
    ] == ["prompt-1"]

    predictions = pq.read_table(
        output_dir / "predictions.parquet"
    ).to_pylist()
    assert [row["selection_id"] for row in predictions] == [
        "s2",
        "s1",
        "s3",
    ]
    assert all(row["success"] for row in predictions)
    resolved = json.loads(
        (output_dir / "resolved_config.json").read_text()
    )
    assert resolved["api_key_env"] == "OPENAI_API_KEY"
    assert "secret" not in json.dumps(resolved)
