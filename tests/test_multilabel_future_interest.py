from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from new_interest.experiments.multilabel_future_interest import (  # noqa: E402
    build_feature_vector,
    build_masked_targets,
    fit_multilabel_baseline,
    masked_bce_loss,
    predict_multilabel_baseline,
)


LABELS = ["a", "b", "c"]


def _write_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    vocabulary = root / "vocabulary.parquet"
    train = root / "train.parquet"
    validation = root / "validation.parquet"
    benchmark = root / "benchmark.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"interest_id": i, "subcategory": label} for i, label in enumerate(LABELS)]
        ),
        vocabulary,
        row_group_size=1,
    )
    rows = [
        {
            "user_id": "u1",
            "history_subcategories": ["a", "a", "b"],
            "future_novel_subcategories": ["b"],
            "future_exposed_unclicked_subcategories": ["a"],
        },
        {
            "user_id": "u2",
            "history_subcategories": ["b"],
            "future_novel_subcategories": ["a"],
            "future_exposed_unclicked_subcategories": ["b"],
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), train, row_group_size=1)
    pq.write_table(
        pa.Table.from_pylist([rows[0]]), validation, row_group_size=1
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"user_id": "u3", "history_subcategories": ["a", "c"]},
                {"user_id": "u4", "history_subcategories": []},
            ]
        ),
        benchmark,
        row_group_size=1,
    )
    return train, validation, benchmark, vocabulary


def test_mask_construction_and_unknown_labels() -> None:
    target, mask = build_masked_targets(
        {
            "user_id": "u",
            "future_novel_subcategories": ["a"],
            "future_exposed_unclicked_subcategories": ["c"],
        },
        LABELS,
    )
    assert target == (1.0, 0.0, 0.0)
    assert mask == (True, False, True)

    logits = torch.tensor([[2.0, -100.0, 0.0]])
    changed = torch.tensor([[2.0, 100.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    mask_tensor = torch.tensor([[True, False, True]])
    assert masked_bce_loss(logits, targets, mask_tensor).item() == pytest.approx(
        masked_bce_loss(changed, targets, mask_tensor).item()
    )


def test_topic_frequency_and_embedding_alignment(tmp_path: Path) -> None:
    row = {"user_id": "u1", "history_subcategories": ["a", "a", "b"]}
    assert build_feature_vector(
        row, LABELS, feature_mode="topic_frequency"
    ) == pytest.approx((2 / 3, 1 / 3, 0.0))
    embeddings = tmp_path / "embeddings.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"user_id": "u2", "embedding": [1.0, 2.0]},
                {"user_id": "u1", "embedding": [3.0, 4.0]},
            ]
        ),
        embeddings,
        row_group_size=1,
    )
    assert build_feature_vector(
        row,
        LABELS,
        feature_mode="history_embedding",
        embedding=(3.0, 4.0),
    ) == (3.0, 4.0)
    with pytest.raises(ValueError, match="missing embedding"):
        build_feature_vector(
            {"user_id": "missing", "history_subcategories": []},
            LABELS,
            feature_mode="history_embedding",
            embeddings={"u1": (3.0, 4.0)},
        )


def test_tiny_training_is_deterministic_and_records_unsupported(tmp_path: Path) -> None:
    train, validation, _, vocabulary = _write_fixture(tmp_path)
    first, first_summary = fit_multilabel_baseline(
        train,
        vocabulary,
        validation_path=validation,
        epochs=4,
        patience=2,
        batch_size=1,
        seed=7,
        device="cpu",
    )
    second, second_summary = fit_multilabel_baseline(
        train,
        vocabulary,
        validation_path=validation,
        epochs=4,
        patience=2,
        batch_size=1,
        seed=7,
        device="cpu",
    )
    assert first_summary["history"] == second_summary["history"]
    for key in first.state_dict():
        assert torch.equal(first.state_dict()[key], second.state_dict()[key])
    assert first_summary["label_status"]["c"] == "no_train_positives_or_negatives"


def test_predictions_batch_forward_preserves_order_and_full_vocabulary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train, validation, benchmark, vocabulary = _write_fixture(tmp_path)
    model, _ = fit_multilabel_baseline(
        train,
        vocabulary,
        validation_path=validation,
        epochs=2,
        batch_size=2,
        device="cpu",
    )
    forward_calls = 0
    original_forward = model.module.forward

    def counted_forward(*args: object, **kwargs: object) -> object:
        nonlocal forward_calls
        forward_calls += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model.module, "forward", counted_forward)
    result = predict_multilabel_baseline(
        model, benchmark, tmp_path / "output", overwrite=True, batch_size=2
    )
    assert forward_calls == 1
    assert result["users"] == 2
    rows = pq.read_table(result["prediction_path"]).to_pylist()
    assert [row["user_id"] for row in rows] == ["u3", "u4"]
    assert all(len(row["ranked_labels"]) == 3 for row in rows)
    assert all(len(row["ranked_scores"]) == 3 for row in rows)
    assert json.loads((tmp_path / "output/config.json").read_text())["label_count"] == 3
    assert (tmp_path / "output/model.pt").exists()


def test_embedding_order_and_coverage_are_validated_streaming(tmp_path: Path) -> None:
    train, validation, benchmark, vocabulary = _write_fixture(tmp_path)
    train_embeddings = tmp_path / "train_embeddings.parquet"
    validation_embeddings = tmp_path / "validation_embeddings.parquet"
    benchmark_embeddings = tmp_path / "benchmark_embeddings.parquet"
    for path, ids in (
        (train_embeddings, ["u1", "u2"]),
        (validation_embeddings, ["u1"]),
        (benchmark_embeddings, ["u3", "u4"]),
    ):
        pq.write_table(
            pa.Table.from_pylist(
                [{"user_id": user_id, "embedding": [1.0, 2.0]} for user_id in ids]
            ),
            path,
            row_group_size=1,
        )
    model, _ = fit_multilabel_baseline(
        train,
        vocabulary,
        validation_path=validation,
        train_embedding_path=train_embeddings,
        validation_embedding_path=validation_embeddings,
        feature_mode="history_embedding",
        epochs=1,
        batch_size=1,
        device="cpu",
    )
    result = predict_multilabel_baseline(
        model,
        benchmark,
        tmp_path / "embedding-output",
        test_embedding_path=benchmark_embeddings,
        batch_size=1,
    )
    assert pq.read_table(result["prediction_path"]).num_rows == 2

    mismatched = tmp_path / "mismatched.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"user_id": "u4", "embedding": [1.0, 2.0]},
                {"user_id": "u3", "embedding": [1.0, 2.0]},
            ]
        ),
        mismatched,
        row_group_size=1,
    )
    with pytest.raises(ValueError, match="order differs"):
        predict_multilabel_baseline(
            model,
            benchmark,
            tmp_path / "bad-order",
            test_embedding_path=mismatched,
        )

    missing = tmp_path / "missing.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [{"user_id": "u3", "embedding": [1.0, 2.0]}]
        ),
        missing,
        row_group_size=1,
    )
    with pytest.raises(ValueError, match="coverage differs"):
        predict_multilabel_baseline(
            model,
            benchmark,
            tmp_path / "bad-coverage",
            test_embedding_path=missing,
        )
