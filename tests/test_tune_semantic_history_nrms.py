from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.experiments.tune_semantic_history_nrms import (
    Architecture,
    HistorySplit,
    NRMSInterestRanker,
    _BucketAccumulator,
    _device_split,
    _evaluation_view,
    _load_dataset_row_splits,
    _score_split,
    _select_configuration,
    _unit_ids,
)


def test_bucket_accumulator_preserves_every_title_and_masks_history() -> None:
    accumulator = _BucketAccumulator(4, multi_target=False)
    accumulator.add(
        user_id="u1",
        title_ids=(7, 2, 7),
        history_ids=(1, 3),
        target=2,
        positive_ids=None,
    )

    bucket = accumulator.finalize(label_count=5, padding_id=99)

    assert bucket.title_ids.tolist() == [[7, 2, 7, 99]]
    assert bucket.history_lengths.tolist() == [3]
    assert bucket.history_mask.tolist() == [[False, True, False, True, False]]
    assert bucket.targets.tolist() == [2]


def test_bucket_accumulator_rejects_target_in_history() -> None:
    accumulator = _BucketAccumulator(4, multi_target=False)
    accumulator.add(
        user_id="u1",
        title_ids=(1,),
        history_ids=(2,),
        target=2,
        positive_ids=None,
    )
    with pytest.raises(ValueError, match="overlaps History"):
        accumulator.finalize(label_count=3, padding_id=9)


def test_nrms_forward_uses_padding_without_truncation() -> None:
    torch = pytest.importorskip("torch")
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    model = NRMSInterestRanker.build(
        embeddings,
        padding_id=3,
        label_count=5,
        architecture=Architecture(8, 2, 0.0),
    )
    model.eval()

    scores = model(torch.tensor([[0, 1, 2, 3], [1, 3, 3, 3]]))

    assert scores.shape == (2, 5)
    assert torch.isfinite(scores).all()
    assert "title_embeddings" not in model.state_dict()


def test_nrms_checkpoint_scoring_can_disable_history_filtering() -> None:
    torch = pytest.importorskip("torch")
    accumulator = _BucketAccumulator(1, multi_target=True)
    accumulator.add(
        user_id="dev-user",
        unit_id="dev-impression",
        title_ids=(0,),
        history_ids=(0,),
        target=None,
        positive_ids=(1,),
    )
    split = HistorySplit(
        name="dev",
        buckets=(accumulator.finalize(label_count=2, padding_id=1),),
        label_count=2,
        unit="impression",
    )

    class FixedScores:
        def eval(self) -> "FixedScores":
            return self

        def __call__(self, title_ids):
            return torch.tensor(
                [[10.0, 9.0]], device=title_ids.device
            ).repeat(len(title_ids), 1)

    device_split = _device_split(split, "cpu")
    filtered = _score_split(
        FixedScores(),
        split,
        device_split,
        batch_sizes={1: 1},
        precision="float32",
        device="cpu",
        labels=("history", "target"),
        k=1,
    )
    unfiltered = _score_split(
        FixedScores(),
        split,
        device_split,
        batch_sizes={1: 1},
        precision="float32",
        device="cpu",
        labels=("history", "target"),
        k=1,
        history_prediction_filtering=False,
    )

    assert filtered.top_k.tolist() == [[1]]
    assert filtered.metrics["history_prediction_filtering"] is True
    assert unfiltered.top_k.tolist() == [[0]]
    assert unfiltered.metrics["history_prediction_filtering"] is False


def test_nrms_selection_enforces_user_floor_and_recall_tolerance() -> None:
    common = {
        "hidden_dimension": 128,
        "attention_heads": 4,
        "dropout": 0.1,
        "class_balance_beta": 0.9,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "epoch": 4,
    }
    rows = [
        {
            **common,
            "interest_macro_recall_at_10": 0.270,
            "interest_macro_ndcg_at_10": 0.130,
            "user_macro_recall_at_10": 0.350,
        },
        {
            **common,
            "hidden_dimension": 256,
            "interest_macro_recall_at_10": 0.269,
            "interest_macro_ndcg_at_10": 0.145,
            "user_macro_recall_at_10": 0.360,
        },
        {
            **common,
            "hidden_dimension": 384,
            "interest_macro_recall_at_10": 0.260,
            "interest_macro_ndcg_at_10": 0.150,
            "user_macro_recall_at_10": 0.370,
        },
    ]

    selected, frontier = _select_configuration(
        rows,
        global_recall=0.12,
        user_recall_floor=0.355,
        recall_tolerance=0.002,
    )

    assert len(frontier) == 2
    assert selected["hidden_dimension"] == 256


def test_nrms_selection_fails_when_user_floor_is_unmet() -> None:
    with pytest.raises(ValueError, match="User-Recall"):
        _select_configuration(
            [
                {
                    "hidden_dimension": 128,
                    "attention_heads": 4,
                    "dropout": 0.1,
                    "class_balance_beta": 0.9,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "epoch": 2,
                    "interest_macro_recall_at_10": 0.2,
                    "interest_macro_ndcg_at_10": 0.1,
                    "user_macro_recall_at_10": 0.2,
                }
            ],
            global_recall=0.1,
            user_recall_floor=0.3,
            recall_tolerance=0.002,
        )


def _dataset_row(
    *,
    user_id: str,
    impression_id: str,
    title: str,
    history_interest: str,
    target: str,
) -> dict[str, object]:
    return {
        "user_id": user_id,
        "impression_id": impression_id,
        "history": [
            {
                "news_id": f"n-{title}",
                "category": "category",
                "subcategory": history_interest,
                "title": title,
            }
        ],
        "target_interest": {
            "category": "category",
            "subcategory": target,
        },
    }


def test_dataset_row_loader_groups_dev_targets_by_impression(tmp_path) -> None:
    train_path = tmp_path / "train.parquet"
    dev_path = tmp_path / "dev.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                _dataset_row(
                    user_id="train-user",
                    impression_id="train-impression",
                    title="train-title",
                    history_interest="history",
                    target="target-a",
                )
            ]
        ),
        train_path,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                _dataset_row(
                    user_id="dev-user",
                    impression_id="dev-impression",
                    title="dev-title",
                    history_interest="history",
                    target="target-a",
                ),
                _dataset_row(
                    user_id="dev-user",
                    impression_id="dev-impression",
                    title="dev-title",
                    history_interest="history",
                    target="target-b",
                ),
            ]
        ),
        dev_path,
    )

    train_splits, dev, audit = _load_dataset_row_splits(
        train_path=train_path,
        dev_path=dev_path,
        train_targets={"train-user": 0},
        labels=("target-a", "target-b"),
        title_to_id={"train-title": 0, "dev-title": 1},
        boundaries=(4,),
        seed=20260815,
        padding_id=2,
    )

    assert sum(split.user_count for split in train_splits.values()) == 1
    assert dev.unit == "impression"
    assert dev.user_count == 1
    assert _unit_ids(dev) == ("dev-impression",)
    assert dev.buckets[0].user_ids == ("dev-user",)
    assert _evaluation_view(dev).positives.tolist() == [[True, True]]
    assert audit["dev_rows"] == 2
    assert audit["dev_impressions"] == 1
    assert audit["dev_users"] == 1


def test_dataset_row_loader_uses_ordered_repeated_interest_inputs(
    tmp_path,
) -> None:
    train_path = tmp_path / "train.parquet"
    dev_path = tmp_path / "dev.parquet"
    train_row = _dataset_row(
        user_id="train-user",
        impression_id="train-impression",
        title="ignored-title",
        history_interest="history-a",
        target="target-a",
    )
    train_row["history"] = [
        {
            "news_id": "n1",
            "category": "category",
            "subcategory": "history-a",
            "title": "first title",
        },
        {
            "news_id": "n2",
            "category": "category",
            "subcategory": "history-b",
            "title": "second title",
        },
        {
            "news_id": "n3",
            "category": "category",
            "subcategory": "history-a",
            "title": "third title",
        },
    ]
    dev_row = _dataset_row(
        user_id="dev-user",
        impression_id="dev-impression",
        title="dev-title",
        history_interest="history-b",
        target="target-a",
    )
    pq.write_table(pa.Table.from_pylist([train_row]), train_path)
    pq.write_table(pa.Table.from_pylist([dev_row]), dev_path)

    train_splits, _, audit = _load_dataset_row_splits(
        train_path=train_path,
        dev_path=dev_path,
        train_targets={"train-user": 0},
        labels=("target-a",),
        input_to_id={"history-a": 4, "history-b": 7},
        history_input="interest_names",
        boundaries=(4,),
        seed=20260815,
        padding_id=9,
    )

    bucket = next(
        bucket
        for split in train_splits.values()
        for bucket in split.buckets
    )
    assert bucket.title_ids.tolist() == [[4, 7, 4, 9]]
    assert bucket.history_lengths.tolist() == [3]
    assert audit["history_input"] == "interest_names"


@pytest.mark.parametrize("failure", ["duplicate-target", "history-mismatch"])
def test_dataset_row_loader_rejects_invalid_impression_groups(
    tmp_path, failure: str
) -> None:
    train_path = tmp_path / "train.parquet"
    dev_path = tmp_path / "dev.parquet"
    train_row = _dataset_row(
        user_id="train-user",
        impression_id="train-impression",
        title="train-title",
        history_interest="history",
        target="target-a",
    )
    first = _dataset_row(
        user_id="dev-user",
        impression_id="dev-impression",
        title="dev-title",
        history_interest="history",
        target="target-a",
    )
    second = _dataset_row(
        user_id="dev-user",
        impression_id="dev-impression",
        title=("other-title" if failure == "history-mismatch" else "dev-title"),
        history_interest="history",
        target=("target-b" if failure == "history-mismatch" else "target-a"),
    )
    pq.write_table(pa.Table.from_pylist([train_row]), train_path)
    pq.write_table(pa.Table.from_pylist([first, second]), dev_path)

    with pytest.raises(
        ValueError,
        match=("duplicate target" if failure == "duplicate-target" else "inconsistent History"),
    ):
        _load_dataset_row_splits(
            train_path=train_path,
            dev_path=dev_path,
            train_targets={"train-user": 0},
            labels=("target-a", "target-b"),
            title_to_id={
                "train-title": 0,
                "dev-title": 1,
                "other-title": 2,
            },
            boundaries=(4,),
            seed=20260815,
            padding_id=3,
        )


def test_nrms_selection_can_disable_user_macro_floor() -> None:
    common = {
        "hidden_dimension": 128,
        "attention_heads": 4,
        "dropout": 0.1,
        "class_balance_beta": 0.0,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "epoch": 2,
        "user_macro_recall_at_10": 0.01,
    }
    selected, _ = _select_configuration(
        [
            {
                **common,
                "interest_macro_recall_at_10": 0.20,
                "interest_macro_ndcg_at_10": 0.10,
            },
            {
                **common,
                "hidden_dimension": 256,
                "interest_macro_recall_at_10": 0.199,
                "interest_macro_ndcg_at_10": 0.12,
            },
        ],
        global_recall=0.08,
        user_recall_floor=None,
        recall_tolerance=0.002,
    )

    assert selected["hidden_dimension"] == 256
