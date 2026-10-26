from __future__ import annotations

import csv
from pathlib import Path

import pytest

from new_interest.data.mind_io import (
    ImpressionCandidate,
    iter_behavior_events,
    load_news_catalog,
)


def _write_tsv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(rows)


def _news(news_id: str, category: str, subcategory: str, title: str) -> list[str]:
    return [news_id, category, subcategory, title, "", "", "[]", "[]"]


def _behavior(
    impression_id: str,
    history: str,
    candidates: str,
    timestamp: str = "11/09/2019 12:00:00 PM",
) -> list[str]:
    return [impression_id, "U1", timestamp, history, candidates]


def _write_fixture(root: Path) -> None:
    rows = [
        _news("N0", "Category A", "Subcategory A", "History"),
        _news("N1", "Category B", "Subcategory B", '"Quoted title'),
        _news("N2", "Category C", "Subcategory C", "Other"),
    ]
    for split in ("train", "dev"):
        _write_tsv(root / split / "news.tsv", rows)


def test_load_catalog_cleans_fields_and_checks_labels(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    catalog = load_news_catalog(tmp_path)

    assert catalog.news["N1"].title == '"Quoted title'
    assert catalog.news["N1"].interest_id == 1
    assert catalog.news_rows == {"train": 3, "dev": 3}


def test_iter_behavior_events_parses_clicks_and_timestamp(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    _write_tsv(
        tmp_path / "train" / "behaviors.tsv",
        [_behavior("1", "N0", "N1-1 N2-0 N1-1")],
    )
    catalog = load_news_catalog(tmp_path)

    event = next(iter_behavior_events(tmp_path, ("train",), catalog))

    assert event.impression_id == "1"
    assert event.history_news_ids == ("N0",)
    assert event.clicked_news_ids == ("N1",)
    assert event.impression_candidates == (
        ImpressionCandidate(news_id="N1", clicked=True),
        ImpressionCandidate(news_id="N2", clicked=False),
        ImpressionCandidate(news_id="N1", clicked=True),
    )
    assert event.timestamp.year == 2019


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (_behavior("1", "N0", "N1"), "Missing click label"),
        (_behavior("1", "N0", "N1-2"), "Unexpected label"),
        (_behavior("1", "UNKNOWN", "N1-1"), "Unknown news ID"),
        (_behavior("1", "N0", "UNKNOWN-1"), "Unknown news ID"),
        (_behavior("1", "N0", "UNKNOWN-0"), "Unknown news ID"),
        (_behavior("1", "N0", "N1-1", "not a timestamp"), "Invalid timestamp"),
    ],
)
def test_iter_behavior_events_reports_malformed_input(
    tmp_path: Path,
    row: list[str],
    message: str,
) -> None:
    _write_fixture(tmp_path)
    _write_tsv(tmp_path / "train" / "behaviors.tsv", [row])
    catalog = load_news_catalog(tmp_path)

    with pytest.raises(ValueError, match=message):
        list(iter_behavior_events(tmp_path, ("train",), catalog))


def test_catalog_rejects_cross_split_interest_changes(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    _write_tsv(
        tmp_path / "dev" / "news.tsv",
        [
            _news("N0", "Category A", "Subcategory A", "History"),
            _news("N1", "Changed", "Subcategory B", "Quoted title"),
            _news("N2", "Category C", "Subcategory C", "Other"),
        ],
    )

    with pytest.raises(ValueError, match="Inconsistent interest labels"):
        load_news_catalog(tmp_path)
