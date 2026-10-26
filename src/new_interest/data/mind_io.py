from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator


BEHAVIOR_TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"

__all__ = [
    "BehaviorEvent",
    "ImpressionCandidate",
    "NewsCatalog",
    "NewsRecord",
    "iter_behavior_events",
    "load_news_catalog",
]


@dataclass(frozen=True)
class NewsRecord:
    news_id: str
    category: str
    subcategory: str
    title: str
    interest_id: int


@dataclass(frozen=True)
class NewsCatalog:
    news: dict[str, NewsRecord]
    interests: list[tuple[str, str]]
    news_rows: dict[str, int]


@dataclass(frozen=True)
class ImpressionCandidate:
    news_id: str
    clicked: bool


@dataclass(frozen=True)
class BehaviorEvent:
    split: str
    impression_id: str
    user_id: str
    timestamp: datetime
    history_text: str
    history_news_ids: tuple[str, ...]
    clicked_news_ids: tuple[str, ...]
    impression_candidates: tuple[ImpressionCandidate, ...]


def _clean_news_field(value: str) -> str:
    if value.startswith('"'):
        value = value[1:]
    if value.endswith('"'):
        value = value[:-1]
    return value.replace('""', '"')


def _read_news_rows(path: Path) -> Iterator[list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = line.rstrip("\r\n").split("\t")
            if len(row) != 8:
                raise ValueError(
                    f"{path}: expected 8 tab-separated fields at physical "
                    f"line {line_number}, got {len(row)}"
                )
            yield [_clean_news_field(value) for value in row]


def load_news_catalog(input_dir: Path | str) -> NewsCatalog:
    input_dir = Path(input_dir)
    raw_news: dict[str, tuple[str, str, str]] = {}
    news_rows: dict[str, int] = {}

    for split in ("train", "dev"):
        path = input_dir / split / "news.tsv"
        rows = 0
        for row in _read_news_rows(path):
            news_id, category, subcategory, title = row[:4]
            previous = raw_news.get(news_id)
            if previous is not None and previous[:2] != (category, subcategory):
                raise ValueError(
                    f"Inconsistent interest labels for {news_id}: "
                    f"{previous[:2]} vs {(category, subcategory)}"
                )
            if previous is None:
                raw_news[news_id] = (category, subcategory, title)
            rows += 1
        news_rows[split] = rows

    interests = sorted(
        {(category, subcategory) for category, subcategory, _ in raw_news.values()}
    )
    interest_to_id = {
        interest: interest_id for interest_id, interest in enumerate(interests)
    }
    news = {
        news_id: NewsRecord(
            news_id=news_id,
            category=category,
            subcategory=subcategory,
            title=title,
            interest_id=interest_to_id[(category, subcategory)],
        )
        for news_id, (category, subcategory, title) in raw_news.items()
    }
    return NewsCatalog(news=news, interests=interests, news_rows=news_rows)


def _lookup_news(
    catalog: NewsCatalog,
    news_id: str,
    context: str,
) -> NewsRecord:
    try:
        return catalog.news[news_id]
    except KeyError as error:
        raise ValueError(f"Unknown news ID {news_id!r} in {context}") from error


def _parse_timestamp(
    value: str,
    *,
    path: Path,
    split: str,
    impression_id: str,
) -> datetime:
    try:
        return datetime.strptime(value, BEHAVIOR_TIME_FORMAT)
    except ValueError as error:
        raise ValueError(
            f"Invalid timestamp {value!r} in {path} {split} "
            f"impression {impression_id}"
        ) from error


def iter_behavior_events(
    input_dir: Path | str,
    splits: Iterable[str],
    catalog: NewsCatalog,
) -> Iterator[BehaviorEvent]:
    input_dir = Path(input_dir)
    for split in splits:
        path = input_dir / split / "behaviors.tsv"
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                if len(row) != 5:
                    raise ValueError(
                        f"{path}: expected 5 columns, got {len(row)} "
                        f"at CSV record {reader.line_num}"
                    )
                impression_id, user_id, raw_time, raw_history, raw_impressions = row
                history_news_ids = tuple(raw_history.split())
                if not history_news_ids:
                    raise ValueError(
                        f"{split} impression {impression_id} has empty history"
                    )
                for news_id in history_news_ids:
                    _lookup_news(
                        catalog,
                        news_id,
                        f"{split} impression {impression_id} history",
                    )

                clicked_news_ids: list[str] = []
                seen_clicked_news: set[str] = set()
                impression_candidates: list[ImpressionCandidate] = []
                for token in raw_impressions.split():
                    if "-" not in token:
                        raise ValueError(
                            f"Missing click label in {split} impression "
                            f"{impression_id}: {token}"
                        )
                    news_id, label = token.rsplit("-", 1)
                    if label not in {"0", "1"}:
                        raise ValueError(
                            f"Unexpected label {label!r} in {split} "
                            f"impression {impression_id}"
                        )
                    _lookup_news(
                        catalog,
                        news_id,
                        f"{split} impression {impression_id}",
                    )
                    clicked = label == "1"
                    impression_candidates.append(
                        ImpressionCandidate(news_id=news_id, clicked=clicked)
                    )
                    if not clicked or news_id in seen_clicked_news:
                        continue
                    clicked_news_ids.append(news_id)
                    seen_clicked_news.add(news_id)

                yield BehaviorEvent(
                    split=split,
                    impression_id=impression_id,
                    user_id=user_id,
                    timestamp=_parse_timestamp(
                        raw_time,
                        path=path,
                        split=split,
                        impression_id=impression_id,
                    ),
                    history_text=raw_history,
                    history_news_ids=history_news_ids,
                    clicked_news_ids=tuple(clicked_news_ids),
                    impression_candidates=tuple(impression_candidates),
                )
