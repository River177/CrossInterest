from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from new_interest.evaluation.metrics import RESPONSE_PREFIX


@dataclass(frozen=True)
class RankedPrediction:
    label: str
    response: str
    score: float


@dataclass(frozen=True)
class UniquePredictionFill:
    predictions: list[RankedPrediction]
    retry_rounds: int
    initial_unique_count: int


def fill_unique_predictions(
    initial_predictions: Sequence[RankedPrediction],
    labels: Sequence[str],
    *,
    top_k: int,
    generate_remaining: Callable[
        [list[str]], Sequence[RankedPrediction]
    ],
) -> UniquePredictionFill:
    """Fill a unique Top-K when beam hypotheses map to duplicate labels."""
    ordered_labels = [str(label) for label in labels]
    if len(set(ordered_labels)) != len(ordered_labels):
        raise ValueError("Label vocabulary must be unique")
    if top_k <= 0 or top_k > len(ordered_labels):
        raise ValueError("top_k must be within the label vocabulary")
    label_set = set(ordered_labels)
    best_by_label: dict[str, RankedPrediction] = {}
    for prediction in initial_predictions:
        if prediction.label not in label_set:
            raise ValueError(
                f"Initial prediction has unknown label {prediction.label!r}"
            )
        previous = best_by_label.get(prediction.label)
        if previous is None or prediction.score > previous.score:
            best_by_label[prediction.label] = prediction
    initial_unique_count = len(best_by_label)
    retry_rounds = 0
    while len(best_by_label) < top_k:
        remaining = [
            label for label in ordered_labels if label not in best_by_label
        ]
        if not remaining:
            break
        retry_predictions = list(generate_remaining(remaining))
        retry_rounds += 1
        before = len(best_by_label)
        remaining_set = set(remaining)
        for prediction in retry_predictions:
            if prediction.label not in remaining_set:
                raise ValueError(
                    "Retry prediction was not drawn from the remaining "
                    f"labels: {prediction.label!r}"
                )
            previous = best_by_label.get(prediction.label)
            if previous is None or prediction.score > previous.score:
                best_by_label[prediction.label] = prediction
        if len(best_by_label) == before:
            raise ValueError(
                "Constrained beam retry produced no additional unique labels"
            )
    if len(best_by_label) < top_k:
        raise ValueError(
            f"Unable to fill Top-{top_k}; got {len(best_by_label)} labels"
        )
    ranked = sorted(
        best_by_label.values(),
        key=lambda prediction: (-prediction.score, prediction.label),
    )[:top_k]
    return UniquePredictionFill(
        predictions=ranked,
        retry_rounds=retry_rounds,
        initial_unique_count=initial_unique_count,
    )


class TokenTrie:
    def __init__(self, sequences: dict[tuple[int, ...], str]) -> None:
        if not sequences:
            raise ValueError("Token trie requires at least one sequence")
        self._children: list[dict[int, int]] = [{}]
        self._labels: dict[int, str] = {}
        for sequence, label in sequences.items():
            if not sequence:
                raise ValueError(f"Empty token sequence for {label!r}")
            node = 0
            for token_id in sequence:
                child = self._children[node].get(token_id)
                if child is None:
                    child = len(self._children)
                    self._children[node][token_id] = child
                    self._children.append({})
                node = child
            if node in self._labels:
                raise ValueError(
                    f"Duplicate token sequence for {label!r} and "
                    f"{self._labels[node]!r}"
                )
            self._labels[node] = label

    def allowed_tokens(self, prefix: list[int]) -> list[int]:
        node = 0
        for token_id in prefix:
            child = self._children[node].get(token_id)
            if child is None:
                return []
            node = child
        return sorted(self._children[node])

    def label_for(self, sequence: list[int]) -> str | None:
        node = 0
        for token_id in sequence:
            child = self._children[node].get(token_id)
            if child is None:
                return None
            node = child
        return self._labels.get(node)


def build_response_trie(
    tokenizer: Any,
    labels: list[str],
) -> tuple[TokenTrie, dict[str, tuple[int, ...]]]:
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    response_tokens: dict[str, tuple[int, ...]] = {}
    sequences: dict[tuple[int, ...], str] = {}
    for label in labels:
        response = f"{RESPONSE_PREFIX}{label}"
        token_ids = tokenizer(
            response,
            add_special_tokens=False,
        )["input_ids"]
        sequence = tuple([*token_ids, eos_token_id])
        response_tokens[label] = sequence
        if sequence in sequences:
            raise ValueError(
                f"Labels {sequences[sequence]!r} and {label!r} have "
                "identical token sequences"
            )
        sequences[sequence] = label
    return TokenTrie(sequences), response_tokens


def make_prefix_allowed_tokens_fn(
    trie: TokenTrie,
    *,
    initial_input_length: int,
) -> Callable[[int, Any], list[int]]:
    cache: dict[tuple[int, ...], list[int]] = {}

    def prefix_allowed_tokens_fn(
        batch_id: int,
        input_ids: Any,
    ) -> list[int]:
        del batch_id
        generated = input_ids[initial_input_length:].tolist()
        key = tuple(generated)
        cached = cache.get(key)
        if cached is not None:
            return cached
        allowed = trie.allowed_tokens(generated)
        if not allowed:
            for end in range(len(generated), 0, -1):
                if (
                    trie.label_for(generated[:end]) is not None
                    and all(
                        token_id == generated[end - 1]
                        for token_id in generated[end:]
                    )
                ):
                    result = [generated[end - 1]]
                    cache[key] = result
                    return result
        if not allowed:
            raise ValueError(
                f"Generated prefix left the response trie: {generated}"
            )
        cache[key] = allowed
        return allowed

    return prefix_allowed_tokens_fn


def decode_ranked_sequences(
    tokenizer: Any,
    trie: TokenTrie,
    generated_sequences: Any,
    sequence_scores: Any,
    *,
    initial_input_length: int,
    num_return_sequences: int,
) -> list[list[RankedPrediction]]:
    if len(generated_sequences) != len(sequence_scores):
        raise ValueError("Sequence and score counts do not match")
    if len(generated_sequences) % num_return_sequences != 0:
        raise ValueError("Generated sequence count is not batch-aligned")

    batch_size = len(generated_sequences) // num_return_sequences
    ranked_batches: list[list[RankedPrediction]] = []
    for batch_index in range(batch_size):
        predictions: list[RankedPrediction] = []
        seen_labels: set[str] = set()
        start = batch_index * num_return_sequences
        stop = start + num_return_sequences
        candidates = sorted(
            zip(
                generated_sequences[start:stop],
                sequence_scores[start:stop],
            ),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        for sequence, score in candidates:
            generated = sequence[initial_input_length:].tolist()
            if tokenizer.eos_token_id in generated:
                eos_index = generated.index(tokenizer.eos_token_id)
                generated = generated[: eos_index + 1]
            label = trie.label_for(generated)
            if label is None:
                raise ValueError(
                    "Beam output is not a complete canonical response: "
                    f"{tokenizer.decode(generated)}"
                )
            if label in seen_labels:
                continue
            seen_labels.add(label)
            predictions.append(
                RankedPrediction(
                    label=label,
                    response=f"{RESPONSE_PREFIX}{label}",
                    score=float(score),
                )
            )
        ranked_batches.append(predictions)
    return ranked_batches
