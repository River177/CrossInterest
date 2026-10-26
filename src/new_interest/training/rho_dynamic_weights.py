"""Pure scalar metadata helpers for class-balanced RHO sample weights.

Callers authenticate score artifacts and their independent-reader receipts
before supplying records here. These functions enforce identity and coverage
of the joins; they do not fetch files, load models, or accept tensor gradients.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from numbers import Real


VERSION = "rho-classwise-bounded-midrank-v1"
SOURCE_FIELDS = ("source_sample_id", "user_id", "target_subcategory", "messages")


def canonical_sha256(value: object) -> str:
    """Match the reference scorer's canonical identity for JSON source fields."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scalar(value, field: str) -> float:
    # Do not float(tensor): that can silently detach a training graph. NumPy
    # real scalars are accepted; Torch tensors are not numbers.Real instances.
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real scalar metadata value, not a tensor or boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _text(row: Mapping, field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing or invalid {field}")
    return value


def _digest(value, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Missing or invalid {field}")
    return value


def source_identity(row: Mapping) -> str:
    """Use complete source messages when present; otherwise require a hash."""
    for field in SOURCE_FIELDS[:3]:
        _text(row, field)
    supplied = row.get("input_sha256")
    if "messages" in row:
        messages = row["messages"]
        if (not isinstance(messages, list) or len(messages) != 2
                or any(not isinstance(message, Mapping) for message in messages)
                or [message.get("role") for message in messages] != ["user", "assistant"]
                or any(not isinstance(message.get("content"), str) or not message["content"] for message in messages)):
            raise ValueError("Source messages must contain the original nonempty user/assistant pair")
        actual = canonical_sha256({field: row[field] for field in SOURCE_FIELDS})
        if supplied is not None and _digest(supplied, "input_sha256") != actual:
            raise ValueError("Source messages differ from supplied input_sha256")
        return actual
    return _digest(supplied, "input_sha256")


def midranks(values: Sequence[Real]) -> list[float]:
    """Ascending zero-based ranks, averaging ranks of exactly equal scalars."""
    scalars = [_scalar(value, "priority") for value in values]
    if not scalars:
        raise ValueError("Cannot rank an empty class")
    order = sorted(range(len(scalars)), key=scalars.__getitem__)
    result = [0.0] * len(scalars)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and scalars[order[stop]] == scalars[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2.0
        for position in order[start:stop]:
            result[position] = rank
        start = stop
    return result


def _rank_details(priorities: Sequence[Real]) -> dict:
    ranks = midranks(priorities)
    n = len(ranks)
    quantiles = [rank / (n - 1) if n > 1 else 0.5 for rank in ranks]
    raw = [0.5 + quantile for quantile in quantiles]
    normalizer = math.fsum(raw) / n
    weights = [value / normalizer for value in raw]
    # Floating-point tolerance preserves exact ties; do not repair a residual
    # on one example, which would arbitrarily split tied examples.
    if any(not 0.5 - 1e-12 <= value <= 1.5 + 1e-12 for value in weights):
        raise ValueError("Classwise bounded weight range violated")
    if not math.isclose(math.fsum(weights), n, rel_tol=0, abs_tol=max(1e-12, n * 1e-14)):
        raise ValueError("Classwise weight mass differs from source class support")
    return {"midranks": ranks, "quantiles": quantiles, "raw_weights": raw,
            "class_normalizer": normalizer, "weights": weights}


def bounded_midrank_weights(priorities: Sequence[Real]) -> list[float]:
    """Map a complete class to positive [0.5, 1.5] weights of mean one."""
    return _rank_details(priorities)["weights"]


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right):
        raise ValueError("Correlation vectors differ in length")
    if len(left) < 2:
        return None
    # Scale before centering so even very large finite metadata does not
    # overflow a covariance or variance calculation.
    left_scale, right_scale = max(map(abs, left)), max(map(abs, right))
    if left_scale == 0 or right_scale == 0:
        return None
    x, y = [v / left_scale for v in left], [v / right_scale for v in right]
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    x, y = [v - mx for v in x], [v - my for v in y]
    xx, yy = math.fsum(v * v for v in x), math.fsum(v * v for v in y)
    if xx == 0 or yy == 0:
        return None
    value = math.fsum(a * b for a, b in zip(x, y)) / math.sqrt(xx * yy)
    return min(1.0, max(-1.0, value))


def correlations(left: Sequence[Real], right: Sequence[Real]) -> dict:
    """Undefined constant-vector correlations are null, not a fabricated zero."""
    x, y = [_scalar(v, "correlation") for v in left], [_scalar(v, "correlation") for v in right]
    if not x or len(x) != len(y):
        raise ValueError("Correlation inputs must have equal nonzero coverage")
    return {"pearson": _pearson(x, y), "spearman": _pearson(midranks(x), midranks(y))}


def _statistics(values: list[float]) -> dict:
    ordered = sorted(values)
    scale = max(map(abs, values))
    scaled = [value / scale for value in values] if scale else [0.0] * len(values)
    mean_scaled = math.fsum(scaled) / len(values)
    mean = mean_scaled * scale
    std = math.sqrt(math.fsum((v - mean_scaled) ** 2 for v in scaled) / len(values)) * scale
    def quantile(q):
        position = (len(ordered) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        fraction = position - low
        return (1 - fraction) * ordered[low] + fraction * ordered[high]
    return {"count": len(values), "mean": mean, "std": std, "min": ordered[0], "max": ordered[-1],
            "p10": quantile(.1), "median": quantile(.5), "p90": quantile(.9)}


def _weight_diagnostics(rows: list[dict]) -> dict:
    weights = [row["sample_weight"] for row in rows]
    priorities = [row["priority"] for row in rows]
    total, squared = math.fsum(weights), math.fsum(value * value for value in weights)
    ess = total * total / squared
    result = {"rows": len(rows), "weight": _statistics(weights), "weight_sum": total,
        "effective_sample_size": ess, "effective_sample_fraction": ess / len(rows),
        "below_one": sum(v < 1 for v in weights), "equal_one": sum(v == 1 for v in weights),
        "above_one": sum(v > 1 for v in weights), "priority": _statistics(priorities),
        "priority_distinct_values": len(set(priorities)),
        "priority_tied_groups": sum(count > 1 for count in Counter(priorities).values()),
        "all_priorities_tied": len(set(priorities)) == 1,
        "correlations": {"priority_vs_weight": correlations(priorities, weights)}}
    for field in ("reference_nll", "current_nll"):
        if rows[0][field] is not None:
            values = [row[field] for row in rows]
            result[field] = _statistics(values)
            result["correlations"][field + "_vs_weight"] = correlations(values, weights)
            result["correlations"][field + "_vs_priority"] = correlations(values, priorities)
    if rows[0]["reference_nll"] is not None and rows[0]["current_nll"] is not None:
        result["correlations"]["reference_vs_current_nll"] = correlations(
            [r["reference_nll"] for r in rows], [r["current_nll"] for r in rows])
    return result


def _score_map(records, expected, *, name, loss_key):
    if records is None:
        return None, []
    result, users, evaluation_ids = {}, set(), set()
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError(f"{name} records must be mappings")
        sample = _text(record, "source_sample_id")
        user = _text(record, "user_id")
        target = _text(record, "target_subcategory")
        if sample in result or user in users:
            raise ValueError(f"Duplicate {name} sample or user identity")
        if sample not in expected:
            raise ValueError(f"Unexpected {name} source_sample_id: {sample}")
        source = expected[sample]
        digest = _digest(record.get("input_sha256"), "input_sha256")
        if (user, target, digest) != (source["user_id"], source["target_subcategory"], source["input_sha256"]):
            raise ValueError(f"{name} user/target/input identity mismatch: {sample}")
        if loss_key not in record:
            raise ValueError(f"{name} score lacks {loss_key}")
        result[sample] = _scalar(record[loss_key], name + "." + loss_key)
        users.add(user)
        if "evaluation_identity" in record:
            evaluation_ids.add(_text(record, "evaluation_identity"))
    if set(result) != set(expected):
        raise ValueError(f"Incomplete {name} coverage: expected {len(expected)}, got {len(result)}")
    return result, sorted(evaluation_ids)


def build_classwise_weights(
    source_rows, *, mode: str, reference_rows=None, current_rows=None,
    expected_classes: int | None = 108, expected_rows_per_class: int | None = 200,
    reference_loss_key: str = "full_vocab_first_sid_nll", current_loss_key: str = "full_vocab_first_sid_nll",
) -> dict:
    """Build source-ordered weights and JSON diagnostics for one full snapshot.

    ``static`` prioritizes ``-reference_nll``. ``dynamic`` prioritizes
    ``current_nll-reference_nll``. ``uniform`` assigns one to every example.
    Any supplied score cohort must have exactly the same source/user/input
    coverage, even when its losses are used only for diagnostics.
    """
    if mode not in ("uniform", "static", "dynamic"):
        raise ValueError("mode must be uniform, static, or dynamic")
    if mode != "uniform" and reference_rows is None:
        raise ValueError(f"{mode} mode requires reference_rows")
    if mode == "dynamic" and current_rows is None:
        raise ValueError("dynamic mode requires current_rows")
    for value, name in ((expected_classes, "expected_classes"), (expected_rows_per_class, "expected_rows_per_class")):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{name} must be a positive integer or None")
    rows, expected, seen_users = [], {}, set()
    computed_identities = 0
    for source in source_rows:
        if not isinstance(source, Mapping):
            raise TypeError("Source records must be mappings")
        sample, user, target = [_text(source, field) for field in SOURCE_FIELDS[:3]]
        if sample in expected or user in seen_users:
            raise ValueError("Duplicate source sample or user identity")
        record = {"source_row_index": len(rows), "source_sample_id": sample, "user_id": user,
                  "target_subcategory": target, "input_sha256": source_identity(source)}
        computed_identities += int("messages" in source)
        expected[sample] = record
        seen_users.add(user)
        rows.append(record)
    if not rows:
        raise ValueError("Cannot weight an empty source cohort")
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["target_subcategory"]].append(index)
    if expected_classes is not None and len(groups) != expected_classes:
        raise ValueError(f"Expected {expected_classes} classes, got {len(groups)}")
    if expected_rows_per_class is not None and any(len(indices) != expected_rows_per_class for indices in groups.values()):
        raise ValueError(f"Expected exactly {expected_rows_per_class} examples in every class")
    reference, reference_ids = _score_map(reference_rows, expected, name="reference", loss_key=reference_loss_key)
    current, current_ids = _score_map(current_rows, expected, name="current", loss_key=current_loss_key)
    for row in rows:
        sample = row["source_sample_id"]
        ref = reference[sample] if reference is not None else None
        cur = current[sample] if current is not None else None
        priority = 0.0 if mode == "uniform" else (-ref if mode == "static" else cur - ref)
        row.update(mode=mode, reference_nll=ref, current_nll=cur, priority=_scalar(priority, "derived priority"))
    per_class = {}
    for target, indices in sorted(groups.items()):
        details = _rank_details([rows[i]["priority"] for i in indices])
        for position, index in enumerate(indices):
            rows[index].update(sample_weight=details["weights"][position],
                within_class_midrank=details["midranks"][position],
                within_class_quantile=details["quantiles"][position],
                raw_weight=details["raw_weights"][position], class_normalizer=details["class_normalizer"],
                class_support=len(indices))
        per_class[target] = _weight_diagnostics([rows[i] for i in indices])
    diagnostics = {"version": VERSION, "mode": mode, "rows": len(rows), "classes": len(groups),
        "rule": "ascending exact-tie midrank; q=midrank/(n-1), n=1:q=0.5; raw=0.5+q; classwise mean-normalize to one",
        "priority_definition": {"uniform": "0", "static": "-reference_nll", "dynamic": "current_nll-reference_nll"}[mode],
        "weight_bounds": [0.5, 1.5], "class_weight_mass_preserved": True,
        "source_order_preserved": True, "one_source_sample_per_user": True,
        "source_identities_computed_from_messages": computed_identities,
        "source_identities_from_provided_hash": len(rows) - computed_identities,
        "source_identity_sha256": canonical_sha256([{key: row[key] for key in
            ("source_sample_id", "user_id", "target_subcategory", "input_sha256")} for row in rows]),
        "reference_score_sha256": canonical_sha256(reference) if reference is not None else None,
        "current_score_sha256": canonical_sha256(current) if current is not None else None,
        "reference_evaluation_identities": reference_ids, "current_evaluation_identities": current_ids,
        "weight_records_sha256": canonical_sha256(rows),
        "global": _weight_diagnostics(rows), "per_class": per_class,
        "verification_scope": "source/user/target/input identity and exact score coverage; artifact manifests must be verified by caller",
        "interpretation": "training priority metadata, not a noise-truth label or evidence of weighting effectiveness"}
    return {"rows": rows, "diagnostics": diagnostics}
