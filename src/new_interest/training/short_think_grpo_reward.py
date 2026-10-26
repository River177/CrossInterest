"""Outcome rewards for the approved MIND thinking-GRPO comparison.

This is separate from the historical 0/.10/.35/1 hierarchical experiment.
Parsing receives the canonical evaluator's validity flag; execution failures must
raise upstream and must never be converted into invalid model responses.
"""
from __future__ import annotations

import math
import statistics

VERSION = "mind-short-think-grpo-reward-v1"


def score(*, valid: bool, predicted: str | None, target: str,
          history: list[str], parent_by_token: dict[str, str],
          thought_tokens: int | None) -> dict:
    if target not in parent_by_token or target in history:
        raise ValueError("Invalid target or target appears in history")
    if valid and (predicted not in parent_by_token or thought_tokens is None
                  or thought_tokens <= 0):
        raise ValueError("Valid response lacks legal SID or nonempty thought")
    exact = bool(valid and predicted == target)
    seen = bool(valid and predicted in history)
    same_parent = bool(valid and parent_by_token[predicted] == parent_by_token[target])
    if not valid:
        outcome, task = "invalid", -0.2
    elif seen:
        outcome, task = "history_seen", -0.1
    elif exact:
        outcome, task = "exact", 1.0
    elif same_parent:
        outcome, task = "novel_same_parent", 0.2
    else:
        outcome, task = "novel_other_parent", 0.0
    penalty = 0.1 * min(1.0, max(0, thought_tokens - 160) / 160) if exact else 0.0
    return {"outcome": outcome, "exact": exact, "history_seen": seen,
            "same_parent": same_parent, "E": float(exact), "H": task,
            "HL": task - penalty, "length_penalty": penalty}


def advantages(rewards: list[float], *, std_floor: float = 0.1) -> list[float]:
    """Population std with a fixed floor, shared across all three arms."""
    if len(rewards) < 2 or not all(math.isfinite(x) for x in rewards):
        raise ValueError("Expected a finite group of at least two rewards")
    if not math.isfinite(std_floor) or std_floor <= 0:
        raise ValueError("Invalid standard-deviation floor")
    mean = statistics.mean(rewards)
    scale = max(statistics.pstdev(rewards), std_floor)
    return [(x - mean) / scale for x in rewards]


def group_statistics(rows: list[dict], *, generations: int) -> dict:
    """Groups are complete, ordered, and identified before reward aggregation."""
    from collections import Counter, defaultdict
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_sample_id"]].append(row)
    if not groups:
        raise ValueError("Empty preflight")
    for group in groups.values():
        if sorted(r["generation_index"] for r in group) != list(range(generations)):
            raise ValueError("Missing or duplicate rollout in group")
    n = len(rows)
    def variable(key, group):
        return len({r["reward"][key] for r in group}) > 1
    def summary(values):
        values = sorted(values)
        return {"n": len(values), "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95": values[max(0, math.ceil(.95 * len(values)) - 1)],
                "max": max(values)} if values else {"n": 0}
    return {"groups": len(groups), "trajectories": n, "generations": generations,
            "valid_rate": sum(r["valid_format"] for r in rows) / n,
            "length_limit_rate": sum(r["budget_exhausted"] for r in rows) / n,
            "exact_rate": sum(r["reward"]["exact"] for r in rows) / n,
            "any_exact_group_rate": sum(any(r["reward"]["exact"] for r in g) for g in groups.values()) / len(groups),
            "variable_group_rates": {k: sum(variable(k, g) for g in groups.values()) / len(groups) for k in ["E", "H", "HL"]},
            "outcomes": dict(Counter(r["reward"]["outcome"] for r in rows)),
            "generated_tokens": summary([r["generated_tokens"] for r in rows]),
            "parsable_thought_tokens": summary([r["think_tokens"] for r in rows if r["think_tokens"] is not None]),
            "task_zero_variance_length_only_groups": sum(not variable("H", g) and variable("HL", g) for g in groups.values())}
