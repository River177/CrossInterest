from __future__ import annotations

import re
from dataclasses import dataclass


INVALID_REWARD = 0.0
NOVEL_REWARD = 0.10
SAME_PARENT_REWARD = 0.35
EXACT_REWARD = 1.0

_OUTPUT = re.compile(
    r"\A<think>\s*(?P<reasoning>[\s\S]*?\S)\s*</think>\s*"
    r"<answer>\s*(?P<token><interest_\d{3}>)\s*</answer>\Z"
)


@dataclass(frozen=True)
class HierarchicalReward:
    reward: float
    valid_format: bool
    legal_token: bool
    novel: bool
    same_parent: bool
    exact: bool
    predicted_token: str | None
    predicted_subcategory: str | None
    predicted_category: str | None


def score_hierarchical_reward(
    solution_str: str,
    extra_info: dict[str, object] | None,
) -> HierarchicalReward:
    match = _OUTPUT.fullmatch(solution_str.strip())
    if match is None or extra_info is None:
        return HierarchicalReward(
            INVALID_REWARD, False, False, False, False, False, None, None, None
        )
    token = match.group("token")
    tokens = [str(value) for value in extra_info.get("interest_tokens", [])]
    subcategories = [
        str(value) for value in extra_info.get("interest_subcategories", [])
    ]
    categories = [
        str(value) for value in extra_info.get("interest_categories", [])
    ]
    if not tokens or not (
        len(tokens) == len(subcategories) == len(categories)
    ):
        raise ValueError("Reward taxonomy arrays are missing or misaligned")
    if len(tokens) != len(set(tokens)):
        raise ValueError("Reward taxonomy contains duplicate tokens")
    by_token = {
        value: (subcategories[index], categories[index])
        for index, value in enumerate(tokens)
    }
    prediction = by_token.get(token)
    if prediction is None:
        return HierarchicalReward(
            INVALID_REWARD, True, False, False, False, False, token, None, None
        )
    predicted_subcategory, predicted_category = prediction
    history = {
        str(value).casefold()
        for value in extra_info.get("history_subcategories", [])
    }
    novel = predicted_subcategory.casefold() not in history
    exact = token == str(extra_info.get("target_token", ""))
    target_category = str(extra_info.get("target_category", ""))
    same_parent = predicted_category == target_category
    if not novel:
        reward = INVALID_REWARD
    elif exact:
        reward = EXACT_REWARD
    elif same_parent:
        reward = SAME_PARENT_REWARD
    else:
        reward = NOVEL_REWARD
    return HierarchicalReward(
        reward=reward,
        valid_format=True,
        legal_token=True,
        novel=novel,
        same_parent=same_parent,
        exact=exact,
        predicted_token=token,
        predicted_subcategory=predicted_subcategory,
        predicted_category=predicted_category,
    )


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
) -> float:
    del data_source, ground_truth
    return score_hierarchical_reward(solution_str, extra_info).reward
