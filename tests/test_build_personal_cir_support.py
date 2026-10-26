from collections import Counter
import copy

import pytest

from new_interest.data.build_personal_cir_support import support_views, parse_history, epoch_plan


def row(categories):
    history = '\n'.join(f'{i}. [{c}] title {i}' for i, c in enumerate(categories, 1))
    return {'user_id': 'u', 'history_length': len(categories), 'prompt': 'header\n\n' + history + '\n\nquery',
            'response': '<future>', 'target_subcategory': 'future', 'timestamp': 'later', 'impression_id': 'i'}


def test_whole_category_deletion_eligibility_and_two_distinct_views():
    original = row(['A'] * 4 + ['B'] * 3 + ['C'])
    views, eligible = support_views(original, {'A': '<a>', 'B': '<b>', 'C': '<c>'})
    assert set(eligible) == {'B', 'C'}
    assert len(views) == 2
    for view in views:
        history = parse_history(view)
        assert view['removed_category'] not in {c for c, _ in history}
        assert view['history_length'] >= 5
        assert history == [x for x in parse_history(original) if x[0] != view['removed_category']]


def test_no_future_label_or_impression_or_timestamp_in_selection():
    original = row(['A', 'B', 'C', 'D'] * 3)
    changed = copy.deepcopy(original)
    changed.update(response='different', target_subcategory='different', source_sample_id='target-dependent',
                   sample_id='target:999', timestamp='different', impression_id='different')
    mapping = {c: '<' + c + '>' for c in 'ABCD'}
    assert support_views(original, mapping) == support_views(changed, mapping)


@pytest.mark.parametrize('categories,mapping,count', [
    (['A'] * 10, {'A': '<a>'}, 0),
    (['A', 'B'] * 2, {'A': '<a>', 'B': '<b>'}, 0),
    (['A'] * 5 + ['B'], {'B': '<b>'}, 1),
    (['A', 'B', 'C'] * 3, {'A': '<a>', 'B': '<b>', 'C': '<c>'}, 2),
])
def test_zero_one_two_supports(categories, mapping, count):
    assert len(support_views(row(categories), mapping)[0]) == count


def test_complete_epochs_and_tail():
    plan = epoch_plan(74)
    assert len(plan) == 9
    for epoch in range(3):
        ids = [i for p in plan if p['epoch'] == epoch for i in p['row_indices']]
        assert Counter(ids) == Counter(range(74))
    assert [len(x['row_indices']) for x in plan] == [32, 32, 10] * 3
    assert plan == epoch_plan(74)


def test_wrong_history_length_rejected():
    r = row(['A', 'B'] * 4)
    r['history_length'] = 5
    with pytest.raises(ValueError):
        support_views(r, {'A': '<a>'})
