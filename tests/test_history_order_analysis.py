import copy

import pytest

from new_interest.experiments.history_order_analysis import aggregate, outcomes, paired_bootstrap, ranking_changes


def fixture():
    records = [{'impression_id': str(i), 'user_id': f'u{i//2}', 'acceptable_subcategories': [c]}
               for i, c in enumerate(['a', 'a', 'a', 'b'])]
    a = [{'impression_id': str(i), 'ranked_labels': [c, 'x'] if c == 'a' else ['x', c]}
         for i, c in enumerate(['a', 'a', 'a', 'b'])]
    return records, {arm: copy.deepcopy(a) for arm in 'ABCD'}


def test_equal_class_macro_is_not_impression_average():
    r, p = fixture()
    values = aggregate(outcomes(r, p))
    assert values['metrics']['A']['interest_macro_recall_at_1'] == 0.5
    assert values['support'] == {'a': 3, 'b': 1}
    assert aggregate(outcomes(r, p), [False, False, False, True])['supported_classes'] == 1


def test_identical_arms_have_exact_zero_paired_intervals_and_shared_user_ids():
    r, p = fixture()
    encoded = outcomes(r, p)
    assert encoded['user_index'].tolist() == [0, 0, 1, 1]
    result = paired_bootstrap(encoded, draws=20)
    for metrics in result['comparisons'].values():
        for metric in metrics.values():
            assert metric['difference_pp'] == 0
            assert metric['bayesian_bootstrap_95_interval_pp'] == [0, 0]
    assert ranking_changes(p)['A/B']['top1_changed_fraction'] == 0


def test_reordered_or_missing_predictions_fail_pairing():
    r, p = fixture()
    p['B'].reverse()
    with pytest.raises(ValueError): outcomes(r, p)
    r, p = fixture()
    p['C'].pop()
    with pytest.raises(ValueError): outcomes(r, p)


def test_changed_ranking_has_expected_direction_and_reproducible_interval():
    r, p = fixture()
    for row in p['B']:
        row['ranked_labels'] = [f'x{i}' for i in range(20)]
    encoded = outcomes(r, p)
    result = paired_bootstrap(encoded, draws=20)
    assert result == paired_bootstrap(encoded, draws=20)
    metric = result['comparisons']['B-A']['interest_macro_recall_at_10']
    assert metric['difference_pp'] == -100
    assert metric['bayesian_bootstrap_95_interval_pp'] == [-100, -100]
