import math

import pytest

from new_interest.experiments.cir_candidate_competition import (
    audit_data, full_rankings, independently_summarize, visible_categories,
)
from new_interest.evaluation.evaluate_beam import record_input_sha256


def record(history, targets):
    return {'impression_id': '1', 'user_id': 'u', 'history_length': len(history),
            'history_subcategories': history, 'acceptable_subcategories': targets,
            'prompt': 'Task\n\n' + '\n'.join(f'{i}. [{c}] title' for i, c in enumerate(history, 1)) + '\n\nAnswer'}


def test_visible_history_and_outside_vocabulary():
    r = record(['a', 'outside', 'a'], ['b'])
    assert visible_categories(r) == {'a', 'outside'}
    audit = audit_data([r], ['a', 'b', 'c'])
    assert audit['minimum_unseen_candidates'] == 2
    assert audit['outside_vocabulary_history_occurrences'] == 1


def test_reject_positive_removal_and_history_mismatch():
    with pytest.raises(ValueError, match='positive'):
        audit_data([record(['a'], ['a'])], ['a', 'b'])
    r = record(['a'], ['b']); r['history_subcategories'] = ['c']
    with pytest.raises(ValueError, match='history'):
        visible_categories(r)


def test_full_scores_preserve_native_top20_with_ties():
    torch = pytest.importorskip('torch')
    from new_interest.evaluation.evaluate_id_token_ranking import rank_restricted_logits
    labels = [f'c{i}' for i in range(108)]; ids = list(range(108)); tokens = labels
    logits = torch.zeros(2, 120)
    logits[1] = torch.arange(120).float()
    native, entropy = rank_restricted_logits(logits, ids, labels, tokens, top_k=20)
    full, ent = full_rankings(logits, ids, labels, tokens, top_k=108)
    assert [r[:20] for r in full] == native
    assert ent == entropy
    assert all(len(set(x['label'] for x in row)) == 108 for row in full)


def test_filtering_promotes_a_target_beyond_original_top20():
    labels = [f'c{i}' for i in range(24)]
    r = record(labels[:5], ['c22'])
    rows = [{'record_index': 0, 'impression_id': '1', 'input_sha256': record_input_sha256(r),
             'ranked_labels': labels, 'ranked_scores': [-i for i in range(24)]}]
    summary, ranks = independently_summarize([r], rows, labels)
    assert summary['treatments']['original']['metrics']['interest_macro_recall_at_20'] == 0
    assert summary['treatments']['unseen']['metrics']['interest_macro_recall_at_20'] == 1
    assert ranks[0]['unseen_rank'] == 18
    assert summary['treatments']['unseen']['metrics']['interest_macro_ndcg_at_20'] == 1/math.log2(19)
    assert summary['historical_prediction_slot_fraction']['20'] == 0.25
    with pytest.raises(ValueError, match='coverage'):
        independently_summarize([r], [{**rows[0], 'ranked_labels': labels[:20], 'ranked_scores': list(range(20))}], labels)


def test_class_macro_weighting_is_not_target_micro_average():
    labels = [f'c{i}' for i in range(28)]
    records = [record(['outside'], [target]) for target in ['c0', 'c0', 'c22']]
    rows = []
    for i, r in enumerate(records):
        r['impression_id'] = str(i)
        rows.append({'record_index': i, 'impression_id': str(i), 'input_sha256': record_input_sha256(r),
                     'ranked_labels': labels, 'ranked_scores': [-i for i in range(28)]})
    summary, _ = independently_summarize(records, rows, labels)
    assert summary['treatments']['original']['metrics']['interest_macro_recall_at_1'] == 0.5
    assert summary['treatments']['original']['class_macro_mean_target_rank'] == 12
    assert summary['treatments']['original']['mean_target_rank'] == 25/3
    with pytest.raises(ValueError, match='coverage'):
        independently_summarize(records, rows[:1], labels)
