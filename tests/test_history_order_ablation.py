import copy
import random

import pytest

from new_interest.data.build_history_order_ablation import audit_pair, parse_history, permutation, shuffle_record


def record(n=20):
    entries = [f"[{i % 3}] Headline {i % 7}" for i in range(n)]
    prompt = 'History in chronological order\n\n' + '\n'.join(f'{i}. {e}' for i, e in enumerate(entries, 1)) + '\n\nPredict new interest'
    return {'user_id': 'u1', 'impression_id': 'i1', 'prompt': prompt, 'history_length': n,
            'history_subcategories': [str(i % 3) for i in range(n)], 'target_subcategory': 'new',
            'response': '<interest_001>', 'messages': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': '<interest_001>'}]}


def test_fixed_permutation_preserves_pairs_repetitions_labels_and_rng():
    a = record(50)
    frozen, rng = copy.deepcopy(a), random.getstate()
    b, ledger = shuffle_record(a, 'train')
    assert a == frozen and random.getstate() == rng
    assert a['prompt'] != b['prompt']
    assert sorted(parse_history(a)[1]) == sorted(parse_history(b)[1])
    assert sorted(ledger['indices']) == list(range(50))
    assert b['messages'][0]['content'] == b['prompt']
    assert b['messages'][1:] == a['messages'][1:]
    assert shuffle_record(a, 'train')[0] == b
    assert audit_pair([a], [b], 'train')['all_rows_input_only']


def test_permutation_never_depends_on_targets_or_impression():
    a, b = record(), record()
    b.update(impression_id='different', target_subcategory='future', response='<interest_002>')
    assert permutation(a, 'dev') == permutation(b, 'dev')
    assert shuffle_record(a, 'dev')[0]['prompt'] == shuffle_record(b, 'dev')[0]['prompt']
    assert permutation(a, 'train') != permutation(a, 'dev')


def test_singleton_is_kept_and_not_resampled():
    a = record(1)
    b, ledger = shuffle_record(a, 'dev')
    assert a == b and ledger['indices'] == [0]
    assert audit_pair([a], [b], 'dev')['length_one'] == 1


@pytest.mark.parametrize('field,value', [('target_subcategory', 'other'), ('user_id', 'other'), ('history_length', 19), ('response', 'bad')])
def test_audit_rejects_changed_frozen_field(field, value):
    a = record()
    b = shuffle_record(a, 'dev')[0]
    b[field] = value
    with pytest.raises(ValueError):
        audit_pair([a], [b], 'dev')


def test_audit_rejects_content_change_even_if_messages_match():
    a = record()
    b = shuffle_record(a, 'dev')[0]
    b['prompt'] = b['prompt'].replace('Headline', 'Different', 1)
    b['messages'][0]['content'] = b['prompt']
    with pytest.raises(ValueError):
        audit_pair([a], [b], 'dev')


def test_audit_rejects_wrong_rule_and_row_reorder():
    a = record()
    b = shuffle_record(a, 'train')[0]
    with pytest.raises(ValueError):
        audit_pair([a], [b], 'dev')
    second = record()
    second['user_id'] = 'u2'
    with pytest.raises(ValueError):
        audit_pair([a, second], [shuffle_record(second, 'dev')[0], shuffle_record(a, 'dev')[0]], 'dev')


def test_invalid_pairing_numbering_and_messages_fail_closed():
    for edit in ('categories', 'numbering', 'messages'):
        a = record()
        if edit == 'categories': a['history_subcategories'].reverse()
        elif edit == 'numbering': a['prompt'] = a['prompt'].replace('20. ', '19. ')
        else: a['messages'][0]['content'] = 'bad'
        with pytest.raises(ValueError): shuffle_record(a, 'train')
