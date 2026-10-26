from collections import Counter
from itertools import product

import pytest

from new_interest.data.build_cir_ni_direct_balanced import balanced_indices


def pairs(edges):
    return [{'source_sample_id':str(i),'cir_target':a,'ni_target':b} for i,(a,b) in enumerate(edges)]


def test_keeps_cycle_and_excludes_unbalanced_path():
    data=pairs([('A','B'),('B','C'),('C','A'),('A','D')])
    selected,_=balanced_indices(data)
    assert selected==[0,1,2]


def test_maximum_matches_exhaustive_small_instance():
    data=pairs([('A','B'),('A','B'),('B','A'),('B','C'),('C','A'),('A','D')])
    selected,_=balanced_indices(data)
    valid=[]
    for mask in product((False,True),repeat=len(data)):
        r=[p for p,keep in zip(data,mask) if keep]
        if Counter(p['cir_target'] for p in r)==Counter(p['ni_target'] for p in r):valid.append(len(r))
    assert len(selected)==max(valid)==5
    assert balanced_indices(data)[0]==selected


def test_rejects_impossible_balance():
    with pytest.raises(ValueError,match='No nonempty'):
        balanced_indices(pairs([('A','B'),('B','C')]))


def test_duplicate_sources_rejected():
    data=pairs([('A','B'),('B','A')]);data[1]['source_sample_id']='0'
    with pytest.raises(ValueError,match='Duplicate'):balanced_indices(data)
