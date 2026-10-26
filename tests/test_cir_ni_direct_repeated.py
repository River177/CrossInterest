from copy import deepcopy
from pathlib import Path

import pytest

from new_interest.data.build_cir_ni_direct_repeated import build, select_pairs


def fixture():
    c=[{"source_sample_id":str(i),"user_id":str(i),"target_subcategory":"A"} for i in range(3)]
    n=[{**x,"target_subcategory":"B"} for x in c]
    p=[{**x,"pair_index":i,"cir_target":"A","ni_target":"B","erased_target_clicks_h50":f}
       for i,(x,f) in enumerate(zip(c,[1,2,4]))]
    return c,n,p


def test_strictly_more_than_one_keeps_same_pair_indices_without_reselection():
    c,n,p=fixture();saved=deepcopy((c,n,p))
    selected,audit=select_pairs(c,n,p)
    assert selected==[1,2] and [x["kept"] for x in audit]==[False,True,True]
    assert (c,n,p)==saved


def test_rejects_reordered_NI_pairing():
    c,n,p=fixture()
    with pytest.raises(ValueError,match="identity or order"):
        select_pairs(c,n[::-1],p)


def test_rejects_missing_counterpart():
    c,n,p=fixture()
    with pytest.raises(ValueError,match="lengths"):
        select_pairs(c,n[:-1],p)


def test_rejects_empty_selection():
    c,n,p=fixture()
    with pytest.raises(ValueError,match="No repeated"):
        select_pairs(c,n,p,minimum=5)


def test_refuses_output_replacement(tmp_path):
    with pytest.raises(FileExistsError):build(Path('missing'),tmp_path)
