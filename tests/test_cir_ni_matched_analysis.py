import copy
import pytest

from new_interest.experiments.cir_ni_matched_analysis import compare


def fixture():
    records=[{"impression_id":"1","user_id":"u","acceptable_subcategories":["c0"]},
             {"impression_id":"2","user_id":"u","acceptable_subcategories":["c1"]},
             {"impression_id":"3","user_id":"v","acceptable_subcategories":["c1"]}]
    ranking=["c"+str(i) for i in range(21)]
    pred=[{"impression_id":r["impression_id"],"ranked_labels":ranking[:20]} for r in records]
    return records,{"ni_prefix":copy.deepcopy(pred),"cir_length_matched":copy.deepcopy(pred)},set(ranking)


def test_identical_predictions_have_exact_zero_paired_intervals():
    records,pred,allowed=fixture()
    result=compare(records,pred,allowed,draws=30)
    assert result["users"]==2 and result["target_occurrences"]==3
    assert result["metrics"]["ni_prefix"]["interest_macro_recall_at_1"]==.5
    assert all(r["difference_pp"]==0 and r["paired_user_bayesian_95_interval_pp"]==[0,0] for r in result["differences"].values())


def test_target_macro_delta_does_not_use_occurrence_average():
    records,pred,allowed=fixture()
    pred["cir_length_matched"][0]["ranked_labels"]=["c"+str(i) for i in range(1,21)]
    result=compare(records,pred,allowed,draws=30)
    assert result["differences"]["interest_macro_recall_at_10"]["difference_pp"]==-.5*100


def test_reject_misaligned_or_illegal_rankings():
    records,pred,allowed=fixture()
    pred["ni_prefix"][0]["impression_id"]="wrong"
    with pytest.raises(ValueError,match="Misaligned"):compare(records,pred,allowed,draws=2)
    records,pred,allowed=fixture()
    pred["ni_prefix"][0]["ranked_labels"][0]="c1"
    with pytest.raises(ValueError,match="Illegal"):compare(records,pred,allowed,draws=2)
