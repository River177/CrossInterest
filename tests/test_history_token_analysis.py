import copy
import pytest
from new_interest.experiments.cir_ni_direct_analysis import compare
from new_interest.experiments.history_token_analysis import validate_pair


def fixture():
    labels=[f'c{i}' for i in range(20)]
    records=[{'impression_id':str(i),'user_id':f'u{i//2}','acceptable_subcategories':['c0'],'prompt':'text'} for i in range(4)]
    predictions=[{'impression_id':str(i),'ranked_labels':labels.copy()} for i in range(4)]
    return labels,records,{'text':copy.deepcopy(predictions),'token':copy.deepcopy(predictions)}


def test_paired_token_minus_text_direction_and_exact_null():
    labels,records,pred=fixture()
    result=compare(records,pred,labels,arms=('text','token'),draws=20)
    assert result['comparison']=='token_minus_text'
    assert all(v['difference_pp']==0 and v['paired_user_bayesian_95_interval_pp']==[0,0] for v in result['differences'].values())
    for row in pred['text']:row['ranked_labels']=row['ranked_labels'][1:]+['c0']
    result=compare(records,pred,labels,arms=('text','token'),draws=20)
    assert result['differences']['interest_macro_recall_at_10']['difference_pp']==100
    assert result['differences']['interest_macro_recall_at_10']['paired_user_bayesian_95_interval_pp']==[100,100]


def test_different_prompts_allowed_but_targets_and_users_must_match():
    _,r,_=fixture();other=copy.deepcopy(r)
    for row in other:row['prompt']='token'
    validate_pair({'text':r,'token':other})
    other[0]['user_id']='different'
    with pytest.raises(ValueError,match='non-prompt'):validate_pair({'text':r,'token':other})


def test_misaligned_predictions_rejected():
    labels,r,p=fixture();p['token'].reverse()
    with pytest.raises(ValueError,match='Misaligned'):compare(r,p,labels,arms=('text','token'),draws=20)
