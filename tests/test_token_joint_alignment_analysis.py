import copy
import json
from pathlib import Path
import pytest

from new_interest.experiments.token_joint_alignment_analysis import validate_pair
from new_interest.experiments.cir_ni_direct_analysis import compare


def test_joint_pair_requires_identical_prompts_and_targets():
    row={'impression_id':'a','user_id':'u','prompt':'history','acceptable_subcategories':['c']}
    records={'baseline':[row],'joint':[copy.deepcopy(row)]}
    validate_pair(records)
    records['joint'][0]['prompt']='changed'
    with pytest.raises(ValueError,match='records differ'):validate_pair(records)


def test_joint_minus_baseline_direction():
    labels=[f'c{i}' for i in range(20)]
    records=[{'impression_id':'a','user_id':'u','acceptable_subcategories':['c0']}]
    predictions={'baseline':[{'impression_id':'a','ranked_labels':labels[1:]+['c0']}],
                 'joint':[{'impression_id':'a','ranked_labels':labels}]}
    result=compare(records,predictions,labels,arms=('baseline','joint'),draws=20)
    assert result['comparison']=='joint_minus_baseline'
    assert result['differences']['interest_macro_recall_at_10']['difference_pp']==100


def test_frozen_analysis_separates_training_budgets():
    p=Path('artifacts/mind-token-joint-alignment-20261001/analysis-bundle-v1/analysis_contract.json')
    if not p.exists():pytest.skip('Analysis bundle unavailable')
    c=json.loads(p.read_text())
    assert c['arms']['baseline']['optimizer_steps']==2025
    assert c['arms']['joint']['optimizer_steps']==2262
    assert c['arms']['joint']['presentations_by_task']=={'new_interest':64800,'interest_name_to_id':3750,'interest_id_to_name':3750}
    assert c['arms']['baseline']['presentations_by_task']=={'new_interest':64800}
