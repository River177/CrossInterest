import json
from pathlib import Path
import pytest

from new_interest.training.short_think_grpo_engine import reward_column,validate_resume_identity
from new_interest.evaluation.preflight_short_think_grpo import sha


def test_length_penalty_starts_only_after_shared_h_step100():
    assert reward_column('HL',99)=='H'
    assert reward_column('HL',100)=='HL'
    assert reward_column('E',0)=='E' and reward_column('H',674)=='H'
    with pytest.raises(ValueError):reward_column('unknown',0)


def test_cross_arm_parent_requires_exact_hundred_step_file(tmp_path):
    path=tmp_path/'state.json';path.write_text(json.dumps({'identity':'H','step':100,'arm':'H'}))
    saved=json.loads(path.read_text())
    contract={'arm':'HL','allowed_parent':{'identity':'H','arm':'H','state_file_sha256':sha(path)}}
    validate_resume_identity(saved,'HL',contract,tmp_path)
    with pytest.raises(ValueError):validate_resume_identity({**saved,'step':99},'HL',contract,tmp_path)
    with pytest.raises(ValueError):validate_resume_identity(saved,'E',{**contract,'arm':'E'},tmp_path)
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError):validate_resume_identity(saved,'HL',contract,tmp_path)
