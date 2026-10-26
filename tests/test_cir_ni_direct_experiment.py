import json
import math

import pytest

from new_interest.experiments.cir_ni_direct import schedule, stage_specs


def project(tmp_path):
    p=tmp_path/'configs/experiments/mind_cir_108_v1.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'direct_training':{'samples':1055,'epochs':3,'optimizer_steps':99,'presentations':3165,'parent':None}}))
    return tmp_path


def test_formal_is_single_cold_stage_not_CIR_then_NI(tmp_path):
    specs=stage_specs(project(tmp_path),tmp_path/'formal',gates=False)
    assert len(specs)==1
    s=specs[0]
    assert s['key']=='ni' and s['parent'] is None and s['samples']==1055
    assert s['epochs']==3 and math.ceil(s['samples']/s['global_batch_size'])*s['epochs']==99
    assert s['world_size']==4 and s['global_batch_size']==32


def test_gates_are_cold_and_world_sizes_correct(tmp_path):
    specs=stage_specs(project(tmp_path),tmp_path/'gates',gates=True)
    assert [s['world_size'] for s in specs]==[1,4]
    assert [math.ceil(s['samples']/s['global_batch_size'])*s['epochs'] for s in specs]==[8,2]
    assert all(s['parent'] is None for s in specs)


def test_changed_budget_rejected(tmp_path):
    root=project(tmp_path);p=root/'configs/experiments/mind_cir_108_v1.json'
    d=json.loads(p.read_text());d['direct_training']['epochs']=5;p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='budget'):schedule(root)
