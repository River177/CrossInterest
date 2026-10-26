import importlib.util
import json
from pathlib import Path

import pytest


def module(name):
    path=Path(__file__).resolve().parents[1]/'scripts/analysis'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_status_parser_handles_duration_column_and_rejects_description():
    pipe=module('run_sid_relation_pipeline');job='qwen35-2b-sid-relation-gate'
    for body,status in [('queued 1m ago','queued'),('7m running 13m ago','running'),('8m failed 14m ago','failed')]:
        assert pipe.parse_status_table(f':{job} {body}',[job])=={job:status}
        assert pipe.parse_status_table(f':0 :{job} {body}',[job])=={job:status}
    with pytest.raises(RuntimeError):pipe.parse_status_table(f'Description {job} running',[job])


def test_formal_staging_requires_matching_real_gate_and_two_four_gpu_jobs(tmp_path):
    import yaml
    pipe=module('run_sid_relation_pipeline');pipe.REPO=tmp_path
    root=tmp_path/'artifacts/run';root.mkdir(parents=True)
    gate=root/'gate';gate.mkdir();(gate/'protocol.json').write_text('{}')
    source={'runtime_sha256':'r','files':{'protocol.json':pipe.sha(gate/'protocol.json')}}
    (gate/'source_manifest.json').write_text(json.dumps(source))
    config=tmp_path/'configs/amlt/gate.yaml';config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump({'code':{'local_dir':'test-only'},'jobs':[{'name':'test-gate',
        'sku':'80G4-A100','sla_tier':'Premium','command':['test-only']}]}))
    acceptance=root/'acceptance.json'
    good={'passed':True,'phase':'gate','condition':'relation','runtime_sha256':'r',
          'protocol_sha256':source['files']['protocol.json']}
    acceptance.write_text(json.dumps({**good,'phase':'full'}))
    with pytest.raises(ValueError):pipe.formal_bundle(gate,acceptance,root,config)
    acceptance.write_text(json.dumps({**good,'runtime_sha256':'wrong'}))
    with pytest.raises(ValueError):pipe.formal_bundle(gate,acceptance,root,config)
    acceptance.write_text(json.dumps(good));dest,yp=pipe.formal_bundle(gate,acceptance,root,config)
    jobs=yaml.safe_load(yp.read_text())['jobs']
    assert len(jobs)==2 and all(j['sku']=='80G4-A100' for j in jobs)
    assert {j['name'] for j in jobs}=={'qwen35-2b-sid-baseline','qwen35-2b-sid-relation'}
    assert (dest/'protocol.json').read_text()=='{}'


def test_paired_bootstrap_uses_matching_users_and_correct_delta():
    collect=module('collect_sid_relation_comparison')
    names=[f'c{i:03d}' for i in range(108)];a=[];b=[]
    for i in range(40):
        row={'record_index':i,'user_id':str(i),'targets':[names[i//20]],
             'input_sha256':str(i),'ranking':list(range(108)),
             'history_subcategories':[names[2],names[3]]}
        a.append(row);b.append({**row,'ranking':list(range(1,21))+[0]+list(range(21,108))})
    result=collect.compare(a,b,names,True)
    assert result['paired_delta']['macro_recall_at_10']==-.5
    assert result['ci95']['macro_recall_at_10']==[-.5,-.5]
    b[0]={**b[0],'user_id':'foreign'}
    with pytest.raises(ValueError):collect.compare(a,b,names,True)
