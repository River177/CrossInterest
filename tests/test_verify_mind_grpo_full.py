import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('full_reader',Path(__file__).resolve().parents[1]/'scripts/analysis/verify_mind_grpo_full.py')
reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)


def checkpoint(root,identity,step,start,arm='H'):
    folder=root/f'checkpoint-{step}';folder.mkdir(parents=True)
    (folder/'state.json').write_text(json.dumps({'identity':identity,'step':step,'arm':arm}))
    (folder/'start-0.json').write_text(json.dumps({'start_step':start}))
    (folder/'checkpoint_manifest.json').write_text(json.dumps({'identity':identity,'files':{'state.json':reader.sha(folder/'state.json'),'start-0.json':reader.sha(folder/'start-0.json')}}))
    return folder


def test_missing_ancestor_is_not_silently_skipped(tmp_path):
    a=tmp_path/'first';b=tmp_path/'retry'
    old=checkpoint(a,'H',25,0);new=checkpoint(b,'H',50,25)
    (b/'initial_parent_reference.json').write_text(json.dumps({'step':25,'path':'cloud/parent25','manifest_sha256':reader.sha(old/'checkpoint_manifest.json')}))
    with pytest.raises(ValueError,match='Missing explicit ancestor'):reader.resolve_chain([b],'H',50)
    chain=reader.resolve_chain([a,b],'H',50)
    assert [x['state']['step'] for x in chain]==[25,50]


def test_hl_only_accepts_bound_h100(tmp_path):
    a=tmp_path/'H';b=tmp_path/'HL'
    previous=0
    for step in [25,50,75,100]:parent=checkpoint(a,'H',step,previous);previous=step
    checkpoint(b,'HL',125,100,arm='HL')
    (b/'initial_parent_reference.json').write_text(json.dumps({'step':100,'path':'cloud/H100','manifest_sha256':reader.sha(parent/'checkpoint_manifest.json')}))
    allowed={'identity':'H','state_file_sha256':reader.sha(parent/'state.json')}
    assert len(reader.resolve_chain([a,b],'HL',125,allowed))==5
    with pytest.raises(ValueError,match='Unapproved'):reader.resolve_chain([a,b],'HL',125,{**allowed,'state_file_sha256':'wrong'})
