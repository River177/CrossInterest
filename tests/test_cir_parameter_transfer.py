from pathlib import Path
import json
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from new_interest.experiments.cir_parameter_transfer import compose_states, verify_tensor_manifest, SID_KEY, stage_spec
from new_interest.data.build_cir_transfer_probes import reconstruction_view, build


@pytest.fixture(autouse=True)
def frozen_state_auditor(monkeypatch):
    # These experiments deliberately execute the accepted frozen training stack.
    import new_interest.training as training
    frozen = Path(__file__).parents[1] / 'artifacts/mind-cir-four-20260918/bundle-v1/src/new_interest/training'
    monkeypatch.setattr(training, '__path__', [*training.__path__, str(frozen)])


def states():
    torch=pytest.importorskip('torch')
    cold={SID_KEY:torch.full((3,4),1.),'layer.lora_A.weight':torch.full((2,4),2.),'layer.lora_B.weight':torch.zeros(4,2)}
    cir={k:v+10 for k,v in cold.items()}
    return cold,cir


@pytest.mark.parametrize('arm',['keep_lora','keep_sid'])
def test_composition_restores_exact_group_without_source_mutation(arm):
    import torch
    from safetensors.torch import save_file,load_file
    from new_interest.training.sft_state_audit import tensor_state_manifest
    import tempfile
    cold,cir=states();result,provenance=compose_states(cold,cir,arm)
    for k,v in result.items():
        use_cir=(k==SID_KEY)==(arm=='keep_sid')
        assert torch.equal(v,(cir if use_cir else cold)[k])
        assert provenance[k]==('cir5' if use_cir else 'cold')
    assert torch.count_nonzero(result['layer.lora_B.weight']).item()==(8 if arm=='keep_lora' else 0)
    with tempfile.TemporaryDirectory() as d:
        f=Path(d)/'adapter.safetensors';save_file(result,str(f));verify_tensor_manifest(load_file(str(f)),tensor_state_manifest(result))
    result[SID_KEY].zero_();assert torch.all(cold[SID_KEY]==1) and torch.all(cir[SID_KEY]==11)


@pytest.mark.parametrize('fault',['key','dtype','shape','unexpected','arm'])
def test_malformed_sources_rejected(fault):
    cold,cir=states();arm='keep_sid'
    if fault=='key':cir.pop(SID_KEY)
    if fault=='dtype':cir[SID_KEY]=cir[SID_KEY].double()
    if fault=='shape':cir[SID_KEY]=cir[SID_KEY][:1]
    if fault=='unexpected':cold['base.weight']=cold[SID_KEY];cir['base.weight']=cir[SID_KEY]
    if fault=='arm':arm='both'
    with pytest.raises(ValueError):compose_states(cold,cir,arm)


def test_wrong_hash_manifest_rejected():
    from new_interest.training.sft_state_audit import tensor_state_manifest
    cold,cir=states();expected=tensor_state_manifest(cold)
    with pytest.raises(ValueError):verify_tensor_manifest(cir,expected)


def row(user='dev0', impression='0'):
    lines=[f'{i+1}. [{"A" if i<2 else "B"}] title {i}' for i in range(7)]
    return {'user_id':user,'impression_id':impression,'history_length':7,'prompt':'Header\n\n'+'\n'.join(lines)+'\n\nFooter', 'acceptable_subcategories':['A'],'acceptable_responses':['<interest_000>']}


def test_reconstruction_does_not_read_future_targets():
    original=row();changed={**original,'acceptable_subcategories':['unrelated-future'],'acceptable_responses':['arbitrary']}
    mapping={'A':'<interest_000>','B':'<interest_001>'}
    a=reconstruction_view(original,mapping);assert a==reconstruction_view(changed,mapping)
    assert a['acceptable_subcategories']==['A'] and a['history_length']==5 and set(a['history_subcategories'])=={'B'}
    assert '[A]' not in a['prompt']


def test_probe_builder_independent_users_and_fixed_coverage(tmp_path):
    ni=tmp_path/'ni.parquet';dev=tmp_path/'dev.parquet';mapping=tmp_path/'map.parquet'
    pq.write_table(pa.Table.from_pylist([{'user_id':'train'}]),ni)
    records=[row('dev'+str(i),str(i)) for i in range(5)]+[row('dev0','9')]
    pq.write_table(pa.Table.from_pylist(records),dev)
    labels=['A','B']+['C'+str(i) for i in range(106)]
    pq.write_table(pa.Table.from_pylist([{'target_subcategory':s,'interest_token':f'<interest_{i:03d}>'} for i,s in enumerate(labels)]),mapping)
    result=build(ni,dev,mapping,tmp_path/'out',maximum=4)
    assert result['reconstruction_users']==4 and result['unique_dev_users']==5 and not result['future_targets_used']
    actual=pq.read_table(tmp_path/'out/heldout_cir.parquet').to_pylist();assert len({x['user_id'] for x in actual})==4
    assert pq.ParquetFile(tmp_path/'out/sid_name.parquet').metadata.num_rows==108
    with pytest.raises(FileExistsError):build(ni,dev,mapping,tmp_path/'out',maximum=4)
    pq.write_table(pa.Table.from_pylist([{'user_id':'dev0'}]),ni)
    with pytest.raises(ValueError,match='overlap'):build(ni,dev,mapping,tmp_path/'overlap',maximum=4)


def test_stage_contract_does_not_freeze_or_change_budget():
    x=stage_spec(Path('/project'))
    assert x['epochs']==3 and x['samples']==21600 and x['global_batch_size']==32 and x['world_size']==4 and x['parent']=='composite'


def test_paired_rank_deltas_and_alignment():
    from new_interest.experiments.cir_parameter_transfer import paired_statistics
    rows=[{'acceptable_subcategories':['A']},{'acceptable_subcategories':['B']}]
    a=[{'record_index':0,'input_sha256':'x','ranked_labels':['C','A']},{'record_index':1,'input_sha256':'y','ranked_labels':['B']}]
    b=[{'record_index':0,'input_sha256':'x','ranked_labels':['A']},{'record_index':1,'input_sha256':'y','ranked_labels':['C']}]
    r=paired_statistics(a,b,rows)
    assert r['top20_rank_direction_counts']=={'better':1,'worse':1}
    assert r['macro_delta']['1']['recall']==0
    assert r['macro_delta']['20']['recall']==-0.5
    b[1]['input_sha256']='wrong'
    with pytest.raises(ValueError,match='identity'):paired_statistics(a,b,rows)


def test_peft_inplace_token_key_rewrite_cannot_mutate_canonical_state():
    from new_interest.experiments.cir_parameter_transfer import loading_copy
    cold,cir=states();composite,provenance=compose_states(cold,cir,'keep_lora')
    scratch=loading_copy(composite)
    # PEFT 0.18 AuxiliaryTrainingWrapper loading rewrites the caller dictionary.
    scratch[SID_KEY+'.default']=scratch.pop(SID_KEY)
    scratch[SID_KEY+'.default'].zero_()
    assert SID_KEY in composite and SID_KEY+'.default' not in composite
    assert composite[SID_KEY].sum().item()==12
    assert set(composite)==set(provenance)
