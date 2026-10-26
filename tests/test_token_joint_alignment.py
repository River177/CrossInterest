import copy
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.data.build_token_joint_alignment import alignment_pairs,TASKS
from new_interest.experiments.token_joint_alignment import gate_indices,stage_specs
from new_interest.training.alignment_token_cursor import expected_supervised_tokens

BUNDLE=Path('artifacts/mind-token-joint-alignment-20261001/bundle-v1')


def cursor_module():
    from new_interest.training import old_new_execution
    return old_new_execution


def test_current_cursor_accepts_legacy_sid_rows_without_auxiliary_lengths(tmp_path):
    cursor = cursor_module()
    cursor.STAGES.add('cir/ni')
    rows = [{'sample_id': str(i), 'target_subcategory': str(i)} for i in range(36)]
    path = tmp_path / 'legacy.parquet'
    pq.write_table(pa.Table.from_pylist(rows), path)
    contract = cursor.make_cursor_contract(path, stage='cir/ni', world_size=4,
        per_device_batch_size=2, epochs=3, global_batch_size=32)
    assert contract['expected_assistant_tokens'] == [3] * 36
    assert cursor.audit_update_batches(contract, 2, rank_batches(contract, 2))['supervised_tokens'] == 12


@pytest.mark.parametrize('length', [0, -1, None])
def test_current_cursor_rejects_invalid_explicit_completion_lengths(tmp_path, length):
    cursor = cursor_module()
    cursor.STAGES.add('cir/ni')
    path = tmp_path / 'invalid.parquet'
    pq.write_table(pa.Table.from_pylist([{'sample_id': 'x', 'target_subcategory': 'x',
        'expected_assistant_tokens': length}]), path)
    with pytest.raises(ValueError, match='completion lengths'):
        cursor.make_cursor_contract(path, stage='cir/ni', world_size=1, per_device_batch_size=1)


def test_bidirectional_pairs_have_correct_roles():
    rows=alignment_pairs([{'interest_index':108,'target_subcategory':'Sample category','interest_token':'<interest_108>'}])
    assert len(rows)==2 and rows[0]['response']=='<interest_108>' and rows[1]['response']=='Sample category'
    assert 'Sample category' in rows[0]['prompt'] and '<interest_108>' in rows[1]['prompt']
    assert all(r['messages'][1]['content']==r['response'] for r in rows)


def test_lengths_are_per_example_not_three_per_answer():
    assert expected_supervised_tokens([0,2,3],[3,7,4])==[0,14,12]
    with pytest.raises(ValueError):expected_supervised_tokens([1],[3,5])
    with pytest.raises(ValueError):expected_supervised_tokens([1],[0])


def test_frozen_data_exact_NI_and_balanced_auxiliary():
    p=Path('data/training/mind_token_joint_alignment_20261001_v1')
    if not p.exists():pytest.skip('Frozen joint data unavailable')
    original=pq.read_table(p/'ni_reference.parquet');table=pq.read_table(p/'joint.parquet');rows=table.to_pylist()
    assert table.num_rows==24100
    assert [{k:r[k] for k in original.column_names} for r in rows[:21600]]==original.to_pylist()
    assert len({r['sample_id'] for r in rows})==24100
    from collections import Counter
    aux=rows[21600:];assert set(Counter(r['base_alignment_id'] for r in aux).values())=={10}
    assert len(set(r['base_alignment_id'] for r in aux))==250
    assert all(r['user_id'] is None and r['impression_id'] is None for r in aux)
    for n,counts in [(32,[24,4,4]),(41,[29,6,6])]:
        selected=[rows[i] for i in gate_indices(rows,n)]
        assert [sum(r['training_task']==task for r in selected) for task in TASKS]==counts


def rank_batches(contract,step):
    cursor=cursor_module();n=contract['samples'];batch=contract['global_batch_size'];world=contract['world_size'];micro=contract['per_device_batch_size']
    import math
    epoch,offset=divmod(step-1,math.ceil(n/batch))
    indices=cursor.epoch_permutation(n,epoch,42)[offset*batch:(offset+1)*batch]
    denom=sum(contract['expected_assistant_tokens'][i] for i in indices)
    padded=indices+[-1]*((-len(indices))%(world*micro))
    result=[[] for _ in range(world)]
    for start in range(0,len(padded),world*micro):
        for rank in range(world):
            selected=padded[start+rank*micro:start+(rank+1)*micro]
            result[rank].append({'indices':selected,'assistant_tokens':[0 if i<0 else contract['expected_assistant_tokens'][i] for i in selected],
                'input_tokens':[20]*micro,'global_supervised_token_denominator':denom})
    return result


@pytest.mark.parametrize('step',[1,2,3,4,5,6])
def test_real_cursor_mixed_answer_lengths_and_epoch_tail(tmp_path,step):
    cursor=cursor_module();cursor.STAGES.add('cir/ni')
    rows=[{'sample_id':str(i),'target_subcategory':str(i),'training_task':TASKS[i%3],'expected_assistant_tokens':[3,3,7][i%3]} for i in range(36)]
    p=tmp_path/'train.parquet';pq.write_table(pa.Table.from_pylist(rows),p)
    c=cursor.make_cursor_contract(p,stage='cir/ni',world_size=4,per_device_batch_size=2,epochs=3,global_batch_size=32)
    batches=rank_batches(c,step);result=cursor.audit_update_batches(c,step,batches)
    assert result['supervised_tokens']==sum(c['expected_assistant_tokens'][i] for i in result['indices'])
    assert result['masked_padding_slots']==(4 if step%2==0 else 0)
    wrong=copy.deepcopy(batches);wrong[0][0]['global_supervised_token_denominator']+=1
    with pytest.raises(ValueError,match='denominator'):cursor.audit_update_batches(c,step,wrong)
    wrong=copy.deepcopy(batches)
    for rank in wrong:
        for batch in rank:
            for i,n in enumerate(batch['assistant_tokens']):
                if n==7:
                    batch['assistant_tokens'][i]=3
                    with pytest.raises(ValueError,match='token counts'):cursor.audit_update_batches(c,step,wrong)
                    return


def test_frozen_full_contract_is_single_cold_joint_stage():
    if not BUNDLE.exists():pytest.skip('Frozen bundle unavailable')
    s=stage_specs(BUNDLE,Path('/tmp/joint-formal'),gates=False)
    assert len(s)==1 and s[0]['parent'] is None and s[0]['samples']==24100
    cursor=cursor_module();cursor.STAGES.add('cir/ni')
    c=cursor.make_cursor_contract(s[0]['train_file'],stage='cir/ni',world_size=4,per_device_batch_size=2,epochs=3,global_batch_size=32)
    assert c['total_steps']==2262
    counts,progress=cursor.expected_counts(24100,2262,epochs=3)
    assert set(counts)=={3} and progress['consumed_total']==72300


def test_checkpoint_resume_validates_variable_supervised_counts(tmp_path):
    cursor=cursor_module();cursor.STAGES.add('cir/ni')
    rows=[{'sample_id':str(i),'target_subcategory':str(i),'training_task':TASKS[i%3],'expected_assistant_tokens':[3,3,7][i%3]} for i in range(36)]
    p=tmp_path/'train.parquet';pq.write_table(pa.Table.from_pylist(rows),p)
    c=cursor.make_cursor_contract(p,stage='cir/ni',world_size=4,per_device_batch_size=2,epochs=3,global_batch_size=32)
    counts,_=cursor.expected_counts(36,3,epochs=3)
    exp={'identity':{'train_sha256':c['train_sha256'],'row_identity_sha256':c['row_identity_sha256'],'expected_epochs':3},
         'optimizer_step':3,'counts':counts,'assistant_tokens':expected_supervised_tokens(counts,c['expected_assistant_tokens']),'input_tokens':[20*n for n in counts]}
    cp=tmp_path/'checkpoint-3';cp.mkdir();(cp/'training_exposure_state.json').write_text(json.dumps(exp));(cp/'trainer_state.json').write_text(json.dumps({'global_step':3,'epoch':1.8}))
    assert cursor.audit_checkpoint_cursor(cp,c)['global_step']==3
    resumed=cursor.make_cursor_contract(p,stage='cir/ni',world_size=4,per_device_batch_size=2,epochs=3,global_batch_size=32,source_checkpoint=cp)
    assert resumed['source_global_step']==3
    exp['assistant_tokens']=[3*n for n in counts];(cp/'training_exposure_state.json').write_text(json.dumps(exp))
    with pytest.raises(ValueError):cursor.audit_checkpoint_cursor(cp,c)
    with pytest.raises(ValueError):cursor.make_cursor_contract(p,stage='cir/ni',world_size=4,per_device_batch_size=2,epochs=3,global_batch_size=32,source_checkpoint=cp)
