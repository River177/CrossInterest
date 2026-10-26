from pathlib import Path
import importlib.util

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.data.build_history_token_dataset import replace_history
from new_interest.training.semantic_id_tokens import load_interest_token_mapping


def test_replacement_preserves_titles_and_targets():
    prompt='Instructions [Zulu]\n1. [Zulu] A [Alpha] title\n2. [Alpha] Zulu\nPredict an interest.'
    row={'prompt':prompt,'history_length':2,'response':'<interest_000>',
         'messages':[{'role':'user','content':prompt},{'role':'assistant','content':'<interest_000>'}]}
    out=replace_history(row,{'Zulu':'<interest_000>','Alpha':'<interest_001>'})
    assert out['prompt']=='Instructions [Zulu]\n1. [<interest_000>] A [Alpha] title\n2. [<interest_001>] Zulu\nPredict an interest.'
    assert out['response']==row['response'] and out['messages'][1]==row['messages'][1]
    assert out['messages'][0]['content']==out['prompt']


def test_replacement_rejects_missing_events():
    with pytest.raises(ValueError,match='every event'):
        replace_history({'prompt':'1. [Zulu] title','history_length':2},{'Zulu':'<interest_000>'})


def test_appended_mapping_preserves_output_ids(tmp_path):
    rows=[{'interest_index':0,'interest_token':'<interest_000>','target_subcategory':'Zulu','is_output_candidate':True},
          {'interest_index':1,'interest_token':'<interest_001>','target_subcategory':'Alpha','is_output_candidate':False}]
    fp=tmp_path/'mapping.parquet';pq.write_table(pa.Table.from_pylist(rows),fp)
    assert load_interest_token_mapping(fp)==rows
    rows[0]['is_output_candidate']=False;rows[1]['is_output_candidate']=True
    pq.write_table(pa.Table.from_pylist(rows),fp)
    with pytest.raises(ValueError,match='prefix'):load_interest_token_mapping(fp)


def test_legacy_mapping_still_rejects_unsorted_labels(tmp_path):
    rows=[{'interest_index':i,'interest_token':f'<interest_{i:03d}>','target_subcategory':label} for i,label in enumerate(['Zulu','Alpha'])]
    fp=tmp_path/'mapping.parquet';pq.write_table(pa.Table.from_pylist(rows),fp)
    with pytest.raises(ValueError,match='not sorted'):load_interest_token_mapping(fp)


@pytest.mark.parametrize('arm',['text','token'])
def test_real_data_has_unchanged_targets_and_complete_mapping(arm):
    root=Path('data/training/mind_history_token_20260930_v1')
    if not root.exists():pytest.skip('Frozen local data unavailable')
    mapping=load_interest_token_mapping(root/'model_token_mapping.parquet')
    original=load_interest_token_mapping(root/'interest_token_mapping.parquet')
    assert [{k:v for k,v in row.items() if k!='is_output_candidate'} for row in mapping[:108]]==original
    assert len(mapping)==125
    for name,n in [('ni.parquet',21600),('dev_full_grouped.parquet',31392)]:
        table=pq.read_table(root/arm/name)
        assert table.num_rows==n
        other=pq.read_table(root/('token' if arm=='text' else 'text')/name)
        columns=[x for x in table.column_names if x not in ['prompt','messages']]
        assert table.select(columns).equals(other.select(columns))
