import json
import re

import pytest

from new_interest.evaluation.evaluate_short_think_ranking import (
    VERSION, analyze_generation, merge, summarize, validate_rows,
)
from new_interest.evaluation.evaluate_beam import record_input_sha256, sha256_file


class Tokenizer:
    def __init__(self):
        self.encode_map = {'<interest_000>':1, '<interest_001>':2, '<eos>':3}
        self.decode_map = {v:k for k,v in self.encode_map.items()}

    def encode(self, text, add_special_tokens=False):
        pieces=re.findall(r'<interest_\d{3}>|<eos>|[\s\S]',text)
        for piece in pieces:
            if piece not in self.encode_map:
                index=len(self.encode_map)+1
                self.encode_map[piece]=index
                self.decode_map[index]=piece
        return [self.encode_map[piece] for piece in pieces]

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(self.decode_map[i] for i in ids)


def analyze(text):
    tokenizer=Tokenizer()
    ids=tokenizer.encode(text)
    return tokenizer,analyze_generation(tokenizer,ids,stop_ids={3},token_by_id={1:'<interest_000>',2:'<interest_001>'})


def test_natural_complete_prefix_excludes_answer_and_future_text():
    tokenizer,row=analyze('A short uncertain bridge.\n</think>\n<answer><interest_001></answer><eos>')
    assert row['valid_format'] and row['generated_sid']=='<interest_001>'
    prefix=tokenizer.decode(row['generated_token_ids'][:row['answer_token_offset']])
    assert prefix.endswith('<answer>') and '<interest_' not in prefix


@pytest.mark.parametrize('text',[
    'unfinished reasoning',
    'brief</think><answer><interest_000></answer>',  # no natural stop
    'brief<answer><interest_000></answer><eos>',
    'brief</think><answer><interest_000></answer>extra<eos>',
    '<think>brief</think><answer><interest_000></answer><eos>',
    'prefer <interest_000></think><answer><interest_001></answer><eos>',
    '  </think><answer><interest_001></answer><eos>',
])
def test_no_repair_of_invalid_or_truncated_outputs(text):
    _,row=analyze(text)
    assert not row['valid_format'] and row['answer_token_offset'] is None


def test_invalid_generation_stays_in_full_macro_denominator():
    records=[{'impression_id':'a','acceptable_subcategories':['A']},
             {'impression_id':'b','acceptable_subcategories':['B']}]
    rows=[{'ranked_labels':['A'],'ranked_tokens':['<interest_000>'],'ranked_scores':[-.1],
           'generation_status':'valid','natural_end':True,'generated_tokens':10},
          {'ranked_labels':[],'ranked_tokens':[],'ranked_scores':[],
           'generation_status':'length_limit','natural_end':False,'generated_tokens':512}]
    metrics,_=summarize(records,rows,{'cutoffs':[10,20],'labels':['A','B']})
    assert metrics['samples']==2 and metrics['supported_interest_count']==2
    assert metrics['interest_macro_recall_at_10']==.5
    assert metrics['interest_macro_ndcg_at_20']==.5
    assert metrics['length_limit_rate']==.5


def test_checkpoint_rejects_changed_record_or_fabricated_invalid_ranking():
    record={'impression_id':'a','acceptable_subcategories':['A']}
    row={'record_index':0,'impression_id':'a','input_sha256':record_input_sha256(record),
         'evaluation_identity':'id','valid_format':False,'natural_end':False,'answer_token_offset':None,
         'generation_status':'length_limit','ranked_labels':[],'ranked_tokens':[],'ranked_scores':[]}
    validate_rows([row],[record],[0],'id',{'A':'<interest_000>'},1)
    with pytest.raises(ValueError,match='identity/order'):
        validate_rows([row],[{**record,'acceptable_subcategories':['B']}],[0],'id',{'A':'<interest_000>'},1)
    with pytest.raises(ValueError,match='cardinality'):
        validate_rows([{**row,'ranked_labels':['A'],'ranked_tokens':['<interest_000>'],'ranked_scores':[0.]}],
                      [record],[0],'id',{'A':'<interest_000>'},1)


def test_actual_shard_merge_keeps_invalid_records_and_rejects_corruption(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    records=[{'impression_id':str(i),'acceptable_subcategories':[label]} for i,label in enumerate(['A','B'])]
    eval_file=tmp_path/'eval.parquet';mapping=tmp_path/'mapping.parquet'
    pq.write_table(pa.Table.from_pylist(records),eval_file)
    pq.write_table(pa.Table.from_pylist([{'target_subcategory':'A','interest_token':'<interest_000>'},
                                      {'target_subcategory':'B','interest_token':'<interest_001>'}]),mapping)
    identity={'identity_version':VERSION,'evaluation_identity':'frozen','num_shards':2,
              'full_record_count':2,'max_examples':None,'top_k':1,'cutoffs':[1],
              'labels':['A','B'],'tokens':['<interest_000>','<interest_001>'],
              'eval_sha256':sha256_file(eval_file),'mapping_sha256':sha256_file(mapping)}
    dirs=[]
    for i,record in enumerate(records):
        d=tmp_path/f'shard{i}';d.mkdir();dirs.append(d)
        row={'record_index':i,'impression_id':record['impression_id'],'input_sha256':record_input_sha256(record),
             'evaluation_identity':'frozen','valid_format':i==0,'natural_end':i==0,'answer_token_offset':3 if i==0 else None,
             'generation_status':'valid' if i==0 else 'length_limit','generated_tokens':10 if i==0 else 512,
             'ranked_labels':['A'] if i==0 else [],'ranked_tokens':['<interest_000>'] if i==0 else [],
             'ranked_scores':[-.1] if i==0 else []}
        pq.write_table(pa.Table.from_pylist([row]),d/'id_token_predictions.parquet')
        (d/'metrics.json').write_text('{}')
        (d/'shard_manifest.json').write_text(json.dumps({**identity,'shard_index':i,'shard_record_count':1,
            'prediction_sha256':sha256_file(d/'id_token_predictions.parquet'),'metrics_sha256':sha256_file(d/'metrics.json')}))
    result=merge(dirs,eval_file,mapping,tmp_path/'merged',expected_records=2)
    assert result['interest_macro_recall_at_1']==.5
    assert pq.read_table(tmp_path/'merged/id_token_predictions.parquet').num_rows==2
    (dirs[1]/'metrics.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='hash mismatch'):
        merge(dirs,eval_file,mapping,tmp_path/'bad',expected_records=2)
