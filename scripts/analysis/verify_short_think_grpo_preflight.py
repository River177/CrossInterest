"""Independent downloaded-token/reward/coverage check for the MIND GRPO preflight."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def verify(bundle,result):
    contract=read(bundle/'contract.json');identity=sha(bundle/'contract.json')
    manifest=read(result/'artifact_manifest.json')
    assert manifest['contract_sha256']==identity
    for name,digest in manifest['files'].items():
        assert sha(result/name)==digest, name
    source=pq.read_table(bundle/'inputs/preflight.parquet').to_pylist()
    predictions=pq.read_table(result/'trajectories.parquet').to_pylist()
    assert len(source)==324 and len(predictions)==5184
    assert len({r['extra_info']['source_sample_id'] for r in source})==324
    assert set(Counter(r['extra_info']['target_token'] for r in source).values())=={3}
    tokenizer=AutoTokenizer.from_pretrained(bundle/'inputs/short_sft_adapter',local_files_only=True)
    stops={tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|im_end|>')}
    closure=tokenizer.encode('</think>',add_special_tokens=False)
    assert len(closure)==1
    outcomes=Counter();groups=defaultdict(list);valid_count=capped_count=exact_count=0
    grammar=re.compile(r'\A(?P<think>\s*\S[\s\S]*?)</think>\s*<answer>\s*(?P<sid><interest_\d{3}>)\s*</answer>\s*\Z')
    for index,row in enumerate(predictions):
        group,generation=divmod(index,16);record=source[group];info=record['extra_info']
        assert row['group_index']==group and row['generation_index']==generation
        assert row['contract_sha256']==identity and row['source_sample_id']==info['source_sample_id']
        # Direct tokenizer template check, independent of the worker helper.
        prompt=tokenizer.apply_chat_template(record['prompt'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
        prompt_ids=tokenizer.encode(prompt,add_special_tokens=False)
        assert row['prompt_token_ids']==prompt_ids and len(prompt_ids)<=2048
        assert row['prompt_mask']==[0]*len(prompt_ids)
        ids=row['rollout_token_ids'];thought_n=row['thought_stage_tokens'];answer_n=row['answer_stage_tokens']
        assert len(ids)==thought_n+answer_n and row['response_mask']==[1]*len(ids)
        assert 0<thought_n<=480 and 0<=answer_n<=32
        thought=ids[:thought_n];answer=ids[thought_n:]
        closed=thought[-1]==closure[0]
        assert row['thought_closed']==closed
        assert all(x not in stops|{closure[0]} for x in thought[:-1])
        if closed:
            assert answer and all(x not in stops for x in answer[:-1])
            capped=answer[-1] not in stops
        else:
            assert not answer
            capped=thought[-1] not in stops
        if contract.get('rollout_constraints')=='phase_sid_v1':
            legal_ids={tokenizer.convert_tokens_to_ids(token) for token in info['interest_tokens']}
            assert len(legal_ids)==108 and not set(thought)&legal_ids
            for j,token_id in enumerate(answer):
                before=tokenizer.decode(answer[:j],skip_special_tokens=False)
                if re.fullmatch(r'\s*<answer>\s*',before):assert token_id in legal_ids
        if capped:assert (answer_n==32 if closed else thought_n==480)
        assert row['budget_exhausted']==capped
        natural=ids[-1] in stops
        text=tokenizer.decode(ids[:-1] if natural else ids,skip_special_tokens=False)
        match=grammar.fullmatch(text) if natural else None
        legal=bool(match and '<think>' not in match['think'] and not re.search(r'<interest_\d{3}>',match['think']))
        legal=legal and match['sid'] in info['interest_tokens'] and not capped
        sid=match['sid'] if legal else None
        assert row['generated_text']==text and row['natural_end']==natural
        assert row['valid_format']==legal and row['generated_sid']==sid
        assert row['generated_tokens']==len(ids)
        thought_tokens=len(tokenizer.encode(match['think'],add_special_tokens=False)) if match else None
        assert row['think_tokens']==thought_tokens
        parents=dict(zip(info['interest_tokens'],info['interest_categories']))
        labels=dict(zip(info['interest_tokens'],info['interest_subcategories']))
        target=info['target_token']
        assert labels[target] not in info['history_subcategories']
        exact=bool(legal and sid==target)
        seen=bool(legal and labels[sid] in info['history_subcategories'])
        parent=bool(legal and parents[sid]==parents[target])
        if not legal:outcome,h='invalid',-.2
        elif seen:outcome,h='history_seen',-.1
        elif exact:outcome,h='exact',1.
        elif parent:outcome,h='novel_same_parent',.2
        else:outcome,h='novel_other_parent',0.
        penalty=.1*min(1,max(0,(thought_tokens-160)/160)) if exact else 0.
        expected={'E':float(exact),'H':h,'HL':h-penalty,'length_penalty':penalty,
                  'exact':exact,'history_seen':seen,'same_parent':parent,'outcome':outcome}
        assert row['reward']==expected
        groups[group].append(expected);outcomes[outcome]+=1
        valid_count+=legal;capped_count+=capped;exact_count+=exact
    metrics=read(result/'metrics.json')
    def close(a,b):assert math.isclose(a,b,rel_tol=0,abs_tol=1e-12),(a,b)
    close(metrics['valid_rate'],valid_count/5184)
    close(metrics['length_limit_rate'],capped_count/5184)
    close(metrics['exact_rate'],exact_count/5184)
    assert metrics['outcomes']==dict(outcomes)
    variable={k:sum(len({r[k] for r in g})>1 for g in groups.values())/324 for k in ['E','H','HL']}
    variable8={k:sum(len({r[k] for r in g[:8]})>1 for g in groups.values())/324 for k in ['E','H','HL']}
    for key in variable:
        close(metrics['variable_group_rates'][key],variable[key])
        close(metrics['first8']['variable_group_rates'][key],variable8[key])
    gate=valid_count/5184>=.95 and capped_count/5184<=.05 and variable['E']>=.2 and variable8['E']>=.1
    assert metrics['reward_density_gate']==gate
    assert metrics['model_updates']==0 and not metrics['training_authorized_by_this_artifact']
    return {'artifact_integrity_passed':True,'independent_rewards_and_token_boundaries_passed':True,
        'artifact_files':len(manifest['files']),'artifact_bytes':sum((result/name).stat().st_size for name in manifest['files']),
        'verifier_sha256':sha(Path(__file__)),'rollout_constraints':contract.get('rollout_constraints','none'),
        'groups':324,'trajectories':5184,'valid_rate':valid_count/5184,'exact_rate':exact_count/5184,
        'reward_density_gate':gate,'outcomes':dict(outcomes),'contract_sha256':identity,
        'artifact_manifest_sha256':sha(result/'artifact_manifest.json'),
        'model_updates':0,'formal_training_acceptance':False,
        'pending':'Numerical trajectory diagnostics and actual training-engine/backward/reload/resume gates remain separate'}


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ['bundle','result','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();report=verify(args.bundle,args.result)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
