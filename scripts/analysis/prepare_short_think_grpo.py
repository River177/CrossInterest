"""Freeze the authorized MIND GRPO preflight without altering previous artifacts."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import shlex
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

REPO=Path(__file__).resolve().parents[2]
OLD=REPO/'artifacts/mind-short-think-sft-20260913/bundle-full-v1'
ROOT=REPO/'artifacts/mind-short-think-grpo-20260914'
BUNDLE=ROOT/'bundle-preflight-v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def validate_environment_inputs(bundle):
    """Follow the actual installer argument before paying for a worker."""
    installer=bundle/'scripts/amlt/common/install_qwen35_fast_kernels.sh'
    tokens=shlex.split(installer.read_text().replace('\\\n',' '))
    name=tokens[tokens.index('--manifest')+1]
    manifest=json.loads((bundle/name).read_text())
    if len(manifest['install_wheels'])!=4:
        raise ValueError('Incomplete pinned kernel manifest')
    return name


def main():
    if BUNDLE.exists():raise FileExistsError(BUNDLE)
    BUNDLE.mkdir(parents=True)
    for rel,digest in json.loads((OLD/'source_manifest.json').read_text())['files'].items():
        if sha(OLD/rel)!=digest:raise ValueError(f'Accepted SFT runtime changed: {rel}')
    shutil.copytree(OLD/'src',BUNDLE/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for rel in ['src/new_interest/training/short_think_grpo_reward.py',
                'src/new_interest/evaluation/preflight_short_think_grpo.py',
                'src/new_interest/experiments/short_think_grpo_preflight.py',
                'scripts/amlt/grpo/run_short_think_grpo_preflight.sh']:
        dest=BUNDLE/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/rel,dest)
    shutil.copytree(OLD/'scripts/amlt/common',BUNDLE/'scripts/amlt/common')
    kernel_manifest=Path('configs/experiments/qwen35_fast_kernel_install_v1.json')
    (BUNDLE/kernel_manifest).parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(OLD/kernel_manifest,BUNDLE/kernel_manifest)
    validate_environment_inputs(BUNDLE)
    shutil.copyfile(OLD/'requirements-short-think-worker.txt',BUNDLE/'requirements-grpo-preflight.txt')
    shutil.copyfile(OLD/'source_snapshot.json',BUNDLE/'source_snapshot.json')
    inputs=BUNDLE/'inputs';inputs.mkdir()
    adapter=REPO/'artifacts/mind-short-think-sft-20260913/readback-full-v1/sft/final_adapter'
    if sha(adapter/'adapter_model.safetensors')!='bc847408fe401920904a3cd8faad8a71393cce378ed1b06fe5ad6674d03e581d':
        raise ValueError('Wrong Short-CoT SFT checkpoint')
    shutil.copytree(adapter,inputs/'short_sft_adapter')
    for filename in ['interest_token_mapping.parquet','dev.parquet']:
        shutil.copyfile(OLD/'inputs'/filename,inputs/filename)
    historic=REPO/'data/training/preference_state_hierarchical_grpo_qwen35_behavior_interest_21600_v1'
    metadata=json.loads((historic/'metadata.json').read_text())
    source=REPO/metadata['source_train'];mapping=REPO/metadata['mapping_file'];vocab=REPO/metadata['vocabulary_file']
    for key,path in [('train',source),('mapping',mapping),('vocabulary',vocab)]:
        if sha(path)!=metadata['source_sha256'][key]:raise ValueError(f'Source changed: {key}')
    if sha(historic/'train.parquet')!=metadata['artifact_sha256']['train.parquet']:raise ValueError('Historical GRPO data changed')
    # Revalidate every logical row; the historical metadata's reward levels are
    # not inherited by this new experiment.
    sys.path.insert(0,str(BUNDLE/'src'))
    from new_interest.data.generate_preference_state_teacher import student_prompt, history_subcategories
    from new_interest.training.sft_data import apply_chat_template
    raw=pq.read_table(source).to_pylist();rows=pq.read_table(historic/'train.parquet').to_pylist()
    if len(rows)!=21600 or len({r['source_sample_id'] for r in raw})!=21600:raise ValueError('Wrong NI source')
    for old,new in zip(raw,rows):
        info=new['extra_info']
        if (old['source_sample_id']!=info['source_sample_id'] or old['response']!=info['target_token']
            or new['prompt']!=[{'role':'user','content':student_prompt(old['prompt'])}]
            or history_subcategories(old['prompt'])!=info['history_subcategories']
            or info['target_subcategory'] in info['history_subcategories']):raise ValueError('Logical NI identity differs')
    counts=Counter(r['extra_info']['target_token'] for r in rows)
    if len(counts)!=108 or set(counts.values())!={200}:raise ValueError('Class balance differs')
    shutil.copyfile(historic/'train.parquet',inputs/'train.parquet')
    shutil.copyfile(source,inputs/'source_ni.parquet')
    shutil.copyfile(vocab,inputs/'vocabulary.parquet')
    groups=defaultdict(list)
    for r in rows:groups[r['extra_info']['target_token']].append(r)
    selected=[]
    for label in sorted(groups):
        selected.extend(sorted(groups[label],key=lambda r:hashlib.sha256(('42:'+r['extra_info']['source_sample_id']).encode()).hexdigest())[:3])
    pq.write_table(pa.Table.from_pylist(selected),inputs/'preflight.parquet',compression='zstd')
    tokenizer=AutoTokenizer.from_pretrained(inputs/'short_sft_adapter',local_files_only=True)
    lengths=[]
    for r in rows:
        prompt=apply_chat_template(tokenizer,r['prompt'],add_generation_prompt=True,enable_thinking=True)
        if not prompt.endswith('<think>\n'):raise ValueError('Native template changed')
        lengths.append(len(tokenizer.encode(prompt,add_special_tokens=False)))
    if max(lengths)>2048:raise ValueError('Overlong prompt; do not silently drop rows')
    write(BUNDLE/'data_audit.json',{'train_rows':21600,'classes':108,'per_class':200,'preflight_rows':324,
        'preflight_per_class':3,'selection':'first3 sorted by SHA256(42:source_sample_id) within each class',
        'max_prompt_tokens':max(lengths),'overlong_prompts':0,'target_history_overlap':0,
        'all_rows_logically_match_source':True,'source_sha256':metadata['source_sha256'],
        'prompt_lengths_sha256':hashlib.sha256(json.dumps(lengths).encode()).hexdigest()})
    contract={'version':'mind-short-think-grpo-preflight-v1','model_name':'Qwen/Qwen3.5-2B',
        'model_revision':'15852e8c16360a2fea060d615a32b45270f8a8fc','seed':42,
        'initial_adapter_file_sha256':sha(inputs/'short_sft_adapter/adapter_model.safetensors'),
        'initial_adapter_tensor_state_sha256':'1ab00cee5802c152718d90e05ee22e0afc1ac8fb9d99727fe8cfa1474c1e56a8',
        'train_sha256':sha(inputs/'train.parquet'),'preflight_sha256':sha(inputs/'preflight.parquet'),
        'mapping_sha256':sha(inputs/'interest_token_mapping.parquet'),'dev_sha256':sha(inputs/'dev.parquet'),
        'max_prompt_tokens':2048,'thought_budget':480,'answer_budget':32,'no_forced_closure':True,
        'temperature':.8,'top_p':.95,'top_k':0,'generations':16,'generation_batch_size_per_gpu':8,
        'preflight_groups':324,'preflight_trajectories':5184,'world_size':4,'model_updates':0,
        'rewards':{'E':'exact 1,otherwise 0','H':{'exact':1.,'novel_same_parent':.2,'novel_other_parent':0.,'history_seen':-.1,'invalid':-.2},
                   'HL':'H - .1 * exact * clip((thought_tokens-160)/160,0,1)'},
        'advantage_rule':'(reward-group_mean)/max(population_group_std,.1), identical all arms',
        'reward_density_gate':{'valid_rate_min':.95,'length_limit_rate_max':.05,'exact_variable16_min':.20,'exact_variable8_min':.10},
        'numerical_gate':'raw-policy cached/fresh diagnosis only; independent behavior-logprob/backward/reload gates required before formal training',
        'planned_training':{'arms':['E-GRPO','H-GRPO','HL-GRPO'],'priority':'H-GRPO','prompts_per_step':32,
          'rollouts_per_prompt':8,'epochs':1,'updates':675,'lr':5e-6,'kl':.01,'reference':'initial Short-CoT SFT',
          'checkpoint_frequency':25,'first_quality_check_step':100,'HL_branch_step':100,'microbatch_per_gpu_initial':4,
          'microbatch_per_gpu_if_verified':8,'dynamic_token_batching':True},
        'evaluation':'unchanged short-think evaluator, greedy512,31392 Dev,108 candidate SIDs,no history mask,invalid rows retained'}
    write(BUNDLE/'contract.json',contract)
    files={str(p.relative_to(BUNDLE)):sha(p) for p in BUNDLE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    runtime={k:v for k,v in files.items() if not k.startswith('inputs/')}
    write(BUNDLE/'source_manifest.json',{'files':files,'runtime_sha256':hashlib.sha256(json.dumps(runtime,sort_keys=True).encode()).hexdigest(),
        'parent_sft_source_manifest_sha256':sha(OLD/'source_manifest.json')})
    write(ROOT/'execution_state.json',{'updated_at_utc':datetime.now(timezone.utc).isoformat(),'status':'prepared',
        'authorization':'User accepted E/H/HL thinking-GRPO design on 2026-09-14; autonomous execution/recovery authorized',
        'bundle':str(BUNDLE.relative_to(REPO)),'contract_sha256':sha(BUNDLE/'contract.json'),'active_jobs':[],
        'completed':['reward_implementation','local_reward_tests','data_and_token_audit','frozen_preflight_bundle'],
        'remaining':['submit_preflight','independent_preflight_readback','training_engine_numerical_and_backward_resume_checks',
                     'H_GRPO','E_GRPO','HL_GRPO','full_Dev_and_independent_acceptance'],
        'formal_training_started':False,'automation_status':'PAUSED'})
    print(json.dumps({'bundle':str(BUNDLE),'files':len(files),'bytes':sum((BUNDLE/p).stat().st_size for p in files),
                      'contract_sha256':sha(BUNDLE/'contract.json'),'max_prompt_tokens':max(lengths)}))


if __name__=='__main__':main()
