"""Two source-verified partial CIR initializations followed by unchanged NI3 SFT."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import math
from collections import Counter
import os
from pathlib import Path
import platform
import shutil
import sys
import pyarrow.parquet as pq
from new_interest.experiments import cir_warmup as core

SID_KEY = 'base_model.model.model.embed_tokens.token_adapter.trainable_tokens_delta'
ARMS = ('keep_lora', 'keep_sid')
CONTRACT = Path('manifests/cir_parameter_transfer.json')


def compose_states(cold, cir, arm):
    if arm not in ARMS or set(cold) != set(cir) or SID_KEY not in cold:
        raise ValueError('Invalid arm or state inventory')
    result, provenance = {}, {}
    for key in sorted(cold):
        if key != SID_KEY and 'lora_' not in key:
            raise ValueError('Unexpected tensor: '+key)
        if cold[key].shape != cir[key].shape or cold[key].dtype != cir[key].dtype:
            raise ValueError('Tensor shape/dtype differs: '+key)
        source = 'cir5' if ((key == SID_KEY) == (arm == 'keep_sid')) else 'cold'
        result[key] = (cir if source == 'cir5' else cold)[key].detach().cpu().clone()
        provenance[key] = source
    return result, provenance


def loading_copy(state):
    """PEFT 0.18 renames trainable-token keys in-place; retain canonical provenance."""
    return {key: value.detach().clone() for key, value in state.items()}


def verify_tensor_manifest(state, expected):
    from new_interest.training.sft_state_audit import tensor_state_manifest
    actual = tensor_state_manifest(state)
    if actual['state_sha256'] != expected['state_sha256'] or actual['tensors'] != expected['tensors']:
        raise ValueError('State differs from frozen source tensor manifest')
    return actual


def verify_bundle(project):
    envelope = core.read(project/'manifests/transfer_bundle.json')
    for name, digest in envelope['files'].items():
        core.require(core.sha256(project/name) == digest, 'Transfer payload changed: '+name)
    contract = core.read(project/CONTRACT)
    core.require(core.sha256(project/'manifests/cir_bundle.json') == contract['source_bundle_sha256'], 'Original bundle changed')
    for name, digest in core.read(project/'manifests/cir_bundle.json')['files'].items():
        core.require(core.sha256(project/name) == digest, 'Original runtime changed: '+name)
    return contract


def stage_spec(project):
    return {'key': 'ni3', 'task': 'ni', 'parent': 'composite', 'samples': 21600, 'epochs': 3,
            'world_size': 4, 'per_device_train_batch_size': 2, 'global_batch_size': 32,
            'accumulation': 32, 'train_file': project/core.DATA/'ni.parquet'}


def build_composite(project, root, arm):
    """Use the frozen training constructor and stop before optimizer construction."""
    import torch
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import get_peft_model_state_dict
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer
    from new_interest.training import train_sft
    from new_interest.training.sft_state_audit import capture_adapter_state, tensor_state_manifest
    from new_interest.training.semantic_id_tokens import validate_saved_interest_tokens, capture_reload_reference
    from new_interest.training.checkpoint_persistence import _copy_file_bytes
    contract = verify_bundle(project)
    source = Path('/mnt/default')/contract['source_results_prefix']/'formal/cir5/final_adapter'
    cached = root/'sources/cir5/final_adapter'
    for name, item in contract['cir5_files'].items():
        target = cached/name
        if target.exists():
            core.require(target.stat().st_size == item['size'] and core.sha256(target) == item['sha256'], 'Cached CIR5 changed')
        else:
            _copy_file_bytes(source/name, target)
            core.require(target.stat().st_size == item['size'] and core.sha256(target) == item['sha256'], 'Copied CIR5 bytes differ')
    folder = root/'formal/composite';adapter = folder/'final_adapter'
    ready = folder/'composition.json'
    if ready.exists():
        receipt = core.read(ready)
        core.require(receipt['arm'] == arm and receipt['adapter_sha256'] == core.sha256(adapter/'adapter_model.safetensors'), 'Composite changed')
        return
    cir = load_file(str(cached/'adapter_model.safetensors'), device='cpu')
    verify_tensor_manifest(cir, core.read(project/'manifests/transfer_sources/cir5_state.json'))
    expected_cold = core.read(project/'manifests/transfer_sources/cold_state.json')
    config = core.read(project/'configs/sft/mind_cir_base_v1.json')
    config.update(train_file=str(project/core.DATA/'ni.parquet'), output_dir=str(root/'cold_constructor'),
                  checkpoint_dir=str(root/'cold_constructor/unused'), run_name='transfer-zero-update-constructor',
                  wandb_mode='disabled')
    config.pop('adapter_path', None)
    captured = []
    class CapturedBeforeOptimization(BaseException):
        pass
    def capture(model, output_dir, checkpoint_identity, *, input_adapter=None):
        core.require(input_adapter is None, 'Cold construction is warm-started')
        cold = get_peft_model_state_dict(model, save_embedding_layers=False)
        verify_tensor_manifest(cold, expected_cold)
        composite, provenance = compose_states(cold, cir, arm)
        expected = tensor_state_manifest(composite)
        core.require(len(composite) == 373 and list(composite[SID_KEY].shape) == [108, 2048], 'Composite dimensions differ')
        set_peft_model_state_dict(model, loading_copy(composite), adapter_name='default')
        actual = capture_adapter_state(model)
        core.require(actual == expected, 'In-memory model differs from composed parameters')
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        core.require(len(trainable) == 373 and sum('trainable_tokens_delta' in n for n in trainable) == 1
                     and sum('lora_' in n for n in trainable) == 372, 'Both parameter groups must remain trainable')
        adapter.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
        for name in contract['cir5_files']:
            if name not in {'adapter_model.safetensors', 'adapter_config.json', 'README.md', 'training_args.bin'}:
                _copy_file_bytes(cached/name, adapter/name)
        verify_tensor_manifest(load_file(str(adapter/'adapter_model.safetensors')), expected)
        tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        spec = validate_saved_interest_tokens(tokenizer, project/core.DATA/'interest_token_mapping.parquet', expected_count=108)
        model.to(torch.device('cuda', 0))
        reference = capture_reload_reference(model, tokenizer, project/core.DATA/'ni.parquet', spec,
                                            enable_thinking=False, sample_count=4, initialization_mode='mean_norm_matched')
        core.write(folder/'semantic_reload_reference.json', reference)
        core.write(folder/'expected_state.json', expected)
        receipt = {'arm': arm, 'cold_state_sha256': expected_cold['state_sha256'],
                   'cir5_state_sha256': contract['cir5_state_sha256'], 'source_adapter_sha256': contract['cir5_adapter_sha256'],
                   'adapter_sha256': core.sha256(adapter/'adapter_model.safetensors'), 'state_sha256': expected['state_sha256'],
                   'tensor_count': 373, 'trainable_lora_tensors': 372, 'trainable_sid_tensors': 1,
                   'optimization_steps': 0, 'provenance': {k: {'source': provenance[k], **expected['tensors'][k]} for k in composite}}
        core.write(ready, receipt)
        captured.append(True)
        raise CapturedBeforeOptimization()
    original = train_sft.record_initial_adapter_state
    train_sft.record_initial_adapter_state = capture
    try:
        train_sft.train(config)
        raise RuntimeError('Cold capture failed to stop before optimization')
    except CapturedBeforeOptimization:
        core.require(captured == [True], 'Initialization capture was not completed')
    finally:
        train_sft.record_initial_adapter_state = original
    print('CIR_TRANSFER_COMPOSED', arm, core.read(ready)['state_sha256'], flush=True)


def evaluate_dataset(project, root, adapter, eval_file, *, normal_dev=False):
    from new_interest.evaluation.merge_id_token_shards import merge_id_token_shards
    from new_interest.evaluation.evaluate_beam import sha256_directory
    folder = root/'real_eval';expected_n = pq.ParquetFile(eval_file).metadata.num_rows
    config = core.read(project/'configs/id_token/mind_cir_v1.json')
    config.update(eval_file=str(eval_file), adapter_path=str(adapter), output_dir=str(folder), batch_size=2,
                  enable_thinking=False, top_k=20, cutoffs=[1,5,10,20], checkpoint_size=256)
    ready=root/'evaluation_readback.json'
    if ready.exists():
        previous=core.read(ready)
        metrics=core.read(folder/'merged/metrics.json')
        core.require(metrics['eval_sha256']==core.sha256(eval_file) and metrics['adapter_sha256']==sha256_directory(adapter), 'Completed evaluation identity differs')
        _, shared=core.bind();shared.DEV=eval_file
        verified=shared.audit_evaluation(project,'2b',root,'full' if normal_dev else 'diagnostic',logical_shards=4)
        core.require(verified==previous, 'Completed evaluation receipt differs')
        return verified
    core.write(root/'evaluation_config.json', config)
    def one(i):
        core.run(core.invocation('infer', '--', '--config', root/'evaluation_config.json', '--output-dir', folder/f'shard-{i:02d}',
                                 '--num-shards', '4', '--shard-index', str(i), '--batch-size', '2'),
                 project, root/'logs'/f'eval-{i}.log', str(i))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, range(4)))
    merge_id_token_shards([folder/f'shard-{i:02d}' for i in range(4)], eval_file,
                         project/core.DATA/'interest_token_mapping.parquet', folder/'merged', expected_records=expected_n)
    _, shared = core.bind();shared.DEV = eval_file
    result = shared.audit_evaluation(project, '2b', root, 'full' if normal_dev else 'diagnostic', logical_shards=4)
    core.require(core.read(folder/'merged/metrics.json')['adapter_sha256'] == sha256_directory(adapter), 'Evaluation adapter differs')
    core.write(root/'evaluation_readback.json', result)
    return result


def diagnostics(project, root, adapter):
    result = {}
    for key in ('sid_name','heldout_cir'):
        result[key] = evaluate_dataset(project, root/key, adapter, project/'data/diagnostics/cir_parameter_transfer_v1'/f'{key}.parquet')
    core.write(root/'complete.json', result)
    return result


def paired_readback(project, root, contract):
    """Recompute matched target rank changes remotely; full predictions stay remote."""
    source = Path('/mnt/default')/contract['source_results_prefix']/contract['reference_prediction_relative_path']
    core.require(core.sha256(source)==contract['reference_prediction_sha256'],'Reference predictions changed')
    before=pq.read_table(source).to_pylist()
    after=pq.read_table(root/'formal/real_eval/merged/id_token_predictions.parquet').to_pylist()
    records=pq.read_table(project/core.DATA/'dev_full_grouped.parquet').to_pylist()
    core.require(len(before)==len(after)==len(records)==31392,'Paired comparison coverage changed')
    result=paired_statistics(before,after,records)
    result.update(reference_prediction_sha256=contract['reference_prediction_sha256'],new_prediction_sha256=core.sha256(root/'formal/real_eval/merged/id_token_predictions.parquet'))
    core.write(root/'paired_vs_full_inheritance.json',result)
    return result


def paired_statistics(before,after,records):
    core.require(len(before)==len(after)==len(records)>0,'Paired coverage differs')
    changes={str(k):{'recall':Counter(),'ndcg':Counter()} for k in (1,5,10,20)}
    support=Counter();direction=Counter()
    for i,(a,b,r) in enumerate(zip(before,after,records)):
        core.require(a['record_index']==b['record_index']==i and a['input_sha256']==b['input_sha256'],'Paired input identity differs')
        for target in r['acceptable_subcategories']:
            ar=a['ranked_labels'].index(target)+1 if target in a['ranked_labels'] else 21
            br=b['ranked_labels'].index(target)+1 if target in b['ranked_labels'] else 21
            support[target]+=1;direction['better' if br<ar else 'worse' if br>ar else 'tied_at_top20_resolution']+=1
            for k in (1,5,10,20):
                changes[str(k)]['recall'][target]+=int(br<=k)-int(ar<=k)
                changes[str(k)]['ndcg'][target]+=(1/math.log2(br+1) if br<=k else 0)-(1/math.log2(ar+1) if ar<=k else 0)
    per_class={k:{metric:{c:value[c]/support[c] for c in support} for metric,value in measures.items()} for k,measures in changes.items()}
    result={'reference':'accepted CIR5->NI3','records':len(records),'targets':sum(support.values()),'supported_classes':len(support),
            'top20_rank_direction_counts':dict(direction),'per_class_delta':per_class,
            'macro_delta':{k:{metric:math.fsum(values.values())/len(support) for metric,values in measures.items()} for k,measures in per_class.items()},
            }
    return result


def audit_final_checkpoint(project, root):
    import torch
    from new_interest.training.checkpoint_persistence import validate_persistent_checkpoint
    from new_interest.training.old_new_execution import make_cursor_contract, STAGES
    cp=root/'formal/checkpoints-ni3/checkpoint-2025'
    marker=core.read(cp/'checkpoint_complete.json')
    validate_persistent_checkpoint(cp,marker['identity'])
    core.require(marker['global_step']==2025 and marker['identity']['epochs']==3,'Final checkpoint duration differs')
    optimizer=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)
    scheduler=torch.load(cp/'scheduler.pt',map_location='cpu',weights_only=False)
    steps=[int(v['step']) for v in optimizer['state'].values()]
    core.require(len(steps)==373 and set(steps)=={2025} and scheduler['last_epoch']==2025,'Final optimizer/scheduler differs')
    core.require(all(torch.isfinite(t).all().item() for v in optimizer['state'].values() for t in v.values() if torch.is_tensor(t)),'Nonfinite final optimizer')
    core.require(marker['files']['adapter_model.safetensors']['sha256']==core.sha256(root/'formal/ni3/final_adapter/adapter_model.safetensors'),'Final adapter is not final checkpoint')
    STAGES.add('cir/ni3')
    cursor=make_cursor_contract(project/core.DATA/'ni.parquet',stage='cir/ni3',world_size=4,per_device_batch_size=2,source_checkpoint=cp,epochs=3,global_batch_size=32)
    counts=core.read(cp/'training_exposure_state.json')['counts']
    core.require(cursor['consumed_total']==64800 and len(counts)==21600 and all(x==3 for x in counts),'Final sample exposure differs')
    result={'status':'accepted','global_step':2025,'optimizer_states':373,'optimizer_step_min':min(steps),'optimizer_step_max':max(steps),
            'scheduler_last_epoch':scheduler['last_epoch'],'presentations':64800,'every_sample_presentations':3,
            'checkpoint_marker_sha256':core.sha256(cp/'checkpoint_complete.json'),'adapter_sha256':marker['files']['adapter_model.safetensors']['sha256'],
            'files':marker['files'],'full_remote_byte_readback':True,'large_payloads':'remote_only'}
    core.write(root/'final_checkpoint_readback.json',result)
    return result


def worker(project, root, arm):
    import torch
    contract = verify_bundle(project);runtime = core.read(project/core.CONTRACT)
    identity = {'version':contract['version'], 'arm':arm,'transfer_bundle_sha256':core.sha256(project/'manifests/transfer_bundle.json'),
                'source_bundle_sha256':contract['source_bundle_sha256']}
    if (root/'identity.json').exists():
        core.require(core.read(root/'identity.json')==identity,'Output identity changed')
    else:core.write(root/'identity.json',identity)
    versions={k:importlib.metadata.version(k) for k in runtime['worker_packages']}
    core.require(versions==runtime['worker_packages'],'Worker packages changed')
    core.require(torch.cuda.device_count()==4 and all('A100' in torch.cuda.get_device_name(i) and torch.cuda.get_device_properties(i).total_memory>=79*1024**3 for i in range(4)), 'Expected four A10080GB')
    core.write(root/'environment.json', {'packages':versions,'python':platform.python_version(),'cuda':torch.version.cuda,'devices':[torch.cuda.get_device_name(i) for i in range(4)]})
    _,shared=core.bind();shared.prepare_snapshot(project,'2b',root/'snapshot_audit.json')
    core.require(core.read(root/'snapshot_audit.json')['files']==core.read(project/'manifests/h50_snapshot.json')['files'],'Base snapshot changed')
    core.write(root/'runtime_gates_reused.json',core.reuse_verified_gates(project,Path(runtime['reuse_verified_gates'])))
    core.run([sys.executable,'-m','new_interest.experiments.cir_parameter_transfer','compose','--project',str(project),'--root',str(root),'--arm',arm],project,root/'logs/composition.log','0')
    comp=root/'formal/composite';adapter=comp/'final_adapter'
    core.run(core.invocation('verify','--','--model-name',runtime['model']['model_name'],'--model-revision',runtime['model']['model_revision'],
                '--adapter-path',adapter,'--mapping-file',project/core.DATA/'interest_token_mapping.parquet','--reference-file',comp/'semantic_reload_reference.json',
                '--output',comp/'semantic_adapter_reload_audit.json','--atol','0','--rtol','0'),project,root/'logs/composite-reload.log','0')
    reload=core.read(comp/'semantic_adapter_reload_audit.json')
    core.require(reload['all_close'] and reload['adapter_state_integrity']['all_tensors_exact'],'Composite reload failed')
    pre=diagnostics(project,root/'diagnostics/pre',adapter)
    print('CIR_TRANSFER_FORMAL_START',arm,flush=True)
    stage=core.train_stage(project,root/'formal',stage_spec(project))
    initial=core.read(root/'formal/ni3/initial_adapter_state.json');composition=core.read(comp/'composition.json')
    core.require(initial['state_sha256']==composition['state_sha256'],'NI initial state differs from composed adapter')
    final_checkpoint=audit_final_checkpoint(project,root)
    result=evaluate_dataset(project,root/'formal',root/'formal/ni3/final_adapter',project/core.DATA/'dev_full_grouped.parquet',normal_dev=True)
    paired=paired_readback(project,root,contract)
    post=diagnostics(project,root/'diagnostics/post',root/'formal/ni3/final_adapter')
    references={}
    if arm=='keep_lora':
        source=Path('/mnt/default')/contract['source_results_prefix']
        from new_interest.evaluation.evaluate_beam import sha256_directory
        core.require(core.sha256(source/'formal/ni3/final_adapter/adapter_model.safetensors')==contract['reference_ni3_adapter_sha256'],'Full reference NI adapter changed')
        core.require(sha256_directory(source/'formal/ni3/final_adapter')==contract['reference_ni3_adapter_directory_sha256'],'Full reference NI directory changed')
        references['pre']=diagnostics(project,root/'diagnostics/reference_full/pre',root/'sources/cir5/final_adapter')
        references['post']=diagnostics(project,root/'diagnostics/reference_full/post',source/'formal/ni3/final_adapter')
    core.write(root/'complete.json',{**identity,'status':'accepted','stage':stage,'composition':composition,
               'final_checkpoint':final_checkpoint,'evaluation':result,'paired':paired,'diagnostics':{'pre':pre,'post':post,'reference_full':references},'large_payloads':'remote_only'})
    print('CIR_TRANSFER_ACCEPTED',arm,json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['worker','compose']);p.add_argument('--project',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=ARMS,required=True);a=p.parse_args()
    project=a.project.resolve();root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    core.reference_kernels()
    if a.command=='compose':build_composite(project,root,a.arm)
    else:worker(project,root,a.arm)


if __name__=='__main__':main()
