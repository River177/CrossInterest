"""One authorized experiment pipeline: gate readback -> parallel formal jobs -> readback.

This is a finite local supervisor, not a recurring Codex automation. It never
cancels jobs or expands the predeclared two-arm budget. Scientific failures stop
for agent diagnosis; status/download transport failures remain retryable.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

import yaml


REPO=Path(__file__).resolve().parents[2]
AMLT='/Users/river/miniforge3/envs/amlt/bin/amlt'
PYTHON='/tmp/crossinterest-sid-relation-env/bin/python'


def read(p):return json.loads(Path(p).read_text())


def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');q.replace(p)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def sanitize(text):
    text=re.sub(r'\x1b\[[0-9;]*m','',text)
    text=re.sub(r'(?i)bearer\s+[A-Za-z0-9._~-]+','Bearer [REDACTED]',text)
    text=re.sub(r'https?://[^\s]+\?[^\s]*(?:sig=|token=)[^\s]*','[SIGNED_URL_REDACTED]',text)
    return text


def command(args,log,timeout=180,env=None):
    proc=subprocess.Popen(args,cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          text=True,start_new_session=True,env=env)
    try:out,_=proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGTERM)
        try:out,_=proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);out,_=proc.communicate()
        Path(log).write_text(sanitize(out));raise TimeoutError('Command transport timeout')
    Path(log).parent.mkdir(parents=True,exist_ok=True);Path(log).write_text(sanitize(out))
    if proc.returncode:raise RuntimeError(f'Command returned{proc.returncode}; see {log}')
    return out


def statuses(exp,jobs,log):
    text=command([AMLT,'status',exp,'--hide-urls'],log)
    return parse_status_table(text,jobs)


def parse_status_table(text,jobs):
    result={}
    for job in jobs:
        lines=[line for line in sanitize(text).splitlines() if re.match(r'^\s*(?::\d+\s+)?:?'+re.escape(job)+r'\s',line)]
        if len(lines)!=1:raise RuntimeError('Status response has missing/ambiguous job row')
        match=re.search(r'\b(queued|preparing|running|pass|failed|killed|canceled|completed)\b',lines[0],re.I)
        if not match:raise RuntimeError('Status response did not contain the expected job')
        result[job]=match[1].lower()
    return result


def download_and_verify(exp,job,bundle,root,label):
    directory=root/'download'/label;directory.mkdir(parents=True,exist_ok=True)
    try:
        command([AMLT,'results','download',exp,':'+job,'--include','*.json','--include','*.safetensors',
            '--include','*.md','--include','*.parquet','--output',str(directory),'--n-workers','4'],root/f'download-{label}.log',timeout=900)
    except RuntimeError as error:raise TimeoutError('Retryable artifact download failure: '+str(error)) from error
    candidates=list(directory.rglob('worker_success.json'))
    if len(candidates)!=1:raise ValueError('Worker success receipt missing or ambiguous')
    worker_root=candidates[0].parent
    receipt=root/f'{label}_acceptance.json'
    # Independent reader improvements do not change or re-run the frozen GPU runtime.
    env={**os.environ,'PYTHONPATH':str(REPO/'src')}
    try:
        command([PYTHON,'-m','new_interest.experiments.read_sid_relation_bias','--project',str(bundle),
            '--root',str(worker_root),'--output',str(receipt)],root/f'verify-{label}.log',timeout=900,env=env)
    except RuntimeError as error:raise ValueError('Independent artifact acceptance failed: '+str(error)) from error
    if not read(receipt)['passed']:raise ValueError('Independent reader rejected artifacts')
    return worker_root,receipt


def formal_bundle(gate,acceptance,root,gate_config):
    dest=root/'bundle-full-v1'
    source=read(gate/'source_manifest.json')
    accepted=read(acceptance)
    if not accepted.get('passed') or accepted.get('phase')!='gate' or accepted.get('condition')!='relation':
        raise ValueError('Expected an accepted relation gate, not another phase or condition')
    if accepted.get('runtime_sha256')!=source['runtime_sha256'] or accepted.get('protocol_sha256')!=source['files']['protocol.json']:
        raise ValueError('Gate acceptance runtime/protocol differs from frozen source')
    if not dest.exists():dest.mkdir()
    for name,expected in source['files'].items():
        inp=gate/name;out=dest/name
        if sha(inp)!=expected:raise ValueError('Frozen gate source changed')
        out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():
            if sha(out)!=expected:raise ValueError('Formal staging conflicts with frozen source')
        else:shutil.copyfile(inp,out)
    shutil.copyfile(acceptance,dest/'gate_acceptance.json')
    files={**source['files'],'gate_acceptance.json':sha(dest/'gate_acceptance.json')}
    write(dest/'source_manifest.json',{**source,'files':files,'source_file_count':len(files),
        'source_bytes':sum((dest/k).stat().st_size for k in files)})
    cfg=yaml.safe_load(gate_config.read_text())
    cfg['description']='Authorized SID-relation pilot: baseline and shared-SID relation, each4A100 in parallel; same Mixed parent,21600 NI rows,675 updates,full native SFT,FP32 full108 ranking.'
    cfg['code']['local_dir']='$CONFIG_DIR/../../artifacts/sid-relation-bias-20260915/bundle-full-v1'
    template=cfg['jobs'][0]
    jobs=[]
    for condition in ['baseline','relation']:
        item=json.loads(json.dumps(template));item['name']='qwen35-2b-sid-'+condition
        item['command']=['set -euo pipefail','cd $$AMLT_CODE_DIR',
                         f'bash scripts/amlt/sft/run_sid_relation_bias.sh full {condition}']
        jobs.append(item)
    cfg['jobs']=jobs
    path=REPO/'configs/amlt/mind_sid_relation_full_0915_v1.yaml'
    path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    return dest,path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true')
    args=parser.parse_args();root=REPO/'artifacts/sid-relation-bias-20260915'
    lock=(root/'pipeline.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('The authorized pipeline supervisor is already running.');return
    state_path=root/'pipeline_state.json'
    state=read(state_path) if state_path.exists() else {'phase':'gate_wait',
        'gate_experiment':'mind-sid-relation-gate-v1-0915','gate_job':'qwen35-2b-sid-relation-gate',
        'formal_experiment':'mind-sid-relation-full-v1-0915','formal_submission_attempted':False,
        'maximum_formal_concurrent_gpus':8,'authorized_formal_updates_per_arm':675}
    gate=Path(state.get('gate_bundle',str(root/'bundle-gate-v1')))
    gate_config=Path(state.get('gate_config',str(REPO/'configs/amlt/mind_sid_relation_gate_0915_v1.yaml')))
    write(root/'pipeline_process.json',{'pid':os.getpid(),'started_unix':time.time(),'command':str(Path(__file__))})
    while state['phase'] not in ['complete','needs_attention']:
        try:
            if state['phase']=='gate_wait':
                job=state['gate_job'];observed=statuses(state['gate_experiment'],[job],root/'gate_status_current.txt')
                state['gate_status']=observed[job]
                if observed[job] in ['pass','completed','failed','killed','canceled']:
                    state['phase']='gate_readback';write(state_path,state)
            if state['phase']=='gate_readback':
                try:worker_root,receipt=download_and_verify(state['gate_experiment'],state['gate_job'],gate,root,'gate')
                except TimeoutError:raise
                except Exception as error:
                    # Preserve output for diagnosis; no blind gate reruns or expanded training.
                    state['phase']='needs_attention';state['reason']='Gate download/acceptance failed: '+str(error);write(state_path,state);break
                state.update(gate_worker_root=str(worker_root),gate_acceptance=str(receipt),phase='formal_prepare')
                write(state_path,state)
            if state['phase']=='formal_prepare':
                full,config=formal_bundle(gate,Path(state['gate_acceptance']),root,gate_config)
                command([AMLT,'run',str(config),state['formal_experiment'],'--dump'],root/'formal_dump.yaml')
                state.update(formal_bundle=str(full),formal_config=str(config),phase='formal_submit')
                write(state_path,state)
            if state['phase']=='formal_submit':
                if state['formal_submission_attempted']:
                    state.update(phase='needs_attention',reason='Ambiguous prior submission; inspect named experiment before retrying')
                    write(state_path,state);break
                state['formal_submission_attempted']=True;write(state_path,state)
                command([AMLT,'run',state['formal_config'],state['formal_experiment'],'-d',
                    'Independently accepted SID-relation gate. Authorized baseline and relation continuation in parallel4GPU each,675 native SFT updates each,complete validation and Dev readback.','-y'],root/'submit_formal_v1.log',timeout=900)
                state['phase']='formal_wait';write(state_path,state)
            if state['phase']=='formal_wait':
                jobs=['qwen35-2b-sid-baseline','qwen35-2b-sid-relation']
                observed=statuses(state['formal_experiment'],jobs,root/'formal_status_current.txt');state['formal_status']=observed
                if all(v in ['pass','completed','failed','killed','canceled'] for v in observed.values()):
                    state['phase']='formal_readback';write(state_path,state)
            if state['phase']=='formal_readback':
                for condition in ['baseline','relation']:
                    worker_root,receipt=download_and_verify(state['formal_experiment'],'qwen35-2b-sid-'+condition,
                        Path(state['formal_bundle']),root,condition)
                    state[condition+'_worker_root']=str(worker_root);state[condition+'_acceptance']=str(receipt)
                    write(state_path,state)
                state['phase']='comparison';write(state_path,state)
            if state['phase']=='comparison':
                try:
                    command([PYTHON,str(REPO/'scripts/analysis/collect_sid_relation_comparison.py'),'--root',str(root)],
                        root/'comparison.log',timeout=900)
                except RuntimeError as error:raise ValueError('Paired analysis failed: '+str(error)) from error
                state['phase']='complete';state['completion_scope']='Both formal arms independently accepted; comparison.json contains paired uncertainty and prespecified screen.'
            state.pop('last_transport_error',None)
        except (TimeoutError,RuntimeError) as error:
            state['last_transport_error']=str(error)
            if state['phase']=='formal_submit':
                state['phase']='needs_attention';state['reason']='Submission outcome ambiguous; never retry automatically'
        except Exception as error:
            state['phase']='needs_attention';state['reason']=type(error).__name__+': '+str(error)
        state['observed_unix']=time.time();write(state_path,state)
        print(json.dumps({'phase':state['phase'],'gate':state.get('gate_status'),
            'formal':state.get('formal_status'),'error':state.get('reason',state.get('last_transport_error'))}),flush=True)
        if args.once:break
        if state['phase'] not in ['complete','needs_attention']:time.sleep(60)


if __name__=='__main__':main()
