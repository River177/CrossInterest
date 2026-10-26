"""Freeze only the SID-relation runtime and its already-audited inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gate-acceptance',type=Path);args=p.parse_args()
    dest=args.output
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True)
    original=Path('artifacts/mind-short-think-sft-20260913/bundle-full-v1')
    files=['src/new_interest/__init__.py','src/new_interest/training/__init__.py',
        'src/new_interest/data/__init__.py','src/new_interest/experiments/__init__.py',
        'src/new_interest/training/sft_data.py','src/new_interest/training/semantic_id_tokens.py',
        'src/new_interest/training/sid_relation_attention.py','src/new_interest/data/build_sid_relation_sidecar.py',
        'src/new_interest/experiments/sid_relation_bias.py','src/new_interest/experiments/read_sid_relation_bias.py',
        'tests/test_sid_relation_attention.py','scripts/amlt/sft/run_sid_relation_bias.sh']
    for name in files:
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(name,target)
    for name in ['scripts/amlt/common/install_qwen35_fast_kernels.sh',
                 'scripts/amlt/common/install_qwen35_fast_kernels.py',
                 'configs/experiments/qwen35_fast_kernel_install_v1.json','source_snapshot.json']:
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original/name,target)
    shutil.copyfile(original/'requirements-short-think-worker.txt',dest/'requirements-worker.txt')
    shutil.copyfile('configs/experiments/mind_sid_relation_bias_pilot_v1.json',dest/'protocol.json')
    shutil.copytree('artifacts/sid-relation-bias-20260915/inputs-v1',dest/'inputs')
    shutil.copytree(original/'inputs/mixed_adapter',dest/'inputs/mixed_adapter')
    shutil.copyfile('data/training/mind_oi_ni_full_sft_v1/interest_token_mapping.parquet',dest/'inputs/interest_token_mapping.parquet')
    if args.gate_acceptance:shutil.copyfile(args.gate_acceptance,dest/'gate_acceptance.json')
    inventory={str(q.relative_to(dest)):sha(q) for q in sorted(dest.rglob('*')) if q.is_file()}
    runtime={k:v for k,v in inventory.items() if k.startswith(('src/','scripts/','tests/','configs/')) or k=='requirements-worker.txt'}
    runtime_sha=hashlib.sha256(json.dumps(runtime,sort_keys=True).encode()).hexdigest()
    if args.gate_acceptance:
        accepted=json.loads(args.gate_acceptance.read_text())
        if not accepted['passed'] or accepted['runtime_sha256']!=runtime_sha:raise ValueError('Source changed since accepted gate')
    result={'files':inventory,'runtime_sha256':runtime_sha,'source_file_count':len(inventory),
        'source_bytes':sum((dest/k).stat().st_size for k in inventory)}
    (dest/'source_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))


if __name__=='__main__':main()
