"""Freeze remote original-H5 acceptance and package it for persistent storage."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile
import importlib.metadata
import os
import platform

from new_interest.experiments.history_length_ablation import read, write, require, sha256

task = Path('/task')
root = task / 'publication-h5-original-profile-v1'
code = task / 'code'
complete = read(root / 'complete.json')
training = read(root / 'original_training_accepted.json')
require(complete['evaluation']['records'] == 31392 and training['status'] == 'passed', 'Acceptance incomplete')
frozen_manifest = read(code / 'manifests/history_ablation_bundle.json')
frozen_sources = {p:h for p,h in frozen_manifest['files'].items() if p.startswith('src/')}
for relative, expected in frozen_sources.items():
    require(sha256(code / relative) == expected, 'Original source changed: ' + relative)
require(read(task / 'source-training/snapshot_audit.json')['files'] == read(code / 'manifests/h50_snapshot.json')['files'], 'Original base snapshot differs')
for expected_path, expected in read(task / 'outputs/source_hashes.json').items():
    if '/hf-cache-accepted-v1/' in expected_path:
        actual_path = Path('/hf') / expected_path.split('/hf-cache-accepted-v1/')[1]
    else:
        actual_path = task / expected_path.split('/history-h5-repair-20260916/')[1]
    require(sha256(actual_path) == expected, 'Actual original source bytes changed')
for source, destination in [('src','runtime/src'),('configs','runtime/configs'),('data','runtime/data'),
                            ('manifests','runtime/manifests'),('migration_kernel','runtime/migration_kernel')]:
    shutil.copytree(code / source, root / destination, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ['evaluate_history_h5_profile.py','audit_history_h5_training.py','verify_history_h5_profile.py',
             'package_history_h5_publication.py','requirements-backbone-qwen35-worker.txt']:
    shutil.copyfile(code / name, root / 'runtime' / name)
shutil.copytree(task / 'source-training', root / 'original_training_evidence', dirs_exist_ok=True)
for name in ['profile_red','profile_green1','profile_green2']:
    shutil.copytree(task / 'outputs' / name, root / 'diagnostic_evidence' / name, dirs_exist_ok=True)
shutil.copyfile(task / 'outputs/source_hashes.json', root / 'diagnostic_evidence/source_hashes.json')
for name in ['norm_profile_w4.json', 'norm_profile_w16.json']:
    shutil.copyfile(task / 'source' / name, root / 'diagnostic_evidence' / name)
import torch
write(root / 'runtime_environment.json', {
    'python':platform.python_version(), 'gpu':torch.cuda.get_device_name(0),
    'cuda_capability':list(torch.cuda.get_device_capability(0)),
    'packages':{name:importlib.metadata.version(name) for name in
        ['torch','transformers','peft','accelerate','triton','flash-linear-attention','fla-core',
         'causal-conv1d','pyarrow','tokenizers','safetensors','huggingface-hub']},
    'environment':{k:os.environ.get(k) for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE',
        'CUBLAS_WORKSPACE_CONFIG','PYTORCH_CUDA_ALLOC_CONF','TOKENIZERS_PARALLELISM','OMP_NUM_THREADS']},
})
write(root / 'runtime_identity.json', {'frozen_original_bundle_sha256':sha256(code / 'manifests/history_ablation_bundle.json'),
    'frozen_source_files_checked':len(frozen_sources), 'frozen_source_files':frozen_sources,
    'new_profile_module_sha256':sha256(code / 'src/new_interest/training/fla_norm_profile.py'),
    'original_evaluator_and_merger_unchanged':True})
write(root / 'restoration.json', {
    'source_results_prefix':'projects/sft4safety/amlt-results/7210551062.55495-3b46c238-52c4',
    'publication_prefix':'projects/sft4safety/manual-results/history-h5-original-profile-20260916-v1',
    'model_revision':'15852e8c16360a2fea060d615a32b45270f8a8fc',
    'restore':'Extract publication.tar.gz remotely. Use runtime/src and the pinned packages, apply sft/final_adapter/fla_norm_profile.json before loading the original adapter. See runtime/evaluate_history_h5_profile.py.',
    'base_weights_and_optimizer':'Remain in original persistent model cache and original AMLT checkpoint artifacts; not duplicated in this evaluation publication',
    'large_artifacts_policy':'All weights and full predictions remain remote; local Mac receives compact JSON only',
})
files = {str(p.relative_to(root)):{'sha256':sha256(p),'bytes':p.stat().st_size}
    for p in root.rglob('*') if p.is_file() and p.name not in {'artifact_manifest.json','complete.json'}
    and not p.name.endswith('.partial')}
write(root / 'artifact_manifest.json', {'files':files})
for name, item in files.items():
    require(sha256(root / name) == item['sha256'], 'Final readback mismatch')
complete.update(status='original_h5_training_reload_full_dev_and_manifest_verified',training=training,
    completed_at_utc=datetime.now(timezone.utc).isoformat(), artifact_manifest_sha256=sha256(root / 'artifact_manifest.json'))
write(root / 'complete.json', complete)
archive = task / 'publication-h5-original-profile-v1.tar.gz'
require(not archive.exists(), 'Do not replace an existing publication archive')
with tarfile.open(archive, 'w:gz') as stream:
    stream.add(root, arcname='publication')
inventory = {'publication.tar.gz':{'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha256(archive)}}
for name in ['complete.json','artifact_manifest.json','evaluation_readback.json','cohort_metrics.json','original_training_accepted.json',
             'restoration.json','runtime_identity.json','sft/final_adapter/fla_norm_profile.json','sft/semantic_adapter_reload_audit.json','real_eval/merged/metrics.json']:
    p = root / name
    inventory[name] = {'path':str(p),'bytes':p.stat().st_size,'sha256':sha256(p)}
write(task / 'h5_transfer_inventory.json', {'files':inventory})
print('H5_PUBLICATION_PACKAGED',len(files),archive.stat().st_size,flush=True)
