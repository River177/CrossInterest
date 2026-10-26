"""Package the approved three-condition experiment with two sequential new runs."""
from pathlib import Path
import json
import shutil

import pyarrow.parquet as pq
import yaml
from transformers import AutoTokenizer

from new_interest.training.semantic_id_tokens import register_semantic_interest_tokens
from new_interest.training.oi_ni_full_sft import encode_completions
from new_interest.experiments.oi_ni_meta_distributed import digest_json
from new_interest.data.build_personal_cir_support import file_hash


def main():
    root = Path.cwd()
    source = root / 'artifacts/mind-cir-parameter-20260923/bundle-v2'
    output = root / 'artifacts/personal-cir-meta-20260924'
    bundle = output / 'bundle-v1'
    if bundle.exists():
        raise FileExistsError(bundle)
    parent = json.loads((source / 'manifests/transfer_bundle.json').read_text())
    for relative, expected in parent['files'].items():
        if file_hash(source / relative) != expected:
            raise ValueError('Original accepted runtime changed: ' + relative)
    shutil.copytree(source, bundle, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.amltignore'))
    additions = [
        'src/new_interest/data/build_personal_cir_support.py',
        'src/new_interest/data/build_oi_ni_full_sft.py',
        'src/new_interest/data/build_oi_ni_meta_dataset.py',
        'src/new_interest/training/personal_cir_meta.py',
        'src/new_interest/experiments/personal_cir_meta.py',
        'src/new_interest/experiments/personal_cir_workflow.py',
        'scripts/amlt/sft/run_personal_cir_meta.sh',
        'tests/test_personal_cir_meta.py', 'tests/test_build_personal_cir_support.py',
    ]
    additions += [str(p.relative_to(root)) for folder in ['training', 'experiments']
                  for p in sorted((root / 'src/new_interest' / folder).glob('oi_ni*.py'))]
    for relative in additions:
        p = root / relative
        if not p.exists():
            raise FileNotFoundError(p)
        if relative in parent['files'] and file_hash(p) != parent['files'][relative]:
            raise ValueError('Would overwrite accepted runtime: ' + relative)
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    data_dir = 'data/training/personal_cir_meta_20260924_v1'
    shutil.copytree(root / data_dir, bundle / data_dir)
    original = json.loads((source / 'configs/experiments/mind_cir_108_v1.json').read_text())
    config = dict(original['model'])
    config.update(version='personal-cir-meta-v1', data_dir=data_dir,
                  data_metadata_sha256=file_hash(bundle / data_dir / 'metadata.json'),
                  hf_cache_dir='/mnt/default/huggingface-cache-mind-backbone-0911/2b/hub',
                  seed=42, steps=2025, epochs=3, global_batch_size=32,
                  lora_r=16, lora_alpha=32, lora_dropout=.05,
                  inner_alpha=.01, outer_learning_rate=2e-5, weight_decay=.01,
                  warmup_ratio=.03, max_grad_norm=1., max_seq_length=2048,
                  modes=['cir5', 'cold'], test_time_adaptation=False,
                  objective='mean_user NI(phi-alpha*grad_LoRA CIR_user, SID)',
                  worker_packages=original['worker_packages'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'], revision=config['model_revision'])
    register_semantic_interest_tokens(tokenizer, bundle / data_dir / 'interest_token_mapping.parquet')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    qrows = pq.read_table(bundle / data_dir / 'ni.parquet').to_pylist()
    frows = pq.read_table(bundle / data_dir / 'supports.parquet').to_pylist()
    q = encode_completions(tokenizer, qrows, 2048)
    f = encode_completions(tokenizer, frows, 2048)
    config['query_encoding_sha256'] = digest_json(q)
    config['support_encoding_sha256'] = digest_json(f)
    lengths = [len(x['input_ids']) for x in q]
    selected = [min(range(len(q)), key=lengths.__getitem__), max(range(len(q)), key=lengths.__getitem__), 0, 1]
    config['probe_indices'] = list(dict.fromkeys(selected))
    path = bundle / 'configs/experiments/personal_cir_meta_v1.json'
    path.write_text(json.dumps(config, indent=2) + '\n')
    shutil.copyfile(path, root / 'configs/experiments/personal_cir_meta_v1.json')
    for relative, expected in parent['files'].items():
        if file_hash(bundle / relative) != expected:
            raise ValueError('Original parent payload changed: ' + relative)
    files = {str(p.relative_to(bundle)): file_hash(p) for p in sorted(bundle.rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    manifest = {'version': config['version'], 'parent_bundle_sha256': file_hash(source / 'manifests/transfer_bundle.json'), 'files': files}
    (bundle / 'personal_cir_bundle_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    cfg = yaml.safe_load((root / 'configs/amlt/mind_cir_parameter_transfer_0923_v2.yaml').read_text())
    cfg['description'] = 'Approved three-condition NI experiment. Reuse CIR5->NI3; train CIR5 and cold per-user exact-meta NI3 serially on four A100s. Deploy shared outer weights; no test-time adaptation.'
    cfg['code']['local_dir'] = '$CONFIG_DIR/../../artifacts/personal-cir-meta-20260924/bundle-v1'
    job = cfg['jobs'][0]
    job['name'] = 'qwen35-2b-personal-cir-meta-three'
    job['submit_args']['env'].pop('CIR_TRANSFER_ARM', None)
    job['submit_args']['env'].update(AMLT_SKIP_GIT_INFO='1', AMLT_CODE_CHECKSUM_SECS='0')
    job['command'][-1] = 'bash scripts/amlt/sft/run_personal_cir_meta.sh'
    cfg['jobs'] = [job]
    yaml_path = root / 'configs/amlt/personal_cir_meta_0924_v1.yaml'
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    tracking = {'status': 'prepared_not_submitted', 'experiment': 'personal-cir-meta-0924-v1',
        'job': job['name'], 'config': str(yaml_path.relative_to(root)),
        'bundle': str(bundle.relative_to(root)), 'bundle_sha256': file_hash(bundle / 'personal_cir_bundle_manifest.json'),
        'conditions': ['cir5_ni3', 'cir5_meta_ni3', 'cold_meta_ni3'],
        'new_training_branches': 2, 'world_size': 4, 'max_gpus': 4,
        'test_time_adaptation': False, 'automatic_monitoring': False,
        'query_max_tokens': max(lengths), 'support_max_tokens': max(len(x['input_ids']) for x in f),
        'original_payload_files_preserved': len(parent['files']), 'runtime_payload_files': len(files)}
    (root / 'manifests/personal_cir_meta_20260924_v1.json').write_text(json.dumps(tracking, indent=2) + '\n')
    (output / 'local_preflight.json').write_text(json.dumps(tracking, indent=2) + '\n')
    print(json.dumps(tracking, indent=2))


if __name__ == '__main__':
    main()
