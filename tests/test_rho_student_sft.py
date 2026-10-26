from contextlib import contextmanager
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from new_interest.training.rho_student_sft import (
    IndexedCollator, IndexedDataset, ROW_INDEX, canonical, expected_rank_order,
    raw_model_context, sha, train_student, validate_weight_rows,
)
from new_interest.training.rho_dynamic_weights import build_classwise_weights, source_identity


def test_metadata_index_collator_never_bakes_in_parent_weights():
    torch = pytest.importorskip('torch')
    from new_interest.training.sft_data import CompletionOnlyCollator
    samples = [{'input_ids': [1, 2], 'attention_mask': [1, 1], 'labels': [-100, 2]},
               {'input_ids': [1, 3, 2], 'attention_mask': [1, 1, 1], 'labels': [-100, -100, 2]}]
    dataset = IndexedDataset(samples)
    collator = IndexedCollator(CompletionOnlyCollator(SimpleNamespace(pad_token_id=0)))
    batch = collator([dataset[1], dataset[0]])
    assert batch[ROW_INDEX].tolist() == [1, 0]
    assert 'sample_weight' not in batch and ROW_INDEX not in samples[0]
    assert torch.equal(batch['labels'], torch.tensor([[-100, -100, 2], [-100, 2, -100]]))


def test_raw_context_restores_exact_forward_wrapper_even_on_failure():
    class Model:
        def forward(self):
            return 'raw'
    model = Model()
    raw = model.forward
    wrapped = lambda: 'wrapped'
    model.forward, model._original_forward = wrapped, raw
    trainer = SimpleNamespace(model_wrapped=model, accelerator=SimpleNamespace(unwrap_model=lambda *a, **k: model))
    with pytest.raises(RuntimeError):
        with raw_model_context(trainer) as selected:
            assert selected.forward() == 'raw' and not hasattr(selected, '_original_forward')
            raise RuntimeError('score failure')
    assert model.forward is wrapped and model._original_forward is raw


def test_expected_native_sampler_partitions_complete_epoch_without_duplicates():
    orders = [expected_rank_order(64, 42, 1, rank, 4) for rank in range(4)]
    assert len(set(sum(orders, []))) == 64
    assert set(sum(orders, [])) == set(range(64))
    assert orders[0] != expected_rank_order(64, 42, 0, 0, 4)


def prepare_cpu_fixture(tmp_path, monkeypatch, *, arm='dynamic', lora_dropout=0.):
    torch = pytest.importorskip('torch')
    pytest.importorskip('peft')
    import importlib
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig, PreTrainedTokenizerFast
    from new_interest.evaluation import score_rho_reference, score_rho_student

    class CpuKernelAudit:
        def __init__(self):
            self.snapshots = []
        def capture(self, stage):
            self.snapshots.append({'stage': stage, 'kernels': {'fla.cpu.fixture': {'cache': {'shape': 'fixed'}}}})
        def verify_coverage(self):
            return None
        def receipt(self, evaluation_identity=None):
            return {**score_rho_reference.kernel_selection_identity('stable_fla_v1'),
                    'evaluation_identity': evaluation_identity, 'snapshots': self.snapshots,
                    'snapshots_sha256': canonical(self.snapshots)}

    @contextmanager
    def cpu_kernel_context(*args, **kwargs):
        yield CpuKernelAudit()

    monkeypatch.setattr(score_rho_reference, 'kernel_selection_context', cpu_kernel_context)
    monkeypatch.setattr(score_rho_student, 'kernel_selection_context', cpu_kernel_context)
    model_root = tmp_path / 'base'
    torch.manual_seed(42)
    Qwen3_5ForCausalLM(Qwen3_5TextConfig(
        vocab_size=7, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        layer_types=['full_attention'], tie_word_embeddings=True, attention_dropout=0,
        use_cache=False,
    )).save_pretrained(model_root)
    backend = Tokenizer(models.WordLevel({name: i for i, name in enumerate([
        '<unk>', '<pad>', '<eos>', '<end>', 'hello', 'movie', 'sport',
    ])}, unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>', pad_token='<pad>', eos_token='<eos>')
    tokenizer.chat_template = '{{ messages[0]["content"] }}{% if messages|length > 1 %} {{ messages[1]["content"] }} <eos> <end>{% endif %}'
    tokenizer.save_pretrained(model_root)
    mapping = tmp_path / 'mapping.parquet'
    pq.write_table(pa.Table.from_pylist([
        {'interest_index': i, 'interest_token': f'<interest_{i:03d}>', 'target_subcategory': name}
        for i, name in enumerate(['movie', 'sport'])]), mapping)
    rows = []
    for i in range(32):
        target = 'movie' if i % 2 == 0 else 'sport'
        prompt = 'hello ' + target
        response = f'<interest_{i%2:03d}>'
        rows.append({'source_sample_id': f's{i}', 'user_id': f'u{i}', 'target_subcategory': target,
            'prompt': prompt, 'response': response,
            'messages': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}]})
    train = tmp_path / 'train.parquet'
    pq.write_table(pa.Table.from_pylist(rows), train)
    refs = [{**{key: row[key] for key in ('source_sample_id', 'user_id', 'target_subcategory')},
             'input_sha256': source_identity(row), 'full_vocab_first_sid_nll': 1.+i*.05}
            for i, row in enumerate(rows)]
    reference = tmp_path / 'reference.parquet'
    pq.write_table(pa.Table.from_pylist(refs), reference)
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps({'version': 'mind-rho-student-v1',
                                    'training_numeric_policy': 'torch-deterministic-algorithms-v1'}))
    transformers = importlib.import_module('transformers')
    original_arguments = transformers.TrainingArguments
    def cpu_arguments(**kwargs):
        kwargs.update(use_cpu=True, bf16=False, tf32=False, optim='adamw_torch',
                      dataloader_pin_memory=False, disable_tqdm=True)
        return original_arguments(**kwargs)
    monkeypatch.setattr(transformers, 'TrainingArguments', cpu_arguments)
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    config = {'model_name': str(model_root), 'model_revision': None, 'train_file': str(train),
        'semantic_id_token_mapping_file': str(mapping), 'semantic_id_token_count': 2,
        'run_name': 'cpu-rho-student', 'epochs': 3, 'per_device_train_batch_size': 1,
        'gradient_accumulation_steps': 8, 'gradient_checkpointing_use_reentrant': False,
        'lora_r': 2, 'lora_alpha': 4, 'lora_dropout': lora_dropout, 'target_modules': 'all-linear',
        'learning_rate': .001, 'warmup_ratio': 0, 'wandb_mode': 'disabled', 'logging_steps': 1,
        'save_strategy': 'steps', 'save_steps': 2, 'save_total_limit': 20, 'persistent_checkpoint_limit': 20,
        'dataloader_num_workers': 0, 'max_seq_length': 64, 'seed': 42,
        'rho_arm': arm, 'rho_protocol_file': str(protocol), 'rho_reference_file': str(reference),
        'rho_reference_file_sha256': sha(reference), 'rho_epoch_weight_dir': str(tmp_path / 'epoch-persistent'),
        'rho_expected_classes': 2, 'rho_expected_rows_per_class': 16, 'rho_score_batch_size': 4,
        'rho_stable_training': True, 'rho_require_native_parity': True,
        'rho_training_numeric_policy': 'torch-deterministic-algorithms-v1'}
    return config, rows


def run_cpu(config, tmp_path, monkeypatch, name, resume=None):
    config = {**config, 'output_dir': str(tmp_path / name), 'checkpoint_dir': str(tmp_path / name / 'checkpoints')}
    if resume:
        config['resume_from_checkpoint'] = str(resume)
    monkeypatch.setenv('RHO_SYNC_DIR', str(tmp_path / (name + '-sync')))
    train_student(config)
    return json.loads((tmp_path / name / 'rho_student_summary.json').read_text())


def test_actual_native_training_step_ga8_and_epoch_refresh_keep_optimizer(tmp_path, monkeypatch):
    config, _ = prepare_cpu_fixture(tmp_path, monkeypatch)
    summary = run_cpu(config, tmp_path, monkeypatch, 'fresh')
    assert summary['optimizer_steps'] == 12 and summary['refresh_steps'] == [0, 4, 8]
    assert summary['optimizer_reinitialized_at_refresh'] is False
    assert summary['training_deterministic_algorithms'] is True
    assert summary['training_deterministic_warn_only'] is False
    assert len(summary['epoch_receipts']) == 3
    assert all(row['all_gradients_exact'] for row in summary['native_gradient_parity'][0]['comparisons'])
    assert [row['scale'] for row in summary['native_gradient_parity'][0]['comparisons']] == [1., 1.5]
    assert summary['native_gradient_parity'][0]['num_items_in_batch'] == 24
    assert all(row['consumed_rows'] == 32 for row in summary['rank_order_history'][0])
    for step, expected_epoch in [(2, 0), (6, 1), (10, 2)]:
        checkpoint = tmp_path / 'fresh/checkpoints' / f'checkpoint-{step}'
        state = json.loads((checkpoint / 'rho_epoch_state.json').read_text())
        assert state['active_receipt']['epoch'] == expected_epoch
        assert sha(checkpoint / 'rho_epoch_weights.parquet') == state['active_receipt']['files']['weights.parquet']


def test_nonzero_epoch_midpoint_resume_reuses_exact_epoch_weights_and_native_order(tmp_path, monkeypatch):
    from new_interest.evaluation import score_rho_student
    from safetensors.torch import load_file
    import torch
    config, _ = prepare_cpu_fixture(tmp_path, monkeypatch)
    original = run_cpu(config, tmp_path, monkeypatch, 'fresh')
    checkpoint = tmp_path / 'fresh/checkpoints/checkpoint-6'
    weights_before = sha(checkpoint / 'rho_epoch_weights.parquet')
    def forbidden_rescore(*args, **kwargs):
        pytest.fail('Existing epoch snapshots must be reused on midpoint resume')
    monkeypatch.setattr(score_rho_student, 'score_current_student', forbidden_rescore)
    resumed = run_cpu(config, tmp_path, monkeypatch, 'resumed', checkpoint)
    assert resumed['epoch_receipts'] == original['epoch_receipts']
    assert resumed['rank_order_history'] == original['rank_order_history']
    assert sha(checkpoint / 'rho_epoch_weights.parquet') == weights_before
    from new_interest.training.rho_student_sft import load_epoch_files
    source = pq.read_table(config['train_file']).to_pylist()
    for epoch in range(3):
        load_epoch_files(tmp_path / 'resumed/rho_epochs' / f'epoch-{epoch:04d}', source,
                         resumed['training_identity'], expected_epoch=epoch)
    first = load_file(tmp_path / 'fresh/final_adapter/adapter_model.safetensors')
    second = load_file(tmp_path / 'resumed/final_adapter/adapter_model.safetensors')
    assert set(first) == set(second) and all(torch.equal(first[k], second[k]) for k in first)


def test_mid_epoch_resume_without_epoch_sidecar_fails_before_updates(tmp_path, monkeypatch):
    config, _ = prepare_cpu_fixture(tmp_path, monkeypatch)
    run_cpu(config, tmp_path, monkeypatch, 'fresh')
    checkpoint = tmp_path / 'fresh/checkpoints/checkpoint-6'
    (checkpoint / 'rho_epoch_state.json').unlink()
    with pytest.raises(FileNotFoundError):
        run_cpu(config, tmp_path, monkeypatch, 'rejected', checkpoint)
    assert not (tmp_path / 'rejected/final_adapter').exists()


def test_uniform_student_matches_actual_native_trainer_ga8_updates_bit_exactly(tmp_path, monkeypatch):
    """This crosses real training_step/GA, unlike compute_loss-only parity."""
    import torch
    from safetensors.torch import load_file
    from new_interest.training import train_sft
    config, _ = prepare_cpu_fixture(tmp_path, monkeypatch, arm='uniform')
    student = run_cpu(config, tmp_path, monkeypatch, 'student')
    baseline = {**config, 'output_dir': str(tmp_path / 'native'),
                'checkpoint_dir': str(tmp_path / 'native/checkpoints')}
    train_sft.train(baseline)
    ours = load_file(tmp_path / 'student/final_adapter/adapter_model.safetensors')
    native = load_file(tmp_path / 'native/final_adapter/adapter_model.safetensors')
    assert student['optimizer_steps'] == 12
    assert set(ours) == set(native) and all(torch.equal(ours[name], native[name]) for name in ours)


def test_epoch_manifest_and_current_score_snapshot_corruption_controls(tmp_path, monkeypatch):
    from new_interest.training.rho_student_sft import load_epoch_files, validate_current_shard
    config, rows = prepare_cpu_fixture(tmp_path, monkeypatch)
    summary = run_cpu(config, tmp_path, monkeypatch, 'fresh')
    folder = tmp_path / 'fresh/rho_epochs/epoch-0000'
    receipt = json.loads((folder / 'receipt.json').read_text())
    metadata = json.loads((folder / 'current-rank-0.json').read_text())
    scores = pq.read_table(folder / 'current-rank-0.parquet').to_pylist()
    args = (rows, list(range(len(rows))), receipt['student_state'], 0, 1)
    validate_current_shard(metadata, scores, *args, {})
    bad_scores = copy.deepcopy(scores)
    bad_scores[0]['evaluation_identity'] = 'foreign'
    with pytest.raises(ValueError, match='identity'):
        validate_current_shard(metadata, bad_scores, *args, {})
    bad_meta = copy.deepcopy(metadata)
    bad_meta['state_audit']['gradients_unchanged'] = False
    with pytest.raises(ValueError, match='student state'):
        validate_current_shard(bad_meta, scores, *args, {})
    with pytest.raises(ValueError, match='different configs'):
        validate_current_shard(metadata, scores, *args, {('fla.cpu.fixture', 'shape'): 'foreign'})
    bad_receipt = copy.deepcopy(receipt)
    del bad_receipt['files']['weights.parquet']
    (folder / 'receipt.json').write_text(json.dumps(bad_receipt))
    with pytest.raises(ValueError, match='authenticate'):
        load_epoch_files(folder, rows, summary['training_identity'], expected_epoch=0)


def test_midpoint_resume_rejects_stale_but_self_consistent_order_prefix(tmp_path, monkeypatch):
    config, _ = prepare_cpu_fixture(tmp_path, monkeypatch)
    run_cpu(config, tmp_path, monkeypatch, 'fresh')
    checkpoint = tmp_path / 'fresh/checkpoints/checkpoint-6'
    file = checkpoint / 'rho_order_rank-0.json'
    order = json.loads(file.read_text())
    order['order_audit']['consumed_rows'] = 8
    order['order_audit']['consumed_order_sha256'] = canonical(expected_rank_order(32, 42, 1, 0, 1)[:8])
    file.write_text(json.dumps(order))
    with pytest.raises(ValueError, match='source-order'):
        run_cpu(config, tmp_path, monkeypatch, 'rejected', checkpoint)
    assert not (tmp_path / 'rejected/final_adapter').exists()


def test_gradient_diagnostic_distinguishes_nonfinite_missing_and_numeric_errors():
    torch = pytest.importorskip('torch')
    from new_interest.training.rho_student_sft import gradient_comparison
    def trial(value):
        return {'loss': 1., 'gradients': {'lora_B': torch.tensor(value)}}
    exact = gradient_comparison(trial([1., 2.]), trial([1., 2.]))
    assert exact['all_gradients_exact'] and exact['global_gradient_relative_l2'] == 0.
    jitter = gradient_comparison(trial([1., 2.]), trial([1., 2.01]))
    assert not jitter['all_gradients_exact'] and jitter['all_gradients_finite']
    assert jitter['loss_absolute_difference'] == 0.
    assert jitter['max_gradient_absolute_difference'] > 0.
    assert jitter['families']['lora_B']['mismatched_tensors'] == 1
    for invalid in (float('nan'), float('inf')):
        bad = gradient_comparison(trial([invalid]), trial([invalid]))
        assert not bad['all_gradients_exact'] and not bad['all_gradients_finite']
        assert bad['global_gradient_relative_l2'] is None
        assert bad['per_parameter']['lora_B']['actual_nonfinite_values'] == 1
        json.dumps(bad, allow_nan=False)
    empty = gradient_comparison({'loss': 1., 'gradients': {}}, {'loss': 1., 'gradients': {}})
    assert not empty['all_gradients_exact']
    missing = gradient_comparison(trial([1.]), {'loss': 1., 'gradients': {}})
    assert missing['missing'] == ['lora_B'] and not missing['all_gradients_exact']
    empty_tensor = gradient_comparison(trial([]), trial([]))
    assert empty_tensor['all_gradients_exact']
    enormous = {'loss': 1., 'gradients': {'lora_B': torch.tensor([1e300], dtype=torch.float64)}}
    large = gradient_comparison(enormous, enormous)
    assert large['all_gradients_exact'] and large['global_reference_gradient_l2'] is None
    json.dumps(large, allow_nan=False)


def test_complete_rng_snapshot_restores_python_numpy_and_torch():
    torch = pytest.importorskip('torch')
    import numpy as np
    import random
    from new_interest.training.rho_student_sft import capture_rng_state, restore_rng_state, rng_receipt
    start = capture_rng_state()
    try:
        expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
        restore_rng_state(start)
        assert rng_receipt(capture_rng_state()) == rng_receipt(start)
        assert (random.random(), float(np.random.rand()), float(torch.rand(()))) == expected
    finally:
        restore_rng_state(start)


def test_diagnostic_only_requires_a_fresh_native_callback():
    from new_interest.training.rho_student_sft import validate_config
    with pytest.raises(ValueError, match='requires native parity'):
        validate_config({'rho_arm': 'uniform', 'rho_native_diagnostic_only': True})
    with pytest.raises(ValueError, match='fresh zero-update'):
        validate_config({'rho_arm': 'uniform', 'rho_native_diagnostic_only': True,
                         'rho_require_native_parity': True, 'resume_from_checkpoint': 'auto'})


def diagnostic_config(config, tmp_path, monkeypatch, *, diagnostic_only):
    from new_interest.evaluation import score_rho_student
    from new_interest.training import train_sft
    import torch
    def forbidden(*args, **kwargs):
        pytest.fail('Zero-step diagnostic must not score epochs, update parameters, or finalize an adapter')
    monkeypatch.setattr(torch.optim.AdamW, 'step', forbidden)
    monkeypatch.setattr(score_rho_student, 'score_current_student', forbidden)
    monkeypatch.setattr(train_sft, 'capture_reload_reference', forbidden)
    monkeypatch.setenv('RHO_SYNC_DIR', str(tmp_path / 'diagnostic-sync'))
    return {**config, 'output_dir': str(tmp_path / 'diagnostic'),
            'checkpoint_dir': str(tmp_path / 'diagnostic/checkpoints'),
            'rho_native_diagnostic_only': diagnostic_only,
            'rho_diagnostic_output': str(tmp_path / 'persistent/native_diagnostic.json')}


def test_actual_hf_diagnostic_stops_at_zero_updates_and_restores_dropout_rng(tmp_path, monkeypatch, capsys):
    config, _ = prepare_cpu_fixture(tmp_path, monkeypatch, lora_dropout=.05)
    cfg = diagnostic_config(config, tmp_path, monkeypatch, diagnostic_only=True)
    result = train_student(cfg)
    report = json.loads(Path(cfg['rho_diagnostic_output']).read_text())
    assert result['diagnostic_complete'] and result['optimizer_steps'] == 0
    assert result['optimizer_state_entries'] == 0
    assert report['diagnostic_complete'] and report['original_guard_passed']
    assert report['all_model_parameters_unchanged']
    assert report['production_gradients_complete_and_finite']
    assert report['global_step'] == report['global_step_after'] == 0
    assert report['optimizer_state_entries'] == report['optimizer_state_entries_after'] == 0
    assert all(report['restored_state'].values())
    assert report['warm_native_reproducible']
    assert all(item['all_gradients_exact'] for item in report['controls'].values())
    assert len(report['trials']) == 14
    assert all(item['upstream_scalar_gradient'] == [item['scale']] for item in report['trials'])
    assert all(item['gradients_complete_and_finite'] for item in report['trials'])
    assert any(item['p'] == .05 for item in report['dropout_modules'])
    warm = {item['label']: item for item in report['trials'] if item['label'].startswith('warm_')}
    assert warm['warm_native_1']['dropout_probes'] == warm['warm_weighted_1']['dropout_probes']
    assert warm['warm_native_1']['rng_after_backward'] == warm['warm_weighted_1']['rng_after_backward']
    assert sha(cfg['rho_diagnostic_output']) == sha(tmp_path / 'diagnostic/rank-0-native-parity.json')
    assert 'RHO_REFERENCE_TRAIN_BEGIN' in capsys.readouterr().out
    assert not (tmp_path / 'diagnostic/final_adapter').exists()
    assert not (tmp_path / 'diagnostic/rho_student_summary.json').exists()


def test_deterministic_training_policy_reaches_real_hf_parity_without_changing_precision(tmp_path, monkeypatch):
    import torch
    from new_interest.training.semantic_id_tokens import capture_torch_numeric_backend
    config, _ = prepare_cpu_fixture(tmp_path, monkeypatch, lora_dropout=.05)
    config['rho_training_numeric_policy'] = 'torch-deterministic-algorithms-v1'
    Path(config['rho_protocol_file']).write_text(json.dumps({
        'version': 'mind-rho-student-v1', 'training_numeric_policy': 'torch-deterministic-algorithms-v1'}))
    cfg = diagnostic_config(config, tmp_path, monkeypatch, diagnostic_only=True)
    original_enabled = torch.are_deterministic_algorithms_enabled()
    original_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(False, warn_only=True)
        before = capture_torch_numeric_backend(torch)
        train_student(cfg)
        report = json.loads(Path(cfg['rho_diagnostic_output']).read_text())
        assert report['numeric_backend_before'] == {
            **before, 'deterministic_algorithms': True, 'deterministic_warn_only': False}
        production = [item for item in report['trials'] if item['label'].startswith(('original_', 'warm_'))]
        assert all(item['numeric_backend'] == report['numeric_backend_before'] for item in production)
        assert report['original_guard_passed'] and report['production_controls_exact']
        assert report['training_identity']['rho_training_numeric_policy'] == 'torch-deterministic-algorithms-v1'
        assert capture_torch_numeric_backend(torch) == before
    finally:
        torch.use_deterministic_algorithms(original_enabled, warn_only=original_warn)


@pytest.mark.parametrize('diagnostic_only', [False, True])
@pytest.mark.parametrize('perturb_call', [2, 7])
def test_parity_failure_is_persisted_and_never_trains(tmp_path, monkeypatch, diagnostic_only, perturb_call):
    from transformers.trainer import Trainer
    original_loss = Trainer.compute_loss
    calls = 0
    def perturb_second_loss(self, *args, **kwargs):
        nonlocal calls
        loss = original_loss(self, *args, **kwargs)
        calls += 1
        return loss + .01 if calls == perturb_call else loss
    monkeypatch.setattr(Trainer, 'compute_loss', perturb_second_loss)
    config, _ = prepare_cpu_fixture(tmp_path, monkeypatch, lora_dropout=.05)
    cfg = diagnostic_config(config, tmp_path, monkeypatch, diagnostic_only=diagnostic_only)
    if diagnostic_only:
        result = train_student(cfg)
        assert result['diagnostic_complete']
        assert result['original_guard_passed'] is (perturb_call == 7)
    else:
        with pytest.raises(ValueError, match='gradient parity failed'):
            train_student(cfg)
    report = json.loads(Path(cfg['rho_diagnostic_output']).read_text())
    assert report['diagnostic_complete']
    assert report['original_guard_passed'] is (perturb_call == 7)
    failure = report['comparisons'][0] if perturb_call == 2 else report['controls']['warm_native_vs_weighted']
    assert not failure['all_gradients_exact']
    assert failure['loss_absolute_difference'] > 0.
    assert failure['max_gradient_absolute_difference'] == 0.
    assert report['warm_native_reproducible'] and report['upstream_scalars_exact']
    assert all(report['restored_state'].values())
    assert not (tmp_path / 'diagnostic/final_adapter').exists()


def test_matching_but_incomplete_gradient_sets_fail_the_native_gate(tmp_path, monkeypatch):
    import torch
    from transformers.trainer import Trainer
    original_init, original_backward = Trainer.__init__, torch.Tensor.backward
    models = []
    def remember_model(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        models.append(self.model)
    def remove_one_gradient(self, *args, **kwargs):
        result = original_backward(self, *args, **kwargs)
        next(p for p in models[0].parameters() if p.requires_grad and p.grad is not None).grad = None
        return result
    monkeypatch.setattr(Trainer, '__init__', remember_model)
    monkeypatch.setattr(torch.Tensor, 'backward', remove_one_gradient)
    config, _ = prepare_cpu_fixture(tmp_path, monkeypatch)
    cfg = diagnostic_config(config, tmp_path, monkeypatch, diagnostic_only=False)
    with pytest.raises(ValueError, match='gradient parity failed'):
        train_student(cfg)
    report = json.loads(Path(cfg['rho_diagnostic_output']).read_text())
    assert all(item['all_gradients_exact'] for item in report['comparisons'])
    assert not report['original_guard_passed']
    assert not report['production_gradients_complete_and_finite']
    assert all(len(item['missing_gradients']) == 1 for item in report['trials'])
    assert not (tmp_path / 'diagnostic/final_adapter').exists()


def test_kernel_receipt_failure_is_durable_and_does_not_abort_remaining_trials(tmp_path, monkeypatch):
    from new_interest.evaluation import score_rho_reference
    config, _ = prepare_cpu_fixture(tmp_path, monkeypatch)
    original_context = score_rho_reference.kernel_selection_context
    @contextmanager
    def bad_first_receipt(*args, **kwargs):
        with original_context(*args, **kwargs) as audit:
            original = audit.receipt
            def receipt(*args, **kwargs):
                if len(audit.snapshots) == 1:
                    raise ValueError('injected missing kernel receipt')
                return original(*args, **kwargs)
            audit.receipt = receipt
            yield audit
    monkeypatch.setattr(score_rho_reference, 'kernel_selection_context', bad_first_receipt)
    cfg = diagnostic_config(config, tmp_path, monkeypatch, diagnostic_only=False)
    with pytest.raises(ValueError, match='gradient parity failed'):
        train_student(cfg)
    report = json.loads(Path(cfg['rho_diagnostic_output']).read_text())
    assert report['diagnostic_complete'] and len(report['trials']) == 8
    assert 'injected missing kernel receipt' in report['trials'][0]['kernel_receipt_error']
    assert not report['production_kernel_receipts_complete']
    assert report['original_guard_passed']
