"""Native completion SFT with checkpoint-bound, epoch-refreshed sample weights."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from new_interest.training.rho_reference_sft import (
    LocalCheckpointCoordinator, TrainingRuntimeAudit, adapter_initial_state,
)
from new_interest.training.rho_dynamic_weights import build_classwise_weights, source_identity


VERSION = 'rho-native-sft-epoch-weights-v1'
ROW_INDEX = '_rho_row_index'
DETERMINISTIC_POLICY = 'torch-deterministic-algorithms-v1'


class NativeParityDiagnosticComplete(Exception):
    """Exit a requested zero-update diagnostic before native SFT continues."""
    def __init__(self, receipt):
        super().__init__('Requested native parity diagnostic completed with zero optimizer updates')
        self.receipt = receipt


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def copy_bytes(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with source.open('rb') as inp, temp.open('wb') as out:
            shutil.copyfileobj(inp, out, 4 * 1024 * 1024)
        if sha(temp) != sha(source):
            raise ValueError('RHO artifact publication changed bytes')
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def validate_config(config):
    if config.get('rho_arm') not in ('uniform', 'static', 'dynamic'):
        raise ValueError('Unknown RHO student arm')
    if config.get('rho_native_diagnostic_only'):
        if config.get('rho_require_native_parity') is not True:
            raise ValueError('Diagnostic-only mode requires native parity before training')
        if config.get('resume_from_checkpoint') not in (None, False):
            raise ValueError('Diagnostic-only mode requires a fresh zero-update model')
    for key in ('adapter_path', 'sample_weight_file', 'mera_relations_file'):
        if config.get(key) is not None:
            raise ValueError('Fresh RHO student forbids ' + key)
    if (config.get('loss_reduction', 'token_mean') != 'token_mean'
            or config.get('semantic_id_loss_scope', 'full_vocabulary') != 'full_vocabulary'
            or config.get('enable_thinking', False)
            or config.get('per_device_train_batch_size') != 1
            or config.get('gradient_checkpointing_use_reentrant') is not False
            or config.get('rho_stable_training') is not True):
        raise ValueError('RHO requires native microbatch-one SFT and stable non-reentrant training')
    for key in ('rho_expected_classes', 'rho_expected_rows_per_class'):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError('Invalid class-balanced RHO data size')
    for key in ('rho_protocol_file', 'rho_reference_file', 'rho_epoch_weight_dir'):
        if not config.get(key):
            raise ValueError('Missing ' + key)
    if sha(config['rho_reference_file']) != config.get('rho_reference_file_sha256'):
        raise ValueError('Accepted reference bytes differ from the frozen input')
    protocol = read(config['rho_protocol_file'])
    if protocol.get('version') != 'mind-rho-student-v1':
        raise ValueError('Unexpected RHO student protocol')
    policy = config.get('rho_training_numeric_policy', 'native-v1')
    if policy not in ('native-v1', DETERMINISTIC_POLICY) or policy != protocol.get('training_numeric_policy', 'native-v1'):
        raise ValueError('RHO training numeric policy differs from its frozen protocol')


@contextmanager
def training_numeric_context(policy):
    """Control backward reproducibility without changing dtype, TF32 or SDPA selection."""
    import torch
    from new_interest.training.semantic_id_tokens import capture_torch_numeric_backend
    if policy not in ('native-v1', DETERMINISTIC_POLICY):
        raise ValueError('Unknown RHO training numeric policy')
    before = capture_torch_numeric_backend(torch)
    try:
        if policy == DETERMINISTIC_POLICY:
            torch.use_deterministic_algorithms(True, warn_only=False)
        active = capture_torch_numeric_backend(torch)
        expected = ({**before, 'deterministic_algorithms': True, 'deterministic_warn_only': False}
                    if policy == DETERMINISTIC_POLICY else before)
        if active != expected:
            raise ValueError('RHO deterministic policy changed an unrelated numerical setting')
        print('RHO_TRAIN_NUMERIC_POLICY ' + json.dumps({'policy': policy, 'backend': active}), flush=True)
        yield active
    finally:
        torch.use_deterministic_algorithms(before['deterministic_algorithms'], warn_only=before['deterministic_warn_only'])


class IndexedDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        result = dict(self.dataset[index])
        if ROW_INDEX in result or 'sample_weight' in result:
            raise ValueError('Unexpected existing sample metadata')
        result[ROW_INDEX] = int(index)
        return result


class IndexedCollator:
    def __init__(self, collator):
        self.collator = collator
        self.tokenizer = collator.tokenizer

    def __call__(self, features):
        import torch
        rows = [int(feature[ROW_INDEX]) for feature in features]
        batch = self.collator([{k: v for k, v in feature.items() if k != ROW_INDEX} for feature in features])
        batch[ROW_INDEX] = torch.tensor(rows, dtype=torch.long)
        return batch


@contextmanager
def raw_model_context(trainer):
    """Temporarily remove only Accelerate's forward wrapper, retaining DDP."""
    model = trainer.accelerator.unwrap_model(trainer.model_wrapped, keep_fp32_wrapper=True, keep_torch_compile=True)
    sentinel = object()
    original = getattr(model, '_original_forward', sentinel)
    forward = model.forward
    try:
        if original is not sentinel:
            model.forward = original
            delattr(model, '_original_forward')
        yield model
    finally:
        model.forward = forward
        if original is not sentinel:
            model._original_forward = original


@contextmanager
def preserve_rng():
    import numpy as np
    import torch
    python = random.getstate()
    numpy = np.random.get_state()
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def adapter_digest(model):
    import torch
    from peft import get_peft_model_state_dict
    values = {}
    for name, tensor in sorted(get_peft_model_state_dict(model, save_embedding_layers=False).items()):
        data = tensor.detach().contiguous().cpu()
        values[name] = {'shape': list(data.shape), 'dtype': str(data.dtype),
            'sha256': hashlib.sha256(data.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    return canonical(values)


def tensor_manifest(values):
    import torch
    records = {}
    for name, tensor in values:
        value = tensor.detach().contiguous().cpu()
        records[name] = {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    return {'state_sha256': canonical(records), 'tensors': records}


def capture_rng_state():
    import numpy as np
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'cpu': torch.get_rng_state().clone(),
            'cuda': [value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def restore_rng_state(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def rng_receipt(state):
    numpy = state['numpy']
    return {'python': canonical(state['python']),
            'numpy': canonical([numpy[0], numpy[1].tolist(), *numpy[2:]]),
            'torch_cpu': tensor_manifest([('rng', state['cpu'])])['state_sha256'],
            'torch_cuda': [tensor_manifest([('rng', value)])['state_sha256'] for value in state['cuda']]}


def gradient_comparison(reference, actual):
    """Compact exact comparison plus useful finite numerical diagnostics."""
    import torch
    if reference is None or actual is None:
        return {'all_gradients_exact': False, 'comparable': False, 'reason': 'a trial failed before gradient capture'}
    expected, observed = reference['gradients'], actual['gradients']
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    tensors = {}
    sum_delta, sum_expected, sum_actual = 0., 0., 0.
    finite_all = bool(expected and observed) and not missing and not extra
    exact_all = finite_all
    for name in sorted(set(expected) & set(observed)):
        left, right = expected[name], observed[name]
        shape_match = left.shape == right.shape and left.dtype == right.dtype
        finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        exact = bool(shape_match and finite and torch.equal(left, right))
        item = {'shape': list(left.shape), 'dtype': str(left.dtype), 'shape_dtype_match': shape_match,
                'actual_shape': list(right.shape), 'actual_dtype': str(right.dtype),
                'finite': finite, 'exact': exact,
                'reference_nonfinite_values': int((~torch.isfinite(left)).sum()),
                'actual_nonfinite_values': int((~torch.isfinite(right)).sum())}
        if finite and shape_match:
            a, b = left.double(), right.double()
            delta = a-b
            d2, a2, b2 = float(delta.square().sum()), float(a.square().sum()), float(b.square().sum())
            maximum = float(delta.abs().max()) if delta.numel() else 0.
            item.update(max_absolute_difference=maximum if math.isfinite(maximum) else None,
                        relative_l2=(math.sqrt(d2/a2) if a2 else (0. if d2 == 0 else None))
                            if math.isfinite(d2) and math.isfinite(a2) else None,
                        reference_l2=math.sqrt(a2) if math.isfinite(a2) else None,
                        actual_l2=math.sqrt(b2) if math.isfinite(b2) else None,
                        statistics_finite=all(math.isfinite(value) for value in (d2, a2, b2, maximum)),
                        changed_values=int(torch.count_nonzero(delta)))
            sum_delta += d2
            sum_expected += a2
            sum_actual += b2
        else:
            item.update(max_absolute_difference=None, relative_l2=None,
                        reference_l2=None, actual_l2=None, changed_values=None, statistics_finite=False)
        tensors[name] = item
        finite_all = finite_all and finite
        exact_all = exact_all and exact and finite
    loss_a, loss_b = reference['loss'], actual['loss']
    loss_finite = math.isfinite(loss_a) and math.isfinite(loss_b)
    family = {}
    for name, item in tensors.items():
        group = 'lora_A' if 'lora_A' in name else ('lora_B' if 'lora_B' in name else ('SID' if 'trainable_tokens_delta' in name else 'other'))
        values = family.setdefault(group, {'tensors': 0, 'mismatched_tensors': 0, 'nonfinite_tensors': 0,
                                         'nonzero_actual_tensors': 0, 'max_absolute_difference': 0.})
        values['tensors'] += 1
        values['mismatched_tensors'] += not item['exact']
        values['nonfinite_tensors'] += not item['finite']
        values['nonzero_actual_tensors'] += item['actual_l2'] not in (None, 0.)
        previous, current = values['max_absolute_difference'], item['max_absolute_difference']
        values['max_absolute_difference'] = max(previous, current) if previous is not None and current is not None else None
    metrics_available = (finite_all and all(x['shape_dtype_match'] and x['statistics_finite'] for x in tensors.values())
                         and all(math.isfinite(value) for value in (sum_delta, sum_expected, sum_actual)))
    return {'comparable': True, 'all_gradients_exact': bool(exact_all and loss_finite and loss_a == loss_b),
        'tensor_count': len(tensors), 'missing': missing, 'unexpected': extra, 'all_gradients_finite': finite_all,
        'reference_loss': loss_a if math.isfinite(loss_a) else None, 'actual_loss': loss_b if math.isfinite(loss_b) else None,
        'loss_absolute_difference': abs(loss_a-loss_b) if loss_finite else None,
        'global_gradient_relative_l2': (math.sqrt(sum_delta/sum_expected) if sum_expected else (0. if sum_delta == 0 else None)) if metrics_available else None,
        'global_reference_gradient_l2': math.sqrt(sum_expected) if metrics_available else None,
        'global_actual_gradient_l2': math.sqrt(sum_actual) if metrics_available else None,
        'max_gradient_absolute_difference': max((x['max_absolute_difference'] or 0. for x in tensors.values()), default=0.) if metrics_available else None,
        'families': family, 'per_parameter': tensors}


def expected_rank_order(rows, seed, epoch, rank, world_size):
    import torch
    if rows % world_size:
        raise ValueError('RHO source must divide evenly across ranks')
    generator = torch.Generator(device='cpu').manual_seed(int(seed) + int(epoch))
    return torch.randperm(rows, generator=generator).tolist()[rank::world_size]


def validate_weight_rows(rows, source):
    from collections import defaultdict
    if len(rows) != len(source):
        raise ValueError('Epoch weights have incomplete source coverage')
    groups = defaultdict(list)
    weights = []
    for record, original in zip(rows, source):
        for key in ('source_sample_id', 'user_id', 'target_subcategory'):
            if record[key] != original[key]:
                raise ValueError('Epoch weights changed source order or identity')
        if record['input_sha256'] != source_identity(original):
            raise ValueError('Epoch weights changed input identity')
        weight = float(record['sample_weight'])
        if not math.isfinite(weight) or not .5 - 1e-12 <= weight <= 1.5 + 1e-12:
            raise ValueError('Epoch weight is outside the positive frozen range')
        weights.append(weight)
        groups[record['target_subcategory']].append(weight)
    if any(not math.isclose(math.fsum(v), len(v), rel_tol=0, abs_tol=max(1e-12, len(v)*1e-14)) for v in groups.values()):
        raise ValueError('Epoch weights changed a class total')
    return weights


def load_epoch_files(folder, source, identity, *, expected_epoch=None):
    folder = Path(folder)
    receipt = read(folder / 'receipt.json')
    if receipt['training_identity'] != identity or (expected_epoch is not None and receipt['epoch'] != expected_epoch):
        raise ValueError('Epoch receipt identity differs from this student')
    world = receipt.get('world_size')
    if type(world) is not int or world < 1:
        raise ValueError('Epoch receipt misses its intended world size')
    required = {'weights.parquet', *[f'current-rank-{rank}.{suffix}' for rank in range(world) for suffix in ('json', 'parquet')]}
    if set(receipt.get('files', {})) != required:
        raise ValueError('Epoch receipt must authenticate weights and every current-score shard')
    for name, digest in receipt['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha(folder / relative) != digest:
            raise ValueError('Epoch artifact bytes differ')
    rows = pq.read_table(folder / 'weights.parquet').to_pylist()
    return receipt, validate_weight_rows(rows, source)


def validate_current_shard(metadata, rows, records, indices, student_state, rank, world, common_configs):
    from new_interest.evaluation.score_rho_reference import kernel_selection_identity, validate_rows
    identity = metadata['identity']
    if (identity.get('student_state') != student_state or identity.get('num_shards') != world
            or identity.get('full_record_count') != len(records)
            or identity.get('mapping_sha256') != student_state['training_identity']['mapping_sha256']
            or identity.get('ordered_source_sha256') != canonical([source_identity(row) for row in records])
            or canonical({k: v for k, v in identity.items() if k != 'evaluation_identity'}) != identity.get('evaluation_identity')):
        raise ValueError('Current-score metadata belongs to a different student/source snapshot')
    validate_rows(rows, [records[i] for i in indices], indices, identity)
    coverage = metadata['coverage']
    if (coverage.get('shard_index') != rank or coverage.get('num_shards') != world
            or coverage.get('scored_rows') != len(indices) or coverage.get('full_record_count') != len(records)
            or coverage.get('record_indices_sha256') != canonical(indices)
            or coverage.get('exact_modulo_coverage') is not True):
        raise ValueError('Current-score shard coverage receipt differs')
    audit = metadata['state_audit']
    if (audit.get('model_updates') != 0 or audit.get('adapter_state_before') != audit.get('adapter_state_after')
            or any(audit.get(k) is not True for k in ('parameters_unchanged', 'gradients_unchanged', 'training_flags_restored', 'rng_restored'))):
        raise ValueError('Current scoring changed the student state')
    kernel = metadata['kernel_receipt']
    for key, value in kernel_selection_identity('stable_fla_v1').items():
        if kernel.get(key) != value or identity.get(key) != value:
            raise ValueError('Current-score kernel source identity differs')
    if (kernel.get('evaluation_identity') != identity['evaluation_identity']
            or canonical(kernel.get('snapshots')) != kernel.get('snapshots_sha256')
            or not kernel.get('snapshots')):
        raise ValueError('Current-score kernel receipt is incomplete or belongs to another snapshot')
    seen = 0
    for snapshot in kernel['snapshots']:
        for name, settings in snapshot['kernels'].items():
            for cache_key, value in settings.get('cache', {}).items():
                seen += 1
                key = (name, cache_key)
                if key in common_configs and common_configs[key] != value:
                    raise ValueError('Current-scoring ranks selected different configs for the same kernel key')
                common_configs[key] = value
    if not seen:
        raise ValueError('Current-score kernel receipt has no selected configurations')


def hydrate_epoch(remote, local, source, identity, *, expected_epoch, expected_receipt=None):
    remote, local = Path(remote), Path(local)
    receipt, _ = load_epoch_files(remote, source, identity, expected_epoch=expected_epoch)
    if expected_receipt is not None and receipt != expected_receipt:
        raise ValueError('Checkpoint epoch receipt differs from immutable remote epoch')
    for name in receipt['files']:
        copy_bytes(remote / name, local / name)
    copy_bytes(remote / 'receipt.json', local / 'receipt.json')
    load_epoch_files(local, source, identity, expected_epoch=expected_epoch)
    return receipt


def student_trainer_class(base, callback_base, config, coordinator, training_identity):
    class StudentCallback(callback_base):
        def __init__(self, trainer):
            self.trainer = trainer

        def on_train_begin(self, args, state, control, **kwargs):
            t = self.trainer
            t.runtime.begin(state, kwargs['optimizer'], kwargs['lr_scheduler'], resumed=t.resume_path is not None)
            if config.get('rho_native_diagnostic_only') and t.resume_path is not None:
                raise ValueError('Diagnostic-only mode cannot enter a resumed training loop')
            if config.get('rho_require_native_parity') and t.resume_path is None:
                t.native_parity()
            return control

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            self.trainer.kernel_audit.verify_coverage()
            if state.global_step == 0:
                self.trainer.kernel_audit.capture('first-backward-before-update')
            return control

        def on_save(self, args, state, control, **kwargs):
            self.trainer.save_epoch_sidecar(Path(args.output_dir) / f'checkpoint-{state.global_step}')
            return control

        def on_train_end(self, args, state, control, **kwargs):
            t = self.trainer
            t.runtime.end(state, kwargs['optimizer'], kwargs['lr_scheduler'])
            t.kernel_audit.capture('training-final')
            write(Path(config['output_dir']) / f'rank-{t.rank}-rho-student-summary.json', {
                'version': VERSION, 'training_identity': training_identity,
                'epochs': t.history, 'rank': t.rank, 'order_audit': t.order_audit(),
                'order_history': t.order_history(),
                'actual_optimizer_steps': state.global_step, 'actual_epoch': state.epoch,
                'optimizer_reinitialized_at_refresh': False,
                'training_kernel_receipt': t.kernel_audit.receipt(canonical(training_identity))})
            return control

    class RhoStudentTrainer(base):
        def __init__(self, *args, **kwargs):
            import torch.distributed as dist
            original_dataset = kwargs['train_dataset']
            original_collator = kwargs['data_collator']
            self.tokenizer_for_rho = original_collator.tokenizer
            kwargs['train_dataset'] = IndexedDataset(original_dataset)
            kwargs['data_collator'] = IndexedCollator(original_collator)
            super().__init__(*args, **kwargs)
            self.rank = dist.get_rank() if dist.is_initialized() else 0
            self.world = dist.get_world_size() if dist.is_initialized() else 1
            self.records = pq.read_table(config['train_file']).slice(0, len(original_dataset)).to_pylist()
            self.references = pq.read_table(config['rho_reference_file']).to_pylist()
            if len(self.records) != config['rho_expected_classes'] * config['rho_expected_rows_per_class']:
                raise ValueError('Student source cardinality differs from protocol')
            build_classwise_weights(self.records, mode='uniform', reference_rows=self.references,
                expected_classes=config['rho_expected_classes'], expected_rows_per_class=config['rho_expected_rows_per_class'])
            if any(sum(x != -100 for x in example['labels'][1:]) != 3 for example in original_dataset.examples):
                raise ValueError('Expected the unchanged three-token completion supervision')
            self.updates_per_epoch = math.ceil(len(self.records) / (self.world * config['gradient_accumulation_steps']))
            initial = adapter_initial_state(self.model)
            if self.world > 1:
                hashes = [None] * self.world
                dist.all_gather_object(hashes, initial['state_sha256'])
                if len(set(hashes)) != 1:
                    raise ValueError('Student initial parameters differ across ranks')
            self.runtime = TrainingRuntimeAudit(config, initial, train_samples=len(self.records), rank=self.rank, world_size=self.world)
            self.weights = [1.] * len(self.records)
            self.epoch_receipt, self.history, self.active_epoch = None, [], None
            self.position = 0
            self.order = []
            self.completed_orders = {}
            self.resume_path = None
            self.parity_running = False
            self.kernel_audit = None
            self.add_callback(StudentCallback(self))

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if config.get('rho_training_numeric_policy') == DETERMINISTIC_POLICY:
                import torch
                if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
                    raise ValueError('RHO deterministic training policy was lost before a gradient update')
            clean = dict(inputs)
            index = clean.pop(ROW_INDEX, None)
            if index is None or index.numel() != 1 or clean['labels'].shape[0] != 1:
                raise ValueError('Native scalar sample weighting requires exactly one metadata-indexed sample')
            row = int(index.item())
            if not 0 <= row < len(self.weights):
                raise ValueError('Student row index is outside frozen source')
            if not self.parity_running:
                if self.active_epoch is None or self.position >= len(self.order) or row != self.order[self.position]:
                    raise ValueError('Actual dataloader order differs from the frozen native seedable sampler')
                self.position += 1
            weight = float(self.weights[row])
            result = super().compute_loss(model, clean, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
            upstream = getattr(self, '_parity_upstream', None)
            if self.parity_running and upstream is not None:
                raw_loss = result[0] if return_outputs else result
                raw_loss.register_hook(lambda value: upstream.append(float(value.detach().cpu())))
            if return_outputs:
                loss, outputs = result
                return loss * loss.new_tensor(weight), outputs
            return result * result.new_tensor(weight)

        def order_audit(self):
            return {'epoch': self.active_epoch, 'consumed_rows': self.position,
                    'expected_rows': len(self.order), 'consumed_order_sha256': canonical(self.order[:self.position])}

        def order_history(self):
            values = dict(self.completed_orders)
            if self.active_epoch is not None:
                values[self.active_epoch] = self.order_audit()
            return [values[key] for key in sorted(values)]

        def install_epoch(self, folder, epoch):
            receipt, self.weights = load_epoch_files(folder, self.records, training_identity, expected_epoch=epoch)
            self.epoch_receipt = receipt
            self.epoch_folder = Path(folder)
            self.active_epoch = epoch
            self.history = [x for x in self.history if x['epoch'] != epoch] + [receipt]
            self.history.sort(key=lambda x: x['epoch'])

        def load_checkpoint_epoch(self, checkpoint):
            state = read(checkpoint / 'rho_epoch_state.json')
            receipt = state['active_receipt']
            cursor = int(read(checkpoint / 'trainer_state.json')['global_step'])
            checkpoint_step = int(checkpoint.name.removeprefix('checkpoint-'))
            active_epoch = int(receipt['epoch'])
            boundary = active_epoch * self.updates_per_epoch
            if (state['training_identity'] != training_identity
                    or receipt['training_identity'] != training_identity
                    or state.get('global_step') != cursor or checkpoint_step != cursor
                    or not boundary < cursor <= boundary + self.updates_per_epoch
                    or receipt['epoch_start_step'] != boundary
                    or receipt['student_state']['global_step'] != boundary
                    or receipt['student_state']['epoch'] != active_epoch
                    or receipt['student_state']['training_identity'] != training_identity
                    or [r['epoch'] for r in state['history']] != list(range(active_epoch + 1))
                    or sha(checkpoint / 'rho_epoch_weights.parquet') != receipt['files']['weights.parquet']):
                raise ValueError('Checkpoint lost or changed its original epoch weights')
            self.weights = validate_weight_rows(pq.read_table(checkpoint / 'rho_epoch_weights.parquet').to_pylist(), self.records)
            self.epoch_receipt, self.active_epoch, self.history = receipt, receipt['epoch'], state['history']
            folder = Path(config['output_dir']) / 'rho_epochs' / f'epoch-{self.active_epoch:04d}'
            def hydrate_history():
                for previous in self.history:
                    e = previous['epoch']
                    hydrate_epoch(Path(config['rho_epoch_weight_dir']) / f'epoch-{e:04d}',
                        Path(config['output_dir']) / 'rho_epochs' / f'epoch-{e:04d}', self.records,
                        training_identity, expected_epoch=e, expected_receipt=previous)
                return str(folder)
            coordinator(hydrate_history)
            if sha(folder / 'weights.parquet') != sha(checkpoint / 'rho_epoch_weights.parquet'):
                raise ValueError('Checkpoint weight bytes differ from their original remote epoch')
            self.epoch_folder = folder
            rank_state = read(checkpoint / f'rho_order_rank-{self.rank}.json')
            expected = expected_rank_order(len(self.records), config.get('seed', 42), self.active_epoch, self.rank, self.world)
            consumed = min((cursor - boundary) * config['gradient_accumulation_steps'], len(expected))
            if (rank_state['rank'] != self.rank or rank_state['training_identity'] != training_identity
                    or rank_state.get('global_step') != cursor
                    or rank_state['order_audit']['epoch'] != active_epoch
                    or rank_state['order_audit']['expected_rows'] != len(expected)
                    or rank_state['order_audit']['consumed_rows'] != consumed
                    or rank_state['order_audit']['consumed_order_sha256'] != canonical(expected[:rank_state['order_audit']['consumed_rows']])):
                raise ValueError('Checkpoint source-order audit differs')
            self.completed_orders = {item['epoch']: item for item in rank_state.get('order_history', [])}
            for item in self.completed_orders.values():
                sequence = expected_rank_order(len(self.records), config.get('seed', 42), item['epoch'], self.rank, self.world)
                expected_consumed = consumed if item['epoch'] == active_epoch else len(sequence)
                if (item['consumed_rows'] != expected_consumed or item['expected_rows'] != len(sequence)
                        or item['consumed_order_sha256'] != canonical(sequence[:item['consumed_rows']])):
                    raise ValueError('Checkpoint contains a different historical source-order prefix')

        def refresh_epoch(self, epoch):
            from new_interest.evaluation.score_rho_student import score_current_student
            from new_interest.training.semantic_id_tokens import validate_saved_interest_tokens
            current_step = int(self.state.global_step)
            boundary = epoch * self.updates_per_epoch
            if current_step > boundary:
                if self.epoch_receipt is None or self.epoch_receipt['epoch'] != epoch:
                    raise ValueError('Mid-epoch resume must restore original weights, never rescore partial training')
                return
            if current_step != boundary:
                raise ValueError('Epoch refresh is not at the expected optimizer boundary')
            local = Path(config['output_dir']) / 'rho_epochs' / f'epoch-{epoch:04d}'
            remote = Path(config['rho_epoch_weight_dir']) / f'epoch-{epoch:04d}'
            with raw_model_context(self) as model:
                state = {'training_identity': training_identity, 'epoch': epoch, 'global_step': current_step,
                         'adapter_state_sha256': adapter_digest(model)}
                def reuse():
                    if not (remote / 'receipt.json').exists():
                        return None
                    receipt, _ = load_epoch_files(remote, self.records, training_identity, expected_epoch=epoch)
                    if receipt['student_state'] != state:
                        raise ValueError('Committed epoch weights belong to a different student boundary')
                    hydrate_epoch(remote, local, self.records, training_identity, expected_epoch=epoch,
                                  expected_receipt=receipt)
                    return str(local)
                reused = coordinator(reuse)
                if reused is not None:
                    receipt = read(local / 'receipt.json')
                    if receipt['student_state'] != state:
                        raise ValueError('This rank restored a different student state at the epoch boundary')
                    self.weights = validate_weight_rows(pq.read_table(local / 'weights.parquet').to_pylist(), self.records)
                    self.epoch_receipt, self.epoch_folder, self.active_epoch = receipt, local, epoch
                    self.history = [x for x in self.history if x['epoch'] != epoch] + [receipt]
                    self.history.sort(key=lambda x: x['epoch'])
                    return
                spec = validate_saved_interest_tokens(self.tokenizer_for_rho, Path(config['semantic_id_token_mapping_file']),
                                                     expected_count=config.get('semantic_id_token_count', 108))
                with preserve_rng():
                    scored = score_current_student(model, self.tokenizer_for_rho, self.records, spec,
                        student_state=state, shard_index=self.rank, num_shards=self.world,
                        batch_size=config.get('rho_score_batch_size', 4), max_seq_length=config.get('max_seq_length', 2048),
                        enable_thinking=False)
                local.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pylist(scored['rows']), local / f'current-rank-{self.rank}.parquet')
                write(local / f'current-rank-{self.rank}.json', {k: v for k, v in scored.items() if k != 'rows'})

            def collect_publish():
                current = []
                score_identities = []
                common_kernel_configs = {}
                for rank in range(self.world):
                    metadata = read(local / f'current-rank-{rank}.json')
                    part = pq.read_table(local / f'current-rank-{rank}.parquet').to_pylist()
                    if [r['record_index'] for r in part] != list(range(rank, len(self.records), self.world)):
                        raise ValueError('Current-student score shard coverage differs')
                    validate_current_shard(metadata, part, self.records, list(range(rank, len(self.records), self.world)),
                                           state, rank, self.world, common_kernel_configs)
                    current.extend(part)
                    score_identities.append(metadata['identity'])
                if any(x != score_identities[0] for x in score_identities):
                    raise ValueError('Ranks scored different student states')
                current.sort(key=lambda row: row['record_index'])
                result = build_classwise_weights(self.records, mode=config['rho_arm'], reference_rows=self.references,
                    current_rows=current, expected_classes=config['rho_expected_classes'],
                    expected_rows_per_class=config['rho_expected_rows_per_class'])
                validate_weight_rows(result['rows'], self.records)
                pq.write_table(pa.Table.from_pylist(result['rows']), local / 'weights.parquet')
                names = ['weights.parquet', *[f'current-rank-{r}.{suffix}' for r in range(self.world) for suffix in ('json', 'parquet')]]
                receipt = {'version': VERSION, 'training_identity': training_identity, 'epoch': epoch,
                    'world_size': self.world, 'restoration': {'epoch_weight_dir': str(remote)},
                    'epoch_start_step': boundary, 'student_state': state, 'score_identity': score_identities[0],
                    'diagnostics': result['diagnostics'], 'files': {name: sha(local / name) for name in names},
                    'all_arms_scored_current_student': True, 'new_optimizer_updates_during_refresh': 0,
                    'optimizer_reinitialized': False, 'large_artifact_policy': 'remote_only'}
                if (remote / 'receipt.json').exists():
                    raise ValueError('Refusing to overwrite committed epoch weights')
                for name in names:
                    copy_bytes(local / name, remote / name)
                write(local / 'receipt.json', receipt)
                copy_bytes(local / 'receipt.json', remote / 'receipt.json')
                load_epoch_files(remote, self.records, training_identity, expected_epoch=epoch)
                return str(local)
            coordinator(collect_publish)
            self.install_epoch(local, epoch)
            print('RHO_STUDENT_EPOCH_REFRESH ' + json.dumps({'rank': self.rank, 'epoch': epoch,
                  'global_step': current_step, 'weights_sha256': self.epoch_receipt['files']['weights.parquet']}), flush=True)

        def _run_epoch(self, *args, **kwargs):
            epoch = kwargs.get('epoch', args[1] if len(args) > 1 else None)
            loader = kwargs.get('train_dataloader', args[2] if len(args) > 2 else None)
            self.refresh_epoch(int(epoch))
            self.order = expected_rank_order(len(self.records), config.get('seed', 42), epoch, self.rank, self.world)
            self.position = (int(self.state.global_step) - epoch * self.updates_per_epoch) * config['gradient_accumulation_steps']
            if self.position > len(self.order):
                raise ValueError('Resumed epoch cursor exceeds source')
            # Preserve the seedable sampler epoch through Accelerate's skip wrapper.
            if hasattr(loader, 'set_epoch'):
                loader.set_epoch(epoch)
            result = super()._run_epoch(*args, **kwargs)
            if self.position != len(self.order) and not self.control.should_training_stop:
                raise ValueError('Student epoch did not consume its exact source partition')
            self.completed_orders[int(epoch)] = self.order_audit()
            return result

        def save_epoch_sidecar(self, checkpoint):
            if self.epoch_receipt is None:
                raise ValueError('Checkpoint cannot precede a bound weight epoch')
            self.kernel_audit.capture(f'checkpoint-{self.state.global_step}')
            write(checkpoint / f'rho_order_rank-{self.rank}.json', {'rank': self.rank,
                  'global_step': int(self.state.global_step), 'training_identity': training_identity,
                  'order_audit': self.order_audit(), 'order_history': self.order_history()})
            def stage():
                copy_bytes(self.epoch_folder / 'weights.parquet', checkpoint / 'rho_epoch_weights.parquet')
                write(checkpoint / 'rho_epoch_state.json', {'training_identity': training_identity,
                    'global_step': int(self.state.global_step), 'active_receipt': self.epoch_receipt, 'history': self.history})
                return str(checkpoint)
            coordinator(stage)

        def native_parity(self):
            import torch
            from new_interest.training.semantic_id_tokens import (
                capture_torch_numeric_backend, apply_torch_numeric_backend,
                deterministic_reload_numeric_backend, deterministic_sdpa_math_context,
            )
            model = self.model
            features = [self.train_dataset[0]]
            batch = self._prepare_inputs(self.data_collator(features))
            clean = {k: v for k, v in batch.items() if k != ROW_INDEX}
            count = self._get_num_items_in_batch([batch] * config['gradient_accumulation_steps'], self.args.device)
            modes = [(module, module.training) for module in model.modules()]
            parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
            if any(p.grad is not None for p in parameters.values()):
                raise ValueError('Native parity requires an empty gradient state')
            full_before = tensor_manifest(model.named_parameters())
            buffers_before = tensor_manifest(model.named_buffers())
            before = adapter_digest(model)
            backend_before = capture_torch_numeric_backend(torch)
            rng_before = capture_rng_state()
            local = Path(config['output_dir']) / f'rank-{self.rank}-native-parity.json'
            remote_base = Path(config.get('rho_diagnostic_output') or
                               (Path(config['rho_epoch_weight_dir']).parent / 'compact' /
                                f'{Path(config["rho_epoch_weight_dir"]).name}-native-parity.json'))
            if not remote_base.is_absolute() or remote_base.suffix != '.json':
                raise ValueError('Native diagnostic output must be an absolute persistent JSON path')
            for key in ('rho_protocol_file', 'rho_reference_file', 'train_file', 'semantic_id_token_mapping_file'):
                if remote_base.resolve() == Path(config[key]).resolve():
                    raise ValueError('Native diagnostic must not overwrite a frozen input')
            report = {'version': 'rho-native-gradient-diagnostic-v2', 'rank': self.rank,
                'training_identity': training_identity, 'global_step': int(self.state.global_step),
                'optimizer_state_entries': len(self.optimizer.state), 'optimizer_updates': 0,
                'num_items_in_batch': None if count is None else int(count),
                'world_size': self.world, 'args_n_gpu': self.args.n_gpu,
                'parallel_mode': str(self.args.parallel_mode),
                'input_tensors': tensor_manifest(clean.items()), 'source_sample_id': self.records[0]['source_sample_id'],
                'parameters_before': full_before, 'buffers_before': buffers_before,
                'numeric_backend_before': backend_before, 'rng_before': rng_receipt(rng_before),
                'dropout_modules': [{'name': name, 'p': module.p, 'training_before': module.training,
                                     'diagnostic_training': True}
                    for name, module in model.named_modules() if isinstance(module, torch.nn.Dropout)],
                'trials': [], 'comparisons': [], 'controls': {}, 'strict_original_threshold_preserved': True,
                'original_v1_detailed_metrics_unavailable': True,
                'diagnostic_only': bool(config.get('rho_native_diagnostic_only')),
                'diagnostic_complete': False, 'expected_gradient_tensors': len(parameters),
                'restoration': str(remote_base), 'large_artifact_policy': 'compact_diagnostics_only'}
            trials = {}
            self.parity_running = True
            old_weight = self.weights[0]

            def publish(stage):
                report['last_stage'] = stage
                write(local, report)
                def durable():
                    for rank in range(self.world):
                        source = Path(config['output_dir']) / f'rank-{rank}-native-parity.json'
                        target = remote_base if self.world == 1 else remote_base.with_name(remote_base.stem + f'-rank-{rank}.json')
                        copy_bytes(source, target)
                    return str(remote_base)
                coordinator(durable)
                print('RHO_NATIVE_PARITY_DIAGNOSTIC ' + json.dumps({'rank': self.rank,
                    'stage': stage, 'global_step': int(self.state.global_step),
                    'optimizer_state_entries': len(self.optimizer.state), 'report': str(remote_base),
                    'report_sha256': sha(local),
                    'comparisons': [{k: row.get(k) for k in ('scale', 'all_gradients_exact',
                        'loss_absolute_difference', 'global_gradient_relative_l2', 'max_gradient_absolute_difference')}
                        for row in report['comparisons']]}), flush=True)

            def one(label, weighted, scale):
                restore_rng_state(rng_before)
                model.zero_grad(set_to_none=True)
                self.weights[0] = scale
                self._parity_upstream = []
                dropout_probes = []
                handles = []
                for name, module in [(n, m) for n, m in model.named_modules()
                                     if isinstance(m, torch.nn.Dropout) and m.p > 0][:3]:
                    def observe(_module, _inputs, output, name=name):
                        values = output.detach().reshape(-1)[:64].eq(0).to('cpu').tolist()
                        dropout_probes.append({'name': name, 'zero_pattern_first64_sha256': canonical(values)})
                    handles.append(module.register_forward_hook(observe))
                item = {'label': label, 'weighted_wrapper': weighted, 'scale': scale,
                        'rng_before': rng_receipt(capture_rng_state()),
                        'numeric_backend': capture_torch_numeric_backend(torch),
                        'input_tensors': tensor_manifest(clean.items()),
                        'buffers_before': tensor_manifest(model.named_buffers())}
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    if weighted:
                        loss = self.compute_loss(model, batch, num_items_in_batch=count)
                    else:
                        raw = super(RhoStudentTrainer, self).compute_loss(model, clean, num_items_in_batch=count)
                        raw.register_hook(lambda value: self._parity_upstream.append(float(value.detach().cpu())))
                        loss = raw * scale
                    item['rng_after_forward'] = rng_receipt(capture_rng_state())
                    loss.backward()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    self.kernel_audit.capture(label)
                    grads = {name: p.grad.detach().cpu().clone() for name, p in parameters.items() if p.grad is not None}
                    loss_value = float(loss.detach().cpu())
                    item.update(loss=loss_value if math.isfinite(loss_value) else None,
                        loss_finite=math.isfinite(loss_value), gradient_tensors=len(grads),
                        gradient_nonfinite_values={name: int((~torch.isfinite(value)).sum()) for name, value in grads.items()},
                        missing_gradients=sorted(set(parameters)-set(grads)),
                        gradient_manifest=tensor_manifest(grads.items()),
                        upstream_scalar_gradient=list(self._parity_upstream),
                        upstream_scalar_exact=self._parity_upstream == [scale])
                    item['gradients_complete_and_finite'] = (bool(parameters) and not item['missing_gradients']
                        and not any(item['gradient_nonfinite_values'].values()) and item['loss_finite'])
                    trials[label] = {'loss': loss_value, 'gradients': grads}
                except Exception as error:
                    item['error'] = type(error).__name__ + ': ' + str(error)
                    trials[label] = None
                finally:
                    for handle in handles:
                        handle.remove()
                    item.update(dropout_probes=dropout_probes, rng_after_backward=rng_receipt(capture_rng_state()),
                                buffers_after=tensor_manifest(model.named_buffers()), adapter_state_sha256=adapter_digest(model))
                    report['trials'].append(item)
                    try:
                        report['kernel_receipt'] = self.kernel_audit.receipt(canonical(training_identity))
                    except Exception as error:
                        item['kernel_receipt_error'] = type(error).__name__ + ': ' + str(error)
                    publish(label)

            def compare(name, first, second):
                result = gradient_comparison(trials.get(first), trials.get(second))
                report['controls'][name] = result
                return result

            try:
                model.train()
                publish('prepared-before-any-backward')
                # Preserve the original v1 comparison order and exact threshold.
                one('original_cold_native_1', False, 1.)
                one('original_weighted_1', True, 1.)
                one('original_native_1_5', False, 1.5)
                one('original_weighted_1_5', True, 1.5)
                report['comparisons'] = [dict(gradient_comparison(trials['original_cold_native_1'], trials['original_weighted_1']), scale=1.),
                    dict(gradient_comparison(trials['original_native_1_5'], trials['original_weighted_1_5']), scale=1.5)]
                publish('original-guard-comparisons')
                for label, weighted in [('warm_native_1', False), ('warm_native_2', False),
                                        ('warm_weighted_1', True), ('warm_weighted_2', True)]:
                    one(label, weighted, 1.)
                compare('cold_vs_warm_native', 'original_cold_native_1', 'warm_native_1')
                compare('warm_native_repeat', 'warm_native_1', 'warm_native_2')
                compare('warm_weighted_repeat', 'warm_weighted_1', 'warm_weighted_2')
                compare('warm_native_vs_weighted', 'warm_native_1', 'warm_weighted_1')
                compare('warm_weighted2_vs_native2', 'warm_weighted_2', 'warm_native_2')
                if config.get('rho_native_diagnostic_only'):
                    try:
                        with deterministic_sdpa_math_context(torch):
                            apply_torch_numeric_backend(torch, deterministic_reload_numeric_backend())
                            one('strict_math_native_1', False, 1.)
                            one('strict_math_native_2', False, 1.)
                            one('strict_math_weighted_1', True, 1.)
                    except Exception as error:
                        report['strict_math_backend_error'] = type(error).__name__ + ': ' + str(error)
                        publish('strict-math-backend-unavailable')
                    finally:
                        apply_torch_numeric_backend(torch, backend_before)
                    compare('strict_math_native_repeat', 'strict_math_native_1', 'strict_math_native_2')
                    compare('strict_math_native_vs_weighted', 'strict_math_native_1', 'strict_math_weighted_1')
                    dropout = [(module, module.p) for module in model.modules() if isinstance(module, torch.nn.Dropout)]
                    try:
                        for module, _ in dropout:
                            module.p = 0.
                        one('dropout_off_native_1', False, 1.)
                        one('dropout_off_native_2', False, 1.)
                        one('dropout_off_weighted_1', True, 1.)
                    finally:
                        for module, p in dropout:
                            module.p = p
                    compare('dropout_off_native_repeat', 'dropout_off_native_1', 'dropout_off_native_2')
                    compare('dropout_off_native_vs_weighted', 'dropout_off_native_1', 'dropout_off_weighted_1')
                report.update(parameters_after=tensor_manifest(model.named_parameters()),
                    buffers_after=tensor_manifest(model.named_buffers()),
                    parameters_unchanged=before == adapter_digest(model),
                    global_step_after=int(self.state.global_step), optimizer_state_entries_after=len(self.optimizer.state))
                report['all_model_parameters_unchanged'] = full_before == report['parameters_after']
                production_trials = [item for item in report['trials'] if item['label'].startswith(('original_', 'warm_'))]
                report['production_gradients_complete_and_finite'] = (len(production_trials) == 8
                    and all(item.get('gradients_complete_and_finite', False) for item in production_trials))
                report['original_guard_passed'] = (report['parameters_unchanged']
                    and all(item.get('gradients_complete_and_finite', False) for item in production_trials[:4])
                    and all(x['all_gradients_exact'] for x in report['comparisons']))
                report['warm_native_reproducible'] = report['controls']['warm_native_repeat']['all_gradients_exact']
                report['production_controls_exact'] = all(item['all_gradients_exact'] for name, item in report['controls'].items()
                    if name == 'cold_vs_warm_native' or name.startswith('warm_'))
                report['production_kernel_receipts_complete'] = all('kernel_receipt_error' not in item for item in production_trials)
                report['upstream_scalars_exact'] = (len(production_trials) == 8
                    and all(item.get('upstream_scalar_exact', False) for item in production_trials))
                publish('complete-zero-updates')
                if int(self.state.global_step) != 0 or len(self.optimizer.state) != 0:
                    raise ValueError('Native diagnostic unexpectedly changed the optimization cursor')
                report['diagnostic_complete'] = True
                if config.get('rho_native_diagnostic_only'):
                    raise NativeParityDiagnosticComplete(report)
                if (not report['original_guard_passed'] or not report['warm_native_reproducible']
                        or not report['production_gradients_complete_and_finite']
                        or not report['production_controls_exact'] or not report['production_kernel_receipts_complete']
                        or not report['all_model_parameters_unchanged'] or not report['upstream_scalars_exact']):
                    raise ValueError('Native unit/scalar-weight gradient parity failed')
            finally:
                model.zero_grad(set_to_none=True)
                self.weights[0] = old_weight
                self.parity_running = False
                self._parity_upstream = None
                restore_rng_state(rng_before)
                apply_torch_numeric_backend(torch, backend_before)
                for module, mode in modes:
                    module.training = mode
                report['restored_state'] = {
                    'rng_unchanged': rng_receipt(capture_rng_state()) == rng_receipt(rng_before),
                    'numeric_backend_unchanged': capture_torch_numeric_backend(torch) == backend_before,
                    'module_modes_unchanged': all(module.training == mode for module, mode in modes),
                    'gradients_cleared': all(p.grad is None for p in parameters.values()),
                    'sample_weight_unchanged': self.weights[0] == old_weight,
                }
                publish('restored-after-diagnostic')

        def train(self, *args, **kwargs):
            from new_interest.evaluation.score_rho_reference import kernel_selection_context
            resume = kwargs.get('resume_from_checkpoint', args[0] if args else None)
            self.resume_path = Path(resume) if resume not in (None, False) else None
            if self.resume_path is not None:
                self.load_checkpoint_epoch(self.resume_path)
            with training_numeric_context(config.get('rho_training_numeric_policy', 'native-v1')), kernel_selection_context('stable_fla_v1') as kernels:
                self.kernel_audit = kernels
                return super().train(*args, **kwargs)

    return RhoStudentTrainer


def train_student(config):
    validate_config(config)
    import torch.distributed as dist
    from transformers.trainer import Trainer
    from transformers.trainer_callback import TrainerCallback
    from new_interest.training import train_sft
    from new_interest.evaluation.score_rho_reference import kernel_selection_context
    protocol_sha = sha(config['rho_protocol_file'])
    binding = {'rho_version': VERSION, 'rho_arm': config['rho_arm'], 'rho_protocol_sha256': protocol_sha,
        'rho_reference_file_sha256': config['rho_reference_file_sha256'], 'rho_stable_training': True,
        'rho_training_numeric_policy': config.get('rho_training_numeric_policy', 'native-v1'),
        'rho_expected_classes': config['rho_expected_classes'], 'rho_expected_rows_per_class': config['rho_expected_rows_per_class']}
    identity = {'model_revision': config.get('model_revision'), 'train_file_sha256': sha(config['train_file']),
                'mapping_sha256': sha(config['semantic_id_token_mapping_file']), **binding}
    sync = os.environ.get('RHO_SYNC_DIR')
    if not sync:
        raise ValueError('Student launcher must provide a unique shared-node RHO_SYNC_DIR')
    coordinator = LocalCheckpointCoordinator(Path(sync), distributed=dist, timeout_seconds=3600)
    original_identity = train_sft._training_checkpoint_identity
    original_coordinator = train_sft._run_rank_zero_checkpoint_operation
    original_capture = train_sft.capture_reload_reference
    def checkpoint_identity(*args, **kwargs):
        return {**original_identity(*args, **kwargs), **binding}
    def capture(*args, **kwargs):
        with kernel_selection_context('stable_fla_v1') as kernels:
            result = original_capture(*args, **kwargs)
            kernels.capture('final-native-reload-reference')
            result['rho_kernel_selection_receipt'] = kernels.receipt(canonical(identity))
            return result

    def aggregate():
        output = Path(config['output_dir'])
        world = int(os.environ.get('WORLD_SIZE', '1'))
        summaries = [read(output / f'rank-{rank}-rho-student-summary.json') for rank in range(world)]
        runtimes = [read(output / f'rank-{rank}-training-runtime.json') for rank in range(world)]
        epochs = int(config['epochs'])
        rows = config['rho_expected_classes'] * config['rho_expected_rows_per_class']
        steps_per_epoch = math.ceil(rows / (world * config['gradient_accumulation_steps']))
        if len({x['initial_state']['state_sha256'] for x in runtimes}) != 1:
            raise ValueError('Final student ranks report different initial states')
        for rank, (summary, runtime) in enumerate(zip(summaries, runtimes)):
            if (summary['rank'] != rank or summary['training_identity'] != identity
                    or summary['actual_optimizer_steps'] != epochs * steps_per_epoch
                    or summary['actual_epoch'] != epochs or not runtime['completed_as_configured']
                    or [x['epoch_start_step'] for x in summary['epochs']] != [i * steps_per_epoch for i in range(epochs)]):
                raise ValueError('Incomplete student epoch/step/rank acceptance')
            if summary['epochs'] != summaries[0]['epochs']:
                raise ValueError('Ranks trained with different epoch weights')
            if len(summary['order_history']) != epochs:
                raise ValueError('Student source-order audit misses an epoch')
            for epoch, order in enumerate(summary['order_history']):
                sequence = expected_rank_order(rows, config.get('seed', 42), epoch, rank, world)
                if order != {'epoch': epoch, 'consumed_rows': len(sequence), 'expected_rows': len(sequence),
                             'consumed_order_sha256': canonical(sequence)}:
                    raise ValueError('Student did not consume the exact native shuffled epoch')
        parity = []
        if config.get('rho_require_native_parity'):
            for rank in range(world):
                file = output / f'rank-{rank}-native-parity.json'
                if file.exists():
                    item = read(file)
                    if not item['parameters_unchanged'] or not all(row['all_gradients_exact'] for row in item['comparisons']):
                        raise ValueError('Student native loss parity failed')
                    parity.append(item)
                elif not runtimes[rank]['train_begin']['resumed_from_checkpoint']:
                    raise ValueError('Fresh gate has no native gradient parity receipt')
        result = {'passed': True, 'version': VERSION, 'arm': config['rho_arm'],
            'training_identity': identity, 'epochs': epochs, 'optimizer_steps': epochs * steps_per_epoch,
            'world_size': world, 'rows': rows, 'classes': config['rho_expected_classes'],
            'rows_per_class': config['rho_expected_rows_per_class'],
            'refresh_steps': [i * steps_per_epoch for i in range(epochs)],
            'reference_file_sha256': config['rho_reference_file_sha256'],
            'train_file_sha256': identity['train_file_sha256'],
            'initial_state_sha256': runtimes[0]['initial_state']['state_sha256'],
            'epoch_receipts': summaries[0]['epochs'], 'rank_order_history': [x['order_history'] for x in summaries],
            'class_weight_mass_preserved': True, 'weight_bounds': [.5, 1.5], 'all_weights_positive': True,
            'current_student_scoring_all_arms': True, 'optimizer_reinitialized_at_refresh': False,
            'native_completion_loss': 'super().compute_loss with unchanged num_items_in_batch, times detached scalar weight',
            'native_gradient_parity': parity, 'training_kernel_selection': 'stable_fla_v1',
            'training_numeric_policy': config.get('rho_training_numeric_policy', 'native-v1'),
            'training_deterministic_algorithms': config.get('rho_training_numeric_policy') == DETERMINISTIC_POLICY,
            'training_deterministic_warn_only': False if config.get('rho_training_numeric_policy') == DETERMINISTIC_POLICY else None,
            'large_artifact_policy': 'remote_only'}
        write(output / 'rho_student_summary.json', result)
        return str(output / 'rho_student_summary.json')
    try:
        train_sft._training_checkpoint_identity = checkpoint_identity
        train_sft._run_rank_zero_checkpoint_operation = coordinator
        train_sft.capture_reload_reference = capture
        trainer = student_trainer_class(Trainer, TrainerCallback, config, coordinator, identity)
        try:
            summary = train_sft.train(config, trainer_base=trainer)
        except NativeParityDiagnosticComplete as completed:
            if not config.get('rho_native_diagnostic_only'):
                raise
            report = completed.receipt
            if (report.get('diagnostic_complete') is not True
                    or report.get('global_step') != 0 or report.get('global_step_after') != 0
                    or report.get('optimizer_state_entries') != 0 or report.get('optimizer_state_entries_after') != 0
                    or not all(report.get('restored_state', {}).get(key) is True for key in (
                        'rng_unchanged', 'numeric_backend_unchanged', 'module_modes_unchanged',
                        'gradients_cleared', 'sample_weight_unchanged'))):
                raise ValueError('Diagnostic completion lacks verified zero-update/restoration evidence') from completed
            return {'diagnostic_complete': True, 'optimizer_steps': report['global_step_after'],
                    'optimizer_state_entries': report['optimizer_state_entries_after'],
                    'original_guard_passed': report['original_guard_passed'],
                    'report': report['restoration'], 'training_identity': identity}
        if config.get('rho_native_diagnostic_only'):
            raise ValueError('Diagnostic-only request did not stop at the native on_train_begin callback')
        coordinator(aggregate)
        return summary
    finally:
        train_sft._training_checkpoint_identity = original_identity
        train_sft._run_rank_zero_checkpoint_operation = original_coordinator
        train_sft.capture_reload_reference = original_capture


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    result = train_student(read(args.config))
    if result.get('diagnostic_complete'):
        print('RHO_NATIVE_DIAGNOSTIC_COMPLETE ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
