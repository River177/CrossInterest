"""Bind FLA norm launch geometry when validating a saved numerical reference."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

PROFILE_VERSION = 'qwen35-fla-norm-launch-profile-v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def apply_fla_norm_profile(profile_file, *, adapter_path, reference_file=None):
    """Select an existing official FLA configuration without changing any tensor."""
    import torch
    from triton.runtime.autotuner import Autotuner
    from fla.modules import fused_norm_gate

    profile_file = Path(profile_file)
    profile = json.loads(profile_file.read_text())
    if profile.get('version') != PROFILE_VERSION:
        raise ValueError('Unknown FLA norm profile version')
    if profile.get('kernel') != 'fla.modules.fused_norm_gate.layer_norm_gated_fwd_kernel':
        raise ValueError('Unexpected FLA profile kernel')
    required_packages = {'torch', 'transformers', 'peft', 'triton', 'flash-linear-attention', 'fla-core', 'causal-conv1d'}
    if set(profile['packages']) != required_packages:
        raise ValueError('Incomplete FLA profile runtime identity')
    if sha256(Path(adapter_path) / 'adapter_model.safetensors') != profile['adapter_sha256']:
        raise ValueError('FLA norm profile belongs to another adapter')
    if reference_file is not None and sha256(reference_file) != profile['reference_sha256']:
        raise ValueError('FLA norm profile belongs to another reference')
    actual = {name: importlib.metadata.version(name) for name in profile['packages']}
    if actual != profile['packages']:
        raise ValueError('FLA norm profile runtime versions differ')
    if sha256(fused_norm_gate.__file__) != profile['module_sha256']:
        raise ValueError('FLA norm implementation changed')
    if tuple(torch.cuda.get_device_capability()) != tuple(profile['cuda_capability']):
        raise ValueError('FLA norm profile GPU architecture differs')
    kernel = fused_norm_gate.layer_norm_gated_fwd_kernel
    while not isinstance(kernel, Autotuner):
        kernel = kernel.fn
    wanted = profile['config']
    matches = [c for c in kernel.configs if c.kwargs == wanted['kwargs']
               and c.num_warps == wanted['num_warps'] and c.num_stages == wanted['num_stages']
               and c.num_ctas == wanted['num_ctas'] and c.maxnreg == wanted['maxnreg']
               and c.pre_hook is None]
    if len(matches) != 1:
        raise ValueError('Profile is not a unique official FLA launch configuration')
    # With one configuration Triton skips autotuning and its process-local cache.
    kernel.configs = matches
    kernel.cache.clear()
    return {'version': PROFILE_VERSION, 'profile_sha256': sha256(profile_file),
            'kernel': profile['kernel'], 'config': wanted, 'packages': actual,
            'adapter_sha256': profile['adapter_sha256'], 'model_tensors_modified': False}
