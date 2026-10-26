"""Compare fresh H5 inference contexts without updates, on an existing GPU runtime."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=('original', 'late_backend', 'autocast_bf16', 'tf32_training', 'import_trainer', 'rope_cuda', 'rope_bf16'), required=True)
    args = parser.parse_args()
    import torch
    if args.case == 'import_trainer':
        from transformers import Trainer  # noqa: F401
    from peft import PeftModel
    from new_interest.training import verify_semantic_id_token_adapter as verifier
    from new_interest.training.semantic_id_tokens import capture_torch_numeric_backend, apply_torch_numeric_backend
    reference = json.loads((args.source / 'reference.json').read_text())
    expected = reference['torch_numeric_backend']
    telemetry = {'case': args.case, 'before_verifier': capture_torch_numeric_backend(torch), 'captures': []}
    loader = PeftModel.from_pretrained
    loaded_models = []

    def tracked_loader(*a, **kw):
        model = loader(*a, **kw)
        loaded_models.append(model)
        if args.case == 'autocast_bf16':
            forward = model.forward
            def wrapped(*inputs, **kwargs):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    return forward(*inputs, **kwargs)
            model.forward = wrapped
        return model
    PeftModel.from_pretrained = tracked_loader
    compare = verifier.compare_adapter_state_dicts

    def state_compare(saved, loaded):
        telemetry['after_construction'] = capture_torch_numeric_backend(torch)
        if args.case in ('rope_cuda', 'rope_bf16'):
            from new_interest.training.semantic_id_tokens import _tensor_sha256
            rope = loaded_models[-1].get_base_model().model.rotary_emb
            old = rope.inv_freq.detach().clone()
            assert rope.rope_type == 'default'
            if args.case == 'rope_cuda':
                inv, scale = rope.compute_default_rope_parameters(rope.config, device=loaded_models[-1].device)
                rope.attention_scaling = scale
            else:
                inv = old.bfloat16().float()
            rope.inv_freq = inv
            rope.original_inv_freq = inv.clone()
            telemetry['rope'] = {'before_sha256': _tensor_sha256(old), 'after_sha256': _tensor_sha256(inv),
                                 'max_delta': float((old-inv).abs().max()), 'values': inv.float().cpu().tolist()}
        if args.case == 'late_backend':
            apply_torch_numeric_backend(torch, expected)
        if args.case == 'tf32_training':
            changed = {**expected, 'cuda_matmul_allow_tf32': True, 'cudnn_allow_tf32': True,
                       'float32_matmul_precision': 'high'}
            apply_torch_numeric_backend(torch, changed)
        telemetry['before_forward'] = capture_torch_numeric_backend(torch)
        telemetry['autocast_enabled'] = torch.is_autocast_enabled()
        telemetry['model_use_cache'] = loaded_models[-1].config.use_cache
        return compare(saved, loaded)
    verifier.compare_adapter_state_dicts = state_compare
    isclose = torch.isclose
    def capture(actual, requested, *a, **kw):
        if tuple(actual.shape) == tuple(requested.shape) == (108,):
            telemetry['captures'].append({'actual': actual.detach().cpu().tolist(), 'expected': requested.detach().cpu().tolist()})
        return isclose(actual, requested, *a, **kw)
    torch.isclose = capture
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        verifier.verify_semantic_id_token_adapter('Qwen/Qwen3.5-2B', args.source / 'adapter',
            args.source / 'adapter/interest_token_mapping.parquet', args.source / 'reference.json',
            args.output / f'{args.case}-audit.json',
            model_revision='15852e8c16360a2fea060d615a32b45270f8a8fc', atol=0, rtol=0)
    except ValueError as exc:
        if not str(exc).startswith('Adapter reload logits mismatch:'):
            raise
    finally:
        torch.isclose = isclose
    assert len(telemetry['captures']) == 4
    errors = [abs(a-b) for row in telemetry['captures'] for a,b in zip(row['actual'], row['expected'])]
    telemetry['max_error'] = max(errors)
    telemetry['exact'] = max(errors) == 0
    telemetry['after_forward'] = capture_torch_numeric_backend(torch)
    prior = args.source / 'previous_fresh.json'
    if prior.is_file():
        previous = json.loads(prior.read_text())
        telemetry['max_difference_from_prior_fresh'] = max(abs(a-b) for x,y in zip(telemetry['captures'], previous['captures'])
                                                         for a,b in zip(x['actual'],y['actual']))
    (args.output / f'{args.case}.json').write_text(json.dumps(telemetry, indent=2) + '\n')
    print(json.dumps({k:v for k,v in telemetry.items() if k != 'captures'}), flush=True)


if __name__ == '__main__':
    main()
