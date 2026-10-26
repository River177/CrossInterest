"""SID-aware merged-weight export for an already-unsharded PEFT actor.

Caller must keep the context open until every streamed tensor has been cloned or
transferred. FSDP gathering, vLLM loading and cross-engine numeric acceptance are
separate integration gates. Export restores the live actor weights on exit.
"""
from contextlib import contextmanager


def normalize_sid_peft_state(state):
    result={}
    for name,value in state.items():
        if any(part in name for part in ('lora_','trainable_tokens_delta','trainable_tokens_original')):
            continue
        name=name.replace('_fsdp_wrapped_module.','')
        if name.startswith('base_model.model.'):
            name=name[len('base_model.model.'):]
        name=name.replace('.base_layer','').replace('.token_adapter','')
        if name in result:raise ValueError(f'Export name collision: {name}')
        if '_flat_param' in name:raise ValueError('Actor must be unsharded before export')
        result[name]=value
    return result


@contextmanager
def merged_sid_lora_context(model):
    """Materialize LoRA and absolute learned token rows, then restore byte-exactly.

Back up each distinct base Parameter once (tied heads share a Parameter). Restore
by copy, not by subtracting a rounded LoRA delta. Learned SID rows themselves are
never modified. No full-vocabulary parameter is made trainable by this export.
"""
    import torch
    from peft.tuners.lora import LoraLayer
    from peft.tuners.trainable_tokens.layer import TrainableTokensLayer
    layers=[layer for layer in model.modules() if isinstance(layer,(LoraLayer,TrainableTokensLayer))]
    if not layers:raise ValueError('No PEFT layers to export')
    if any(layer.merged for layer in layers):raise ValueError('Actor was already merged')
    backup={}
    with torch.no_grad():
        for layer in layers:
            base=layer.get_base_layer()
            for parameter in (base.weight,getattr(base,'bias',None)):
                if parameter is None:continue
                if hasattr(parameter,'full_tensor'):
                    raise ValueError('Export requires materialized local tensors, not DTensor shards')
                if id(parameter) not in backup:backup[id(parameter)]=(parameter,parameter.detach().clone())
    try:
        with torch.no_grad():
            for layer in layers:layer.merge(safe_merge=True)
        yield
    finally:
        with torch.no_grad():
            for parameter,original in backup.values():parameter.copy_(original)
            for layer in layers:layer.merged_adapters.clear()


def export_merged_sid_state(model):
    """Materialize a detached snapshot; never return aliases to restored storage."""
    with merged_sid_lora_context(model):
        return {name:value.detach().clone() for name,value in normalize_sid_peft_state(model.state_dict()).items()}
