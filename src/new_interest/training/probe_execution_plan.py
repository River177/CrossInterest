"""Replay the actual FLA kernel choices associated with a saved raw probe.

This affects serialization probes only. Training/evaluation autotuner state is
restored afterward. Unknown keys fail rather than silently retuning a probe.
"""
import ast
from contextlib import contextmanager
import importlib

from new_interest.training.probe_kernel_diagnostic import discover_tuners,snapshot


def math_flags():
    import torch
    return {'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
        'matmul_precision':torch.get_float32_matmul_precision(),'default_dtype':str(torch.get_default_dtype()),
        'deterministic':torch.are_deterministic_algorithms_enabled()}


def capture_plan():
    return {'version':'fla-probe-execution-plan-v1','math_flags':math_flags(),'tuners':snapshot(discover_tuners())}


def _reject_missing(label,tuner):
    def missing(*args,**kwargs):
        combined={**(getattr(tuner,'nargs',None) or {}),**kwargs}
        fields={k:combined.get(k) for k in getattr(tuner,'keys',[])}
        safe={k:(str(v) if isinstance(v,(int,float,bool,str,type(None))) else type(v).__name__) for k,v in fields.items()}
        dtypes=[str(v.dtype) for v in combined.values() if hasattr(v,'dtype')]
        raise ValueError(f'Probe requested an unrecorded FLA kernel key: {label}; fields={safe}; dtypes={dtypes}; retuning is forbidden')
    return missing


def _resolve(label):
    from triton.runtime.autotuner import Autotuner
    symbol,depth=label.rsplit(':',1);module,name=symbol.rsplit('.',1)
    if not module.startswith('fla.'):raise ValueError('Unexpected probe kernel namespace')
    value=getattr(importlib.import_module(module),name)
    for _ in range(int(depth)):value=value.fn
    if not isinstance(value,Autotuner):raise ValueError('Recorded kernel is no longer an autotuner')
    return value


@contextmanager
def replay_plan(plan,tuners=None):
    if plan['version']!='fla-probe-execution-plan-v1' or plan['math_flags']!=math_flags():
        raise ValueError('Probe version or math flags differ')
    recorded=plan['tuners']
    if not recorded:raise ValueError('Empty probe execution plan')
    selected={name:(_resolve(name) if tuners is None else tuners[name]) for name in recorded}
    all_tuners=discover_tuners() if tuners is None else tuners
    unique={id(t):t for t in [*all_tuners.values(),*selected.values()]}
    labels={id(t):name for name,t in {**all_tuners,**selected}.items()}
    saved=[]
    attrs=['cache','cache_results','_bench','best_config','configs_timings','bench_time']
    try:
        for tuner in unique.values():
            saved.append((tuner,{k:(hasattr(tuner,k),getattr(tuner,k,None)) for k in attrs}))
            tuner.cache={};tuner.cache_results=False;tuner._bench=_reject_missing(labels[id(tuner)],tuner)
        for name,description in recorded.items():
            tuner=selected[name]
            for encoded,choice in description['cache'].items():
                key=ast.literal_eval(encoded)
                if not isinstance(key,tuple):raise ValueError('Invalid saved tuner key')
                matches=[c for c in tuner.configs if str(c)==choice]
                if len(matches)!=1:raise ValueError('Saved kernel configuration is unavailable or ambiguous')
                tuner.cache[key]=matches[0]
        yield
        if tuners is None:
            unexpected=[name for name,tuner in discover_tuners().items() if id(tuner) not in unique and tuner.cache]
            if unexpected:raise ValueError(f'Unrecorded probe kernels were loaded: {unexpected}')
    finally:
        for tuner,before in reversed(saved):
            for name,(present,value) in before.items():
                if present:setattr(tuner,name,value)
                elif hasattr(tuner,name):delattr(tuner,name)


def checkpoint_plan(project,checkpoint,state):
    """Recover only missing execution metadata, never replace saved logits."""
    from new_interest.evaluation.preflight_short_think_grpo import read,sha
    if state.get('probe_execution_plan') is not None:return state['probe_execution_plan']
    path=project/'recovered_probe_plan.json'
    if not path.exists():raise ValueError('Checkpoint has no recorded probe execution plan')
    recovered=read(path)
    if (recovered['checkpoint_state_file_sha256']!=sha(checkpoint/'state.json')
        or recovered['adapter_state_sha256']!=state['adapter_state_sha256']
        or recovered['optimizer_state_sha256']!=state['optimizer_state_sha256']
        or recovered['original_probe']!=state['reload_probe']):
        raise ValueError('Recovered execution plan is not bound to this exact checkpoint/reference')
    return recovered['execution_plan']
