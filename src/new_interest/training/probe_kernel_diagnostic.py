"""Read-only experimental controls for FLA autotuner history sensitivity.

No production probe protocol is changed by importing this module. The fixed
selection context retains Triton's ordinary candidate pruning and runs the real
benchmark to reject invalid configurations, but ranks finite candidates by a
stable configuration digest. Disk timing writes are disabled and all mutable
autotuner selections are restored on exit.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import sys


def discover_tuners():
    from triton.runtime.autotuner import Autotuner,Heuristics
    found={};seen=set()
    for name,module in sorted(list(sys.modules.items())):
        if not name.startswith('fla.') or module is None:continue
        for symbol,value in sorted(list(vars(module).items())):
            chain=set()
            for depth in range(8):
                if not isinstance(value,(Autotuner,Heuristics)):break
                if id(value) in chain:break
                chain.add(id(value))
                if isinstance(value,Autotuner) and id(value) not in seen:
                    found[f'{name}.{symbol}:{depth}']=value;seen.add(id(value))
                value=value.fn
    return found


def snapshot(tuners):
    return {name:{'cache':{str(k):str(v) for k,v in tuner.cache.items()},
        'best_config':str(getattr(tuner,'best_config',None)),
        'cache_results':getattr(tuner,'cache_results',None)} for name,tuner in tuners.items() if tuner.cache}


def stable_benchmark(original):
    def wrapped(*args,config,**kwargs):
        measured=original(*args,config=config,**kwargs)
        values=measured if isinstance(measured,(list,tuple)) else [measured]
        if not all(math.isfinite(float(x)) for x in values):return measured
        encoded=json.dumps(config.all_kwargs(),sort_keys=True,default=str).encode()
        score=float(int(hashlib.sha256(encoded).hexdigest()[:12],16))
        return [score]*len(values) if isinstance(measured,(list,tuple)) else score
    return wrapped


@contextmanager
def fixed_selection(tuners=None):
    tuners=discover_tuners() if tuners is None else tuners
    if not tuners:raise ValueError('No FLA autotuners discovered')
    saved=[]
    names=['cache','cache_results','_bench','best_config','configs_timings','bench_time']
    try:
        for name,tuner in tuners.items():
            before={key:(hasattr(tuner,key),getattr(tuner,key,None)) for key in names}
            saved.append((tuner,before))
            tuner.cache={};tuner.cache_results=False;tuner._bench=stable_benchmark(tuner._bench)
        yield tuners
    finally:
        for tuner,before in reversed(saved):
            for key,(present,value) in before.items():
                if present:setattr(tuner,key,value)
                elif hasattr(tuner,key):delattr(tuner,key)
