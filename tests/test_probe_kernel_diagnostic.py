from types import SimpleNamespace
import pytest

from new_interest.training.probe_kernel_diagnostic import fixed_selection,stable_benchmark


class Config:
    def __init__(self,x):self.x=x
    def all_kwargs(self):return {'num_warps':self.x}


def test_stable_selection_keeps_real_candidate_validity_and_restores_state():
    calls=[]
    def bench(*args,config,**kwargs):
        calls.append(config.x)
        return [float('inf')]*3 if config.x==8 else [1.,2.,3.]
    cache={'old':'plan'}
    tuner=SimpleNamespace(cache=cache,cache_results=True,_bench=bench,best_config='old',configs=[Config(4),Config(8)])
    configs=tuner.configs
    with pytest.raises(RuntimeError):
        with fixed_selection({'test':tuner}):
            assert tuner.cache=={} and not tuner.cache_results and tuner.configs is configs
            first=tuner._bench(config=configs[0]);second=tuner._bench(config=Config(4))
            assert first==second and tuner._bench(config=configs[1])==[float('inf')]*3
            tuner.cache['new']='canonical';tuner.best_config='new'
            raise RuntimeError('injected')
    assert tuner.cache is cache and tuner._bench is bench and tuner.best_config=='old' and tuner.cache_results
    assert calls==[4,4,8]
