from types import SimpleNamespace
import pytest

from new_interest.training.probe_execution_plan import math_flags,replay_plan,checkpoint_plan


class Config:
    def __init__(self,n):self.n=n
    def __str__(self):return f'warps={self.n}'


def test_replay_exact_choice_rejects_miss_and_restores_cache():
    old={('old',):Config(8)};bench=lambda *a,**k:0
    tuner=SimpleNamespace(cache=old,cache_results=True,_bench=bench,configs=[Config(4),Config(8)])
    plan={'version':'fla-probe-execution-plan-v1','math_flags':math_flags(),
          'tuners':{'test':{'cache':{"(128, 'torch.bfloat16')":'warps=4'}}}}
    with replay_plan(plan,{'test':tuner}):
        assert tuner.cache[(128,'torch.bfloat16')].n==4 and not tuner.cache_results
        with pytest.raises(ValueError,match='unrecorded'):tuner._bench()
    assert tuner.cache is old and tuner._bench is bench and tuner.cache_results
    plan['tuners']['test']['cache']["(128, 'torch.bfloat16')"]='warps=32'
    with pytest.raises(ValueError,match='unavailable'):
        with replay_plan(plan,{'test':tuner}):pass
    assert tuner.cache is old


def test_missing_plan_recovery_binds_original_reference_and_bytes(tmp_path):
    import json
    from new_interest.evaluation.preflight_short_think_grpo import sha
    project=tmp_path/'project';project.mkdir();checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    state={'adapter_state_sha256':'adapter','optimizer_state_sha256':'optimizer','reload_probe':[[1.25]]}
    (checkpoint/'state.json').write_text(json.dumps(state))
    recovery={'checkpoint_state_file_sha256':sha(checkpoint/'state.json'),**{k:v for k,v in state.items() if k!='reload_probe'},
        'original_probe':state['reload_probe'],'execution_plan':{'test':'plan'}}
    (project/'recovered_probe_plan.json').write_text(json.dumps(recovery))
    assert checkpoint_plan(project,checkpoint,state)=={'test':'plan'}
    with pytest.raises(ValueError):checkpoint_plan(project,checkpoint,{**state,'reload_probe':[[1.5]]})
    (checkpoint/'state.json').write_text((checkpoint/'state.json').read_text()+' ')
    with pytest.raises(ValueError):checkpoint_plan(project,checkpoint,state)
