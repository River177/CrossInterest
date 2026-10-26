import pytest
from new_interest.experiments.short_think_sft import validate_contract


@pytest.mark.parametrize('record',[None,False])
def test_initial_state_requirement_cannot_be_set_without_recording(record):
    config={'enable_thinking':True,'epochs':2}
    if record is not None:config['record_initial_adapter_state']=record
    with pytest.raises(ValueError,match='record_initial_adapter_state'):
        validate_contract({'sft':config})


def test_recording_and_thinking_are_explicit():
    config={'record_initial_adapter_state':True,'enable_thinking':True,'epochs':2}
    validate_contract({'sft':config})
    with pytest.raises(ValueError,match='contract changed'):
        validate_contract({'sft':{**config,'enable_thinking':False}})
