import json
from pathlib import Path
import pytest

from new_interest.experiments.short_think_grpo_full import choose_checkpoint,hydrate
from new_interest.experiments.short_think_grpo_engine import publish


def test_real_publisher_hydrator_and_checkpoint_choice(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'state.json').write_text(json.dumps({'identity':'H','step':25}))
    (source/'model.bin').write_bytes(b'checkpoint payload')
    output=tmp_path/'output';dest=output/'attempts/one/checkpoint-25'
    publish(source,dest,'H')
    assert choose_checkpoint(output,'H',100)==dest
    state=hydrate(dest,tmp_path/'fresh');assert state['step']==25
    # A partial newer checkpoint without its producer commit is not resumable.
    (output/'attempts/one/checkpoint-50').mkdir()
    assert choose_checkpoint(output,'H',100)==dest
    (dest/'model.bin').write_bytes(b'corrupt')
    with pytest.raises(ValueError):hydrate(dest,tmp_path/'bad')


def test_conflicting_same_step_and_wrong_identity_are_rejected(tmp_path):
    for attempt,payload in [('a','first'),('b','second')]:
        src=tmp_path/attempt;src.mkdir();(src/'state.json').write_text(json.dumps({'identity':'H','step':25,'payload':payload}))
        publish(src,tmp_path/'output/attempts'/attempt/'checkpoint-25','H')
    with pytest.raises(ValueError,match='Conflicting'):choose_checkpoint(tmp_path/'output','H',100)
    with pytest.raises(ValueError,match='another experiment'):choose_checkpoint(tmp_path/'output','E',100)
