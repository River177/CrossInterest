from pathlib import Path
import hashlib
import json
import pytest
from new_interest.experiments.oi_ni_full_sft_storage import LocalTrainingJournal,hydrate_recovery_parent
from new_interest.experiments.oi_ni_meta import sha256_file


def test_cloud_snapshot_failure_does_not_touch_local_training_prefix(tmp_path,monkeypatch):
    prefix=[{'step':i,'row_indices':[i],'loss':1.} for i in range(1,401)]
    journal=LocalTrainingJournal(tmp_path/'node',prefix)
    original=journal.path.read_bytes()
    import new_interest.experiments.oi_ni_full_sft_storage as storage
    def fail(*args):raise OSError('Connection to destination failed')
    with monkeypatch.context() as m:
        m.setattr(storage.shutil,'copyfile',fail)
        with pytest.raises(OSError):journal.snapshot(tmp_path/'cloud'/'training.jsonl')
        journal.append({'step':401,'row_indices':[401],'loss':.9})
    assert journal.path.read_bytes().startswith(original)
    assert len(journal.path.read_text().splitlines())==401
    (tmp_path/'cloud').mkdir()
    journal.snapshot(tmp_path/'cloud'/'training.jsonl')
    assert (tmp_path/'cloud'/'training.jsonl').read_bytes()==journal.path.read_bytes()
    with pytest.raises(ValueError):journal.append({'step':401})


def parent_fixture(tmp_path):
    attempt=tmp_path/'source'/'attempt'
    cp=attempt/'checkpoints'/'checkpoint-000400';cp.mkdir(parents=True)
    config={'seed':42,'epochs':3,'loss':'full_vocabulary'}
    (attempt/'source_manifest.json').write_text('{"version":"fixture"}')
    source=sha256_file(attempt/'source_manifest.json')
    identity=hashlib.sha256(json.dumps({'source':source,'config':config,'mode':'mixed'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (attempt/'resolved_config.json').write_text(json.dumps({'identity':identity,'mode':'mixed','config':config,'source_sha256':source}))
    (cp/'state.json').write_text(json.dumps({'step':400,'tensor_sha256':'test-digest','identity':identity}))
    (cp/'payload.bin').write_bytes(b'original-state')
    (cp/'manifest.json').write_text(json.dumps({'identity':identity,'files':{n:sha256_file(cp/n) for n in ['state.json','payload.bin']}}))
    binding={'mode':'mixed','identity':identity,'source_manifest_sha256':source,'checkpoint_path':str(cp),
        'checkpoint_manifest_sha256':sha256_file(cp/'manifest.json'),'step':400,'tensor_sha256':'test-digest'}
    scratch=tmp_path/'node';scratch.mkdir()
    return cp,config,binding,scratch


def test_recovery_hydrates_exact_parent_without_rewriting_it(tmp_path):
    cp,c,b,s=parent_fixture(tmp_path)
    before={p.name:p.read_bytes() for p in cp.iterdir()}
    local=hydrate_recovery_parent(b,c,'mixed',s)
    assert {p.name:p.read_bytes() for p in cp.iterdir()}==before
    assert {p.name:p.read_bytes() for p in local.iterdir()}==before


@pytest.mark.parametrize('tamper',['payload','config','mode','manifest','source','identity','step','tensor'])
def test_recovery_rejects_mismatched_parent(tmp_path,tamper):
    cp,c,b,s=parent_fixture(tmp_path)
    if tamper=='payload':(cp/'payload.bin').write_bytes(b'changed')
    if tamper=='config':c['epochs']=4
    if tamper=='mode':b['mode']='meta'
    if tamper=='manifest':b['checkpoint_manifest_sha256']='0'*64
    if tamper=='source':b['source_manifest_sha256']='0'*64
    if tamper=='identity':b['identity']='0'*64
    if tamper=='step':b['step']=399
    if tamper=='tensor':b['tensor_sha256']='wrong'
    with pytest.raises(ValueError):hydrate_recovery_parent(b,c,'mixed',s)
