"""Final upload must not touch the logs of a live managed run."""

import json
import pytest
import finalize_with_drive as recovery


@pytest.fixture
def closed(tmp_path,monkeypatch):
    run=tmp_path/'batch_sac_test';run.mkdir()
    (run/'manifest.json').write_text('{}')
    state=dict(phase='final_upload',training_pid=101,supervisor_pid=102,
               run_dir=str(run),training_exit_code=1,stop_reason='low_disk_space')
    (tmp_path/'status.json').write_text(json.dumps(state))
    monkeypatch.setattr(recovery,'process_alive',lambda pid:False)
    return tmp_path,run,state


def test_closed_failure_keeps_original_exit_code(closed):
    parent,run,state=closed
    actual,loaded=recovery.validate_closed_run(parent)
    assert actual==run
    assert loaded['training_exit_code']==1
    assert loaded['stop_reason']=='low_disk_space'
    assert json.loads((parent/'status.json').read_text())==state


@pytest.mark.parametrize('alive',[101,102])
def test_active_writer_or_supervisor_is_rejected(closed,monkeypatch,alive):
    monkeypatch.setattr(recovery,'process_alive',lambda pid:pid==alive)
    with pytest.raises(ValueError,match='stopped'):
        recovery.validate_closed_run(closed[0])


def test_training_phase_is_rejected(closed):
    parent,_,state=closed;state['phase']='training'
    (parent/'status.json').write_text(json.dumps(state))
    with pytest.raises(ValueError,match='final upload'):
        recovery.validate_closed_run(parent)


def test_missing_writer_pid_is_rejected(closed):
    parent,_,state=closed;state.pop('training_pid')
    (parent/'status.json').write_text(json.dumps(state))
    with pytest.raises(ValueError,match='recorded'):
        recovery.validate_closed_run(parent)


def test_run_outside_managed_parent_is_rejected(closed,tmp_path):
    parent,_,state=closed;state['run_dir']=str(tmp_path.parent)
    (parent/'status.json').write_text(json.dumps(state))
    with pytest.raises(ValueError,match='direct non-symlink child'):
        recovery.validate_closed_run(parent)


def test_symlinked_run_is_rejected(closed):
    parent,run,state=closed;link=parent/'linked';link.symlink_to(run,target_is_directory=True)
    state['run_dir']=str(link);(parent/'status.json').write_text(json.dumps(state))
    with pytest.raises(ValueError,match='non-symlink'):
        recovery.validate_closed_run(parent)


@pytest.mark.parametrize('scope,checkpoint_only',[
    ('checkpoint_contract_logs_only',True),('pilot_payloads',False),(None,False)])
def test_final_retry_preserves_original_backup_scope(closed,monkeypatch,scope,checkpoint_only):
    parent,run,state=closed
    if scope is not None:(parent/'launch.json').write_text(json.dumps(dict(backup_scope=scope)))
    calls=[]
    monkeypatch.setattr(recovery,'archive_batched',lambda *a,**kw:calls.append((a,kw)))
    assert recovery.finalize_once(parent,'example:HumanoidScene-RL')==run
    assert calls==[((run,'example:HumanoidScene-RL',True),dict(checkpoint_log_only=checkpoint_only))]
    actual=json.loads((parent/'status.json').read_text())
    assert actual['training_exit_code']==state['training_exit_code']
    assert actual['phase']=='finished' and actual['final_upload_verified']
    assert actual['stop_reason']=='low_disk_space'


def test_unknown_scope_cannot_widen_payloads(closed,monkeypatch):
    parent,_,state=closed
    (parent/'launch.json').write_text(json.dumps(dict(backup_scope='unknown')))
    monkeypatch.setattr(recovery,'archive_batched',lambda *a,**kw:pytest.fail('Must not upload'))
    with pytest.raises(ValueError,match='scope'):recovery.finalize_once(parent,'example:HumanoidScene-RL')
    assert json.loads((parent/'status.json').read_text())==state


def test_failed_upload_retains_local_source_and_original_exit(closed,monkeypatch):
    parent,run,state=closed;checkpoint=run/'checkpoint_00000001.pt';checkpoint.write_bytes(b'not yet backed up')
    def fail(*args,**kwargs):raise RuntimeError('disconnected')
    monkeypatch.setattr(recovery,'archive_batched',fail)
    with pytest.raises(RuntimeError):recovery.finalize_once(parent,'example:HumanoidScene-RL')
    actual=json.loads((parent/'status.json').read_text())
    assert actual['training_exit_code']==state['training_exit_code']
    assert actual['phase']=='final_upload' and not actual.get('final_upload_verified')
    assert checkpoint.read_bytes()==b'not yet backed up'
