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
