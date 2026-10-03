"""Restart races and stale launch pointers must not overlap GPU experiments."""
import json

import pytest

import queued_experiment as queue


@pytest.mark.parametrize('active,pid', [('active', 123), ('activating', 0),
                                      ('deactivating', 123), ('inactive', 123)])
def test_live_or_restarting_dependency_cannot_start_a_gpu_child(monkeypatch, tmp_path, active, pid):
    monkeypatch.setattr(queue, 'unit_state', lambda _: {'ActiveState': active, 'MainPID': str(pid)})
    assert queue.verified_dependency('our-unit', tmp_path/'absent.json') is None


def closed_attempt(tmp_path):
    run=tmp_path/'parent'/'run';run.mkdir(parents=True)
    launch=tmp_path/'launch.json'
    launch.write_text(json.dumps({'parent':str(run.parent),'run':str(run)}))
    status={'supervisor_pid':123,'phase':'finished','final_upload_verified':True,
            'training_exit_code':0,'run_dir':str(run)}
    (run.parent/'status.json').write_text(json.dumps(status))
    (run/'verification.json').write_text(json.dumps({'writers_stopped_at':'2026-10-03T19:00:00+09:00',
                                                     'training_exit_code':0}))
    return launch,run,status


def test_closed_verified_attempt_can_release_dependency(monkeypatch, tmp_path):
    launch,run,_=closed_attempt(tmp_path)
    monkeypatch.setattr(queue,'unit_state',lambda _:dict(ActiveState='inactive',MainPID='0',ExecMainPID='123'))
    assert queue.verified_dependency('our-unit',launch)['run_dir']==run


@pytest.mark.parametrize('field,value,message', [
    ('supervisor_pid',99,'different supervisor'),
    ('final_upload_verified',False,'verified backup'),
    ('training_exit_code',1,'execution failed'),
])
def test_failed_unarchived_or_stale_attempt_cannot_release_dependency(monkeypatch,tmp_path,field,value,message):
    launch,run,status=closed_attempt(tmp_path);status[field]=value
    (run.parent/'status.json').write_text(json.dumps(status))
    monkeypatch.setattr(queue,'unit_state',lambda _:dict(ActiveState='inactive',MainPID='0',ExecMainPID='123'))
    with pytest.raises(RuntimeError,match=message):queue.verified_dependency('our-unit',launch)
