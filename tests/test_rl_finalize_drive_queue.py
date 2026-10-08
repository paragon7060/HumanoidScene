"""A closed-run queue cannot broaden payloads or touch ongoing writers."""
import json
import pytest
import finalize_drive_queue as queue
import finalize_with_drive as recovery


@pytest.fixture
def queued(tmp_path,monkeypatch):
    parent=tmp_path/'managed';parent.mkdir();run=parent/'batch_sac_test';run.mkdir()
    (run/'manifest.json').write_text('{}')
    state=dict(phase='final_upload',training_pid=101,supervisor_pid=102,run_dir=str(run),training_exit_code=0)
    (parent/'status.json').write_text(json.dumps(state))
    (parent/'launch.json').write_text(json.dumps(dict(backup_scope='checkpoint_contract_logs_only')))
    entry=dict(experiment_dir=str(parent),run_dir=str(run),training_pid=101,supervisor_pid=102,
               backup_scope='checkpoint_contract_logs_only')
    path=tmp_path/'queue.json';path.write_text(json.dumps(dict(format='closed_managed_Drive_backups_v1',entries=[entry])))
    monkeypatch.setattr(recovery,'process_alive',lambda pid:False)
    return path,parent,run,entry


def test_queue_keeps_original_requested_identities(queued):
    path,parent,run,entry=queued
    assert queue.read_queue(path)==[entry]
    assert queue.validate_entry(entry)[:2]==(parent,run)


def test_live_writer_never_archived(queued,monkeypatch):
    entry=queued[3]
    monkeypatch.setattr(recovery,'process_alive',lambda pid:pid==101)
    monkeypatch.setattr(queue,'finalize_once',lambda *a:pytest.fail('Must not upload active logs'))
    monkeypatch.setattr(queue,'remote_ready',lambda *a:pytest.fail('No eligible jobs'))
    actual=queue.queue_pass([entry],'example:HumanoidScene-RL')
    assert actual['verified']==0
    assert actual['entries'][0]['state']=='waiting_original_processes_or_metadata'


def test_authentication_failure_preserves_originals_and_does_not_retry_each_job(queued,monkeypatch):
    _,_,run,entry=queued;payload=run/'executed_transitions.hdf5';payload.write_bytes(b'keep this')
    calls=[]
    monkeypatch.setattr(queue,'remote_ready',lambda *a:calls.append(a) or False)
    monkeypatch.setattr(queue,'finalize_once',lambda *a:pytest.fail('Unavailable existing remote'))
    actual=queue.queue_pass([entry],'example:HumanoidScene-RL')
    assert len(calls)==1 and actual['verified']==0
    assert actual['entries'][0]['state']=='waiting_existing_Drive_connection'
    assert payload.read_bytes()==b'keep this'


def test_scope_or_identity_changes_are_rejected(queued):
    entry=queued[3]
    with pytest.raises(ValueError,match='identity or payload scope'):
        queue.validate_entry(entry|dict(backup_scope='pilot_payloads'))
    with pytest.raises(ValueError,match='identity or payload scope'):
        queue.validate_entry(entry|dict(supervisor_pid=999))


def test_successful_queue_pass_uses_original_scope_and_preserves_training_result(queued,monkeypatch):
    _,parent,run,entry=queued
    calls=[]
    monkeypatch.setattr(queue,'remote_ready',lambda *a:True)
    monkeypatch.setattr(recovery,'archive_batched',lambda *a,**kw:calls.append(kw))
    actual=queue.queue_pass([entry],'example:HumanoidScene-RL')
    assert actual['verified']==1 and calls==[dict(checkpoint_log_only=True)]
    assert json.loads((parent/'status.json').read_text())['training_exit_code']==0
    queue.queue_pass([entry],'example:HumanoidScene-RL')
    assert len(calls)==1


def test_duplicate_parents_and_symlinked_queue_are_rejected(queued,tmp_path):
    path,_,_,entry=queued
    path.write_text(json.dumps(dict(format='closed_managed_Drive_backups_v1',entries=[entry,entry])))
    with pytest.raises(ValueError,match='Distinct'):queue.read_queue(path)
    linked=tmp_path/'link';linked.symlink_to(path)
    with pytest.raises(ValueError,match='non-symlink'):queue.read_queue(linked)
