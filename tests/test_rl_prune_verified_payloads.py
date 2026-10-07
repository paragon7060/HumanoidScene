import json
from pathlib import Path
import pytest
from prune_verified_payloads import prune_experiment


def experiment(tmp_path):
    parent=tmp_path/'managed';parent.mkdir();run=parent/'run_unique';run.mkdir()
    (parent/'launch.json').write_text(json.dumps(dict(run=str(run))))
    (parent/'status.json').write_text(json.dumps(dict(phase='finished',final_upload_verified=True,
        training_exit_code=0,training_pid=99999991,supervisor_pid=99999992)))
    (run/'manifest.json').write_text('{}')
    for name in ('staged_goal_experience.pt','checkpoint_00000001.pt','checkpoint_00000002.pt','console.log','eval.mp4'):
        (run/name).write_bytes(b'original')
    return parent,run


class Verified:
    def verify(self,path,destination,digest):
        assert destination=='test:HumanoidScene-RL/run_unique/staged_goal_experience.pt'
        assert len(digest)==32


def test_only_closed_verified_payload_removed_and_model_logs_media_kept(tmp_path,monkeypatch):
    parent,run=experiment(tmp_path)
    monkeypatch.setattr('prune_verified_payloads.process_inventory',lambda:([],set()))
    result=prune_experiment(parent,'test:HumanoidScene-RL',Verified())
    assert len(result['removed'])==1 and not (run/'staged_goal_experience.pt').exists()
    assert all((run/name).read_bytes()==b'original' for name in ('checkpoint_00000001.pt','checkpoint_00000002.pt','console.log','eval.mp4'))
    marker=json.loads((run/'local_payload_cleanup.json').read_text())
    assert marker['files'][0]['local_removal_complete']
    assert prune_experiment(parent,'test:HumanoidScene-RL',Verified())['removed']==[]


@pytest.mark.parametrize('condition',['active','not_verified','referenced','open','protected','checkpoint_only'])
def test_active_unverified_used_or_protected_payload_never_deleted(tmp_path,monkeypatch,condition):
    parent,run=experiment(tmp_path);file=run/'staged_goal_experience.pt'
    state=json.loads((parent/'status.json').read_text())
    if condition=='active':state['phase']='training'
    if condition=='not_verified':state['final_upload_verified']=False
    if condition=='checkpoint_only':
        (parent/'launch.json').write_text(json.dumps(dict(run=str(run),backup_scope='checkpoint_contract_logs_only')))
    (parent/'status.json').write_text(json.dumps(state))
    inode=(file.stat().st_dev,file.stat().st_ino)
    monkeypatch.setattr('prune_verified_payloads.process_inventory',lambda:([str(run).encode()] if condition=='referenced' else [], {inode} if condition=='open' else set()))
    class NeverVerify:
        def verify(self,*a):raise AssertionError('Protected payload must not be inspected remotely')
    result=prune_experiment(parent,'test:HumanoidScene-RL',NeverVerify(),protected=[file] if condition=='protected' else [])
    assert not result['removed'] and file.read_bytes()==b'original'


@pytest.mark.parametrize('condition',['remote_error','source_change'])
def test_failed_checksum_or_source_change_keeps_file(tmp_path,monkeypatch,condition):
    parent,run=experiment(tmp_path);file=run/'staged_goal_experience.pt'
    monkeypatch.setattr('prune_verified_payloads.process_inventory',lambda:([],set()))
    class Changed:
        def verify(self,*a):
            if condition=='remote_error':raise RuntimeError('checksum differs')
            file.write_bytes(b'changed')
    result=prune_experiment(parent,'test:HumanoidScene-RL',Changed())
    assert not result['removed'] and file.exists()


@pytest.mark.parametrize('condition',['matching','different_command','different_output','conflicting_run'])
def test_legacy_supervisor_requires_exact_run_command_binding(tmp_path,monkeypatch,condition):
    parent,run=experiment(tmp_path)
    command=['python','reference.py','--output-dir',str(run)]
    launch={'command':command}
    state=json.loads((parent/'status.json').read_text())
    state.update(run_dir=str(run),command=command.copy())
    if condition=='different_command':state['command'][1]='other.py'
    if condition=='different_output':
        launch['command'][-1]=str(parent/'other')
        state['command']=launch['command'].copy()
    if condition=='conflicting_run':launch['run']=str(parent/'other')
    (parent/'launch.json').write_text(json.dumps(launch))
    (parent/'status.json').write_text(json.dumps(state))
    monkeypatch.setattr('prune_verified_payloads.process_inventory',lambda:([],set()))
    if condition=='matching':
        assert len(prune_experiment(parent,'test:HumanoidScene-RL',Verified())['removed'])==1
    else:
        with pytest.raises(ValueError):
            prune_experiment(parent,'test:HumanoidScene-RL',Verified())
        assert (run/'staged_goal_experience.pt').exists()
