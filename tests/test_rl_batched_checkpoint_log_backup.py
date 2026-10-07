"""An explicit minimal backup must never export raw replay or rollout media."""
import hashlib
import json
import sys
from pathlib import Path
import pytest

import batched_staged_goal_with_drive as managed
import train_with_drive as storage


def test_checkpoint_log_scope_keeps_physical_payloads_local(tmp_path, monkeypatch):
    source=tmp_path/'unique_run';source.mkdir()
    allowed=('manifest.json','env.yaml','agent.yaml','verification.json',
             'checkpoint_00003072.pt','console.log','metrics.json','metrics.jsonl',
             'status.json','resources.jsonl','events.out.tfevents.example')
    local=('staged_goal_experience.pt','executed_transitions.hdf5',
           'eval_wave_0000_env_007_h264.mp4','eval_wave_0000_videos.json')
    for name in (*allowed,*local):
        (source/name).write_bytes(b'{}' if name.endswith('.json') else name.encode())
    original={name:(source/name).read_bytes() for name in local}
    transferred=[];checked=[]
    class Remote:
        def __init__(self, wrapper):pass
        def upload(self,path,destination):transferred.append(Path(path).name)
        def verify(self,path,destination,digest):
            assert digest==hashlib.md5(Path(path).read_bytes()).hexdigest()
            checked.append(Path(path).name)
    monkeypatch.setattr(storage,'Rclone',Remote)
    def forbidden(*args,**kwargs):raise AssertionError('Extended pilot exporter must not run')
    monkeypatch.setattr(managed,'archive_pilot',forbidden)
    result=managed.archive_batched(source,'test:HumanoidScene-RL',True,checkpoint_log_only=True)
    assert result==[] and set(transferred)==set(allowed) and transferred==checked
    assert all((source/name).read_bytes()==original[name] for name in local)


def test_existing_pilot_backup_scope_is_preserved(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(managed,'archive_pilot',lambda *args:calls.append(args) or ['verified-old.pt'])
    def forbidden(*args,**kwargs):raise AssertionError('Minimal exporter is opt-in')
    monkeypatch.setattr(managed,'archive',forbidden)
    assert managed.archive_batched(tmp_path,'test:HumanoidScene-RL',False)==['verified-old.pt']
    assert calls==[(tmp_path,'test:HumanoidScene-RL',False)]


@pytest.mark.parametrize('minimal',[False,True])
def test_manager_forwards_checkpoint_exactly_without_consuming_child_flag(tmp_path,monkeypatch,minimal):
    parent=tmp_path/'new_run';checkpoint=tmp_path/'checkpoint_00019396.pt';calls=[]
    def simulated(command,*args,**kwargs):calls.append(command);return 0
    monkeypatch.setattr(managed,'supervise',simulated)
    argv=['manager','--experiment-dir',str(parent),'--gpu','3','--python',sys.executable,
          '--remote-root','test:HumanoidScene-RL','--checkpoint',str(checkpoint),'--training']
    if minimal:argv.append('--checkpoint-log-backup-only')
    monkeypatch.setattr(sys,'argv',argv)
    assert managed.main()==0 and len(calls)==1
    command=calls[0]
    assert command[command.index('--checkpoint')+1]==str(checkpoint)
    assert '--checkpoint-log-backup-only' not in command and '--training' in command
    actual=json.loads((parent/'launch.json').read_text())
    assert actual['backup_scope']==('checkpoint_contract_logs_only' if minimal else 'pilot_payloads')
