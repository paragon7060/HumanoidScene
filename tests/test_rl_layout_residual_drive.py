"""Training failures are real data; held-out trials cannot optimize the actor."""
import importlib.util
import json
from pathlib import Path
import sys


def module():
    scripts=Path(__file__).resolve().parents[1]/'scripts/rl';sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('layout_drive_test',scripts/'layout_residual_with_drive.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def test_varied_training_keeps_failure_data_then_freezes_one_checkpoint_for_holdout(tmp_path,monkeypatch):
    script=module();layouts=tmp_path/'layouts';layouts.mkdir();parent=tmp_path/'suite';calls=[];uploads=[]
    for split in ('train','holdout'):
        for i in range(2):
            (layouts/f'{split}_{i:02d}.json').write_text(json.dumps(dict(seed=i,split=split,lateral_m=-.02-.01*i)))
    def simulate(command,trial,environment,*args,**kwargs):
        calls.append(command);run=Path(command[command.index('--output-dir')+1]);run.mkdir()
        train='--residual-training' in command
        (trial/'status.json').write_text(json.dumps(dict(run_dir=str(run),training_exit_code=0,final_upload_verified=True)))
        (run/'metrics.json').write_text(json.dumps(dict(steps=100,
            outcomes=dict(success=int(len(calls)>1),unsafe=int(len(calls)==1),invalid_reset=0,time_out=0),
            residual_sac=dict(training=train,actor_updates=min(len(calls),2)*100))))
        if train:(run/f'checkpoint_{len(calls)*100:08d}.pt').write_bytes(b'actual model')
        assert environment['CUDA_VISIBLE_DEVICES']=='3'
        frozen=Path(command[command.index('--layout-json')+1]);assert frozen.parent==parent/'layouts'
        return 0
    monkeypatch.setattr(script,'supervise',simulate)
    monkeypatch.setattr(script,'Rclone',lambda *a:object())
    monkeypatch.setattr(script,'archive_file',lambda path,*args:uploads.append(path.name))
    monkeypatch.setattr(sys,'argv',['suite','--experiment-dir',str(parent),'--layout-dir',str(layouts),
        '--train-count','2','--eval-count','2','--python',sys.executable,
        '--remote-root','test-remote:HumanoidScene-RL','--residual-sac'])
    assert script.main()==0 and len(calls)==4
    assert all('--no-residual-training' in c and '--residual-training' not in c for c in calls[2:])
    checkpoints=[c[c.index('--residual-checkpoint')+1] for c in calls[2:]]
    assert checkpoints[0]==checkpoints[1]
    state=json.loads((parent/'status.json').read_text())
    assert state['train_unsafe']==1 and state['holdout_successes']==2
    assert state['final_upload_verified'] and uploads.count('status.json')==1


def test_pose_goal_suite_uses_one_frozen_model_and_no_reference_residual(tmp_path,monkeypatch):
    script=module();layouts=tmp_path/'layouts';layouts.mkdir();parent=tmp_path/'suite';calls=[]
    for split,count in [('train',1),('holdout',2)]:
        for i in range(count):
            (layouts/f'{split}_{i:02d}.json').write_text(json.dumps(dict(seed=i,split=split,lateral_m=-.025)))
    warm=tmp_path/'student.pt';warm.write_bytes(b'bc warm start')
    def simulate(command,trial,environment,*args,**kwargs):
        calls.append(command);run=Path(command[command.index('--output-dir')+1]);run.mkdir()
        train='--pose-student-training' in command
        assert environment['CUDA_VISIBLE_DEVICES']=='0' and kwargs['run_prefix']=='pose_sac_'
        assert '--residual-controller' not in command and '--residual-checkpoint' not in command
        (trial/'status.json').write_text(json.dumps(dict(run_dir=str(run),training_exit_code=0,final_upload_verified=True)))
        (run/'metrics.json').write_text(json.dumps(dict(steps=411,
            outcomes=dict(success=1,unsafe=0,invalid_reset=0,time_out=0),
            pose_goal_sac=dict(training=train,actor_updates=696))))
        if train:(run/'checkpoint_00000696.pt').write_bytes(b'executed goal SAC')
        return 0
    monkeypatch.setattr(script,'supervise',simulate)
    monkeypatch.setattr(script,'Rclone',lambda *a:object())
    monkeypatch.setattr(script,'archive_file',lambda *a:None)
    monkeypatch.setattr(sys,'argv',['suite','--experiment-dir',str(parent),'--layout-dir',str(layouts),
        '--policy-mode','pose-goal','--checkpoint',str(warm),'--gpu','0',
        '--train-count','1','--eval-count','2','--python',sys.executable,
        '--remote-root','test-remote:HumanoidScene-RL','--pose-student-native-seed','measured.hdf5'])
    assert script.main()==0 and len(calls)==3
    assert all('--no-pose-student-training' in c and '--pose-student-training' not in c for c in calls[1:])
    assert calls[1][calls[1].index('--pose-student-checkpoint')+1]==calls[2][calls[2].index('--pose-student-checkpoint')+1]
    assert json.loads((parent/'manifest.json').read_text())['physical_reference_dependency'] is False
