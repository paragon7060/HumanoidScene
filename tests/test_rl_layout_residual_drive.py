"""Training failures are real data; held-out trials cannot optimize the actor."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest


def module():
    scripts=Path(__file__).resolve().parents[1]/'scripts/rl';sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('layout_drive_test',scripts/'layout_residual_with_drive.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


@pytest.mark.parametrize('policy_mode',['pose-goal','staged-goal'])
def test_mixed_shelf_reset_provenance_keeps_final_policy_frozen(tmp_path,monkeypatch,policy_mode):
    script=module();layouts=tmp_path/'layouts';layouts.mkdir();parent=tmp_path/'suite';calls=[]
    mapping={'1100':0,'1300':1,'3200':0,'3300':1}
    for split,seeds in [('train',[1100,1300]),('holdout',[3200,3300])]:
        for i,seed in enumerate(seeds):
            (layouts/f'{split}_{i:02d}.json').write_text(json.dumps(dict(seed=seed,split=split,lateral_m=-.025)))
    mapping_file=tmp_path/'episodes.json';mapping_file.write_text(json.dumps(mapping))
    warm=tmp_path/'student.pt';warm.write_bytes(b'BC model')
    def simulate(command,trial,environment,*args,**kwargs):
        calls.append(command);run=Path(command[command.index('--output-dir')+1]);run.mkdir()
        layout=json.loads(Path(command[command.index('--layout-json')+1]).read_text())
        assert command.count('--episode-index')==1
        assert int(command[command.index('--episode-index')+1])==mapping[str(layout['seed'])]
        assert not any(value.startswith('--episode-index=') for value in command)
        staged=policy_mode=='staged-goal'
        train=('--staged-goal-training' if staged else '--pose-student-training') in command
        assert train==(layout['split']=='train') and '--residual-sac' not in command
        if staged:
            assert '--staged-goal-sac' in command and '--pose-student-training' not in command
        (trial/'status.json').write_text(json.dumps(dict(run_dir=str(run),training_exit_code=0,final_upload_verified=True)))
        (run/'metrics.json').write_text(json.dumps(dict(steps=410,
            outcomes=dict(success=1,unsafe=0,invalid_reset=0,time_out=0),
            **{('staged_goal_sac' if staged else 'pose_goal_sac'):
                dict(training=train,actor_updates=700*min(len(calls),2))})))
        if train:(run/f'checkpoint_{len(calls)*700:08d}.pt').write_bytes(b'goal SAC model')
        return 0
    monkeypatch.setattr(script,'supervise',simulate)
    monkeypatch.setattr(script,'Rclone',lambda *a:object())
    monkeypatch.setattr(script,'archive_file',lambda *a:None)
    monkeypatch.setattr(sys,'argv',['suite','--experiment-dir',str(parent),'--layout-dir',str(layouts),
        '--policy-mode',policy_mode,'--checkpoint',str(warm),'--reference-episode-map',str(mapping_file),
        '--train-count','2','--eval-count','2','--python',sys.executable,
        '--remote-root','test-remote:HumanoidScene-RL','--episode-index=7',
        '--pose-student-native-seed','actual_lower.hdf5','--pose-student-native-seed','actual_upper.hdf5',
        *(['--staged-base-waypoints','measured_templates.json'] if policy_mode=='staged-goal' else [])])
    assert script.main()==0 and len(calls)==4
    assert calls[2][calls[2].index('--pose-student-checkpoint')+1]==calls[3][calls[3].index('--pose-student-checkpoint')+1]
    manifest=json.loads((parent/'manifest.json').read_text())
    assert manifest['reference_episode_by_seed']==mapping
    assert manifest['reference_episodes_used_only_for_initial_scene'] is True
    assert manifest['physical_reference_dependency'] is False


@pytest.mark.parametrize('mapping',[{'1':0},{'1':True,'2':0},{'1':0,'2':-1},{'01':0,'2':1},[0,1]])
def test_reference_episode_map_rejects_missing_or_ambiguous_reset_sources(tmp_path,mapping):
    script=module();files=[]
    for seed in [1,2]:
        p=tmp_path/f'{seed}.json';p.write_text(json.dumps({'seed':seed}));files.append(p)
    p=tmp_path/'map.json';p.write_text(json.dumps(mapping))
    with pytest.raises(ValueError):script.reference_episode_map(p,{'train':files})
    arguments=['--episode-index','0','--steps','900']
    assert script.child_reference_episode(arguments,None)==arguments


def test_upper_gain_does_not_hide_lower_regression():
    script=module()
    rows=[dict(reference_episode_index=0,outcomes={'success':1}),
          dict(reference_episode_index=0,outcomes={'success':0}),
          dict(reference_episode_index=1,outcomes={'success':1}),
          dict(reference_episode_index=1,outcomes={'success':0})]
    latest=script.episode_validation_successes(rows)
    best={'0':2,'1':0}
    assert sum(latest.values())==sum(best.values())
    assert script.episode_validation_regressed(latest,best,0)
    assert not script.episode_validation_regressed({'0':2,'1':1},best,0)
    assert not script.episode_validation_regressed(latest,None,0)


def test_requested_stop_never_launches_another_layout(tmp_path,monkeypatch):
    script=module();layouts=tmp_path/'layouts';layouts.mkdir();parent=tmp_path/'suite';calls=[]
    for split in ('train','holdout'):
        for index in range(2):
            (layouts/f'{split}_{index:02d}.json').write_text(json.dumps(
                dict(seed=index,split=split,lateral_m=-.025)))
    def stop(command,trial,*args,**kwargs):
        calls.append(command)
        (trial/'status.json').write_text(json.dumps(dict(training_exit_code=0,
            final_upload_verified=True,stop_reason='requested_stop')))
        return 0
    monkeypatch.setattr(script,'supervise',stop)
    monkeypatch.setattr(sys,'argv',['suite','--experiment-dir',str(parent),'--layout-dir',str(layouts),
        '--train-count','2','--eval-count','2','--python',sys.executable,
        '--remote-root','test-remote:HumanoidScene-RL','--residual-sac'])
    assert script.main()==0 and len(calls)==1
    assert json.loads((parent/'status.json').read_text())['phase']=='stopped'


def test_right_gain_does_not_hide_left_regression_on_the_same_shelf():
    script=module()
    rows=[dict(reference_episode_index=0,layout={'target_region':'shelf_2_left'},outcomes={'success':0}),
          dict(reference_episode_index=0,layout={'target_region':'shelf_2_right'},outcomes={'success':1})]
    latest=script.episode_validation_successes(rows)
    best={'0:shelf_2_left':1,'0:shelf_2_right':0}
    assert sum(latest.values())==sum(best.values())
    assert script.episode_validation_regressed(latest,best,0)


def test_missing_native_seed_audit_is_rejected_before_starting_isaac(tmp_path,monkeypatch):
    script=module();checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_bytes(b'native SAC')
    (tmp_path/'manifest.json').write_text(json.dumps({'artifact_type':'pose_goal_sac_no_live_reference'}))
    parent=tmp_path/'must_not_start'
    monkeypatch.setattr(sys,'argv',['suite','--experiment-dir',str(parent),'--layout-dir',str(tmp_path),
        '--policy-mode','pose-goal','--evaluation-only','--checkpoint',str(checkpoint),
        '--python',sys.executable,'--remote-root','test-remote:HumanoidScene-RL'])
    with pytest.raises(SystemExit) as error:script.main()
    assert error.value.code==2 and not parent.exists()


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


def test_bc_comparison_preserves_box_and_base_randomization_without_updates(tmp_path,monkeypatch):
    script=module();layouts=tmp_path/'layouts';layouts.mkdir();parent=tmp_path/'suite';calls=[]
    for i in range(2):
        (layouts/f'holdout_{i:02d}.json').write_text(json.dumps(dict(seed=i,split='holdout',
            lateral_m=-.025-.005*i,yaw_rad=.01,base_lateral_m=.12,base_outward_m=.18,base_yaw_rad=.2)))
    frozen=tmp_path/'student.pt';frozen.write_bytes(b'frozen BC')
    previous=tmp_path/'previous';previous.mkdir()
    (previous/'status.json').write_text(json.dumps(dict(phase='finished',final_upload_verified=True)))
    def simulate(command,trial,environment,*args,**kwargs):
        calls.append(command);run=Path(command[command.index('--output-dir')+1]);run.mkdir()
        assert '--no-pose-student-training' in command and '--pose-student-training' not in command
        assert command[command.index('--pose-student-checkpoint')+1]==str(frozen)
        assert '--pose-student-native-seed' not in command
        (trial/'status.json').write_text(json.dumps(dict(run_dir=str(run),training_exit_code=0,final_upload_verified=True)))
        (run/'metrics.json').write_text(json.dumps(dict(steps=412,
            outcomes=dict(success=1,unsafe=0,invalid_reset=0,time_out=0),
            pose_student=dict(sac_actor_updates=0,sac_critic_updates=0))))
        return 0
    monkeypatch.setattr(script,'supervise',simulate)
    monkeypatch.setattr(script,'Rclone',lambda *a:object())
    monkeypatch.setattr(script,'archive_file',lambda *a:None)
    monkeypatch.setattr(sys,'argv',['suite','--experiment-dir',str(parent),'--layout-dir',str(layouts),
        '--policy-mode','pose-goal','--checkpoint',str(frozen),'--evaluation-only',
        '--layout-distribution','initial-base-and-box','--eval-count','2','--python',sys.executable,
        '--wait-for-verified-experiment',str(previous),'--remote-root','test-remote:HumanoidScene-RL'])
    assert script.main()==0 and len(calls)==2
    manifest=json.loads((parent/'manifest.json').read_text())
    assert not manifest['train_layouts'] and manifest['initial_base_distribution']
    assert abs(manifest['heldout_layouts'][1]['lateral_m']+.03)<1e-7
    assert json.loads((parent/'status.json').read_text())['holdout_successes']==2


def test_development_regression_recovers_before_unseen_final_holdout(tmp_path,monkeypatch):
    script=module();layouts=tmp_path/'layouts';layouts.mkdir();dev=tmp_path/'dev';dev.mkdir()
    parent=tmp_path/'suite';calls=[];recoveries=[]
    for split,seeds in [('train',[1]),('holdout',[30,31])]:
        for i,seed in enumerate(seeds):
            (layouts/f'{split}_{i:02d}.json').write_text(json.dumps(dict(seed=seed,split=split,lateral_m=-.025)))
    for i in range(2):
        (dev/f'holdout_{i:02d}.json').write_text(json.dumps(dict(seed=10+i,split='holdout',lateral_m=-.025)))
    warm=tmp_path/'warm.pt';warm.write_bytes(b'known actor')
    def simulate(command,trial,environment,*args,**kwargs):
        calls.append(command);run=Path(command[command.index('--output-dir')+1]);run.mkdir()
        train='--pose-student-training' in command
        failure=len(calls) in (4,5)
        (trial/'status.json').write_text(json.dumps(dict(final_upload_verified=True)))
        (run/'metrics.json').write_text(json.dumps(dict(steps=900 if failure else 410,
            outcomes=dict(success=int(not failure),unsafe=0,time_out=int(failure)),
            pose_goal_sac=dict(training=train,actor_updates=100))))
        if train:(run/'checkpoint_00000100.pt').write_bytes(b'latest actor')
        return 0
    def recovery(checkpoint,best,output,python):
        recoveries.append((checkpoint,best));output.mkdir(parents=True)
        result=output/'checkpoint_00000100.pt';result.write_bytes(b'restored actor and latest Q')
        return result
    monkeypatch.setattr(script,'supervise',simulate)
    monkeypatch.setattr(script,'recover_actor',recovery)
    monkeypatch.setattr(script,'archive_pilot',lambda *a:None)
    monkeypatch.setattr(script,'Rclone',lambda *a:object())
    monkeypatch.setattr(script,'archive_file',lambda *a:None)
    monkeypatch.setattr(sys,'argv',['suite','--experiment-dir',str(parent),'--layout-dir',str(layouts),
        '--policy-mode','pose-goal','--checkpoint',str(warm),'--train-count','1','--eval-count','2',
        '--validation-layout-dir',str(dev),'--validation-every','1','--validation-count','2',
        '--python',sys.executable,'--remote-root','test-remote:HumanoidScene-RL',
        '--pose-student-native-seed','measured.hdf5'])
    assert script.main()==0 and len(calls)==7 and len(recoveries)==1
    assert recoveries[0][1]==warm
    assert all('--no-pose-student-training' in calls[i] for i in [0,1,3,4,5,6])
    assert all('actor_recoveries' in c[c.index('--pose-student-checkpoint')+1] for c in calls[5:])
    rows=json.loads((parent/'results.json').read_text())
    assert sum(r['outcomes']['time_out'] for r in rows)==2
    assert [r['layout']['seed'] for r in rows if r['split']=='holdout']==[30,31]
    assert json.loads((parent/'status.json').read_text())['actor_recoveries']==1
