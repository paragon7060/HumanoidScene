#!/usr/bin/env python3
"""Sequential layout SAC training, then frozen held-out evaluation on one GPU.

Uses the shared five-minute verified Drive lifecycle. Task failures are data;
runtime/backup failures stop this batch. No optimizer updates occur on holdout.
"""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

from drive_backup import ROOT, Rclone, archive_file
from reference_residual_with_drive import archive_pilot
from train_with_drive import supervise, write_status


def validation_regressed(successes, best_successes, allowed_drop):
    return successes < best_successes-allowed_drop


def episode_validation_successes(rows):
    result={}
    for row in rows:
        key=str(row['reference_episode_index'])
        region=row.get('layout',{}).get('target_region')
        if region is not None:
            key+=':'+region
        result[key]=result.get(key,0)+int(row['outcomes']['success'])
    return result


def episode_validation_regressed(latest, best, allowed_drop):
    return best is not None and any(
        validation_regressed(latest.get(key,0),value,allowed_drop)
        for key,value in best.items())


def reference_episode_map(path, layouts):
    """Explicit reset-scene provenance for a mixed-shelf learned goal policy."""
    if path is None:
        return None
    mapping=json.loads(Path(path).read_text())
    if not isinstance(mapping,dict) or any(
            not isinstance(key,str) or not key.isdecimal() or str(int(key))!=key
            or type(value) is not int or value<0 for key,value in mapping.items()):
        raise ValueError('Reference episode map must contain nonnegative seed and episode integers')
    seeds={str(json.loads(layout.read_text())['seed'])
           for group in layouts.values() for layout in group}
    if not seeds.issubset(mapping):
        raise ValueError('Every train/development/final layout needs an explicit reference episode')
    return {seed:mapping[seed] for seed in sorted(seeds,key=int)}


def child_reference_episode(arguments, episode):
    """Replace only the reset-scene episode; preserve every policy argument."""
    if episode is None:
        return list(arguments)
    result=[];index=0
    while index<len(arguments):
        value=arguments[index]
        if value=='--episode-index':
            if index+1>=len(arguments):raise ValueError('Missing reference episode argument')
            index+=2
        elif value.startswith('--episode-index='):
            index+=1
        else:
            result.append(value);index+=1
    return result+['--episode-index',str(episode)]


def recover_actor(checkpoint, best_checkpoint, output_dir, python):
    """Restore a validated actor while retaining latest measured replay and Q."""
    command=[str(python),str(ROOT/'scripts/rl/recover_pose_goal_actor.py'),
             '--checkpoint',str(checkpoint),'--best-checkpoint',str(best_checkpoint),
             '--output-dir',str(output_dir)]
    subprocess.run(command,check=True)
    return sorted(output_dir.glob('checkpoint_*.pt'))[-1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir',type=Path,required=True)
    parser.add_argument('--layout-dir',type=Path,required=True)
    parser.add_argument('--policy-mode',choices=('reference-residual','pose-goal'),default='reference-residual',
                        help='Pose-goal mode trains a learned BC-warmed SAC actor without a live demo path.')
    parser.add_argument('--train-count',type=int,default=8)
    parser.add_argument('--evaluation-only',action='store_true',
                        help='Evaluate a frozen BC/goal-SAC policy on heldout layouts without training.')
    parser.add_argument('--layout-distribution',choices=('inward-box','initial-base','initial-base-and-box'),default='inward-box',
                        help='Initial-base keeps boxes fixed and varies robot XY/yaw at episode start.')
    parser.add_argument('--max-initial-base-offset-m',type=float,default=.25)
    parser.add_argument('--max-initial-base-yaw-deg',type=float,default=15.)
    parser.add_argument('--max-layout-depth-m',type=float,default=0.,
                        help='Explicit allowed depth displacement for this fixed layout distribution; default preserves lateral-only runs.')
    parser.add_argument('--eval-count',type=int,default=8)
    parser.add_argument('--passes',type=int,default=1)
    parser.add_argument('--validation-layout-dir',type=Path,
                        help='Reusable development layouts; keep final holdout seeds separate.')
    parser.add_argument('--validation-every',type=int,default=4)
    parser.add_argument('--validation-count',type=int,default=4)
    parser.add_argument('--allowed-validation-success-drop',type=int,default=0)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--reference-episode-map',type=Path,
                        help='Pose-goal only: JSON seed->demo episode for the reset scene of mixed lower/upper layouts; no live path.')
    parser.add_argument('--after-verified-experiment',type=Path,
                        help='Wait for this own pilot series to finish verified; inherit its final checkpoint.')
    parser.add_argument('--wait-for-verified-experiment',type=Path,
                        help='Wait for a prior own suite to close and verify its backup before using this GPU; keep the explicit checkpoint.')
    parser.add_argument('--gpu',type=int,default=3)
    parser.add_argument('--python',type=Path,default=Path.home()/'miniconda3/envs/env_isaaclab_232/bin/python')
    parser.add_argument('--remote-root',default=os.environ.get('RL_DRIVE_REMOTE_ROOT'))
    args,child_args=parser.parse_known_args()
    if args.evaluation_only:
        args.train_count=0
    if args.checkpoint and args.after_verified_experiment:
        parser.error('Choose an explicit checkpoint or an earlier verified experiment')
    if args.wait_for_verified_experiment and args.after_verified_experiment:
        parser.error('Choose one dependency mode')
    if (args.train_count<0 or (not args.evaluation_only and args.train_count<1)
            or min(args.eval_count,args.passes)<1 or args.gpu<0 or not args.python.is_file()):
        parser.error('Positive split sizes/passes and a valid Isaac Python are required')
    if not math.isfinite(args.max_layout_depth_m) or not 0<=args.max_layout_depth_m<=.02:
        parser.error('Maximum layout depth must be within0..2cm')
    if (not math.isfinite(args.max_initial_base_offset_m) or not 0<=args.max_initial_base_offset_m<=.25
            or not math.isfinite(args.max_initial_base_yaw_deg) or not 0<=args.max_initial_base_yaw_deg<=15):
        parser.error('Initial base bounds must be within25cm and15degrees')
    reserved={'--output-dir','--device','--layout-json','--residual-checkpoint',
              '--residual-controller','--residual-training','--no-residual-training','--residual-zero',
              '--pose-student-checkpoint','--pose-student-training','--no-pose-student-training'}
    if any(value.split('=')[0] in reserved for value in child_args):
        parser.error('This supervisor owns layout, checkpoint, device and training/evaluation mode')
    pose_mode=args.policy_mode=='pose-goal'
    if args.reference_episode_map and not pose_mode:
        parser.error('Mixed reference episodes require a learned pose-goal policy')
    guarded=args.validation_layout_dir is not None
    if guarded and (not pose_mode or args.evaluation_only or args.validation_every<1
                    or args.validation_count<1 or args.allowed_validation_success_drop<0
                    or args.allowed_validation_success_drop>=args.validation_count):
        parser.error('Development validation requires training goal-SAC and valid counts/drop')
    if not pose_mode and '--residual-sac' not in child_args:
        parser.error('Supply --residual-sac and the current measured reference/physical manifest')
    if args.evaluation_only and (not pose_mode or not args.checkpoint):
        parser.error('Frozen evaluation needs pose-goal mode and an explicit BC/goal-SAC checkpoint')
    if pose_mode and (not args.checkpoint or (not args.evaluation_only and '--pose-student-native-seed' not in child_args)
                      or '--residual-sac' in child_args or args.after_verified_experiment):
        parser.error('Pose-goal SAC needs its own checkpoint and measured native seed, without residual options')
    if pose_mode and args.checkpoint:
        checkpoint_manifest=args.checkpoint.parent/'manifest.json'
        if checkpoint_manifest.is_file():
            checkpoint_type=json.loads(checkpoint_manifest.read_text()).get('artifact_type')
            has_seed_audit=any(value.split('=')[0]=='--pose-student-native-seed' for value in child_args)
            if checkpoint_type=='pose_goal_sac_no_live_reference' and not has_seed_audit:
                parser.error('Frozen goal-SAC evaluation also needs its declared physical native seed audit')
    layouts={split:sorted(args.layout_dir.resolve().glob(split+'_*.json'))[:count]
             for split,count in [('train',args.train_count),('holdout',args.eval_count)]}
    if guarded:
        layouts['validation']=sorted(args.validation_layout_dir.resolve().glob('holdout_*.json'))[:args.validation_count]
        if len(layouts['validation'])!=args.validation_count:
            parser.error('Not enough development validation layouts')
        validation_records=[json.loads(p.read_text()) for p in layouts['validation']]
        final_records=[json.loads(p.read_text()) for p in layouts['holdout']]
        if (any(r['split']!='holdout' for r in validation_records)
                or {r['seed'] for r in validation_records}&{r['seed'] for r in final_records}):
            parser.error('Development layouts must have a separate namespace from final holdout')
    expected_counts=[('train',args.train_count),('holdout',args.eval_count)]
    if guarded:expected_counts.append(('validation',args.validation_count))
    for split,count in expected_counts:
        expected_split='holdout' if split=='validation' else split
        if len(layouts[split])!=count or any(json.loads(path.read_text())['split']!=expected_split for path in layouts[split]):
            parser.error('Each requested layout must have its correct explicit split')
        for path in layouts[split]:
            layout=json.loads(path.read_text())
            if args.layout_distribution=='initial-base':
                if any(abs(layout.get(key,0))>1e-7 for key in ('lateral_m','yaw_rad','depth_m')):
                    parser.error('Initial-base experiments keep target boxes fixed')
            elif not -.040001<=layout['lateral_m']<=-.019999 or abs(layout.get('yaw_rad',0))>math.radians(1)+1e-7:
                parser.error('The footprint-valid sampling distribution is inward2..4cm and yaw+/-1degree')
            if (any(not math.isfinite(layout.get(key,0)) or abs(layout.get(key,0))>args.max_initial_base_offset_m+1e-7
                    for key in ('base_lateral_m','base_outward_m'))
                    or not math.isfinite(layout.get('base_yaw_rad',0))
                    or abs(layout.get('base_yaw_rad',0))>math.radians(args.max_initial_base_yaw_deg)+1e-7):
                parser.error('Initial base pose exceeds the declared fixed distribution')
            if args.layout_distribution=='inward-box' and any(layout.get(key,0) for key in
                    ('base_lateral_m','base_outward_m','base_yaw_rad')):
                parser.error('Use initial-base distribution explicitly to move the robot start')
            if not math.isfinite(layout.get('depth_m',0.)) or abs(layout.get('depth_m',0.))>args.max_layout_depth_m+1e-7:
                parser.error('Layout depth exceeds the explicitly allowed fixed distribution')
    try:
        episodes=reference_episode_map(args.reference_episode_map,layouts)
    except (ValueError,OSError) as exc:
        parser.error(str(exc))
    if not args.remote_root:
        remotes=subprocess.run(['bash',str(ROOT/'scripts/rl/gdrive.sh'),'listremotes'],
                               check=True,capture_output=True,text=True).stdout.splitlines()
        if len(remotes)!=1:
            parser.error('Set RL_DRIVE_REMOTE_ROOT to the existing host-local remote')
        args.remote_root=remotes[0]+'HumanoidScene-RL'
    if not re.fullmatch(r'[\w-]+:HumanoidScene-RL',args.remote_root.rstrip('/')):
        parser.error('Use the existing remote and dedicated HumanoidScene-RL root')
    parent=args.experiment_dir.resolve();parent.mkdir(parents=True,exist_ok=False)
    frozen=parent/'layouts';frozen.mkdir()
    for split,paths in layouts.items():
        copies=[]
        for path in paths:
            copy=frozen/(('validation_' if split=='validation' else '')+path.name)
            copy.write_text(path.read_text());copies.append(copy)
        layouts[split]=copies
    recipe=args.layout_dir.resolve()/'recipe.json'
    (parent/'manifest.json').write_text(json.dumps(dict(artifact_type=('layout_pose_goal_sac_suite' if pose_mode else 'layout_reference_residual_sac_suite'),
        gpu=args.gpu,train_layouts=[json.loads(p.read_text()) for p in layouts['train']],
        reference_episode_by_seed=episodes,
        reference_episodes_used_only_for_initial_scene=bool(episodes),
        heldout_layouts=[json.loads(p.read_text()) for p in layouts['holdout']],
        validation_layouts=[json.loads(p.read_text()) for p in layouts.get('validation',[])],
        validation_every=args.validation_every if guarded else None,
        allowed_validation_success_drop=args.allowed_validation_success_drop if guarded else None,
        validation_reuses_development_only=guarded,
        passes=args.passes,physical_reference_dependency=not pose_mode,curriculum=False,
        evaluation_only=args.evaluation_only,frozen_checkpoint=str(args.checkpoint.resolve()) if args.evaluation_only else None,
        wait_for_verified_experiment=str(args.wait_for_verified_experiment.resolve()) if args.wait_for_verified_experiment else None,
        initial_base_distribution=args.layout_distribution!='inward-box',
        max_initial_base_offset_m=args.max_initial_base_offset_m,
        max_initial_base_yaw_deg=args.max_initial_base_yaw_deg,
        layout_recipe=json.loads(recipe.read_text()) if recipe.exists() else None,
        sampled_distribution=('random_boxes_initial_base_XY_yaw_v1' if args.layout_distribution=='initial-base-and-box' else
                              'fixed_boxes_initial_base_XY_yaw_v1' if args.layout_distribution=='initial-base' else
                              'lower_small_inward2to4cm_yaw1deg_depth_rear_upper_distractors_v3'
                              if args.max_layout_depth_m else 'lower_small_inward2to4cm_yaw1deg_rear_upper_distractors_v2'),
        max_layout_depth_m=args.max_layout_depth_m,
        sampling_reason='2..6cm crossed the assigned half-shelf boundary; physical reset bounds unchanged'),indent=2)+'\n')
    environment=os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OMNI_KIT_ACCEPT_EULA='YES',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'scripts/rl'))
    checkpoint=args.checkpoint.resolve() if args.checkpoint else None
    if args.wait_for_verified_experiment:
        previous=args.wait_for_verified_experiment.resolve()
        write_status(parent,phase='waiting_for_verified_dependency',previous_experiment=str(previous))
        while True:
            state=json.loads((previous/'status.json').read_text())
            if state.get('phase')=='finished' and state.get('final_upload_verified'):
                break
            if state.get('phase') in {'failed','performance_gate_failed','runtime_failed','previous_pilot_failed'}:
                write_status(parent,phase='dependency_failed',previous_experiment=str(previous))
                return 2
            time.sleep(30)
    if args.after_verified_experiment:
        previous=args.after_verified_experiment.resolve()
        write_status(parent,phase='waiting_for_verified_pilot',previous_experiment=str(previous))
        while True:
            state=json.loads((previous/'status.json').read_text())
            phase=state.get('phase')
            if phase=='finished':
                if state.get('training_exit_code')!=0 or not state.get('final_upload_verified') or not state.get('episodes_verified'):
                    raise RuntimeError('The earlier pilot did not finish with verified frozen success')
                checkpoint=Path(state['latest_checkpoint'])
                if not checkpoint.is_file():
                    raise RuntimeError('Earlier verified checkpoint is missing')
                break
            if phase in {'failed','performance_gate_failed','runtime_failed'}:
                write_status(parent,phase='previous_pilot_failed',previous_experiment=str(previous))
                return 2
            time.sleep(30)
    rows=[]
    training_schedule=[('train',p,i) for i in range(args.passes) for p in layouts['train']]
    schedule=[]
    if guarded:schedule.extend(('validation',p,-1) for p in layouts['validation'])
    for count,item in enumerate(training_schedule,1):
        schedule.append(item)
        if guarded and (count%args.validation_every==0 or count==len(training_schedule)):
            schedule.extend(('validation',p,count) for p in layouts['validation'])
    schedule += [('holdout',p,0) for p in layouts['holdout']]
    best_checkpoint=None;best_successes=-1;best_episode_successes=None;recoveries=0
    for index,(split,layout,pass_index) in enumerate(schedule):
        trial=parent/f'{index+1:03d}_{split}_p{pass_index+1}_{layout.stem}';trial.mkdir()
        prefix='pose_sac_' if pose_mode else 'residual_'
        run=trial/(prefix+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        extras=(['--layout-json',str(layout),'--pose-student-training' if split=='train' else '--no-pose-student-training']
                if pose_mode else ['--residual-controller','retargeted-goal','--layout-json',str(layout),
                                  '--residual-training' if split=='train' else '--no-residual-training'])
        if checkpoint:
            extras.extend(('--pose-student-checkpoint' if pose_mode else '--residual-checkpoint',str(checkpoint)))
        episode=episodes[str(json.loads(layout.read_text())['seed'])] if episodes else None
        arguments=child_reference_episode(child_args,episode)
        command=[str(args.python),'-u',str(ROOT/'scripts/rl/replay_v2_grasp_reference.py'),
                 *arguments,*extras,'--output-dir',str(run),'--device','cuda:0','--headless']
        (trial/'launch.json').write_text(json.dumps(dict(command=command,gpu=args.gpu),indent=2)+'\n')
        write_status(parent,phase='training' if split=='train' else
                     'development_validation' if split=='validation' else 'heldout_evaluation',
                     active_trial=str(trial),completed_trials=len(rows),total_trials=len(schedule),
                     latest_checkpoint=str(checkpoint) if checkpoint else None)
        code=supervise(command,trial,environment,
            lambda source,finished:archive_pilot(source,args.remote_root,finished),
            interval=300,run_prefix=prefix,require_run_status=True)
        state=json.loads((trial/'status.json').read_text())
        if code!=0 or not state.get('final_upload_verified'):
            write_status(parent,phase='runtime_failed',failed_trial=str(trial),training_exit_code=code)
            return code or 2
        metrics=json.loads((run/'metrics.json').read_text())
        if pose_mode and args.evaluation_only and 'pose_student' in metrics:
            if metrics['pose_student']['sac_actor_updates']!=0 or metrics['pose_student']['sac_critic_updates']!=0:
                raise RuntimeError('BC comparison unexpectedly performed RL updates')
            learning=dict(training=False,actor_updates=0)
        else:
            learning=metrics['pose_goal_sac' if pose_mode else 'residual_sac']
        if split!='train' and learning['training']:
            raise RuntimeError('Frozen validation/evaluation unexpectedly enabled optimizer updates')
        row=dict(split=split,layout=json.loads(layout.read_text()),pass_index=pass_index,
                 run_dir=str(run),outcomes=metrics['outcomes'],steps=metrics['steps'],
                 actor_updates=learning['actor_updates'],final_upload_verified=True)
        if episodes:row['reference_episode_index']=episode
        rows.append(row)
        if split=='train':
            checkpoints=sorted(run.glob('checkpoint_*.pt'))
            if not checkpoints:
                raise RuntimeError('Completed training did not save a checkpoint')
            checkpoint=checkpoints[-1]
        if split=='validation' and (index+1==len(schedule) or schedule[index+1][0]!='validation'):
            block=rows[-args.validation_count:]
            successes=sum(r['outcomes']['success'] for r in block)
            episode_successes=episode_validation_successes(block) if episodes else None
            if (validation_regressed(successes,best_successes,args.allowed_validation_success_drop)
                    or (episodes and episode_validation_regressed(episode_successes,best_episode_successes,
                                                                  args.allowed_validation_success_drop))):
                recovery=parent/'actor_recoveries'/('recovery_'+uuid.uuid4().hex[:10])
                recovery.parent.mkdir(exist_ok=True)
                checkpoint=recover_actor(checkpoint,best_checkpoint,recovery,args.python)
                # The recovery has no live writers: archive real replay as well
                # as the recovered checkpoint before allowing another rollout.
                archive_pilot(recovery,args.remote_root,True)
                recoveries+=1
            elif successes>=best_successes:
                best_successes=successes;best_checkpoint=checkpoint
                best_episode_successes=episode_successes
            write_status(parent,latest_validation_successes=successes,
                         best_validation_successes=best_successes,
                         best_validation_checkpoint=str(best_checkpoint),actor_recoveries=recoveries)
            if episodes:
                write_status(parent,latest_validation_by_episode=episode_successes,
                             best_validation_by_episode=best_episode_successes)
        (parent/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
        for group in ('train','validation','holdout'):
            subset=[r for r in rows if r['split']==group]
            write_status(parent,**{group+'_attempts':len(subset),
                group+'_successes':sum(r['outcomes']['success'] for r in subset),
                group+'_unsafe':sum(r['outcomes']['unsafe'] for r in subset)})
    state=json.loads((parent/'status.json').read_text())
    write_status(parent,phase='finished',training_exit_code=0,latest_checkpoint=str(checkpoint),
                 best_validation_checkpoint=str(best_checkpoint) if guarded else None,
                 completed_trials=len(rows),heldout_success_rate=state['holdout_successes']/args.eval_count)
    remote=Rclone(ROOT/'scripts/rl/gdrive.sh')
    destination=args.remote_root.rstrip('/')+'/'+parent.name
    for name in ('manifest.json','results.json','status.json'):
        archive_file(parent/name,destination,remote)
    write_status(parent,final_upload_verified=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
