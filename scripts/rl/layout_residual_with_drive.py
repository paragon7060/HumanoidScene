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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir',type=Path,required=True)
    parser.add_argument('--layout-dir',type=Path,required=True)
    parser.add_argument('--train-count',type=int,default=8)
    parser.add_argument('--eval-count',type=int,default=8)
    parser.add_argument('--passes',type=int,default=1)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--after-verified-experiment',type=Path,
                        help='Wait for this own pilot series to finish verified; inherit its final checkpoint.')
    parser.add_argument('--gpu',type=int,default=3)
    parser.add_argument('--python',type=Path,default=Path.home()/'miniconda3/envs/env_isaaclab_232/bin/python')
    parser.add_argument('--remote-root',default=os.environ.get('RL_DRIVE_REMOTE_ROOT'))
    args,child_args=parser.parse_known_args()
    if args.checkpoint and args.after_verified_experiment:
        parser.error('Choose an explicit checkpoint or an earlier verified experiment')
    if min(args.train_count,args.eval_count,args.passes)<1 or args.gpu<0 or not args.python.is_file():
        parser.error('Positive split sizes/passes and a valid Isaac Python are required')
    reserved={'--output-dir','--device','--layout-json','--residual-checkpoint',
              '--residual-controller','--residual-training','--no-residual-training','--residual-zero'}
    if any(value.split('=')[0] in reserved for value in child_args):
        parser.error('This supervisor owns layout, checkpoint, device and training/evaluation mode')
    if '--residual-sac' not in child_args:
        parser.error('Supply --residual-sac and the current measured reference/physical manifest')
    layouts={split:sorted(args.layout_dir.resolve().glob(split+'_*.json'))[:count]
             for split,count in [('train',args.train_count),('holdout',args.eval_count)]}
    for split,count in [('train',args.train_count),('holdout',args.eval_count)]:
        if len(layouts[split])!=count or any(json.loads(path.read_text())['split']!=split for path in layouts[split]):
            parser.error('Each requested layout must have its correct explicit split')
        for path in layouts[split]:
            layout=json.loads(path.read_text())
            if not -.040001<=layout['lateral_m']<=-.019999 or abs(layout.get('yaw_rad',0))>math.radians(1)+1e-7:
                parser.error('The footprint-valid sampling distribution is inward2..4cm and yaw+/-1degree')
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
            copy=frozen/path.name;copy.write_text(path.read_text());copies.append(copy)
        layouts[split]=copies
    (parent/'manifest.json').write_text(json.dumps(dict(artifact_type='layout_reference_residual_sac_suite',
        gpu=args.gpu,train_layouts=[json.loads(p.read_text()) for p in layouts['train']],
        heldout_layouts=[json.loads(p.read_text()) for p in layouts['holdout']],
        passes=args.passes,physical_reference_dependency=True,curriculum=False,
        sampled_distribution='lower_small_inward2to4cm_yaw1deg_rear_upper_distractors_v2',
        sampling_reason='2..6cm crossed the assigned half-shelf boundary; physical reset bounds unchanged'),indent=2)+'\n')
    environment=os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OMNI_KIT_ACCEPT_EULA='YES',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'scripts/rl'))
    checkpoint=args.checkpoint.resolve() if args.checkpoint else None
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
    schedule=[('train',p,i) for i in range(args.passes) for p in layouts['train']]
    schedule += [('holdout',p,0) for p in layouts['holdout']]
    for index,(split,layout,pass_index) in enumerate(schedule):
        trial=parent/f'{index+1:03d}_{split}_p{pass_index+1}_{layout.stem}';trial.mkdir()
        run=trial/('residual_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        extras=['--residual-controller','retargeted-goal','--layout-json',str(layout),
                '--residual-training' if split=='train' else '--no-residual-training']
        if checkpoint:
            extras.extend(('--residual-checkpoint',str(checkpoint)))
        command=[str(args.python),'-u',str(ROOT/'scripts/rl/replay_v2_grasp_reference.py'),
                 *child_args,*extras,'--output-dir',str(run),'--device','cuda:0','--headless']
        (trial/'launch.json').write_text(json.dumps(dict(command=command,gpu=args.gpu),indent=2)+'\n')
        write_status(parent,phase='training' if split=='train' else 'heldout_evaluation',
                     active_trial=str(trial),completed_trials=len(rows),total_trials=len(schedule),
                     latest_checkpoint=str(checkpoint) if checkpoint else None)
        code=supervise(command,trial,environment,
            lambda source,finished:archive_pilot(source,args.remote_root,finished),
            interval=300,run_prefix='residual_',require_run_status=True)
        state=json.loads((trial/'status.json').read_text())
        if code!=0 or not state.get('final_upload_verified'):
            write_status(parent,phase='runtime_failed',failed_trial=str(trial),training_exit_code=code)
            return code or 2
        metrics=json.loads((run/'metrics.json').read_text())
        if split=='holdout' and metrics['residual_sac']['training']:
            raise RuntimeError('Held-out evaluation unexpectedly enabled optimizer updates')
        row=dict(split=split,layout=json.loads(layout.read_text()),pass_index=pass_index,
                 run_dir=str(run),outcomes=metrics['outcomes'],steps=metrics['steps'],
                 actor_updates=metrics['residual_sac']['actor_updates'],final_upload_verified=True)
        rows.append(row)
        if split=='train':
            checkpoints=sorted(run.glob('checkpoint_*.pt'))
            if not checkpoints:
                raise RuntimeError('Completed training did not save a checkpoint')
            checkpoint=checkpoints[-1]
        (parent/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
        for group in ('train','holdout'):
            subset=[r for r in rows if r['split']==group]
            write_status(parent,**{group+'_attempts':len(subset),
                group+'_successes':sum(r['outcomes']['success'] for r in subset),
                group+'_unsafe':sum(r['outcomes']['unsafe'] for r in subset)})
    state=json.loads((parent/'status.json').read_text())
    write_status(parent,phase='finished',training_exit_code=0,latest_checkpoint=str(checkpoint),
                 completed_trials=len(rows),heldout_success_rate=state['holdout_successes']/args.eval_count)
    remote=Rclone(ROOT/'scripts/rl/gdrive.sh')
    destination=args.remote_root.rstrip('/')+'/'+parent.name
    for name in ('manifest.json','results.json','status.json'):
        archive_file(parent/name,destination,remote)
    write_status(parent,final_upload_verified=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
