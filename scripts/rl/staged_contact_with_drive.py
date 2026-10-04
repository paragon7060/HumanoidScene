#!/usr/bin/env python3
"""Collect one actual TRAIN diagnostic with staged base and a local IK teacher.

No optimizer runs; physical failures are still recorded failures. The shared
Drive supervisor verifies closed video/data/logs and keeps its normal retention.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import uuid

from drive_backup import ROOT
from reference_residual_with_drive import archive_pilot
from train_with_drive import supervise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--layout-json',type=Path,required=True)
    parser.add_argument('--waypoints',type=Path,required=True)
    parser.add_argument('--demo-dataset',type=Path,required=True)
    parser.add_argument('--training-manifest',type=Path,required=True)
    parser.add_argument('--native-seed',type=Path,action='append',required=True)
    parser.add_argument('--episode-index',type=int,required=True)
    parser.add_argument('--contact-mode',choices=('near-contact','after-base-hold'),default='near-contact')
    parser.add_argument('--contact-orientation',choices=('full','closing-axis'),default='full')
    parser.add_argument('--contact-velocity-feedforward',action='store_true',
                        help='Diagnostic only: pair bounded IK position with its executed velocity target.')
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--python',type=Path,default=Path.home()/'miniconda3/envs/env_isaaclab_232/bin/python')
    parser.add_argument('--remote-root',default=os.environ.get('RL_DRIVE_REMOTE_ROOT'))
    args=parser.parse_args()
    if args.gpu<0 or args.episode_index<0 or not all(p.is_file() for p in (
            args.python,args.checkpoint,args.layout_json,args.waypoints,args.demo_dataset,
            args.training_manifest,*args.native_seed)):
        parser.error('Existing files and nonnegative GPU/episode index are required')
    if json.loads(args.layout_json.read_text()).get('split')!='train':
        parser.error('This collection probe requires an explicitly separate TRAIN layout')
    if not args.remote_root:
        remotes=subprocess.run(['bash',str(ROOT/'scripts/rl/gdrive.sh'),'listremotes'],
            check=True,capture_output=True,text=True).stdout.splitlines()
        if len(remotes)!=1:parser.error('Set RL_DRIVE_REMOTE_ROOT to the existing connection')
        args.remote_root=remotes[0]+'HumanoidScene-RL'
    if not re.fullmatch(r'[\w-]+:HumanoidScene-RL',args.remote_root.rstrip('/')):
        parser.error('Use the existing remote and dedicated HumanoidScene-RL folder')
    parent=args.experiment_dir.expanduser().resolve()
    parent.mkdir(parents=True,exist_ok=False)
    run=parent/('contact_probe_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    command=[str(args.python),'-u',str(ROOT/'scripts/rl/replay_v2_grasp_reference.py'),
        '--pose-student-checkpoint',str(args.checkpoint),'--no-pose-student-training',
        '--staged-base-waypoints',str(args.waypoints),'--layout-json',str(args.layout_json),
        '--staged-contact-ik-mode',args.contact_mode,
        '--staged-contact-ik-orientation',args.contact_orientation,
        '--demo-dataset',str(args.demo_dataset),'--training-manifest',str(args.training_manifest),
        '--episode-index',str(args.episode_index),'--torso-extra-height-m','.06',
        '--steps','900','--capture-every','90','--contact-diagnostics',
        '--output-dir',str(run),'--device','cuda:0','--headless','--kit_args',
        f'--/renderer/activeGpu={args.gpu} --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false']
    for seed in args.native_seed:
        command.extend(('--pose-student-native-seed',str(seed),
                        '--staged-contact-ik-native-seed',str(seed)))
    if args.contact_velocity_feedforward:
        command.append('--staged-contact-ik-velocity-feedforward')
    environment=os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OMNI_KIT_ACCEPT_EULA='YES',
        PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'scripts/rl'),
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    (parent/'launch.json').write_text(json.dumps(dict(command=command,gpu=args.gpu,
        contact_mode=args.contact_mode,
        contact_orientation=args.contact_orientation,
        contact_velocity_feedforward=args.contact_velocity_feedforward,
        diagnostic_only=True,optimizer_updates=0,teacher_has_privileged_pinch_confirmation=True,
        old_goal_Q_import_forbidden=True),indent=2)+'\n')
    return supervise(command,parent,environment,
        lambda source,finished:archive_pilot(source,args.remote_root,finished),
        interval=300,run_prefix='contact_probe_',require_run_status=True)


if __name__=='__main__':
    raise SystemExit(main())
