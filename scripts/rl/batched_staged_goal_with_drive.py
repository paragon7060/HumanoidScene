#!/usr/bin/env python3
"""Own a batched physical SAC run using existing verified Drive storage."""
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
    parser.add_argument('--gpu',type=int,default=3)
    parser.add_argument('--python',type=Path,default=Path.home()/'miniconda3/envs/env_isaaclab_232/bin/python')
    parser.add_argument('--remote-root',default=os.environ.get('RL_DRIVE_REMOTE_ROOT'))
    args,child=parser.parse_known_args()
    if args.gpu<0 or not args.python.is_file():parser.error('Valid GPU/Isaac Python required')
    if any(s.split('=')[0] in ('--output-dir','--device','--kit_args') for s in child):
        parser.error('Managed run owns its output/device/renderer isolation')
    if not args.remote_root:
        remotes=subprocess.run(['bash',str(ROOT/'scripts/rl/gdrive.sh'),'listremotes'],capture_output=True,text=True,check=True).stdout.splitlines()
        if len(remotes)!=1:parser.error('Select the existing remote with RL_DRIVE_REMOTE_ROOT')
        args.remote_root=remotes[0]+'HumanoidScene-RL'
    if not re.fullmatch(r'[\w-]+:HumanoidScene-RL',args.remote_root.rstrip('/')):
        parser.error('Existing remote and dedicated folder required')
    parent=args.experiment_dir.expanduser().resolve();parent.mkdir(parents=True,exist_ok=False)
    run=parent/('batch_sac_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    command=[str(args.python),'-u',str(ROOT/'scripts/rl/train_batched_staged_goal.py'),*child,
        '--output-dir',str(run),'--device','cuda:0','--headless','--kit_args',
        f'--/renderer/activeGpu={args.gpu} --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false']
    environment=os.environ.copy();environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OMNI_KIT_ACCEPT_EULA='YES',
        PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'scripts/rl'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    (parent/'launch.json').write_text(json.dumps(dict(command=command,gpu=args.gpu,run=str(run)),indent=2)+'\n')
    return supervise(command,parent,environment,
        lambda source,finished:archive_pilot(source,args.remote_root,finished),
        interval=300,run_prefix='batch_sac_',require_run_status=True)


if __name__=='__main__':raise SystemExit(main())
