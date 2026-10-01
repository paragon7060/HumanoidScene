#!/usr/bin/env python3
"""Own one fixed-scene residual SAC pilot and its existing Drive backups."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import uuid

from drive_backup import ROOT, Rclone, archive_file
from train_with_drive import archive, supervise


def archive_pilot(source, remote_root, finished):
    removed = archive(source, remote_root, finished)
    if finished:
        # supervise has waited for the child and closed the console writer.
        with (source/'.drive-backup.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            remote = Rclone(ROOT/'scripts/rl/gdrive.sh')
            destination = remote_root.rstrip('/')+'/'+source.name
            for name in ('executed_transitions.hdf5','reference.mp4','policy.mp4','preview.png',
                         'failure.json','residual_experience.pt'):
                path = source/name
                if path.exists():
                    archive_file(path,destination,remote)
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=3)
    parser.add_argument('--python',type=Path,default=Path.home()/'miniconda3/envs/env_isaaclab_232/bin/python')
    parser.add_argument('--remote-root',default=os.environ.get('RL_DRIVE_REMOTE_ROOT'))
    args, child_args = parser.parse_known_args()
    if args.gpu < 0 or not args.python.is_file():
        parser.error('Select an existing Isaac Python and a nonnegative physical GPU')
    if any(v.split('=')[0] in {'--output-dir','--device'} for v in child_args):
        parser.error('The supervisor sets the unique output directory and internal cuda:0')
    if '--residual-sac' not in child_args:
        parser.error('This supervisor requires --residual-sac')
    if not args.remote_root:
        remotes = subprocess.run(['bash',str(ROOT/'scripts/rl/gdrive.sh'),'listremotes'],
                                 check=True,capture_output=True,text=True).stdout.splitlines()
        if len(remotes)!=1:
            parser.error('Set RL_DRIVE_REMOTE_ROOT to an existing host-local remote')
        args.remote_root = remotes[0]+'HumanoidScene-RL'
    if not re.fullmatch(r'[\w-]+:HumanoidScene-RL',args.remote_root.rstrip('/')):
        parser.error('Use the existing remote and dedicated HumanoidScene-RL root')
    parent=args.experiment_dir.expanduser().resolve()
    parent.mkdir(parents=True,exist_ok=False)
    run=parent/('residual_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    environment=os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OMNI_KIT_ACCEPT_EULA='YES',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'scripts/rl'))
    command=[str(args.python),'-u',str(ROOT/'scripts/rl/replay_v2_grasp_reference.py'),
             *child_args,'--output-dir',str(run),'--device','cuda:0','--headless']
    # No account alias is checked into Git; this launch file is local run metadata.
    (parent/'launch.json').write_text(json.dumps({'command':command,'gpu':args.gpu},indent=2)+'\n')
    code=supervise(command,parent,environment,
        lambda source,finished:archive_pilot(source,args.remote_root,finished),
        interval=300,run_prefix='residual_',require_run_status=True)
    return code


if __name__=='__main__':
    raise SystemExit(main())
