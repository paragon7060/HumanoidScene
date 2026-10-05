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
from train_with_drive import archive, supervise, write_status


def archive_pilot(source, remote_root, finished):
    removed = archive(source, remote_root, finished)
    if finished:
        # supervise has waited for the child and closed the console writer.
        with (source/'.drive-backup.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            remote = Rclone(ROOT/'scripts/rl/gdrive.sh')
            destination = remote_root.rstrip('/')+'/'+source.name
            for name in ('executed_transitions.hdf5','reference.mp4','policy.mp4','preview.png',
                         'failure.json','residual_experience.pt','pose_goal_experience.pt',
                         'staged_goal_experience.pt','gripper_drive_audit.json',
                         'actual_train_goal_collection.pt','physical_body_experience.pt',
                         'grasp_observation_audit.jsonl.gz','grasp_observation_audit_summary.json'):
                path = source/name
                if path.exists():
                    archive_file(path,destination,remote)
            for path in sorted(source.glob('reset_failure_diagnostics_wave_*.json')):
                archive_file(path,destination,remote)
            for path in sorted(source.glob('initial_layout_guard_wave_*.json')):
                archive_file(path,destination,remote)
    return removed


def verified_success(parent, require_checkpoint=True):
    state=json.loads((parent/'status.json').read_text())
    if state.get('training_exit_code')!=0 or not state.get('final_upload_verified'):
        return None
    source=Path(state['run_dir'])
    metrics=json.loads((source/'metrics.json').read_text())
    outcomes=metrics['outcomes']
    if not metrics.get('completed_attempt') or not outcomes.get('success') \
            or any(outcomes.get(key,0) for key in ('unsafe','invalid_reset','time_out')):
        return None
    checkpoints=sorted(source.glob('checkpoint_*.pt'))
    return checkpoints[-1] if require_checkpoint and checkpoints else source if not require_checkpoint else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=3)
    parser.add_argument('--episodes',type=int,default=1,
                        help='For 2..5 episodes, automatically check each frozen actor before continuing; fixed scene only.')
    parser.add_argument('--python',type=Path,default=Path.home()/'miniconda3/envs/env_isaaclab_232/bin/python')
    parser.add_argument('--remote-root',default=os.environ.get('RL_DRIVE_REMOTE_ROOT'))
    args, child_args = parser.parse_known_args()
    if args.gpu < 0 or not args.python.is_file() or not 1<=args.episodes<=5:
        parser.error('Select an existing Isaac Python and a nonnegative physical GPU')
    if args.episodes>1 and '--no-residual-training' in child_args:
        parser.error('Multiple episodes require training; each has its own frozen evaluation')
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
    environment=os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OMNI_KIT_ACCEPT_EULA='YES',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'scripts/rl'))
    def run_trial(trial,extras):
        run=trial/('residual_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        command=[str(args.python),'-u',str(ROOT/'scripts/rl/replay_v2_grasp_reference.py'),
                 *child_args,*extras,'--output-dir',str(run),'--device','cuda:0','--headless']
        (trial/'launch.json').write_text(json.dumps({'command':command,'gpu':args.gpu},indent=2)+'\n')
        return supervise(command,trial,environment,
            lambda source,finished:archive_pilot(source,args.remote_root,finished),
            interval=300,run_prefix='residual_',require_run_status=True)
    if args.episodes==1:
        return run_trial(parent,[])
    checkpoint=None
    for index in range(1,args.episodes+1):
        train=parent/f'train_{index:02d}';train.mkdir()
        write_status(parent,phase='training',episode=index,episodes=args.episodes,active_trial=str(train))
        extras=['--residual-training']
        if checkpoint:
            extras.extend(('--residual-checkpoint',str(checkpoint)))
        code=run_trial(train,extras)
        checkpoint=verified_success(train) if code==0 else None
        if checkpoint is None:
            write_status(parent,phase='performance_gate_failed' if code==0 else 'failed',
                         failed_trial=str(train),training_exit_code=code)
            return code or 2
        evaluation=parent/f'eval_{index:02d}';evaluation.mkdir()
        write_status(parent,phase='evaluating',active_trial=str(evaluation),latest_checkpoint=str(checkpoint))
        code=run_trial(evaluation,['--no-residual-training','--residual-checkpoint',str(checkpoint)])
        if code!=0 or verified_success(evaluation,require_checkpoint=False) is None:
            write_status(parent,phase='performance_gate_failed' if code==0 else 'failed',
                         failed_trial=str(evaluation),training_exit_code=code)
            return code or 2
        write_status(parent,episodes_verified=index)
    write_status(parent,phase='finished',training_exit_code=0,final_upload_verified=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
