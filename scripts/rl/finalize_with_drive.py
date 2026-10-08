#!/usr/bin/env python3
"""Retry a closed managed run's final Drive upload using current backup code."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import time

from drive_backup import ROOT
from batched_staged_goal_with_drive import archive_batched
from train_with_drive import write_status


def process_alive(pid):
    return Path(f'/proc/{pid}').exists()


def validate_closed_run(parent):
    """Reject an active writer or supervisor before including final logs."""
    parent=Path(parent).resolve()
    state=json.loads((parent/'status.json').read_text())
    if state.get('phase') not in ('final_upload','finished','stopped'):
        raise ValueError('Managed run has not reached final upload')
    for key in ('training_pid','supervisor_pid'):
        pid=state.get(key)
        if type(pid) is not int or pid<=0 or process_alive(pid):
            raise ValueError(f'Original {key} must be recorded and stopped')
    original=Path(state.get('run_dir',''))
    run=original.resolve()
    if original.is_symlink() or run.parent!=parent or not run.is_dir():
        raise ValueError('Recorded run must be a direct non-symlink child of its parent')
    if parent.stat().st_uid!=os.getuid() or run.stat().st_uid!=os.getuid():
        raise ValueError('Only owned managed runs may be finalized')
    manifest=run/'manifest.json'
    if manifest.is_symlink() or not manifest.is_file() or not isinstance(json.loads(manifest.read_text()),dict):
        raise ValueError('Run manifest is required')
    return run,state


def backup_scope(parent):
    """Preserve the original manager's explicit payload scope on recovery."""
    launch=Path(parent)/'launch.json'
    if launch.is_symlink():
        raise ValueError('Original launch metadata cannot be a symlink')
    scope=json.loads(launch.read_text()).get('backup_scope','pilot_payloads') if launch.exists() else 'pilot_payloads'
    if scope not in ('pilot_payloads','checkpoint_contract_logs_only'):
        raise ValueError('Unknown original backup scope; do not widen it on recovery')
    return scope


def finalize_once(parent,remote_root):
    """One locked, checksum-verified attempt; failure preserves local sources."""
    parent=Path(parent).resolve()
    validate_closed_run(parent)
    with (parent/'.drive-finalize.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run,state=validate_closed_run(parent)
        scope=backup_scope(parent)
        if state.get('phase')=='finished' and state.get('final_upload_verified'):
            return run
        write_status(parent,phase='final_upload',finalizer_pid=os.getpid(),
                     finalizer_started_at=datetime.now().astimezone().isoformat(),
                     finalizer_backup_scope=scope)
        archive_batched(run,remote_root,True,
            checkpoint_log_only=scope=='checkpoint_contract_logs_only')
        write_status(parent,phase='finished',final_upload_verified=True,backup_error=None,
                     finalizer_finished_at=datetime.now().astimezone().isoformat())
        return run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir',type=Path,required=True)
    parser.add_argument('--remote-root',default=os.environ.get('RL_DRIVE_REMOTE_ROOT'))
    parser.add_argument('--retry-seconds',type=int,default=300)
    args=parser.parse_args()
    if args.retry_seconds<30:parser.error('Retry interval must be at least30seconds')
    parent=args.experiment_dir.resolve()
    validate_closed_run(parent)
    if not args.remote_root:
        names=subprocess.check_output(['bash',str(ROOT/'scripts/rl/gdrive.sh'),'listremotes'],text=True).splitlines()
        if len(names)!=1:parser.error('Select the existing connection using RL_DRIVE_REMOTE_ROOT')
        args.remote_root=names[0]+'HumanoidScene-RL'
    if not re.fullmatch(r'[\w-]+:HumanoidScene-RL',args.remote_root.rstrip('/')):
        parser.error('Dedicated HumanoidScene-RL destination required')
    while True:
        validate_closed_run(parent)
        try:
            finalize_once(parent,args.remote_root)
        except Exception as error:
            # Keep credentials and the host-specific remote alias out of
            # diagnostics. Local source files are retained on failure.
            write_status(parent,backup_error=type(error).__name__)
            print(f'[Finalize] Upload failed ({type(error).__name__}); retrying in {args.retry_seconds}s',flush=True)
            time.sleep(args.retry_seconds)
            continue
        print('[Finalize] Closed payloads verified within the original backup scope; training exit code preserved.',flush=True)
        return 0


if __name__=='__main__':raise SystemExit(main())
