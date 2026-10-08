#!/usr/bin/env python3
"""One CPU queue for final backups of explicitly listed, stopped managed runs.

This worker never stops jobs, changes authentication, widens payload scope or
deletes raw data. Existing verified checkpoint retention remains authoritative.
"""
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
from finalize_with_drive import backup_scope,finalize_once,validate_closed_run


def read_queue(path):
    path=Path(path)
    if path.is_symlink() or path.stat().st_uid!=os.getuid():
        raise ValueError('Only an owned, non-symlink queue file is accepted')
    data=json.loads(path.read_text())
    if not isinstance(data,dict) or data.get('format')!='closed_managed_Drive_backups_v1':
        raise ValueError('Explicit closed-run queue format required')
    entries=data.get('entries')
    if not isinstance(entries,list) or not entries:
        raise ValueError('At least one explicit managed run is required')
    seen=set()
    for entry in entries:
        if not isinstance(entry,dict):
            raise ValueError('Queue entries must record original run identities')
        parent=Path(entry.get('experiment_dir',''))
        if not parent.is_absolute() or parent.is_symlink() or parent.resolve() in seen:
            raise ValueError('Distinct absolute non-symlink managed parents required')
        seen.add(parent.resolve())
        if parent.stat().st_uid!=os.getuid():
            raise ValueError('Queue cannot manage another owner\'s run')
        if any(type(entry.get(k)) is not int or entry[k]<=0 for k in ('training_pid','supervisor_pid')):
            raise ValueError('Original writer and supervisor identities required')
        if entry.get('backup_scope') not in ('pilot_payloads','checkpoint_contract_logs_only'):
            raise ValueError('Explicit original backup scope required')
    return entries


def validate_entry(entry):
    parent=Path(entry['experiment_dir'])
    run,state=validate_closed_run(parent)
    if any(state[k]!=entry[k] for k in ('training_pid','supervisor_pid')) \
            or str(run)!=entry.get('run_dir') or backup_scope(parent)!=entry['backup_scope']:
        raise ValueError('Original queued run identity or payload scope changed')
    return parent,run,state


def remote_ready(remote_root):
    # Discard rclone diagnostics rather than expose account aliases or tokens.
    result=subprocess.run(['bash',str(ROOT/'scripts/rl/gdrive.sh'),'about',remote_root.split(':',1)[0]+':'],
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=45)
    return result.returncode==0


def queue_pass(entries,remote_root):
    rows=[];eligible=[]
    for entry in entries:
        row=dict(experiment_dir=entry['experiment_dir'],run_dir=entry.get('run_dir'),
                 backup_scope=entry['backup_scope'],state='pending')
        try:
            _,_,state=validate_entry(entry)
            if state.get('phase')=='finished' and state.get('final_upload_verified'):
                row['state']='verified'
            else:eligible.append((entry,row))
        except (ValueError,OSError,KeyError) as error:
            row.update(state='waiting_original_processes_or_metadata',error_kind=type(error).__name__)
        rows.append(row)
    if eligible:
        try:available=remote_ready(remote_root)
        except (OSError,subprocess.SubprocessError):available=False
        if not available:
            for _,row in eligible:row['state']='waiting_existing_Drive_connection'
        else:
            for entry,row in eligible:
                try:
                    validate_entry(entry)
                    finalize_once(Path(entry['experiment_dir']),remote_root)
                    row['state']='verified'
                except Exception as error:
                    row.update(state='pending_upload',error_kind=type(error).__name__)
    return dict(recorded_at=datetime.now().astimezone().isoformat(),queue_pid=os.getpid(),
        total=len(rows),verified=sum(row['state']=='verified' for row in rows),
        local_unverified_sources_retained=True,no_processes_signalled=True,entries=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--remote-root',default=os.environ.get('RL_DRIVE_REMOTE_ROOT'))
    parser.add_argument('--retry-seconds',type=int,default=300)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':parser.error('CPU queue requires CUDA_VISIBLE_DEVICES empty')
    if args.retry_seconds<30:parser.error('Retry interval must be at least30seconds')
    entries=read_queue(args.queue)
    if not args.remote_root:
        remotes=subprocess.check_output(['bash',str(ROOT/'scripts/rl/gdrive.sh'),'listremotes'],text=True).splitlines()
        if len(remotes)!=1:parser.error('Select the existing remote using RL_DRIVE_REMOTE_ROOT')
        args.remote_root=remotes[0]+'HumanoidScene-RL'
    if not re.fullmatch(r'[\w-]+:HumanoidScene-RL',args.remote_root.rstrip('/')):
        parser.error('Existing dedicated HumanoidScene-RL destination required')
    status_path=args.queue.with_suffix('.status.json')
    with args.queue.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            state=queue_pass(entries,args.remote_root)
            temporary=status_path.with_suffix('.pending')
            temporary.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n');temporary.replace(status_path)
            print(f"[Drive queue] {state['verified']}/{state['total']} final backups verified; unverified originals retained",flush=True)
            if state['verified']==state['total']:return 0
            if args.once:return 2
            time.sleep(args.retry_seconds)


if __name__=='__main__':raise SystemExit(main())
