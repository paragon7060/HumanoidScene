#!/usr/bin/env python3
"""Prune unused closed RL payloads only after existing Drive MD5 verification.

Explicit experiment paths only. Never uploads, touches checkpoints/logs/media,
or removes an active/unverified file. No Isaac or GPU imports.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time

from drive_backup import ROOT, Rclone, md5, signature

PAYLOADS = ('staged_goal_experience.pt', 'physical_body_experience.pt',
    'pose_goal_experience.pt', 'residual_experience.pt', 'executed_transitions.hdf5')


def process_inventory():
    """Inspect this UID only; absence of a process is never a training restart."""
    commands, opened = [], set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            commands.append((proc/'cmdline').read_bytes())
            for fd in (proc/'fd').iterdir():
                try:
                    value = fd.stat()
                    opened.add((value.st_dev, value.st_ino))
                except OSError:
                    pass
        except OSError:
            pass
    return commands, opened


def closed_run(parent, commands):
    parent = Path(parent)
    if parent.is_symlink() or not parent.is_dir() or parent.stat().st_uid != os.getuid():
        raise ValueError('Owned direct experiment directory required')
    for name in ('status.json', 'launch.json'):
        p = parent/name
        if p.is_symlink() or not p.is_file() or p.stat().st_uid != os.getuid():
            raise ValueError('Owned regular managed metadata required')
    state = json.loads((parent/'status.json').read_text())
    if state.get('phase') != 'finished' or not state.get('final_upload_verified') \
            or state.get('training_exit_code') is None:
        return None
    launch = json.loads((parent/'launch.json').read_text())
    if launch.get('backup_scope') == 'checkpoint_contract_logs_only':
        return None
    recorded_run = launch.get('run')
    if recorded_run is None:
        # Earlier reference supervisors recorded the run in status, with the
        # exact launch command repeated there. Require that binding rather
        # than guessing a child directory or trusting a run name alone.
        recorded_run = state.get('run_dir')
        command = launch.get('command')
        if not isinstance(command, list) or command != state.get('command') \
                or command.count('--output-dir') != 1 or not recorded_run:
            raise ValueError('Legacy run requires matching recorded launch command')
        index = command.index('--output-dir')
        if index + 1 >= len(command) or command[index + 1] != recorded_run:
            raise ValueError('Legacy output directory must match recorded run')
    elif state.get('run_dir') not in (None, recorded_run):
        raise ValueError('Conflicting recorded run directories')
    run = Path(recorded_run)
    if run.parent != parent or run.is_symlink() or not run.is_dir() or run.stat().st_uid != os.getuid():
        raise ValueError('Owned direct recorded run required')
    manifest = run/'manifest.json'
    if manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_uid != os.getuid():
        raise ValueError('Owned regular manifest required')
    json.loads(manifest.read_text())
    if any(str(run).encode() in command or str(parent).encode() in command for command in commands):
        return None
    for key in ('training_pid', 'supervisor_pid'):
        pid = state.get(key)
        if type(pid) is not int or pid <= 0:
            raise ValueError('Recorded writer and supervisor are required')
        proc = Path('/proc', str(pid))
        if proc.exists():
            # Fail closed even when a numeric PID may have been reused.
            return None
    return run


def prune_experiment(parent, remote_root, remote, *, protected=()):
    commands, opened = process_inventory()
    run = closed_run(parent, commands)
    if run is None:
        return dict(experiment=str(parent), skipped='not_closed_or_still_referenced', removed=[])
    protected = {Path(p).absolute() for p in protected}
    candidates = [run/name for name in PAYLOADS if (run/name).exists()]
    if not candidates:
        return dict(experiment=str(parent), skipped='no_remaining_payloads', removed=[])
    with (run/'.drive-backup.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(experiment=str(parent), skipped='backup_lock_busy', removed=[])
        marker = run/'local_payload_cleanup.json'
        if marker.is_symlink():
            raise ValueError('Cleanup marker must not be a symlink')
        previous = json.loads(marker.read_text()) if marker.exists() else {}
        records = {r['file']: r for r in previous.get('files', [])}
        removed, retained = [], []
        for path in candidates:
            value = path.lstat()
            if path.absolute() in protected or not stat.S_ISREG(value.st_mode) \
                    or value.st_uid != os.getuid() or value.st_nlink != 1 \
                    or (value.st_dev, value.st_ino) in opened:
                retained.append(dict(file=path.name, reason='protected_nonregular_shared_or_open'))
                continue
            before = signature(path)
            digest = md5(path)
            try:
                remote.verify(path, remote_root+'/'+run.name+'/'+path.name, digest)
            except Exception as error:
                retained.append(dict(file=path.name, reason='remote_verification_failed', error_type=type(error).__name__))
                continue
            commands, opened = process_inventory()
            if closed_run(parent, commands) != run or signature(path) != before \
                    or (value.st_dev, value.st_ino) in opened:
                retained.append(dict(file=path.name, reason='source_or_runtime_changed'))
                continue
            record = dict(file=path.name, bytes=value.st_size, allocated_bytes=value.st_blocks*512,
                MD5=digest, existing_Drive_size_MD5_reverified=True,
                local_removed_at=datetime.now().astimezone().isoformat())
            # Journal the verified destination before unlinking; after crash a
            # future call still verifies any surviving local payload again.
            records[path.name] = record | dict(local_removal_complete=False)
            def write_marker():
                data = dict(run_directory=run.name, dedicated_remote_folder='HumanoidScene-RL/'+run.name,
                    files=list(records.values()), local_checkpoints_logs_videos_kept=True,
                    resume_requires_restoring_payloads_from_existing_remote=True)
                temporary = marker.with_suffix('.tmp')
                temporary.write_text(json.dumps(data, indent=2)+'\n')
                temporary.replace(marker)
            write_marker()
            path.unlink()
            records[path.name]['local_removal_complete'] = True
            write_marker()
            removed.append(record)
        return dict(experiment=str(parent), run=run.name, removed=removed, retained=retained,
            allocated_bytes_removed=sum(r['allocated_bytes'] for r in removed))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True,
        help='Owned JSON with explicit experiment_dirs and protected_paths; no directory-wide discovery')
    p.add_argument('--remote-root', default=os.environ.get('RL_DRIVE_REMOTE_ROOT'))
    p.add_argument('--watch', type=int, default=0)
    args = p.parse_args()
    if args.watch and args.watch < 300:
        p.error('Watch interval must be at least300seconds')
    if args.config.is_symlink() or not args.config.is_file() or args.config.stat().st_uid != os.getuid():
        p.error('Owned regular configuration required')
    config = json.loads(args.config.read_text())
    experiments = [Path(x) for x in config['experiment_dirs']]
    protected = config.get('protected_paths', [])
    if not experiments or len(set(experiments)) != len(experiments) or any(not x.is_absolute() for x in experiments):
        p.error('Distinct explicit absolute experiment directories required')
    if not args.remote_root:
        names = subprocess.check_output(['bash', str(ROOT/'scripts/rl/gdrive.sh'), 'listremotes'], text=True).splitlines()
        if len(names) != 1:
            p.error('Select the existing remote with RL_DRIVE_REMOTE_ROOT')
        args.remote_root = names[0]+'HumanoidScene-RL'
    if not re.fullmatch(r'[\w-]+:HumanoidScene-RL', args.remote_root.rstrip('/')):
        p.error('Existing dedicated HumanoidScene-RL remote required')
    remote = Rclone(ROOT/'scripts/rl/gdrive.sh')
    with args.config.with_suffix('.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            results = []
            for experiment in experiments:
                try:
                    results.append(prune_experiment(experiment, args.remote_root, remote, protected=protected))
                except Exception as error:
                    results.append(dict(experiment=str(experiment), skipped='check_failed', error_type=type(error).__name__))
            data = dict(recorded_at=datetime.now().astimezone().isoformat(), worker_pid=os.getpid(),
                no_GPU_or_Isaac=True, no_upload_or_remote_deletion=True, watch_seconds=args.watch,
                allocated_bytes_removed_this_pass=sum(r.get('allocated_bytes_removed', 0) for r in results), results=results)
            args.config.with_suffix('.status.json').write_text(json.dumps(data, indent=2)+'\n')
            print(json.dumps(dict(worker_pid=os.getpid(), removed_files=sum(len(r.get('removed', [])) for r in results),
                allocated_bytes_removed=data['allocated_bytes_removed_this_pass'])), flush=True)
            if not args.watch:
                return
            time.sleep(args.watch)


if __name__ == '__main__':
    main()
