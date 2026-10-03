"""Wait for an owned experiment's real terminal state and verified archive.

CPU-only dependency gate for sequential experiments. Restart/stop transitions
are live states, never permission to start a second GPU child. No process is
killed, no authentication or GPU runtime is imported.
"""
import json
from pathlib import Path
import subprocess
import time


def unit_state(unit):
    result = subprocess.run(['systemctl', '--user', 'show', unit,
        '-p', 'ActiveState', '-p', 'MainPID', '-p', 'ExecMainPID', '-p', 'LoadState'],
        check=True, capture_output=True, text=True)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def verified_dependency(unit, launch_record):
    """Return a closed run or None while its service is still live."""
    state = unit_state(unit)
    if state.get('ActiveState') not in {'inactive', 'failed'} \
            or int(state.get('MainPID', '0')):
        return None
    launch_record = Path(launch_record)
    if not launch_record.is_file():
        raise RuntimeError(f'{unit} ended before publishing its launch record')
    launch = json.loads(launch_record.read_text())
    parent = Path(launch['parent']).resolve()
    status = json.loads((parent / 'status.json').read_text())
    previous_pid = int(state.get('ExecMainPID', '0'))
    if previous_pid and previous_pid != status.get('supervisor_pid'):
        raise RuntimeError('Launch pointer belongs to a different supervisor attempt')
    if status.get('phase') != 'finished' or status.get('final_upload_verified') is not True:
        raise RuntimeError('Terminal dependency lacks its completed verified backup')
    if status.get('training_exit_code') != 0:
        raise RuntimeError('Dependency execution failed; repair it before starting the next experiment')
    run = Path(status['run_dir']).resolve()
    if not run.is_relative_to(parent) or run != Path(launch['run']).resolve():
        raise RuntimeError('Dependency launch/run directories disagree')
    verification = json.loads((run / 'verification.json').read_text())
    if not verification.get('writers_stopped_at') or verification.get('training_exit_code') != 0:
        raise RuntimeError('Dependency has no verified writer termination')
    return dict(parent=parent, run_dir=run, status=status)


def wait_for_verified_run(unit, launch_record, *, poll_seconds=20):
    if not 1 <= poll_seconds <= 60:
        raise ValueError('Dependency poll interval must be within1..60seconds')
    while True:
        result = verified_dependency(unit, launch_record)
        if result is not None:
            return result
        time.sleep(poll_seconds)
