#!/usr/bin/env python3
"""Wait for one owned experiment, then start an isolated SAC comparison.

No process signalling, credential copying, raw upload/deletion, or inference
of a running job from prepared configuration. The queue itself has no GPU.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def owned_process(pid, hint, mask):
    proc = Path('/proc', str(pid))
    if not proc.exists(): return False
    if proc.stat().st_uid != os.getuid() or hint.encode() not in (proc/'cmdline').read_bytes() \
            or ('CUDA_VISIBLE_DEVICES='+mask).encode() not in (proc/'environ').read_bytes().split(b'\0'):
        raise ValueError('Predecessor PID ownership, command or GPU mask changed')
    return True


def atomic_json(path, value):
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def verified_input(plan):
    code, initial = Path(plan['training_source_root']), Path(plan['initialization_dir'])
    for relative, digest in plan['source_files_SHA256'].items():
        path = code/relative
        if path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid() \
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen trainer input changed')
    checkpoint = initial/'checkpoint_00000000.pt'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != plan['initial_checkpoint_SHA256']:
        raise ValueError('Prepared initial model changed')
    for filename, digest in plan['initial_metadata_SHA256'].items():
        if hashlib.sha256((initial/filename).read_bytes()).hexdigest() != digest:
            raise ValueError('Prepared initial metadata changed')
    return code, initial


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('The CPU queue requires an empty CUDA mask')
    plan = json.loads(args.plan.read_text())
    if plan['format'] != 'owned_staged_SAC_comparison_queue_v1' or args.plan.stat().st_uid != os.getuid():
        raise ValueError('Explicit owned comparison plan required')
    code, initial = verified_input(plan)
    pointer = Path(plan['predecessor_launch']); launch_output = Path(plan['launch_output'])
    if launch_output.exists(): raise ValueError('Never launch a duplicate experiment')
    def status(phase, **details):
        atomic_json(args.status, dict(recorded_at=datetime.now().astimezone().isoformat(),
            phase=phase, queue_PID=os.getpid(), CUDA_VISIBLE_DEVICES='', target_GPU=plan['gpu'],
            physical_training_started=phase=='launched', no_process_signals=True,
            no_raw_upload_or_deletion=True, goal_not_complete=True, **details))
    while not pointer.exists():
        status('waiting_predecessor_actual_launch')
        time.sleep(30)
    predecessor = json.loads(pointer.read_text()); parent = Path(predecessor['parent'])
    managed = json.loads((parent/'launch.json').read_text()); run = Path(managed['run'])
    if predecessor['gpu'] != plan['gpu'] or managed['backup_scope'] != 'checkpoint_contract_logs_only':
        raise ValueError('Predecessor or backup scope changed')
    while True:
        previous = json.loads((parent/'status.json').read_text())
        if previous['training_pid'] != predecessor['writer_pid_at_launch'] \
                or previous['supervisor_pid'] != predecessor['supervisor_pid']:
            raise ValueError('Original managed writer identity changed')
        live = [pid for pid, hint in ((previous['training_pid'], str(run)),
            (previous['supervisor_pid'], str(parent))) if owned_process(pid, hint, predecessor['CUDA_VISIBLE_DEVICES'])]
        if live:
            status('waiting_predecessor_normal_exit', actual_predecessor_live_PIDs=live)
            time.sleep(30); continue
        if previous.get('training_exit_code') != 0:
            raise RuntimeError('Predecessor failed; retain inputs for diagnosis instead of repeating its failure')
        if previous.get('final_upload_verified') is not True:
            status('waiting_predecessor_final_Drive_verification'); time.sleep(30); continue
        free = {int(r.split(',')[0]): int(r.split(',')[1]) for r in subprocess.check_output(
            ['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'], text=True).splitlines()}
        root_free, shm_free = shutil.disk_usage(ROOT).free/2**30, shutil.disk_usage('/dev/shm').free/2**30
        if free[plan['gpu']] < plan['minimum_GPU_free_MiB'] or root_free < plan['minimum_ROOT_free_GiB'] \
                or shm_free < plan['minimum_SHM_free_GiB']:
            status('waiting_actual_resource_capacity', GPU_free_MiB=free[plan['gpu']],
                ROOT_free_GiB=root_free, SHM_free_GiB=shm_free)
            time.sleep(30); continue
        break
    verified_input(plan)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = Path(plan['run_root'])/(plan['label']+'_'+stamp)
    unit = 'humanoid-rl-'+plan['label'].lower().replace('_','-')+'-'+stamp+'.service'
    command = [sys.executable, '-u', str(ROOT/'scripts/rl/batched_staged_goal_with_drive.py'),
        '--experiment-dir', str(output), '--gpu', str(plan['gpu']), '--physics-device', 'cpu',
        '--learner-device', 'cuda:0', '--training-source-root', str(code),
        '--checkpoint-log-backup-only', *plan['child_arguments']]
    status('requesting_distinct_comparison', requested_unit=unit, requested_parent=str(output))
    subprocess.run(['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+str(ROOT),
        '--setenv=CUDA_VISIBLE_DEVICES='+str(plan['gpu']), '--setenv=OMP_NUM_THREADS=1',
        '--setenv=MKL_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1',
        '--setenv=PYTHONPATH='+str(ROOT/'src')+':'+str(ROOT/'scripts/rl'), *command],
        check=True, capture_output=True, text=True)
    for _ in range(90):
        if (output/'status.json').exists() and (output/'launch.json').exists():
            current = json.loads((output/'status.json').read_text())
            if current.get('training_pid'): break
        time.sleep(1)
    else: raise RuntimeError('Launch was requested but not observed; never duplicate it')
    actual = json.loads((output/'launch.json').read_text())
    actual_mask = current['CUDA_VISIBLE_DEVICES']
    if not owned_process(current['training_pid'], actual['run'], actual_mask) \
            or not owned_process(current['supervisor_pid'], str(output), actual_mask):
        raise RuntimeError('Actual new owned writer not observed')
    receipt = dict(recorded_at=datetime.now().astimezone().isoformat(), parent=str(output), run=actual['run'],
        unit=unit, writer_pid_at_launch=current['training_pid'], supervisor_pid=current['supervisor_pid'],
        gpu=plan['gpu'], CUDA_VISIBLE_DEVICES=actual_mask, source_commit=plan['source_commit'],
        initial_checkpoint_SHA256=plan['initial_checkpoint_SHA256'],
        backup_scope=actual['backup_scope'], original_randomization_success_safety_preserved=True,
        fresh_TRAIN_requested=plan['fresh_TRAIN_requested'], independent_FINAL_unused=True,
        no_process_signals=True, goal_not_complete=True)
    atomic_json(launch_output, receipt)
    status('launched', actual_parent=receipt['parent'], actual_run=receipt['run'],
        actual_writer_PID=receipt['writer_pid_at_launch'], actual_supervisor_PID=receipt['supervisor_pid'],
        actual_writer_CUDA_VISIBLE_DEVICES=actual_mask, actual_unit=receipt['unit'])


if __name__ == '__main__': main()
