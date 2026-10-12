#!/usr/bin/env python3
"""Prepare an explicit compatible SAC resume only after closed verified source.

No training launch, GPU initialization, uploads, deletion or authentication.
Raw replay is loaded solely as the next learner's unchanged resume input.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from queued_experiment import verified_dependency


def prepare(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Resume preparation must use CUDA_VISIBLE_DEVICES empty')
    dependency = verified_dependency(args.unit, args.launch_record)
    if dependency is None:
        raise RuntimeError('Source is still live; refusing before Torch or raw replay access')
    for name in ('training_pid', 'supervisor_pid'):
        pid = dependency['status'].get(name)
        if pid and Path(f'/proc/{pid}').exists():
            raise RuntimeError('Registered source writer/supervisor is still present')
    run = dependency['run_dir']
    checkpoint = args.checkpoint.absolute()
    experience = run/'staged_goal_experience.pt'
    destination = args.destination.absolute()
    if checkpoint.parent != run or not checkpoint.name.startswith('checkpoint_') \
            or not checkpoint.is_file() or not experience.is_file() or destination.exists():
        raise ValueError('A closed run checkpoint/replay and a new unique destination are required')
    if shutil.disk_usage(destination.parent).free < checkpoint.stat().st_size+experience.stat().st_size+512*1024**2:
        raise RuntimeError('Insufficient space for the exact resume input copy and reserve')
    # Neither Torch nor the raw input is touched until the real closure gate.
    import torch
    from kuavo_isaaclab_scene.rl.multi_box.experiments.region_size_sac import (
        migrate_region_size_checkpoint, migrate_region_size_resume_input,
    )
    before = (checkpoint.stat(), experience.stat())
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    source = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    target = migrate_region_size_checkpoint(source, sha)
    actual_resume = torch.load(experience, map_location='cpu', weights_only=True, mmap=True)
    resume = migrate_region_size_resume_input(actual_resume, source, target)
    destination.mkdir()
    target_checkpoint = destination/checkpoint.name
    for path, state in ((target_checkpoint, target), (destination/'staged_goal_experience.pt', resume)):
        pending = path.with_suffix(path.suffix+'.pending')
        torch.save(state, pending)
        pending.replace(path)
    after = (checkpoint.stat(), experience.stat())
    if any((a.st_size, a.st_mtime_ns, a.st_ino) != (b.st_size, b.st_mtime_ns, b.st_ino)
           for a, b in zip(before, after)):
        raise RuntimeError('Closed source changed while preparing resume; do not use output')
    manifest = dict(artifact_type=target['artifact_type'],source_checkpoint_SHA256=sha,
        checkpoint_SHA256=hashlib.sha256(target_checkpoint.read_bytes()).hexdigest(),
        checkpoint=target_checkpoint.name,source_run=str(run),source_all_writers_closed_final_Drive_verified=True,
        actor_updates=target['actor_updates'],critic_updates=target['critic_updates'],
        actor_only_import=False,all_Q_entropy_optimizers_bank_cohort_resumed=True,
        exact_closed_replay_for_training_resume_only=True,no_transition_relabel=True,
        physical_task_reward_DR_success_safety_curriculum_unchanged=True,
        raw_replay_HDF_uploaded_or_deleted=False,CUDA_initialized=torch.cuda.is_initialized(),
        prepared_resume_not_training_launch=True)
    if manifest['CUDA_initialized']:
        raise RuntimeError('CPU preparation unexpectedly initialized CUDA')
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit',required=True)
    parser.add_argument('--launch-record',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args),indent=2))


if __name__ == '__main__':
    main()
