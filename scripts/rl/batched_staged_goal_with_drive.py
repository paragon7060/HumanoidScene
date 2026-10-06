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


def validate_managed_physics_device(device, child):
    """CPU dynamics cannot silently become matching TRAIN or FINAL data."""
    if device not in ('cpu', 'cuda:0'):
        raise ValueError('Managed physics device must be cuda:0 or cpu')
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--training', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--reset-failure-diagnostics', action='store_true')
    parser.add_argument('--frozen-physics-backend-eval', action='store_true')
    parser.add_argument('--steps', type=int, default=900)
    parser.add_argument('--waves-json', type=Path)
    audit, _ = parser.parse_known_args(child)
    if device == 'cuda:0' and not audit.frozen_physics_backend_eval:
        return None
    explicit_frozen = '--no-training' in child and '--training' not in child
    if audit.frozen_physics_backend_eval:
        from kuavo_isaaclab_scene.rl.multi_box.experiments.physics_backend_eval import (
            INCOMPATIBLE_FLAGS,validate_frozen_backend_policy_eval)
        if audit.waves_json is None:
            raise ValueError('Backend policy evaluation requires original DEV waves')
        return validate_frozen_backend_policy_eval(json.loads(audit.waves_json.read_text()),
            enabled=True, device=device, training=audit.training, steps=audit.steps,
            explicit_frozen=explicit_frozen,
            other_probe=any(s.split('=')[0] in INCOMPATIBLE_FLAGS for s in child))
    if audit.training or not explicit_frozen or not audit.reset_failure_diagnostics or audit.steps != 1:
        raise ValueError('CPU dynamics requires explicit --no-training --reset-failure-diagnostics --steps 1')
    if audit.waves_json is None:
        raise ValueError('CPU dynamics requires original DEV waves')
    waves = json.loads(audit.waves_json.read_text())
    if not isinstance(waves, list) or not waves or any(wave.get('split') != 'validation' for wave in waves):
        raise ValueError('CPU dynamics cannot collect TRAIN or independent FINAL waves')
    return dict(name='frozen_DEV_reset_CPU_PhysX', training=False, Q_import_eligible=False,
        physics_device='cpu', renderer_GPU_isolation_unchanged=True,
        constructor_and_contact_solver_history_not_matched=True,
        not_a_grasp_performance_evaluation=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=3)
    parser.add_argument('--physics-device',choices=('cuda:0','cpu'),default='cuda:0',
        help='CPU requires explicit frozen DEV reset diagnostics or full backend policy evaluation; training retains cuda:0')
    parser.add_argument('--python',type=Path,default=Path.home()/'miniconda3/envs/env_isaaclab_232/bin/python')
    parser.add_argument('--remote-root',default=os.environ.get('RL_DRIVE_REMOTE_ROOT'))
    args,child=parser.parse_known_args()
    if args.gpu<0 or not args.python.is_file():parser.error('Valid GPU/Isaac Python required')
    if any(s.split('=')[0] in ('--output-dir','--device','--kit_args') for s in child):
        parser.error('Managed run owns its output/device/renderer isolation')
    try:device_audit=validate_managed_physics_device(args.physics_device,child)
    except (ValueError,OSError) as error:parser.error(str(error))
    if not args.remote_root:
        remotes=subprocess.run(['bash',str(ROOT/'scripts/rl/gdrive.sh'),'listremotes'],capture_output=True,text=True,check=True).stdout.splitlines()
        if len(remotes)!=1:parser.error('Select the existing remote with RL_DRIVE_REMOTE_ROOT')
        args.remote_root=remotes[0]+'HumanoidScene-RL'
    if not re.fullmatch(r'[\w-]+:HumanoidScene-RL',args.remote_root.rstrip('/')):
        parser.error('Existing remote and dedicated folder required')
    parent=args.experiment_dir.expanduser().resolve();parent.mkdir(parents=True,exist_ok=False)
    run=parent/('batch_sac_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    command=[str(args.python),'-u',str(ROOT/'scripts/rl/train_batched_staged_goal.py'),*child,
        '--output-dir',str(run),'--device',args.physics_device,'--headless','--kit_args',
        f'--/renderer/activeGpu={args.gpu} --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false']
    environment=os.environ.copy();environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu),OMNI_KIT_ACCEPT_EULA='YES',
        PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'scripts/rl'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    (parent/'launch.json').write_text(json.dumps(dict(command=command,gpu=args.gpu,run=str(run),
        physics_device=args.physics_device,frozen_device_diagnostic=device_audit),indent=2)+'\n')
    return supervise(command,parent,environment,
        lambda source,finished:archive_pilot(source,args.remote_root,finished),
        interval=300,run_prefix='batch_sac_',require_run_status=True)


if __name__=='__main__':raise SystemExit(main())
