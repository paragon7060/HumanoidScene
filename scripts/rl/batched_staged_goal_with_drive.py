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
from train_with_drive import archive, supervise


def archive_batched(source, remote_root, finished, *, checkpoint_log_only=False):
    """Allow checkpoint/log backups without exporting physical replay payloads."""
    if checkpoint_log_only:
        return archive(source, remote_root, finished)
    return archive_pilot(source, remote_root, finished)


def validate_managed_physics_device(device, child, *, learner_device=None):
    """CPU dynamics cannot silently become matching TRAIN or FINAL data."""
    if device not in ('cpu', 'cuda:0'):
        raise ValueError('Managed physics device must be cuda:0 or cpu')
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--training', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--reset-failure-diagnostics', action='store_true')
    parser.add_argument('--frozen-physics-backend-eval', action='store_true')
    parser.add_argument('--cpu-physics-training', action='store_true')
    parser.add_argument('--cpu-workplace-probe', action='store_true')
    parser.add_argument('--base-waypoint-probe', action='store_true')
    parser.add_argument('--unmeasured-size-workplace-probe', action='store_true')
    parser.add_argument('--workplace-reset-diagnostics', action='store_true')
    parser.add_argument('--base-substep-trace-env-indices',type=int,nargs='+',default=None)
    parser.add_argument('--base-attitude-gain-probe',choices=('soft15_2',),default=None)
    parser.add_argument('--steps', type=int, default=900)
    parser.add_argument('--waves-json', type=Path)
    parser.add_argument('--training-manifest', type=Path)
    audit, _ = parser.parse_known_args(child)
    if audit.unmeasured_size_workplace_probe and not audit.cpu_workplace_probe:
        raise ValueError('Unmeasured size candidates require the explicit frozen workplace route')
    if audit.workplace_reset_diagnostics and not audit.cpu_workplace_probe:
        raise ValueError('Workplace reset capture requires the explicit frozen TRAIN workplace route')
    if audit.base_substep_trace_env_indices is not None and not audit.cpu_workplace_probe:
        raise ValueError('Base substep trace requires the explicit frozen TRAIN workplace route')
    if audit.cpu_workplace_probe:
        from kuavo_isaaclab_scene.rl.multi_box.experiments.cpu_workplace_probe import (
            INCOMPATIBLE_FLAGS, validate_cpu_workplace_probe)
        if audit.waves_json is None or audit.training_manifest is None:
            raise ValueError('CPU workplace search requires TRAIN candidate waves and its physical contract')
        waves=json.loads(audit.waves_json.read_text())
        workplace=validate_cpu_workplace_probe(waves,
            json.loads(audit.training_manifest.read_text()), enabled=True, device=device,
            training=audit.training, steps=audit.steps, waypoint_enabled=audit.base_waypoint_probe,
            unmeasured_size_probe=audit.unmeasured_size_workplace_probe,
            workplace_reset_diagnostics=audit.workplace_reset_diagnostics,
            explicit_frozen='--no-training' in child and '--training' not in child,
            other_probe=any(s.split('=')[0] in INCOMPATIBLE_FLAGS for s in child))
        from kuavo_isaaclab_scene.rl.multi_box.debug.base_substep_trace import validate_base_substep_trace
        validate_base_substep_trace(audit.base_substep_trace_env_indices,
            workplace=workplace,training=audit.training,num_envs=len(waves[0]['layouts']))
        from kuavo_isaaclab_scene.rl.multi_box.experiments.base_attitude_probe import validate_base_attitude_probe
        validate_base_attitude_probe(audit.base_attitude_gain_probe,
            workplace=workplace,training=audit.training,num_envs=len(waves[0]['layouts']))
        return workplace
    if audit.base_attitude_gain_probe is not None:
        raise ValueError('Attitude gain comparison requires the complete frozen TRAIN workplace route')
    if audit.cpu_physics_training:
        from kuavo_isaaclab_scene.rl.multi_box.experiments.cpu_physics_training import (
            INCOMPATIBLE_FLAGS, validate_cpu_physics_training)
        if audit.waves_json is None or audit.training_manifest is None:
            raise ValueError('CPU physics learning requires waves and its new training manifest')
        return validate_cpu_physics_training(json.loads(audit.waves_json.read_text()),
            json.loads(audit.training_manifest.read_text()), enabled=True, physics_device=device,
            learner_device=learner_device, training=audit.training, steps=audit.steps,
            other_probe=any(s.split('=')[0] in INCOMPATIBLE_FLAGS for s in child))
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
    # Child flags such as --checkpoint must not abbreviate manager flags.
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--experiment-dir',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=3)
    parser.add_argument('--physics-device',choices=('cuda:0','cpu'),default='cuda:0',
        help='CPU requires explicit frozen diagnostics or a separate CPU-physics-training contract')
    parser.add_argument('--learner-device', choices=('cuda:0','cpu'), default=None,
        help='Managed GPU learner uses cuda:0 within CUDA_VISIBLE_DEVICES; default follows physics')
    parser.add_argument('--python',type=Path,default=Path.home()/'miniconda3/envs/env_isaaclab_232/bin/python')
    parser.add_argument('--training-source-root', type=Path, default=ROOT,
        help='Use a separate immutable trainer source; authentication and uploader remain in this checkout')
    parser.add_argument('--isolate-graphics-devices', action=argparse.BooleanOptionalAction, default=True,
        help='Expose only the selected GPU to this process; isolates Vulkan/GL as well as CUDA')
    parser.add_argument('--remote-root',default=os.environ.get('RL_DRIVE_REMOTE_ROOT'))
    parser.add_argument('--checkpoint-log-backup-only',action='store_true',
        help='Upload checkpoints, contract metadata and closed logs; keep replay/HDF/media local')
    args,child=parser.parse_known_args()
    if args.gpu<0 or not args.python.is_file():parser.error('Valid GPU/Isaac Python required')
    isolation = None
    if args.isolate_graphics_devices:
        from single_gpu_runtime import ensure_single_gpu_namespace
        isolation = ensure_single_gpu_namespace(args.gpu)
    code = args.training_source_root.resolve()
    trainer = code/'scripts/rl/train_batched_staged_goal.py'
    if not trainer.is_file() or trainer.is_symlink() or trainer.stat().st_uid != os.getuid():
        parser.error('An owned regular trainer source is required')
    if any(s.split('=')[0] in ('--output-dir','--device','--learner-device','--kit_args') for s in child):
        parser.error('Managed run owns its output/device/renderer isolation')
    learner_device=args.learner_device or args.physics_device
    if '--cpu-physics-training' not in child and learner_device != args.physics_device:
        parser.error('Separate learner device requires explicit CPU physics learning')
    try:device_audit=validate_managed_physics_device(args.physics_device,child,learner_device=learner_device)
    except (ValueError,OSError) as error:parser.error(str(error))
    if not args.remote_root:
        remotes=subprocess.run(['bash',str(ROOT/'scripts/rl/gdrive.sh'),'listremotes'],capture_output=True,text=True,check=True).stdout.splitlines()
        if len(remotes)!=1:parser.error('Select the existing remote with RL_DRIVE_REMOTE_ROOT')
        args.remote_root=remotes[0]+'HumanoidScene-RL'
    if not re.fullmatch(r'[\w-]+:HumanoidScene-RL',args.remote_root.rstrip('/')):
        parser.error('Existing remote and dedicated folder required')
    parent=args.experiment_dir.expanduser().resolve();parent.mkdir(parents=True,exist_ok=False)
    run=parent/('batch_sac_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    renderer_gpu = isolation['renderer_gpu'] if isolation else args.gpu
    cuda_mask = isolation['CUDA_VISIBLE_DEVICES'] if isolation else str(args.gpu)
    command=[str(args.python),'-u',str(trainer),*child,
        '--output-dir',str(run),'--device',args.physics_device,'--learner-device',learner_device,'--headless','--kit_args',
        f'--/renderer/activeGpu={renderer_gpu} --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1']
    environment=os.environ.copy();environment.update(CUDA_VISIBLE_DEVICES=cuda_mask,OMNI_KIT_ACCEPT_EULA='YES',
        PYTHONPATH=str(code/'src')+':'+str(code/'scripts/rl'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    (parent/'launch.json').write_text(json.dumps(dict(command=command,gpu=args.gpu,run=str(run),
        physics_device=args.physics_device,learner_device=learner_device,frozen_device_diagnostic=device_audit,
        backup_scope='checkpoint_contract_logs_only' if args.checkpoint_log_backup_only else 'pilot_payloads',
        training_source_root=str(code), original_Drive_wrapper_root=str(ROOT),
        single_GPU_graphics_isolation=isolation),indent=2)+'\n')
    return supervise(command,parent,environment,
        lambda source,finished:archive_batched(source,args.remote_root,finished,
            checkpoint_log_only=args.checkpoint_log_backup_only),
        interval=300,run_prefix='batch_sac_',require_run_status=True)


if __name__=='__main__':raise SystemExit(main())
