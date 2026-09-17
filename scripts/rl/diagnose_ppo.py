#!/usr/bin/env python3
"""Bounded, read-only checkpoint rollout; compare mean and exploration noise.

Uses the production environment and its exact checkpoint contract. No training,
physics overrides, reward changes, object teleports, or policy weight changes.
"""

import json
import os
from pathlib import Path
import time
import traceback

from kuavo_isaaclab_scene.rl.runners.common import (
    parse_args, build_configs, write_run_config, check_checkpoint, install_stop_handlers,
)


def main():
    def extra(parser):
        parser.add_argument('--diagnostic-dir', type=Path, required=True)
        parser.add_argument('--diagnostic-steps', type=int, default=900)
        parser.add_argument('--mesh-video', action='store_true')
        parser.add_argument('--training-source-dir', type=Path,
                            help='Verified copies of shared modules from this training launch')

    args = parse_args('play', extra)
    if args.num_envs % 4 or not 1 <= args.diagnostic_steps <= 1800:
        raise ValueError('Use a multiple of four envs and 1..1800 steps')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('This diagnostic is authorized on physical GPU 1 only')
    directory = args.diagnostic_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report = {'status': 'starting', 'pid': os.getpid(), 'checkpoint': str(args.checkpoint),
              'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']}
    app = env = writer = None
    started = time.monotonic()
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(args).app
        install_stop_handlers()
        import numpy as np
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from kuavo_isaaclab_scene.rl.mdp.geometry import unrotate

        if args.training_source_dir:
            import hashlib
            import importlib
            import importlib.util
            source = args.training_source_dir.resolve()
            hashes = json.loads((source/'source_sha256.json').read_text())
            report['training_source_sha256'] = {}
            for name in ('rl.mdp.flap_grasp', 'rl.mdp.rewards', 'rl.managers.rewards'):
                module_name = 'kuavo_isaaclab_scene.' + name
                relative = 'src/' + module_name.replace('.', '/') + '.py'
                path = source/relative
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != hashes[relative]:
                    raise ValueError(f'Training source checksum mismatch: {relative}')
                parent = importlib.import_module(module_name.rsplit('.', 1)[0])
                spec = importlib.util.spec_from_file_location(module_name, path)
                module = importlib.util.module_from_spec(spec)
                import sys
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                setattr(parent, module_name.rsplit('.', 1)[1], module)
                report['training_source_sha256'][relative] = digest

        cfg, agent = build_configs(args)
        env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg), clip_actions=agent.clip_actions)
        manifest = write_run_config(directory, cfg, agent, env.unwrapped)
        check_checkpoint(args.checkpoint, manifest)
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
        runner.load(str(args.checkpoint), load_optimizer=False)
        policy = runner.alg.policy
        policy.eval()
        obs, _ = env.reset()
        torch.manual_seed(args.seed)
        t = env.unwrapped.command_manager.get_term('workcell')
        arm = env.unwrapped.action_manager.get_term('upper_body')
        gripper = env.unwrapped.action_manager.get_term('right_gripper')
        groups = ('mean', 'stochastic', 'quarter_noise', 'hold')
        width = env.num_envs // 4
        report.update(checkpoint_iteration=runner.current_learning_iteration, groups=list(groups),
                      envs_per_group=width, step_dt=env.unwrapped.step_dt,
                      action_std=policy.std.detach().cpu().tolist(),
                      network=str(policy), setup_seconds=time.monotonic()-started)
        print('[PPO_DIAG] loaded '+json.dumps(report), flush=True)
        renderer = None
        if args.mesh_video:
            import imageio.v2 as imageio
            from cpu_scene_video import SceneVideo
            renderer = SceneVideo(env.unwrapped)
            # The renderer's reference camera is for a one-env cell at zero;
            # the first cell of a parallel grid is translated by its origin.
            renderer.eye = renderer.eye + env.unwrapped.scene.env_origins[0].cpu().numpy()
            writer = imageio.get_writer(str(directory/'policy.mp4'), fps=2,
                                        codec='libx264', quality=7, macro_block_size=16)
        arrays = {}
        sampled_observations = []
        episode_count = torch.zeros(env.num_envs, device=env.device)
        unsafe = episode_count.clone()
        success = episode_count.clone()
        representatives = [i*width for i in range(4)]
        with torch.inference_mode():
            for step in range(args.diagnostic_steps):
                t.refresh()
                mean = policy.act_inference(obs)
                raw = mean + torch.randn_like(mean)*policy.std
                raw[:width] = mean[:width]
                raw[2*width:3*width] = mean[2*width:3*width] + .25*(raw[2*width:3*width]-mean[2*width:3*width])
                raw[3*width:] = 0
                normal = policy.actor_obs_normalizer(policy.get_actor_obs(obs))
                halves = t.flap_grasp.halves[t.active_box, 1]
                centers = t.flap_grasp.centers[t.active_box, 1]
                axes = t.flap_grasp.normal_axes[t.active_box, 1]
                local_pads = unrotate(t.flap_quat[:,1,None].expand(-1,2,-1), t.pad_points[:,1]-t.flap_pos[:,1,None])
                signed = (local_pads-centers[:,None]).gather(-1,axes[:,None,None].expand(-1,2,1)).squeeze(-1)
                values = dict(ready=t.settling.ready, distance=t.hand_target_distance[:,1],
                    alignment=t.grasp_alignment[:,1], gap=t.jaw_gap[:,1],
                    gap_error=t.jaw_gap_error[:,1], capture=t.ready_to_close[:,1],
                    straddling=signed[:,0]*signed[:,1]<=0,
                    contact=t.finger_grasp_contacts[:,1], force=t.contact_force[:,2:4],
                    grasp=t.hand_grasp_flags[:,1], signed_target=(1-2*gripper.raw_actions[:,0]),
                    pad_signed=signed, raw_action=raw, mean_action=mean,
                    arm_tracking=arm.processed_actions-t.robot.data.joint_pos[:,arm._joint_ids],
                    lift=t.centers[t.ids,t.active_box,2]-env.unwrapped.scene.env_origins[:,2]-t.initial_z[t.ids,t.active_box],
                    dwell=t.dwell, pads=t.pad_points[:,1]-env.unwrapped.scene.env_origins[:,None],
                    target=t.grips[:,1]-env.unwrapped.scene.env_origins,
                    normal_absmax=normal.abs().amax(-1), value=policy.evaluate(obs).squeeze(-1))
                for key, value in values.items():
                    arrays.setdefault(key, []).append(value.detach().cpu().numpy().copy())
                if step % 15 == 0:
                    sampled_observations.append(obs['policy'].cpu().numpy().copy())
                # Preview is constructed from live USD mesh geometry at live
                # PhysX poses. It is explicitly labelled as a CPU mesh preview.
                if renderer is not None and step < 450 and step % 15 == 0:
                    import cv2
                    frame = renderer.frame(env.unwrapped,step,float(values['distance'][0]),int(values['grasp'][0]))
                    if np.std(frame[70:].astype(float)) < 1:
                        raise RuntimeError('CPU camera contains no visible scene geometry')
                    cv2.rectangle(frame,(0,0),(960,30),(23,28,36),-1)
                    cv2.putText(frame,f'PPO {runner.current_learning_iteration} mean | live PhysX | CPU mesh preview',
                                (15,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
                    writer.append_data(frame)
                    if step in (0,225,435):
                        from PIL import Image
                        Image.fromarray(frame).save(directory/f'preview_{step:04d}.png')
                obs, reward, dones, extras = env.step(raw)
                episode_count += dones
                unsafe += env.unwrapped.termination_manager.get_term('unsafe')
                success += env.unwrapped.termination_manager.get_term('success')
                arrays.setdefault('reward', []).append(reward.cpu().numpy().copy())
                arrays.setdefault('done', []).append(dones.cpu().numpy().copy())
                if (step+1) % 150 == 0:
                    print('[PPO_DIAG] '+json.dumps({'step':step+1,'distances':t.hand_target_distance[representatives,1].tolist(),
                        'gripper_target':(1-2*gripper.raw_actions[representatives,0]).tolist(),
                        'successes':success.sum().item(), 'elapsed_s':time.monotonic()-started}), flush=True)
        if writer is not None:
            writer.close()
            writer = None
        arrays = {k: np.stack(v) for k,v in arrays.items()}
        np.savez_compressed(directory/'rollout.npz', **arrays)
        np.savez_compressed(directory/'observations.npz', policy=np.stack(sampled_observations))
        report['results'] = {}
        for i, name in enumerate(groups):
            sl = slice(i*width,(i+1)*width)
            mask = arrays['ready'][:,sl].astype(bool)
            def avg(key):
                return np.asarray(arrays[key][:,sl][mask]).mean(axis=0).tolist()
            result = dict(episodes=episode_count[sl].sum().item(), unsafe=unsafe[sl].sum().item(),
                successes=success[sl].sum().item(), ready_samples=int(mask.sum()),
                mean_distance=avg('distance'), mean_alignment=avg('alignment'), mean_gap=avg('gap'),
                capture_samples=int(arrays['capture'][:,sl][mask].sum()),
                grasp_samples=int(arrays['grasp'][:,sl][mask].sum()),
                straddling_fraction=avg('straddling'), finger_contact_fraction=avg('contact'),
                mean_signed_target=avg('signed_target'), mean_raw_action=avg('raw_action'),
                clipped_fraction_per_dim=(np.abs(arrays['raw_action'][:,sl][mask])>1).mean(0).tolist(),
                min_distance_per_env=np.where(mask,arrays['distance'][:,sl],np.inf).min(0).tolist(),
                near_25mm_samples=int((arrays['distance'][:,sl][mask]<.025).sum()),
                max_normalized_observation=float(arrays['normal_absmax'][:,sl][mask].max()))
            report['results'][name] = result
        report['status'] = 'completed'
    except BaseException:
        report.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        (directory/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print('[PPO_DIAG_RESULT] '+json.dumps(report),flush=True)
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == '__main__':
    main()
