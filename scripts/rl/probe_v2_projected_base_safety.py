#!/usr/bin/env python3
"""Two-env finite telemetry fault injection, never a training/success trial."""
import argparse,json
from pathlib import Path
from dataclasses import replace


def main():
    from isaaclab.app import AppLauncher
    from kuavo_isaaclab_scene.robots.robot_model import add_robot_model_cli_args,export_robot_model_cli
    from kuavo_isaaclab_scene.robots.gripper_config import add_gripper_cli_args,export_gripper_cli
    from kuavo_isaaclab_scene.robots.base_drive import add_base_drive_cli_args,export_base_drive_cli
    from kuavo_isaaclab_scene.workcell.rack_rollers import add_rack_roller_cli_args,export_rack_roller_cli
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    add_robot_model_cli_args(p);add_gripper_cli_args(p);add_base_drive_cli_args(p);add_rack_roller_cli_args(p)
    AppLauncher.add_app_launcher_args(p)
    p.set_defaults(headless=True,robot_model='s63',gripper='leju-twofinger',rack_rollers=True)
    args=p.parse_args();export_robot_model_cli(args);export_gripper_cli(args);export_base_drive_cli(args);export_rack_roller_cli(args)
    app=AppLauncher(args).app;env=None
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from kuavo_isaaclab_scene.rl.envs.terminal_observation import TerminalObservationMixin
        from kuavo_isaaclab_scene.rl.multi_box.training_env_cfg import MultiBoxGraspAssemblyEnvCfg
        from kuavo_isaaclab_scene.rl.runners.train_asymmetric_sac import _settle_initial_resets
        from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_student import PoseGoalCoordinates
        from kuavo_isaaclab_scene.rl.multi_box.geometry.projected_base import planar_projection_determinant, outside_projected_base_domain
        class Env(TerminalObservationMixin,ManagerBasedRLEnv):pass
        cfg=MultiBoxGraspAssemblyEnvCfg(num_envs=2);cfg.multi_box=replace(cfg.multi_box,self_collision_enabled=False)
        cfg.sim.device=args.device or 'cuda:0';cfg.sim.physx.solver_type=0
        env=Env(cfg);env.enable_numerical_dynamics_recovery();env.enable_projected_base_safety()
        obs,_=env.reset(seed=42);obs,settled=_settle_initial_resets(env,obs)
        assert env._multi_box_reset_settling.ready.all()
        resets=[];ordinary_reset=env._reset_idx
        def tracked_reset(ids):resets.extend(ids.tolist());return ordinary_reset(ids)
        env._reset_idx=tracked_reset
        robot=env.scene['robot'];data=robot.data;original_update=robot.update;count=0
        angle=torch.tensor(torch.pi/2-.01,device=env.device)
        quaternion=torch.stack((torch.cos(angle/2),angle*0,torch.sin(angle/2),angle*0))
        def inject_finite_read_pose(dt):
            nonlocal count
            original_update(dt);count+=1
            if count==cfg.decimation:
                # Deterministic final-substep read-cache fault. No bad pose
                # is written into PhysX, and no probe row can enter Q.
                data.root_pose_w;data.root_state_w
                for name,buffer in vars(data).items():
                    values=getattr(buffer,'data',None)
                    if name.startswith('_root_') and isinstance(values,torch.Tensor) and values.ndim==2 and values.shape[0]==2 and values.shape[1] in (7,13):
                        buffer.data=values.clone();buffer.data[0,3:7]=quaternion
                assert outside_projected_base_domain(planar_projection_determinant(data.root_quat_w,env.scene['rack'].data.root_quat_w)).tolist()==[True,False]
        robot.update=inject_finite_read_pose
        action=torch.zeros((2,env.action_manager.total_action_dim),device=env.device)
        obs,reward,done,timeout,info=env.step(action)
        assert done.tolist()==[True,False],done
        assert env.termination_manager.get_term('unsafe').tolist()==[True,False]
        assert not env.termination_manager.get_term('success').any()
        assert not env.termination_manager.get_term('invalid_reset').any()
        assert not info['transition_numerical_failure'].any()
        assert set(resets)=={0},resets
        assert info['transition_safety']['base_projection_invalid'].tolist()==[True,False]
        assert info['transition_safety']['workspace_limit'].tolist()==[True,False]
        assert env._multi_box_grasp_reward_breakdown.terms['workspace_limit'][0]<0
        assert torch.isfinite(info['transition_next_observations']['policy']).all()
        coordinates=PoseGoalCoordinates(exact_projected_base=True)
        terminal=info['transition_next_observations']['policy']
        _,_,_,_,rotation=coordinates.current(terminal)
        assert outside_projected_base_domain(torch.linalg.det(rotation[:,:2,:2])).tolist()==[True,False]
        robot.update=original_update
        obs,_,_,_,next_info=env.step(action)
        assert torch.isfinite(obs['policy']).all() and not next_info['transition_numerical_failure'].any()
        result=dict(passed=True,num_envs=2,projection_failed_envs=[0],unaffected_envs=[1],
                    finite_tilt_read_cache_fault_injection=True,not_a_physical_success_or_training_trial=True,
                    invalid_pose_never_written_to_PhysX=True,unsafe_workspace_failure=True,
                    existing_workspace_penalty_applied=True,partial_reset_ids=resets,
                    finite_original_terminal_observation_preserved=True,failed_transition_Q_eligible=True,
                    no_numerical_quarantine_or_invalid_reset=True,healthy_environment_continued=True,
                    no_Q_or_optimizer_updates=True,initial_settling_steps=settled)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
        print('[PROJECTED BASE SAFETY PROBE] '+json.dumps(result),flush=True)
    finally:
        if env is not None:env.close()
        app.close()

if __name__=='__main__':main()
