#!/usr/bin/env python3
"""Inject one non-finite dynamics row and verify isolated v2 SAC recovery."""
import argparse
from dataclasses import replace
import json
from pathlib import Path


def main():
    from isaaclab.app import AppLauncher
    from kuavo_isaaclab_scene.robots.robot_model import add_robot_model_cli_args, export_robot_model_cli
    from kuavo_isaaclab_scene.robots.gripper_config import add_gripper_cli_args, export_gripper_cli
    from kuavo_isaaclab_scene.robots.base_drive import add_base_drive_cli_args, export_base_drive_cli
    from kuavo_isaaclab_scene.workcell.rack_rollers import add_rack_roller_cli_args, export_rack_roller_cli
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    add_robot_model_cli_args(parser)
    add_gripper_cli_args(parser)
    add_base_drive_cli_args(parser)
    add_rack_roller_cli_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True, robot_model='s63', gripper='leju-twofinger', rack_rollers=True)
    args = parser.parse_args()
    export_robot_model_cli(args)
    export_gripper_cli(args)
    export_base_drive_cli(args)
    export_rack_roller_cli(args)
    app = AppLauncher(args).app
    env = None
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from kuavo_isaaclab_scene.rl.envs.terminal_observation import TerminalObservationMixin
        from kuavo_isaaclab_scene.rl.multi_box.training_env_cfg import MultiBoxGraspAssemblyEnvCfg
        from kuavo_isaaclab_scene.rl.runners.train_asymmetric_sac import _settle_initial_resets
        class Env(TerminalObservationMixin, ManagerBasedRLEnv):
            pass
        cfg = MultiBoxGraspAssemblyEnvCfg(num_envs=2)
        cfg.multi_box = replace(cfg.multi_box, self_collision_enabled=False)
        cfg.sim.device = args.device or 'cuda:0'
        env = Env(cfg)
        env.enable_numerical_dynamics_recovery()
        obs, _ = env.reset(seed=42)
        obs, settling_steps = _settle_initial_resets(env, obs)
        resets = []
        ordinary_reset = env._reset_idx
        def tracked_reset(ids):
            resets.extend(ids.tolist())
            return ordinary_reset(ids)
        env._reset_idx = tracked_reset
        robot = env.scene['robot']
        original_view = robot._root_physx_view
        class InjectedView:
            pending = True
            def __getattr__(self, name):
                return getattr(original_view, name)
            def get_gravity_compensation_forces(self):
                value = original_view.get_gravity_compensation_forces()
                if self.pending:
                    self.pending = False
                    value = value.clone()
                    value[0, -1] = float('nan')
                return value
        robot._root_physx_view = InjectedView()
        action = torch.zeros((2, env.action_manager.total_action_dim), device=env.device)
        obs, reward, done, timeout, info = env.step(action)
        first_mask = info['transition_numerical_failure'].tolist()
        assert first_mask == [True, False], first_mask
        assert done.tolist() == [True, False], done
        assert not env.termination_manager.get_term('success').any()
        assert resets and set(resets) == {0}, resets
        assert env.termination_manager.get_term('invalid_reset').tolist() == [True, False]
        assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
        assert torch.isfinite(robot.total_feedforward_torque).all()
        assert env._multi_box_reset_settling.ready[1]
        obs, _, _, _, info = env.step(action)
        assert not info['transition_numerical_failure'].any()
        assert torch.isfinite(obs['policy']).all()
        result = dict(passed=True, num_envs=2, settling_steps=settling_steps,
            failed_envs=[0], unaffected_envs=[1], first_failure_mask=first_mask,
            reset_ids=resets, finite_observations=True, finite_feedforward=True,
            next_step_failure_cleared=True, failed_transition_excluded_by_invalid_reset=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print('[NUMERICAL RECOVERY PROBE]', json.dumps(result), flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
