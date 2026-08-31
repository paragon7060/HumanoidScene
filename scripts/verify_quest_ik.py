"""Run with Isaac Lab Python; compare target retention without a Quest connection."""

import os
import sys
from types import MethodType

from isaaclab.app import AppLauncher

launcher = AppLauncher(headless=True, enable_cameras=False)

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from kuavo_isaaclab_scene.teleop_env import KuavoQuestTeleopEnvCfg, set_domain_randomization


def main():
    cfg = KuavoQuestTeleopEnvCfg()
    cfg.scene.robustness_camera = None
    cfg.scene.left_wrist_camera = None
    cfg.scene.right_wrist_camera = None
    cfg.observations.vision = None
    set_domain_randomization(cfg, False)
    env = ManagerBasedRLEnv(cfg)
    terms = [env.action_manager.get_term(name) for name in ("left_arm", "right_arm")]
    originals = [term.process_actions for term in terms]
    robot = env.scene["robot"]
    ids = [robot.find_bodies(name)[0][0] for name in ("zarm_l7_end_effector", "zarm_r7_end_effector")]
    results = {}
    try:
        for mode in ("legacy", "persistent"):
            for term, original in zip(terms, originals):
                term.process_actions = (MethodType(DifferentialInverseKinematicsAction.process_actions, term)
                                        if mode == "legacy" else original)
            env.reset(seed=42)
            command = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
            for _ in range(100):
                env.step(command)
            for term in terms:
                term.hold_current_pose()
            target_starts = [term._target_position.clone() for term in terms]
            start = robot.data.body_pos_w[0, ids].clone()
            # A slow 5 cm command in robot-root +X, then hold still.
            for step in range(350):
                command[0, 0] = command[0, 6] = 0.001 if step < 50 else 0.0
                env.step(command)
            displacement = (robot.data.body_pos_w[0, ids] - start).norm(dim=-1)
            results[mode] = displacement.detach().cpu().tolist()
            print(f"[IK VERIFY] {mode}: hand displacement L/R (m)={results[mode]}", flush=True)
            if mode == "persistent":
                errors = [float(term.target_position_error()[0]) for term in terms]
                print(f"[IK VERIFY] final target errors (m)={errors}", flush=True)
                # Test retained command, not perfect physical tracking: gravity,
                # contacts and joint limits can leave a real residual error.
                for term, target_start in zip(terms, target_starts):
                    expected = target_start.clone()
                    expected[:, 0] += 0.05
                    torch.testing.assert_close(term._target_position, expected, atol=1e-5, rtol=0)
        assert sum(results["persistent"]) > 2 * sum(results["legacy"]), results
        for term in terms:
            term.process_actions(torch.ones((1, 6), device=env.device))
            position, _ = term._compute_frame_pose()
            assert float((term._ik_controller.ee_pos_des - position).norm()) <= 0.15001
            assert float(term.target_position_error()[0]) > 1.0  # Goal was retained, not clipped away.
            term.hold_current_pose()
            term.process_actions(torch.zeros((1, 6), device=env.device))
            assert float(term.target_position_error()[0]) < 1e-5
        print("[IK VERIFY] PASS: target retention, bounded correction, and explicit pause.", flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
