"""Measure stationary arm jitter and recovery in real physics, without XR."""
import argparse
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--legacy-ik", type=Path, help="Previous teleop_ik.py for an A/B run")
parser.add_argument("--legacy-physics", action="store_true", help="Use source USD inertials and old solver settings")
args = parser.parse_args()
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, device="cpu").app
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from kuavo_isaaclab_scene.envs.teleop_env import KuavoQuestTeleopEnvCfg, set_domain_randomization
from kuavo_isaaclab_scene.teleop.teleop_scene import configure_scene_detail


def main():
    cfg = KuavoQuestTeleopEnvCfg()
    cfg.sim.device = "cpu"
    cfg.scene.robot.init_state.pos = (0., -5., 0.)
    cfg.scene.robustness_camera = cfg.scene.left_wrist_camera = cfg.scene.right_wrist_camera = None
    cfg.actions.left_arm.controller.use_relative_mode = cfg.actions.right_arm.controller.use_relative_mode = False
    if args.legacy_physics:
        from isaaclab.sim.spawners.from_files import spawn_from_usd
        cfg.scene.robot.spawn.func = spawn_from_usd
        cfg.scene.robot.actuators["integrated_grippers"].armature = None
        cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 8
        cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 2
    if args.legacy_ik:
        import importlib.util
        spec = importlib.util.spec_from_file_location("legacy", str(args.legacy_ik))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg.actions.left_arm.class_type = cfg.actions.right_arm.class_type = module.PersistentTeleopIKAction
    set_domain_randomization(cfg, False)
    configure_scene_detail(cfg, "compact")
    env = ManagerBasedRLEnv(cfg)
    env.reset(seed=42)
    robot = env.scene["robot"]
    terms = [env.action_manager.get_term(side + "_arm") for side in ("left", "right")]
    action = torch.zeros((1, env.action_manager.total_action_dim))
    action[:, 16:18] = 1
    for index, term in enumerate(terms):
        pos, quat = term._compute_frame_pose()
        action[:, index * 7:index * 7 + 7] = torch.cat((pos, quat), -1)
    # Store a guaranteed reachable forward grasp from FK, then restore reset.
    q = robot.data.default_joint_pos.clone()
    for term in terms:
        q[:, term._joint_ids] = torch.tensor([-.15, 0., 0., -1., 0., 0., -.42])
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    env.sim.forward()
    env.scene.update(env.step_dt)
    reachable = torch.cat([torch.cat(term._compute_frame_pose(), -1) for term in terms], -1)
    env.reset(seed=42)
    results = []
    try:
        for phase, x, z in (("raised", .35, 1.), ("outside_reach", .65, 1.5),
                            ("return", .35, 1.), ("forward_grasp", None, None)):
            if phase == "forward_grasp":
                action[:, :14] = reachable
            else:
                action[0, :3] = torch.tensor([x, .25, z])
                action[0, 7:10] = torch.tensor([x, -.25, z])
            positions, joints, errors = [], [], []
            for _ in range(240):
                env.step(action)
                positions.append(torch.cat([t._compute_frame_pose()[0] for t in terms], -1)[0].numpy().copy())
                joints.append(robot.data.joint_pos[0, terms[0]._joint_ids + terms[1]._joint_ids].numpy().copy())
                errors.append([float(t.target_position_error()[0]) for t in terms])
            pos, q = np.array(positions)[-90:], np.array(joints)[-90:]
            metrics = {
                "phase": phase,
                "position_error_m": np.mean(errors[-90:], 0).tolist(),
                # Residual motion, not purely sensor jitter: outside-reach
                # phases may still be converging in this last 3-second window.
                "mean_step_mm": float(np.linalg.norm(np.diff(pos, axis=0).reshape(-1, 2, 3), axis=-1).mean() * 1000),
                "joint_accel_rms": float(np.sqrt(np.mean((np.diff(q, n=2, axis=0) / env.step_dt ** 2) ** 2))),
            }
            results.append(metrics)
            print("[STABILITY]", metrics, flush=True)
        name = "legacy" if args.legacy_ik else "new"
        path = Path(f"artifacts/quest-stability-{name}.json")
        path.write_text(json.dumps(results, indent=2) + "\n")
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
