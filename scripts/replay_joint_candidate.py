"""Replay a bounded VR candidate in Isaac only; no ROS or real commands."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "real_robot_calibration/s63_body_arms"))
from vr_trajectory import load_limits, validate_candidate
from trajectory import NAMES, sample
from kuavo_isaaclab_scene.robots.robot_model import (
    add_robot_model_cli_args, export_robot_model_cli, resolve_robot_model,
)
from kuavo_isaaclab_scene.robots.gripper_config import add_gripper_cli_args, export_gripper_cli


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New response JSONL")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, help="Smoke test only; explicitly reports an incomplete replay")
    add_robot_model_cli_args(parser)
    add_gripper_cli_args(parser)
    args = parser.parse_args()
    if args.robot_model != "s63" or (args.max_steps is not None and args.max_steps < 1):
        parser.error("Initial candidate replay requires S63 and positive max-steps")
    args.output = args.output.expanduser().resolve()
    summary_path = args.output.with_suffix(".summary.json")
    if args.output.exists() or summary_path.exists():
        parser.error("Choose a new response output filename")
    export_robot_model_cli(args)
    export_gripper_cli(args)
    model = resolve_robot_model()
    plan = validate_candidate(json.loads(args.candidate.read_text()), load_limits(model.urdf_path))

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device=args.device).app
    env = probe = None
    report = {"output":str(args.output),"complete":False,"steps":0,
              "real_commands_sent":False,"collision_and_balance_validated":False}
    try:
        import torch
        import isaaclab.envs.mdp as mdp
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.utils import configclass
        from kuavo_isaaclab_scene.envs.teleop_env import KuavoQuestTeleopEnvCfg, set_domain_randomization
        from kuavo_isaaclab_scene.teleop.teleop_scene import configure_scene_detail
        from kuavo_isaaclab_scene.recording.joint_response import JointResponseProbe

        @configclass
        class ReplayActionsCfg:
            position = mdp.JointPositionActionCfg(
                asset_name="robot", joint_names=NAMES, preserve_order=True,
                scale=1., use_default_offset=False,
            )
            velocity = mdp.JointVelocityActionCfg(
                asset_name="robot", joint_names=NAMES, preserve_order=True,
                scale=1., use_default_offset=False,
            )

        cfg = KuavoQuestTeleopEnvCfg()
        cfg.sim.device = args.device
        cfg.actions = ReplayActionsCfg()
        cfg.scene.robot.init_state.joint_pos.update(dict(zip(NAMES, plan["baseline_rad"])))
        for name in ("robustness_camera", "left_wrist_camera", "right_wrist_camera",
                     "xr_left_eye_camera", "xr_right_eye_camera"):
            setattr(cfg.scene, name, None)
        set_domain_randomization(cfg, False)
        configure_scene_detail(cfg, "compact")
        env = ManagerBasedRLEnv(cfg)
        env.reset(seed=42)
        robot = env.scene["robot"]
        # Unlike the normal teleop actions, this replay does not write head or
        # hand commands. Hold their reset targets instead of Isaac's zero buffer.
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.set_joint_velocity_target(robot.data.default_joint_vel)
        probe = JointResponseProbe(env, model, args.output, "processed_candidate_sim_replay")
        steps = math.ceil(plan["duration_s"] / env.step_dt) + 1
        actual_steps = min(steps, args.max_steps) if args.max_steps is not None else steps
        report.update(required_steps=steps, control_dt_s=env.step_dt)
        max_target_error = 0.
        ids = [env.scene["robot"].joint_names.index(n) for n in NAMES]
        for i in range(actual_steps):
            t = min(i * env.step_dt, plan["duration_s"])
            q, v, label = sample(plan, t)
            action = torch.tensor([q + v], dtype=torch.float32, device=env.device)
            probe.begin(True)
            env.step(action)
            probe.end(tracking_valid=True, context={"candidate":str(args.candidate.resolve()),
                "candidate_elapsed_s":t,"phase":label,"collision_validation_performed":False})
            report["steps"] = i + 1
            target = env.scene["robot"].data.joint_pos_target[0, ids]
            max_target_error = max(max_target_error, float(torch.max(torch.abs(target-action[0,:18]))))
        if max_target_error > 1e-6:
            raise RuntimeError("Simulation action changed the logical replay command")
        report.update(complete=actual_steps == steps, max_logical_command_error_rad=max_target_error)
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if env is not None:
            try:
                robot = env.scene["robot"]
                for field in ("joint_pos", "joint_vel"):
                    values = getattr(robot.data, field)[0].detach().cpu().tolist()
                    report["nonfinite_" + field] = [n for n,v in zip(robot.joint_names, values) if not math.isfinite(v)]
                values = robot.root_physx_view.get_gravity_compensation_forces()[0].detach().cpu().tolist()
                report["nonfinite_gravity_joints"] = [n for n,v in zip(robot.joint_names, values) if not math.isfinite(v)]
            except Exception as diagnostic_error:
                report["diagnostic_error"] = str(diagnostic_error)
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        try:
            if probe is not None:
                probe.close()
        finally:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("x") as stream:
                stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
            print("[REPLAY] " + json.dumps(report), flush=True)
            # Kit shutdown can suppress Python exceptions or crash when a
            # physics state is invalid. Save the result first and preserve exit
            # status, as the other standalone physics verification scripts do.
            if "error" not in report and env is not None:
                env.close()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1 if "error" in report else 0)


if __name__ == "__main__":
    main()
