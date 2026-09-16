"""Replay a real S63 /joint_cmd trajectory in Isaac; never contacts ROS."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "real_robot_calibration/s63_body_arms"))
from real_vr_replay import sample_command, validate
from trajectory import NAMES
from kuavo_isaaclab_scene.robots.robot_model import (
    add_robot_model_cli_args, export_robot_model_cli, resolve_robot_model,
)
from kuavo_isaaclab_scene.robots.gripper_config import add_gripper_cli_args, export_gripper_cli


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--arm-stiffness", type=float)
    parser.add_argument("--arm-damping", type=float)
    parser.add_argument("--arm-effort-limit", type=float)
    parser.add_argument("--servo-config", type=Path,
                        help="S63 simulation-gain JSON used only for this replay")
    parser.add_argument("--feedforward-mode", choices=("profile","none","recorded-total","gravity-plus-recorded","inverse-dynamics"), default="profile",
                        help="profile keeps --dynamics-profile; none explicitly forces gravity-only")
    parser.add_argument("--feedforward-joints", choices=("arms","all"), default="arms")
    parser.add_argument("--feedforward-scale", type=float, default=1.0)
    parser.add_argument("--scene", choices=("robot-only", "workcell"), default="robot-only",
                        help="Use robot-only for free-space actuator calibration")
    add_robot_model_cli_args(parser); add_gripper_cli_args(parser)
    args = parser.parse_args()
    if args.robot_model != "s63" or args.gripper != "leju-twofinger":
        parser.error("Replay is pinned to --robot-model s63 --gripper leju-twofinger")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max-steps must be positive")
    for name in ("arm_stiffness", "arm_damping", "arm_effort_limit"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(name.replace("_", "-")+" must be positive finite")
    if not math.isfinite(args.feedforward_scale) or args.feedforward_scale < 0:
        parser.error("feedforward-scale must be finite and nonnegative")
    args.output = args.output.expanduser().resolve()
    summary_path = args.output.with_suffix(".summary.json")
    if args.output.exists() or summary_path.exists():
        parser.error("Choose a new response output filename")
    plan = validate(json.loads(args.plan.read_text()))
    if args.feedforward_mode in ("recorded-total", "gravity-plus-recorded") and plan.get("schema") != "kuavo_real_vr_replay_v2":
        parser.error("Recorded feedforward requires a v2 replay plan")
    plan["_validated"] = True
    if args.servo_config is not None:
        args.servo_config = args.servo_config.expanduser().resolve()
        if not args.servo_config.is_file():
            parser.error("servo-config does not exist")
        os.environ["KUAVO_S63_SERVO_CONFIG"] = str(args.servo_config)
    export_robot_model_cli(args); export_gripper_cli(args)
    model = resolve_robot_model()

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device=args.device).app
    env = probe = None
    report = {"plan":str(args.plan.resolve()),"output":str(args.output),"complete":False,
              "steps":0,"real_commands_sent":False,"robot_model":"s63","gripper":"leju-twofinger"}
    if args.servo_config is not None:
        report["servo_config"] = str(args.servo_config)
        report["servo_config_sha256"] = hashlib.sha256(args.servo_config.read_bytes()).hexdigest()
    try:
        import torch
        import isaaclab.envs.mdp as mdp
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass
        from kuavo_isaaclab_scene.envs.teleop_env import KuavoQuestTeleopEnvCfg, set_domain_randomization
        from kuavo_isaaclab_scene.teleop.teleop_scene import configure_scene_detail
        from kuavo_isaaclab_scene.recording.joint_response import JointResponseProbe

        @configclass
        class ReplayActionsCfg:
            position = mdp.JointPositionActionCfg(asset_name="robot", joint_names=NAMES,
                preserve_order=True, scale=1., use_default_offset=False)
            velocity = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=NAMES,
                preserve_order=True, scale=1., use_default_offset=False)

        cfg = KuavoQuestTeleopEnvCfg(); cfg.sim.device=args.device; cfg.actions=ReplayActionsCfg()
        if args.scene == "robot-only":
            replay_robot_cfg = cfg.scene.robot

            @configclass
            class RobotOnlyReplaySceneCfg(InteractiveSceneCfg):
                robot = replay_robot_cfg

            cfg.scene = RobotOnlyReplaySceneCfg(
                num_envs=1, env_spacing=5.0, replicate_physics=True
            )
        arms = cfg.scene.robot.actuators["arms"]
        if args.arm_stiffness is not None: arms.stiffness=args.arm_stiffness
        if args.arm_damping is not None: arms.damping=args.arm_damping
        if args.arm_effort_limit is not None: arms.effort_limit_sim=args.arm_effort_limit
        report["arm_actuator_override"]={"stiffness":args.arm_stiffness,
            "damping":args.arm_damping,"effort_limit_sim":args.arm_effort_limit}
        cfg.scene.robot.init_state.joint_pos.update(dict(zip(NAMES,plan["samples"][0]["target_q_rad"])))
        for name in ("robustness_camera","left_wrist_camera","right_wrist_camera",
                     "xr_left_eye_camera","xr_right_eye_camera"):
            setattr(cfg.scene,name,None)
        set_domain_randomization(cfg,False)
        if args.scene == "workcell":
            configure_scene_detail(cfg,"compact")
        env=ManagerBasedRLEnv(cfg); env.reset(seed=42)
        robot=env.scene["robot"]
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.set_joint_velocity_target(robot.data.default_joint_vel)
        probe=JointResponseProbe(env,model,args.output,"real_vr_joint_cmd_sim_replay")
        required=math.ceil(plan["duration_s"]/env.step_dt)+1
        actual=min(required,args.max_steps) if args.max_steps is not None else required
        report.update(required_steps=required,control_dt_s=env.step_dt)
        report["scene"] = args.scene
        report.update(feedforward_mode=args.feedforward_mode,
                      feedforward_joints=args.feedforward_joints,
                      feedforward_scale=args.feedforward_scale)
        ids=[robot.joint_names.index(n) for n in NAMES];max_error=0.
        ff_mode={"profile":None,"none":"off","recorded-total":"replace_gravity",
                 "gravity-plus-recorded":"add_to_gravity",
                 "inverse-dynamics":"inverse_dynamics"}[args.feedforward_mode]
        for index in range(actual):
            elapsed=min(index*env.step_dt,plan["duration_s"])
            command=sample_command(plan,elapsed)
            q=command["target_q_rad"];v=command["target_v_rad_s"]
            real_q=command["real_q_rad"];real_v=command["real_v_rad_s"]
            source_elapsed=command["source_elapsed_s"];phase=command["phase"]
            feedforward=torch.zeros_like(robot.data.joint_pos_target)
            feedforward_mask=torch.zeros_like(robot.data.joint_pos_target,dtype=torch.bool)
            start=4 if args.feedforward_joints == "arms" else 0
            for real_index in range(start,18):
                feedforward[:,ids[real_index]]=float(command["target_tau"][real_index])*args.feedforward_scale
                feedforward_mask[:,ids[real_index]]=True
            if ff_mode is not None:
                robot.set_command_feedforward_torque(feedforward,ff_mode,feedforward_mask)
            action=torch.tensor([q+v],dtype=torch.float32,device=env.device)
            probe.begin(True);env.step(action)
            probe.end(tracking_valid=True,context={"plan":str(args.plan.resolve()),
                "phase":phase,"replay_elapsed_s":elapsed,"source_elapsed_s":source_elapsed,
                "real_q_rad":real_q,"real_v_rad_s":real_v,
                "target_tau":command["target_tau"],"control_modes":command["control_modes"]})
            report["steps"]=index+1
            target=robot.data.joint_pos_target[0,ids]
            max_error=max(max_error,float(torch.max(torch.abs(target-action[0,:18]))))
        if max_error>1e-6: raise RuntimeError("Simulation changed the logical replay command")
        report.update(complete=actual==required,max_logical_command_error_rad=max_error)
    except BaseException as exc:
        report["error"]=f"{type(exc).__name__}: {exc}";traceback.print_exc();sys.stderr.flush();raise
    finally:
        try:
            if probe is not None: probe.close()
        finally:
            summary_path.parent.mkdir(parents=True,exist_ok=True)
            with summary_path.open("x") as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+"\n")
            print("[REAL VR REPLAY] "+json.dumps(report),flush=True)
            if "error" not in report and env is not None:env.close()
            sys.stdout.flush();sys.stderr.flush();os._exit(1 if "error" in report else 0)


if __name__ == "__main__": main()
