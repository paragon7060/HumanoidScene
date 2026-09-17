#!/usr/bin/env python3
"""Measure actual Isaac body poses throughout a calibrated S63 hand cycle.

Only runs a standalone, fixed-base simulated claw. No SSH or hardware control.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ["KUAVO_ROBOT_MODEL"] = "s63"
os.environ["KUAVO_GRIPPER"] = "leju-twofinger"
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "reports/isaac_mapping_validation.json")
parser.add_argument("--steps-per-target", type=int, default=360)
parser.add_argument("--side", choices=("left", "right"), default="left")
parser.add_argument("--protocol", choices=("full_cycle", "calibration"), default="full_cycle",
                    help="calibration adds the measured minor-loop paths after the full cycle")
parser.add_argument("--mode", choices=("continuous", "incremental"), default="incremental",
                    help="incremental: actual RL action with 6 percent command steps; continuous: abrupt absolute targets")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps_per_target < 120:
    parser.error("At least 120 physics steps per target are required")
launcher = AppLauncher(args)
app = launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from kuavo_isaaclab_scene.robots.claw_assets.isaaclab import make_claw_cfg
from kuavo_isaaclab_scene.robots.gripper_config import load_gripper_settings
from kuavo_isaaclab_scene.robots.gripper_runtime import build_gripper_action_cfg
from kuavo_isaaclab_scene.robots.claw_assets.geometry import TwoFingerGeometry
from kuavo_isaaclab_scene.robots.claw_assets.linkage import FINGER_PIN, FOLLOWER_PIN, initial_passive_positions, pin_for
from kuavo_isaaclab_scene.rl.mdp.actions import IncrementalGripperCfg


def main():
    settings = load_gripper_settings("leju-twofinger")
    geometry = TwoFingerGeometry(args.side)
    prefix = args.side[0]
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120, device=args.device))
    cfg = make_claw_cfg(args.side, "/World/Claw")
    cfg.spawn.articulation_props.solver_position_iteration_count = 32
    cfg.spawn.articulation_props.solver_velocity_iteration_count = 8
    cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
    opened = settings.command_for(args.side, settings.open_command)
    cfg.init_state.joint_pos = {**opened, **initial_passive_positions(opened)}
    cfg.actuators["drivers"].armature = .001
    cfg.actuators["passive"].damping = .01
    cfg.actuators["passive"].armature = .001
    cfg.actuators["passive"].velocity_limit_sim = 5.0
    claw = Articulation(cfg)
    sim.reset()
    env = SimpleNamespace(scene={"robot": claw}, num_envs=1, device=claw.device,
                          cfg=SimpleNamespace(task=SimpleNamespace(reset_settle_seconds=0, reset_bank=False)))
    action_cfg = build_gripper_action_cfg(settings, args.side, continuous=True)
    if args.mode == "incremental":
        action_cfg = IncrementalGripperCfg(asset_name=action_cfg.asset_name,
            joint_names=action_cfg.joint_names, open_command_expr=action_cfg.open_command_expr,
            close_command_expr=action_cfg.close_command_expr, position_mapping=action_cfg.position_mapping,
            target_filter=action_cfg.target_filter)
    action = action_cfg.class_type(action_cfg, env)
    action.reset()
    finger_ids = [claw.body_names.index(f"{prefix}_{jaw}_finger") for jaw in "fb"]
    bar_ids = [claw.body_names.index(f"{prefix}_{jaw}_bar_4") for jaw in "fb"]
    base_id = claw.body_names.index(f"{prefix}_twofinger_base")
    pin0 = torch.tensor([pin_for(jaw, FINGER_PIN) for jaw in "fb"], device=claw.device)
    pin1 = torch.tensor([pin_for(jaw, FOLLOWER_PIN) for jaw in "fb"], device=claw.device)
    tips = {jaw: torch.tensor(points, device=claw.device, dtype=torch.float32)
            for jaw, points in geometry.tip_points.items()}
    rows = []
    trajectory = []
    for index, percent in enumerate([0,25,50,75,100,75,50,25,0]):
        direction = "closing" if index <= 4 else "opening"
        sample = load_samples[direction]
        trajectory.append({"command_percent":percent,"direction":direction,"case":"full_cycle",
                           "reference_width_mm":sample["width_mm"][sample["command_percent"].index(percent)],
                           "tolerance_mm":2.0})
    if args.protocol == "calibration":
        source = Path(__file__).resolve().parent / "data/s63_gripper_reversal_01/reversal_measurements.jsonl"
        previous = 0
        for line in source.read_text().splitlines():
            observation = json.loads(line)
            if observation["kind"] != "width_measurement": continue
            percent = observation["command_percent"]
            direction = "closing" if percent >= previous else "opening"
            trajectory.append({"command_percent":percent,"direction":direction,"case":observation["case"],
                               "reference_width_mm":observation[f"{args.side}_width_mm"],"tolerance_mm":4.03})
            previous = percent
    maximum_closure = 0.0
    maximum_settled_closure = 0.0
    maximum_reset_error = 0.0
    closure_peak = None
    def physics_tick(percent, tick):
        nonlocal maximum_closure, closure_peak
        action.apply_actions()
        claw.write_data_to_sim()
        sim.step(render=False)
        claw.update(sim.get_physics_dt())
        p0 = claw.data.body_link_pos_w[0, finger_ids] + quat_apply(claw.data.body_link_quat_w[0, finger_ids], pin0)
        p1 = claw.data.body_link_pos_w[0, bar_ids] + quat_apply(claw.data.body_link_quat_w[0, bar_ids], pin1)
        error = float(torch.linalg.vector_norm(p0-p1, dim=-1).max())
        if error > maximum_closure:
            maximum_closure = error
            closure_peak = {"target_percent": percent, "physics_tick": tick}
        return error
    try:
        for step, planned in enumerate(trajectory):
            percent = planned["command_percent"]
            direction = planned["direction"]
            expected = planned["reference_width_mm"]
            desired = torch.tensor([[1 - percent / 50]], device=claw.device)
            if args.mode == "continuous":
                action.process_actions(desired)
            else:
                for control_step in range(64):
                    if torch.allclose(action._signed_target, desired, atol=1e-6):
                        break
                    delta = ((desired - action._signed_target) / action.cfg.delta_scale).clamp(-1, 1)
                    action.process_actions(delta)
                    for substep in range(4):
                        physics_tick(percent, f"transition:{control_step*4+substep}")
                else:
                    raise RuntimeError("Incremental command did not reach requested value")
                action.process_actions(torch.zeros_like(desired))
            tail = []
            for tick in range(args.steps_per_target):
                closure = physics_tick(percent, tick)
                if tick >= args.steps_per_target - 60:
                    maximum_settled_closure = max(maximum_settled_closure, closure)
                    points = {}
                    base_pos = claw.data.body_link_pos_w[0, base_id]
                    base_quat = claw.data.body_link_quat_w[0, base_id]
                    for jaw, body_id in zip("fb", finger_ids):
                        local = tips[jaw]
                        world = quat_apply(claw.data.body_link_quat_w[0, body_id].expand(len(local), -1), local) + claw.data.body_link_pos_w[0, body_id]
                        points[jaw] = quat_apply_inverse(base_quat.expand(len(local), -1), world - base_pos)
                    tail.append(float(points["f"][:, 0].min() - points["b"][:, 0].max()) * 1000)
            gap = sum(tail) / len(tail)
            row = {"direction": direction, "command_percent": percent, "measured_width_mm": expected,
                   "case":planned["case"],"tolerance_mm":planned["tolerance_mm"],
                   "isaac_settled_width_mm": gap, "error_mm": gap - expected,
                   "tail_range_mm": max(tail)-min(tail),
                   "target_driver_rad": action.processed_actions[0].tolist(),
                   "actual_driver_rad": claw.data.joint_pos[0, action._joint_ids].tolist()}
            rows.append(row)
            print(json.dumps(row), flush=True)
            if args.mode == "incremental" and planned["case"] == "full_cycle" and step in (3, 7):
                q = claw.data.joint_pos[:, action._joint_ids].clone()
                action.reset()
                action.process_actions(torch.zeros_like(desired))
                maximum_reset_error = max(maximum_reset_error, float((action.processed_actions - q).abs().max()))
        report = {"robot_model": "s63", "side": args.side, "device": args.device,
                  "width_reference_source": "Full-cycle nodes use left measurements/copied right mapping, except closing-75 uses repeated both-hand data; minor-loop comparisons use side-specific measured widths",
                  "geometry_sha256": geometry.sha256,
                  "test": "Fixed-base standalone packaged claw; same runtime action and PD parameters as integrated S63; no object, gravity disabled",
                  "steps_per_target": args.steps_per_target, "physics_dt_s": sim.get_physics_dt(),
                  "mode": args.mode, "incremental_delta_scale": .12 if args.mode == "incremental" else None,
                  "protocol":args.protocol,
                  "incremental_control_decimation": 4 if args.mode == "incremental" else None,
                  "max_closure_error_mm": maximum_closure*1000,
                  "closure_peak": closure_peak,
                  "max_settled_closure_error_mm": maximum_settled_closure*1000,
                  "max_reset_zero_action_error_rad": maximum_reset_error,
                  "max_width_error_mm": max(abs(row["error_mm"]) for row in rows),
                  "max_full_cycle_fit_error_mm":max(abs(row["error_mm"]) for row in rows if row["case"] == "full_cycle"),
                  "acceptance": "Full-cycle fit error <=2 mm; independent minor-loop measurements <=4.03 mm (two bounded ±2 mm errors plus CAD lookup allowance); closure <=1 mm throughout",
                  "comparison": rows}
        report["static_gap_passed"] = all(abs(row["error_mm"]) <= row["tolerance_mm"] for row in rows) and report["max_settled_closure_error_mm"] <= 1
        report["passed"] = report["static_gap_passed"] and report["max_closure_error_mm"] <= 1 and maximum_reset_error <= 1e-4
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"passed": report["passed"], "max_width_error_mm": report["max_width_error_mm"],
                          "max_closure_error_mm": report["max_closure_error_mm"], "report": str(args.output)}), flush=True)
        if not report["passed"]:
            raise RuntimeError(f"Isaac calibration verification failed: {report}")
    finally:
        sim.clear_all_callbacks()
        sim.clear_instance()


load_samples = json.loads((Path(__file__).resolve().parent / "measured_parameters.json").read_text())["directional_width_samples"]
try:
    main()
except Exception:
    traceback.print_exc()
    import omni.kit.app
    omni.kit.app.get_app().post_quit(1)
finally:
    app.close()
