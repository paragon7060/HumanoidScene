#!/usr/bin/env python3
"""Capture both fixed-base Isaac claws' time response. No robot or SSH access."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import traceback
import time
from statistics import median

FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["KUAVO_ROBOT_MODEL"] = "s63"
os.environ["KUAVO_GRIPPER"] = "leju-twofinger"
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("continuous", "incremental"), default="continuous")
parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=3)
parser.add_argument("--filter-stages", type=int, choices=(1, 2),
                    help="Experimental filter stages; does not edit grippers.json")
parser.add_argument("--benchmark-physics", action="store_true",
                    help="Compare stationary two-claw physics-loop cost for one/two stages")
parser.add_argument("--output", type=Path)
parser.add_argument("--target-filter", type=float, nargs=4, metavar=("LEFT_CLOSE", "LEFT_OPEN", "RIGHT_CLOSE", "RIGHT_OPEN"),
                    help="Experimental time constants in seconds; does not edit grippers.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output is None:
    args.output = FOLDER / f"reports/isaac_dynamics_current_{args.mode}.json"
launcher = AppLauncher(args)
app = launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from kuavo_isaaclab_scene.robots.claw_assets.isaaclab import make_claw_cfg
from kuavo_isaaclab_scene.robots.gripper_config import load_gripper_settings
from kuavo_isaaclab_scene.robots.gripper_runtime import build_gripper_action_cfg, FilteredBinaryJointPositionActionCfg
from kuavo_isaaclab_scene.robots.claw_assets.geometry import TwoFingerGeometry
from kuavo_isaaclab_scene.robots.claw_assets.linkage import FINGER_PIN, FOLLOWER_PIN, initial_passive_positions, pin_for
from kuavo_isaaclab_scene.rl.mdp.actions import IncrementalGripperCfg
from analyze_dynamics import timing_metrics


def main():
    settings = load_gripper_settings("leju-twofinger")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120, device=args.device))
    hands = {}
    for side, offset in (("left", -.2), ("right", .2)):
        cfg = make_claw_cfg(side, f"/World/{side.title()}Claw", pos=(offset, 0, 0))
        cfg.spawn.articulation_props.solver_position_iteration_count = 32
        cfg.spawn.articulation_props.solver_velocity_iteration_count = 8
        cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
        opened = settings.command_for(side, settings.open_command)
        cfg.init_state.joint_pos = {**opened, **initial_passive_positions(opened)}
        cfg.actuators["drivers"].armature = .001
        cfg.actuators["passive"].damping = .01
        cfg.actuators["passive"].armature = .001
        cfg.actuators["passive"].velocity_limit_sim = 5.
        hands[side] = dict(claw=Articulation(cfg), geometry=TwoFingerGeometry(side), cfg=cfg)
    sim.reset()
    for side, h in hands.items():
        claw = h["claw"]
        # A standalone Articulation has no scene reset manager to apply its
        # configured default joint state. Start at the authored open linkage.
        claw.write_joint_state_to_sim(claw.data.default_joint_pos, claw.data.default_joint_vel)
        env = SimpleNamespace(scene={"robot":claw}, num_envs=1, device=claw.device,
                              cfg=SimpleNamespace(task=SimpleNamespace(reset_settle_seconds=0, reset_bank=False)))
        action_cfg = build_gripper_action_cfg(settings, side, continuous=True)
        if args.target_filter is not None:
            index = 0 if side == "left" else 2
            action_cfg.target_filter = dict(closing_time_constant_s=args.target_filter[index],
                                           opening_time_constant_s=args.target_filter[index+1])
        if args.filter_stages is not None:
            if action_cfg.target_filter is None:
                raise ValueError("Filter stages require configured or experimental time constants")
            action_cfg.target_filter = {**action_cfg.target_filter, "stages": args.filter_stages}
        if args.mode == "incremental":
            action_cfg = IncrementalGripperCfg(asset_name=action_cfg.asset_name, joint_names=action_cfg.joint_names,
                open_command_expr=action_cfg.open_command_expr, close_command_expr=action_cfg.close_command_expr,
                position_mapping=action_cfg.position_mapping,target_filter=action_cfg.target_filter)
        h["action"] = action_cfg.class_type(action_cfg, env)
        h["action"].reset()
        h["binary_semantics_checked"] = False
        if action_cfg.target_filter is not None:
            binary_cfg = FilteredBinaryJointPositionActionCfg(asset_name=action_cfg.asset_name,
                joint_names=action_cfg.joint_names,open_command_expr=action_cfg.open_command_expr,
                close_command_expr=action_cfg.close_command_expr,target_filter=action_cfg.target_filter)
            binary = binary_cfg.class_type(binary_cfg,env)
            for value,dtype,closed in ((0.,torch.float32,False),(-1.,torch.float32,True),
                                       (False,torch.bool,True),(True,torch.bool,False)):
                binary.process_actions(torch.tensor([[value]],dtype=dtype,device=claw.device))
                expected = binary._close_command if closed else binary._open_command
                if not torch.allclose(binary._desired_actions,expected):
                    raise RuntimeError("Filtered binary action changed Isaac sign/bool semantics")
            binary.reset()
            if not torch.equal(binary._desired_actions,claw.data.joint_pos[:,binary._joint_ids]):
                raise RuntimeError("Binary filter reset did not clear pending targets")
            h["binary_semantics_checked"] = True
        prefix = side[0]
        h["finger_ids"] = [claw.body_names.index(f"{prefix}_{jaw}_finger") for jaw in "fb"]
        h["bar_ids"] = [claw.body_names.index(f"{prefix}_{jaw}_bar_4") for jaw in "fb"]
        h["base_id"] = claw.body_names.index(f"{prefix}_twofinger_base")
        h["pin0"] = torch.tensor([pin_for(jaw,FINGER_PIN) for jaw in "fb"],device=claw.device)
        h["pin1"] = torch.tensor([pin_for(jaw,FOLLOWER_PIN) for jaw in "fb"],device=claw.device)
        h["tips"] = {jaw:torch.tensor(p,device=claw.device,dtype=torch.float32) for jaw,p in h["geometry"].tip_points.items()}
        h["samples"] = []
        h["max_closure_mm"] = 0.
    tick_count = 0
    args.output.parent.mkdir(parents=True,exist_ok=True)
    trace_path = args.output.with_suffix(".jsonl")
    comparisons = []
    real = json.loads((FOLDER / "reports/dynamics_speed_comparison.json").read_text())
    dt = sim.get_physics_dt()
    try:
        with trace_path.open("w") as trace:
            def tick(trial, step, percent):
                nonlocal tick_count
                for h in hands.values():
                    h["action"].apply_actions()
                    h["claw"].write_data_to_sim()
                sim.step(render=False)
                tick_count += 1
                for side,h in hands.items():
                    claw,action = h["claw"],h["action"]
                    claw.update(dt)
                    q = claw.data.joint_pos[0,action._joint_ids]
                    feedback = ((q-action._open_command)/(action._close_command-action._open_command)*100).tolist()
                    p0 = claw.data.body_link_pos_w[0,h["finger_ids"]]+quat_apply(claw.data.body_link_quat_w[0,h["finger_ids"]],h["pin0"])
                    p1 = claw.data.body_link_pos_w[0,h["bar_ids"]]+quat_apply(claw.data.body_link_quat_w[0,h["bar_ids"]],h["pin1"])
                    closure = float(torch.linalg.vector_norm(p0-p1,dim=-1).max())*1000
                    h["max_closure_mm"] = max(h["max_closure_mm"],closure)
                    points = {}
                    base_pos = claw.data.body_link_pos_w[0,h["base_id"]]
                    base_quat = claw.data.body_link_quat_w[0,h["base_id"]]
                    for jaw,body_id in zip("fb",h["finger_ids"]):
                        local = h["tips"][jaw]
                        world = quat_apply(claw.data.body_link_quat_w[0,body_id].expand(len(local),-1),local)+claw.data.body_link_pos_w[0,body_id]
                        points[jaw] = quat_apply_inverse(base_quat.expand(len(local),-1),world-base_pos)
                    gap = float(points["f"][:,0].min()-points["b"][:,0].max())*1000
                    h["samples"].append((tick_count*dt,feedback))
                    trace.write(json.dumps(dict(time_s=tick_count*dt,side=side,trial=trial,step=step,
                        command_percent=percent,driver_feedback_fraction_percent=feedback,
                        driver_target_rad=action.processed_actions[0].tolist(),driver_actual_rad=q.tolist(),
                        tip_gap_mm=gap,closure_mm=closure))+"\n")
            for _ in range(180):
                tick(0,0,0)
            for trial in range(1,args.repeats+1):
                for step,percent in enumerate((0,25,75,25),1):
                    command_time = tick_count*dt
                    for h in hands.values():
                        h["samples"] = [s for s in h["samples"] if s[0]>=command_time-1.1]
                    for target_tick in range(720):
                        for h in hands.values():
                            action = h["action"]
                            desired = torch.tensor([[1-percent/50]],device=action.device)
                            if args.mode == "continuous" and target_tick == 0:
                                action.process_actions(desired)
                            elif args.mode == "incremental" and target_tick % 4 == 0:
                                delta = ((desired-action._signed_target)/action.cfg.delta_scale).clamp(-1,1)
                                action.process_actions(delta)
                        tick(trial,step,percent)
                    if step not in (3,4):
                        continue
                    for side,h in hands.items():
                        direction = "closing" if step == 3 else "opening"
                        reference = next(r for r in real["pooled"] if r["side"]==side and r["direction"]==direction)
                        metrics = timing_metrics(h["samples"],command_time,0)
                        rear_metrics = timing_metrics(h["samples"],command_time,1)
                        duration = metrics.get("travel_10_90_s")
                        reference_time = reference["travel_10_90_s"]["median"]
                        row = dict(side=side,trial=trial,direction=direction,command_percent=percent,
                            isaac_primary_driver=metrics,isaac_secondary_driver=rear_metrics,
                            real_feedback_10_90_s=reference_time,
                            relative_time_difference=duration/reference_time-1 if duration is not None else None)
                        comparisons.append(row)
                        print(json.dumps(row),flush=True)
        physics_cost = None
        if args.benchmark_physics:
            rounds = {1: [], 2: []}
            original_stages = {side:h["action"]._target_filter.stages for side,h in hands.items()}
            for stages in (1, 2, 2, 1, 1, 2):
                for h in hands.values():
                    action = h["action"]
                    action._target_filter.stages = stages
                    q = h["claw"].data.joint_pos[:,action._joint_ids].clone()
                    action._target_filter.reset(q)
                    action._desired_actions.copy_(q)
                for count in (60, 400):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    for _ in range(count):
                        for h in hands.values():
                            h["action"].apply_actions()
                            h["claw"].write_data_to_sim()
                        sim.step(render=False)
                        for h in hands.values():h["claw"].update(dt)
                    torch.cuda.synchronize()
                    if count == 400:rounds[stages].append((time.perf_counter()-started)/count)
            for side,h in hands.items():h["action"]._target_filter.stages = original_stages[side]
            physics_cost = dict(one_stage_ms=median(rounds[1])*1000,
                two_stage_ms=median(rounds[2])*1000,
                relative_difference=median(rounds[2])/median(rounds[1])-1,
                rounds_s=rounds,scope="Stationary fixed-base two-claw physics/action loop, no trace writes or rendering; not full scene or RL throughput")
        report = dict(mode=args.mode,physics_dt_s=dt,repeats=args.repeats,trace=str(trace_path.resolve()),
            physics_loop_cost=physics_cost,
            initialization="Configured open driver and analytically consistent passive joints explicitly written before startup settling",
            test="Two independent fixed-base claws, no object, gravity disabled; same static mapping/runtime actions and packaged S63 driver PD",
            control_decimation=4 if args.mode=="incremental" else None,
            incremental_delta_scale=.12 if args.mode=="incremental" else None,
            driver_stiffness=hands["left"]["cfg"].actuators["drivers"].stiffness,
            driver_damping=hands["left"]["cfg"].actuators["drivers"].damping,
            target_filter={side:h["action"].cfg.target_filter for side,h in hands.items()},
            binary_semantics_checked={side:h["binary_semantics_checked"] for side,h in hands.items()},
            max_closure_mm={side:h["max_closure_mm"] for side,h in hands.items()},comparison=comparisons,
            scope="Compare normalized Isaac primary driver timing to real encoder feedback as a proxy. Donor/real transmission relation and optical tip timing are not verified. Real onset includes ROS overhead; Isaac time excludes wall-clock startup. No transport delay fitted or PD gains changed. RL reference is a single real setpoint and does not share RL's incremental input trajectory.")
        report["closure_passed"] = all(h["max_closure_mm"]<=1 for h in hands.values())
        report["all_metrics_valid"] = all(r["isaac_primary_driver"]["valid"] and r["isaac_secondary_driver"]["valid"] for r in comparisons)
        args.output.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(dict(report=str(args.output.resolve()),closure_passed=report["closure_passed"],all_metrics_valid=report["all_metrics_valid"])),flush=True)
    finally:
        sim.clear_all_callbacks()
        sim.clear_instance()


try:
    main()
except Exception:
    traceback.print_exc()
    import omni.kit.app
    omni.kit.app.get_app().post_quit(1)
finally:
    app.close()
