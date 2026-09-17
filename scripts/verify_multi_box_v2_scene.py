#!/usr/bin/env python3
"""Low-memory one-environment smoke test for the multi-box v2 scene."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sys
from types import SimpleNamespace

from kuavo_isaaclab_scene.robots.gripper_config import (
    add_gripper_cli_args,
    export_gripper_cli,
)
from kuavo_isaaclab_scene.robots.robot_model import (
    add_robot_model_cli_args,
    export_robot_model_cli,
)


parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--box-count", type=int, default=12)
add_robot_model_cli_args(parser)
add_gripper_cli_args(parser)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1 or args.steps > 120:
    parser.error("--steps must be between 1 and 120")
if not 1 <= args.box_count <= 12:
    parser.error("--box-count must be between 1 and 12")
if args.robot_model is None:
    args.robot_model = "s63"
if args.gripper is None:
    args.gripper = "leju-twofinger"
if args.robot_model != "s63" or args.gripper != "leju-twofinger":
    parser.error("This v2 smoke test currently verifies s63 + leju-twofinger only")
export_robot_model_cli(args)
export_gripper_cli(args)
os.environ["KUAVO_RACK_ROLLERS"] = "0"
args.headless = True
args.enable_cameras = False
launcher = AppLauncher(args)
app = launcher.app


def main():
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene

    from kuavo_isaaclab_scene.rl.multi_box.scene.randomization import reset_randomized_scene
    from kuavo_isaaclab_scene.rl.multi_box.scene.scene_cfg import build_scene
    from kuavo_isaaclab_scene.rl.multi_box.scene.spawn import physical_asset_names
    from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec

    torch.manual_seed(17)
    device = args.device or "cuda:0"
    dt = 1.0 / 120.0
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=dt,
        render_interval=4,
        device=device,
    ))
    spec = replace(MultiBoxSpec(), full_spawn_count_range=(args.box_count, args.box_count))
    scene_cfg, pool = build_scene(spec, num_envs=1, env_spacing=8.0)
    from kuavo_isaaclab_scene.rl.multi_box.debug.contact_sensors import add_quest_contact_sensors
    add_quest_contact_sensors(scene_cfg)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()
    wrapper = SimpleNamespace(
        scene=scene,
        num_envs=1,
        device=torch.device(device),
        cfg=SimpleNamespace(multi_box=spec, scene=scene_cfg),
    )
    reset_randomized_scene(wrapper, torch.tensor([0], device=device))

    robot = scene["robot"]
    for _ in range(args.steps):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)

    names = physical_asset_names()
    if len(pool) != 18 or len(names) != 18 or len(set(names)) != 18:
        raise AssertionError("Expected 18 distinct physical box-pool assets")
    counts = wrapper._multi_box_counts
    active = wrapper._multi_box_active
    pool_ids = wrapper._multi_box_pool_ids
    if not torch.equal(active.sum(-1), counts):
        raise AssertionError("Active logical-box mask does not match sampled count")
    selected_pool = pool_ids[active]
    if len(torch.unique(selected_pool)) != len(selected_pool):
        raise AssertionError("Active logical boxes mapped to duplicate physical assets")
    for name in ("rack", *names, "conveyor_surface"):
        if not torch.isfinite(scene[name].data.root_pos_w).all():
            raise AssertionError(f"Non-finite root position: {name}")

    from kuavo_isaaclab_scene.rl.multi_box.debug.isaac_metrics import IsaacMultiBoxMetricAdapter
    metrics = IsaacMultiBoxMetricAdapter(wrapper).measure()
    for phase, values in metrics.potentials_by_phase.items():
        if not all(torch.isfinite(value).all() for value in values.values()):
            raise AssertionError(f"Non-finite {phase} shadow potential")

    report = {
        "schema": "multi_box_v2_scene_smoke_v1",
        "device": device,
        "robot_model": args.robot_model,
        "gripper": args.gripper,
        "num_envs": 1,
        "steps": args.steps,
        "physical_box_assets": len(pool),
        "active_logical_boxes": int(counts.item()),
        "active_pool_ids": selected_pool.cpu().tolist(),
        "rack_xy_delta_m": wrapper._multi_box_rack_xy_delta[0].cpu().tolist(),
        "rack_yaw_delta_rad": float(wrapper._multi_box_rack_yaw_delta[0].item()),
        "conveyor_xy_delta_m": wrapper._multi_box_conveyor_xy_delta[0].cpu().tolist(),
        "conveyor_yaw_delta_rad": float(wrapper._multi_box_conveyor_yaw_delta[0].item()),
        "shadow_target": metrics.target_logical_id,
        "shadow_target_type": metrics.target_box_type,
        "contact_adapter_available": bool(
            metrics.diagnostics.get("contact_adapter_available", 0.0)),
        "shadow_potential_keys": {
            phase: sorted(values) for phase, values in metrics.potentials_by_phase.items()
        },
    }
    print("[MULTI BOX V2 SMOKE] " + json.dumps(report), flush=True)


exit_code = 0
try:
    main()
except BaseException:
    exit_code = 1
    import traceback
    traceback.print_exc()
finally:
    # This is a disposable smoke process with no Replicator writers.  Fast
    # shutdown prevents a failed/finished verifier from lingering beside a VR run.
    app.close(wait_for_replicator=False, skip_cleanup=True)
sys.stdout.flush()
sys.stderr.flush()
os._exit(exit_code)
