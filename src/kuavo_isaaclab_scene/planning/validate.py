"""GPU-scoped live Isaac model/world inspection; pregrasp execution is gated."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import traceback

from .config import load_yaml, output_directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    # Keep the CUDA mapping explicit: Isaac sees the first visible GPU as
    # logical ``cuda:0``. This works on KT (for example, physical GPU 2) and
    # on a local workstation (for example, physical GPU 0) without changing
    # the simulation code below.
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        parser.error("set CUDA_VISIBLE_DEVICES to the physical GPU for this lane")
    physical_gpu = visible_devices.split(",", 1)[0].strip()
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.device != "cuda:0" or not args.headless:
        parser.error("use --device cuda:0 --headless; logical cuda:0 is the first visible GPU")
    configs = {name: load_yaml(args.config_dir / f"{name}.yaml")
               for name in ("robot", "task1", "collection")}
    output = output_directory(configs["collection"], args.run_name)
    output.mkdir(parents=True, exist_ok=False)
    for name in configs:
        shutil.copyfile(args.config_dir / f"{name}.yaml", output / f"{name}.yaml")
    os.environ["KUAVO_ROBOT_MODEL"] = configs["robot"]["robot_model"]
    os.environ["KUAVO_GRIPPER"] = configs["robot"]["hand_model"]
    args.kit_args = (args.kit_args or "") + (
        " --/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=4"
        f" --/log/file={output}/kit.log")
    app = None
    report = {"physical_gpu": physical_gpu, "cuda_visible_devices": visible_devices, "status": "starting",
              "planning_success": "NOT_RUN", "execution_success": "NOT_RUN",
              "pregrasp_verified": "NOT_RUN"}
    try:
        app = AppLauncher(args).app
        report.update(inspect_live_scene(configs, output))
    except Exception as exc:
        report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (output / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if app is not None:
            app.close(wait_for_replicator=False)
    print(json.dumps({"report": str(output / "validation.json"), "status": report["status"]}))


def inspect_live_scene(configs, output):
    # These imports must remain after AppLauncher.
    import numpy as np
    import torch
    from isaaclab.sim import SimulationContext, SimulationCfg
    from isaaclab.scene import InteractiveScene
    from ..rl.scenes.scene_cfg import build_scene
    from ..rl.scenes.layout import active_rack_box_scene_keys
    from ..rl.tasks.specs import TaskSpec
    from ..robots.initial_states import load_initial_state, apply_initial_state
    from ..robots.robot_model import resolve_robot_model
    from .geometry import inverse_transform, pose_matrix, matrix_pose
    from .robot_model import UrdfModel, joint_indices
    from .world import snapshot_colliders, world_config
    from types import SimpleNamespace

    cfg = configs["robot"]
    task = configs["task1"]
    dt = float(cfg["physics_dt_s"])
    spec = TaskSpec(name="pick", control_mode="arms-only",
                    box_names=active_rack_box_scene_keys(), cargo_per_box=0, prefill_count=0,
                    grasp_mode="body", randomization=False)
    sim = SimulationContext(SimulationCfg(dt=dt, device="cuda:0", use_fabric=False))
    scene_cfg, geometry = build_scene(spec, num_envs=1, cameras=False)
    # Local validation override: do not inherit the RL self-collision-off default.
    scene_cfg.robot.spawn.articulation_props.enabled_self_collisions = True
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    state = load_initial_state(task["initial_state"], robot_model=cfg["robot_model"], gripper=cfg["hand_model"])
    env = SimpleNamespace(scene=scene, num_envs=1, device="cuda:0")
    apply_initial_state(env, None, state, task["initial_state"])
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(dt)
    robot = scene["robot"]
    model = UrdfModel(resolve_robot_model().urdf_path)
    arm_names = [f"zarm_{s}{j}_joint" for s in ("l", "r") for j in range(1, 8)]
    arm_ids = joint_indices(arm_names, robot.joint_names)
    commanded = robot.data.joint_pos.clone()
    # Hold actual reset posture. No post-reset state writes, no kinematic base moves.
    contacts, fk_rows, frames = [], [], []
    for step in range(task["settle_steps"]):
        robot.set_joint_position_target(commanded)
        robot.set_joint_velocity_target(torch.zeros_like(commanded))
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)
        if step % 4 == 0:
            frames.append({"time_s": (step+1)*dt,
                           "robot_body_poses_w": robot.data.body_link_pose_w[0].cpu().tolist(),
                           "boxes": {name: scene[name].data.body_link_pose_w[0].cpu().tolist()
                                     for name in spec.box_names}})
        if step % 20 == 0 or step == task["settle_steps"] - 1:
            q = dict(zip(robot.joint_names, robot.data.joint_pos[0].cpu().tolist(), strict=True))
            base = pose_matrix(robot.data.root_pose_w[0].cpu().tolist())
            for side in ("left", "right"):
                link = cfg[f"{side}_tcp_link"]
                ids, _ = robot.find_bodies(link)
                if len(ids) != 1:
                    raise ValueError(f"TCP body is not unique in Isaac: {link}")
                measured = pose_matrix(robot.data.body_link_pose_w[0, ids[0]].cpu().tolist())
                predicted = base @ model.fk(link, q)
                delta = inverse_transform(predicted) @ measured
                rotation = float(np.arccos(np.clip((np.trace(delta[:3, :3])-1)/2, -1, 1)))
                fk_rows.append({"step": step, "side": side,
                                "position_error_m": float(np.linalg.norm(delta[:3, 3])),
                                "rotation_error_rad": rotation})
            contacts.append({"step": step, "net_force_max_n": float(
                scene["robot_contact"].data.net_forces_w.norm(dim=-1).max().item())})
    root_pose = robot.data.root_pose_w[0].cpu().tolist()
    runtime = {
        "joint_names": list(robot.joint_names), "joint_positions": robot.data.joint_pos[0].cpu().tolist(),
        "joint_limits": robot.data.joint_pos_limits[0].cpu().tolist(),
        "body_names": list(robot.body_names), "body_poses_w": robot.data.body_link_pose_w[0].cpu().tolist(),
        "root_pose_w": root_pose, "arm_joint_names": arm_names, "arm_joint_indices": arm_ids,
        "box_geometry": {name: {"center": g.center, "half_size": g.half_size,
                                 "body_path": g.body_path} for name, g in geometry.items()},
        "box_body_names": {name: list(scene[name].body_names) for name in spec.box_names},
        "box_body_poses_w": {name: scene[name].data.body_link_pose_w[0].cpu().tolist() for name in spec.box_names},
    }
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2, allow_nan=False))
    (output / "frames.json").write_text(json.dumps(frames, allow_nan=False))
    (output / "fk_report.json").write_text(json.dumps({"measurements": fk_rows, "contacts": contacts}, indent=2))
    snapshot = snapshot_colliders(sim.stage, "/World/envs/env_0/Kuavo")
    (output / "collision_snapshot.json").write_text(json.dumps(snapshot, indent=2))
    (output / "world.json").write_text(json.dumps(world_config(snapshot, root_pose), indent=2))
    passed = all(r["position_error_m"] <= cfg["fk_position_tolerance_m"]
                 and r["rotation_error_rad"] <= cfg["fk_rotation_tolerance_rad"] for r in fk_rows)
    return {"status": "inspected", "scene": "existing minimal workcell; all configured rack boxes; no flap locking",
            "self_collision_enabled": True, "fk_validation_passed": passed, "fk_measurements": fk_rows,
            "contacts_during_settling": contacts, "arm_joint_indices": arm_ids,
            "enabled_collider_count": len(snapshot["colliders"]),
            "robot_collider_count": sum(c["robot"] for c in snapshot["colliders"]),
            "disabled_collider_count": len(snapshot["disabled_colliders"]),
            "physics_dt_s": dt, "torch_device": torch.cuda.get_device_name(0),
            "note": "Settling and kinematic consistency only. No planned motion or pregrasp claim."}


if __name__ == "__main__":
    main()
