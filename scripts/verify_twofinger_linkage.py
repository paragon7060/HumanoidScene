#!/usr/bin/env python3
"""Measure physical four-bar closure through repeated open/close/reset cycles.

Run with the Isaac Lab conda Python and --headless. Optional MP4 is a close-up
of the real simulated left/right hand meshes, not an analytic animation.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--robot-model", choices=("s56", "s200062"), default="s56")
parser.add_argument("--cycles", type=int, default=3)
parser.add_argument("--steps-per-cycle", type=int, default=480)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--video-out", type=Path)
parser.add_argument("--overwrite-video", action="store_true")
parser.add_argument("--contact-probe", action="store_true",
                    help="Place kinematic blocks between the jaws and require measured contact forces")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.cycles < 1 or args.steps_per_cycle < 24:
    parser.error("Use at least one cycle with at least 24 physics steps")
os.environ["KUAVO_ROBOT_MODEL"] = args.robot_model
os.environ["KUAVO_GRIPPER"] = "s56_twofinger" if args.robot_model == "s56" else "s200062_integrated"
args.enable_cameras = bool(args.video_out)
launcher = AppLauncher(args)
app = launcher.app

import torch
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply
from kuavo_isaaclab_scene.display.eval_video import FfmpegVideoWriter
from kuavo_isaaclab_scene.robots.gripper_config import FingerContactSettings, load_gripper_settings
from kuavo_isaaclab_scene.envs.manager_env import KUAVO_CFG
from kuavo_isaaclab_scene.robots.twofinger_linkage import FINGER_PIN, FOLLOWER_PIN, pin_for


def verify_contact_materials():
    """Check resolved physics bindings on actual collision meshes, not just JSON."""
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = sim_utils.get_current_stage()
    root = stage.GetPrimAtPath("/World/Robot")
    fingers = {f"{side}_{jaw}_finger" for side in "lr" for jaw in "fb"}
    housing = {f"{side}_twofinger_base" for side in "lr"} | {"zarm_l7_link", "zarm_r7_link"}
    configured = load_gripper_settings().finger_contact
    report = {}
    for link in root.GetChildren():
        name = link.GetName()
        if name not in fingers | housing:
            continue
        expected = configured if name in fingers else FingerContactSettings()
        material_name = "FingerContactMaterial" if name in fingers else "HandContactMaterial"
        meshes = []
        for prim in Usd.PrimRange(link):
            if not prim.IsA(UsdGeom.Mesh) or "/visuals/" not in str(prim.GetPath()):
                continue
            collision = UsdPhysics.CollisionAPI(prim)
            if not collision or not collision.GetCollisionEnabledAttr().Get():
                raise RuntimeError(f"Missing enabled collision on {prim.GetPath()}")
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            expected_path = f"/World/Robot/{material_name}"
            if not material or str(material.GetPath()) != expected_path:
                raise RuntimeError(f"Incorrect physics material on {prim.GetPath()}: expected {expected_path}")
            api = UsdPhysics.MaterialAPI(material.GetPrim())
            static = api.GetStaticFrictionAttr().Get()
            dynamic = api.GetDynamicFrictionAttr().Get()
            combine = PhysxSchema.PhysxMaterialAPI(material.GetPrim()).GetFrictionCombineModeAttr().Get()
            if (not math.isclose(static, expected.static_friction, rel_tol=1e-6, abs_tol=1e-7)
                    or not math.isclose(dynamic, expected.dynamic_friction, rel_tol=1e-6, abs_tol=1e-7)
                    or combine != expected.friction_combine_mode):
                raise RuntimeError(f"Unexpected surface friction on {prim.GetPath()}: {static}/{dynamic}/{combine}")
            meshes.append(str(prim.GetPath()))
        if not meshes:
            raise RuntimeError(f"No contact meshes found under {link.GetPath()}")
        report[name] = {"static_friction": static, "dynamic_friction": dynamic,
                        "friction_combine_mode": combine, "collision_meshes": meshes}
    if set(report) != fingers | housing:
        raise RuntimeError(f"Missing contact links: {(fingers | housing) - set(report)}")
    return report


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120, device=args.device))
    cfg = KUAVO_CFG.copy()
    cfg.prim_path = "/World/Robot"
    cfg.init_state.pos = (0.0, 0.0, 1.0 if args.robot_model == "s56" else 0.0)
    cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    cfg.spawn.activate_contact_sensors = bool(args.contact_probe)
    robot = Articulation(cfg)
    probes = []
    contacts = None
    if args.contact_probe:
        for side in "lr":
            probes.append(RigidObject(RigidObjectCfg(
                prim_path=f"/World/Probe_{side}",
                spawn=sim_utils.CuboidCfg(
                    size=(0.04, 0.024, 0.02),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001, rest_offset=0.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=.05),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15,0.5,0.9)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0,0.0,-10.0)),
            )))
        contacts = ContactSensor(ContactSensorCfg(
            prim_path="/World/Robot/[lr]_[fb]_finger", update_period=0.0, history_length=1,
        ))
    light = sim_utils.DomeLightCfg(intensity=250.0, color=(0.85, 0.9, 1.0))
    light.func("/World/Light", light)
    cameras = []
    if args.video_out:
        # Diagnostic colors only, never saved to packaged assets: blue jaws,
        # orange passive central bars, cyan motor cranks. This makes pin
        # continuity visible even on the original white CAD materials.
        from pxr import UsdShade
        stage = sim_utils.get_current_stage()
        materials = {}
        for label, color in (("motor", (.05,.45,.6)), ("jaw", (.08,.2,.65)),
                             ("follower", (.85,.25,.025))):
            path = f"/World/LinkageDebugMaterials/{label}"
            mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=.6)
            mat_cfg.func(path, mat_cfg)
            materials[label] = UsdShade.Material(stage.GetPrimAtPath(path))
        for link in stage.GetPrimAtPath("/World/Robot").GetChildren():
            name = link.GetName()
            label = ("follower" if name.endswith("bar_4") else
                     "jaw" if name.endswith(("bar_3", "_finger")) else
                     "motor" if name.endswith(("bar_1", "bar_2")) else None)
            if label:
                UsdShade.MaterialBindingAPI.Apply(link).Bind(
                    materials[label], bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        for side in "lr":
            cameras.append(Camera(CameraCfg(
                prim_path=f"/World/Inspect_{side}", height=480, width=640,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955,
                                                clipping_range=(0.01, 10.0)),
            )))
    sim.reset()
    contact_materials = verify_contact_materials()
    dt = sim.get_physics_dt()
    motor_names = [f"{s}_{j}_bar_1_joint" for s in "lr" for j in "fb"]
    motor_ids, _ = robot.find_joints(motor_names, preserve_order=True)
    follower_ids, _ = robot.find_joints([f"{s}_{j}_bar_4_joint" for s in "lr" for j in "fb"], preserve_order=True)
    finger_ids = [robot.body_names.index(f"{s}_{j}_finger") for s in "lr" for j in "fb"]
    bar_ids = [robot.body_names.index(f"{s}_{j}_bar_4") for s in "lr" for j in "fb"]
    pin0 = torch.tensor([pin_for(j, FINGER_PIN) for s in "lr" for j in "fb"], device=robot.device)
    pin1 = torch.tensor([pin_for(j, FOLLOWER_PIN) for s in "lr" for j in "fb"], device=robot.device)
    rows = []
    writer = None
    try:
        if args.video_out:
            writer = FfmpegVideoWriter(args.video_out, width=1280, height=480,
                                       fps=30, overwrite=args.overwrite_video)
        for cycle in range(args.cycles):
            robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),
                                          robot.data.default_joint_vel.clone())
            robot.reset()
            if probes:
                for side, probe in zip("lr", probes):
                    index = robot.body_names.index(f"{side}_twofinger_base")
                    quat = robot.data.body_link_quat_w[:,index]
                    pos = robot.data.body_link_pos_w[:,index] + quat_apply(
                        quat, torch.tensor([[0.0,0.0,-0.17]], device=robot.device))
                    probe.write_root_pose_to_sim(torch.cat([pos,quat], dim=-1))
                contacts.reset()
            for step in range(args.steps_per_cycle):
                phase = step / (args.steps_per_cycle - 1)
                q = -0.125 * (1 + math.cos(2 * math.pi * phase))
                target = robot.data.default_joint_pos.clone()
                target[:, motor_ids] = torch.tensor([q, -q, q, -q], device=robot.device)
                robot.set_joint_position_target(target)
                robot.write_data_to_sim()
                render = bool(cameras) and step % 4 == 0
                sim.step(render=render)
                robot.update(dt)
                contact_force = 0.0
                if contacts:
                    contacts.update(dt)
                    contact_force = torch.linalg.vector_norm(contacts.data.net_forces_w, dim=-1).max().item()
                p = robot.data.body_link_pos_w[0]
                rot = robot.data.body_link_quat_w[0]
                a = p[finger_ids] + quat_apply(rot[finger_ids], pin0)
                b = p[bar_ids] + quat_apply(rot[bar_ids], pin1)
                gap = torch.linalg.vector_norm(a-b, dim=-1)
                if not torch.isfinite(robot.data.joint_pos).all() or not torch.isfinite(gap).all():
                    raise RuntimeError("Non-finite physical state")
                rows.append({"cycle": cycle, "step": step, "driver_target": q,
                             "pin_gap_m": gap.tolist(),
                             "motor_actual": robot.data.joint_pos[0, motor_ids].tolist(),
                             "contact_force_n": contact_force,
                             "follower_actual": robot.data.joint_pos[0, follower_ids].tolist()})
                if render:
                    for side, camera in zip("lr", cameras):
                        index = robot.body_names.index(f"{side}_twofinger_base")
                        base = p[index:index+1]
                        orient = rot[index:index+1]
                        # Stay outside the torso/legs; the inward view would
                        # put the camera inside the robot's thigh mesh.
                        y = 0.34 if side == "l" else -0.34
                        eye = base + quat_apply(orient, torch.tensor([[0.10,y,-0.12]], device=robot.device))
                        look = base + quat_apply(orient, torch.tensor([[0.0,0.0,-0.10]], device=robot.device))
                        camera.set_world_poses_from_view(eye, look)
                    sim.render()
                    for camera in cameras:
                        camera.update(dt*4, force_recompute=True)
                    frame = torch.cat([c.data.output["rgb"][0,...,:3] for c in cameras], dim=1).cpu()
                    writer.write(frame)
            print(f"[LINKAGE] completed {args.robot_model} cycle {cycle+1}", flush=True)
        gaps = torch.tensor([row["pin_gap_m"] for row in rows])
        followers = torch.tensor([row["follower_actual"] for row in rows])
        motion = followers.max(dim=0).values-followers.min(dim=0).values
        max_force = max(row["contact_force_n"] for row in rows)
        motion_threshold = .05 if args.contact_probe else .35
        report = {"robot_model": args.robot_model, "physics_hz": 120,
                  "cycles": args.cycles, "steps": len(rows),
                  "max_pin_gap_m": gaps.max().item(),
                  "p95_pin_gap_m": gaps.flatten().quantile(.95).item(),
                  "follower_motion_rad": motion.tolist(),
                  "contact_probe": args.contact_probe,
                  "contact_materials": contact_materials,
                  "max_contact_force_n": max_force,
                  "passed": bool(gaps.max() < .001 and torch.all(motion > motion_threshold)
                                 and (not args.contact_probe or max_force > 1.0)),
                  "video_path": str(args.video_out.resolve()) if args.video_out else None,
                  "samples": rows}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print("[LINKAGE]", json.dumps({k:v for k,v in report.items() if k != "samples"}), flush=True)
        if not report["passed"]:
            raise RuntimeError("Pin gap or follower-motion acceptance check failed")
    finally:
        if writer:
            writer.close()
        # Follow ManagerBasedEnv.close(): unsubscribe the standalone STOP
        # renderer before app.close stops the timeline. Calling sim.stop()
        # first would enter Isaac Lab's keep-rendering-on-STOP loop forever.
        sim.clear_all_callbacks()
        sim.clear_instance()


try:
    main()
finally:
    app.close(wait_for_replicator=False)
