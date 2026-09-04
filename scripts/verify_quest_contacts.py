"""Physics regression: grip a 2mm box flap, lift, release, then press a button.

No attachment constraints or object pose writes are used after grasp setup.
The robot's real torso joints perform the lift. Synthetic fixtures stay here;
the production scene is not moved or simplified by this check.
"""
import os
import sys
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, device="cpu").app
import numpy as np
import torch
import isaaclab.sim as sim
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.math import quat_apply
from kuavo_isaaclab_scene.core.paths import ASSET_DIR, BOX_ATLAS_ASSETS
from kuavo_isaaclab_scene.teleop.teleop_body import TeleopBodyMapper
from kuavo_isaaclab_scene.envs.teleop_env import KuavoQuestTeleopEnvCfg, set_domain_randomization
from kuavo_isaaclab_scene.workcell.rack_box_layout import BOX_DIMENSIONS_M, BOX_FLAP_LENGTH_M


TEST_BOX_WIDTH, TEST_BOX_DEPTH, TEST_BOX_HEIGHT = BOX_DIMENSIONS_M["small"]
TEST_BOX_FLAP = BOX_FLAP_LENGTH_M["small"]


def body_bottom(box):
    # The final SmallBox atlas wrapper is already authored at measured size.
    points = torch.tensor([
        [x, y, z]
        for x in (-TEST_BOX_WIDTH / 2, TEST_BOX_WIDTH / 2)
        for y in (-TEST_BOX_DEPTH / 2, TEST_BOX_DEPTH / 2)
        for z in (-0.005 * TEST_BOX_HEIGHT, TEST_BOX_HEIGHT)
    ])
    rotation = box.data.root_quat_w.expand(len(points), -1)
    return float((quat_apply(rotation, points) + box.data.root_pos_w)[:, 2].min())


def main():
    cfg = KuavoQuestTeleopEnvCfg()
    cfg.sim.device = "cpu"
    cfg.scene.robot.init_state.pos = (0., -5., 0.)
    cfg.scene.robot.spawn.activate_contact_sensors = True
    cfg.scene.button_station.prim_path = "{ENV_REGEX_NS}/ButtonStation"
    cfg.scene.contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Kuavo/l_.*", update_period=0)
    cfg.scene.robustness_camera = cfg.scene.left_wrist_camera = cfg.scene.right_wrist_camera = None
    cfg.actions.left_arm.controller.use_relative_mode = cfg.actions.right_arm.controller.use_relative_mode = False
    cfg.scene.test_box = cfg.scene.small_box_0.copy()
    cfg.scene.test_box.prim_path = "{ENV_REGEX_NS}/GraspBox"
    cfg.scene.test_box.spawn.usd_path = str(BOX_ATLAS_ASSETS["small"])
    cfg.scene.test_box.spawn.scale = (1.0, 1.0, 1.0)
    cfg.scene.test_box.init_state.pos = (0., -9., 1.)
    # Authored support position for the known FK pose below. Do not teleport
    # a kinematic support at reset: its USD transform can supersede tensor writes.
    support_z = .31735087
    cfg.scene.platform = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GraspPlatform",
        spawn=sim.CuboidCfg(size=(TEST_BOX_WIDTH + .04, TEST_BOX_DEPTH + .04, .04),
                           rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True),
                           collision_props=sim.CollisionPropertiesCfg(contact_offset=.001, rest_offset=0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(.34146118, -4.74729967, support_z)),
    )
    cfg.scene.probe = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ButtonProbe",
        spawn=sim.CuboidCfg(size=(.04, .04, .04),
                           rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                           collision_props=sim.CollisionPropertiesCfg(contact_offset=.001, rest_offset=0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0., -8., 1.)),
    )
    set_domain_randomization(cfg, False)
    keep = {"robot", "button_station", "test_box", "platform", "probe"}
    for name, asset in vars(cfg.scene).items():
        if (isinstance(asset, AssetBaseCfg) or hasattr(asset, "rigid_objects")) and name not in keep:
            setattr(cfg.scene, name, None)
    env = ManagerBasedRLEnv(cfg)
    try:
        env.reset(seed=42)
        robot, box, platform = (env.scene[n] for n in ("robot", "test_box", "platform"))
        arms = [env.action_manager.get_term(n) for n in ("left_arm", "right_arm")]
        q = robot.data.default_joint_pos.clone()
        for arm in arms:
            q[:, arm._joint_ids] = torch.tensor([-.65, 0., 0., 0., 0., 0., .65])
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        env.sim.forward()
        env.scene.update(env.step_dt)
        action = torch.zeros((1, env.action_manager.total_action_dim))
        action[:, 16:18] = 1
        for i, arm in enumerate(arms):
            arm.set_following(False)
            action[:, i * 7:(i + 1) * 7] = torch.cat(arm._compute_frame_pose(), -1)

        def steps(count):
            for _ in range(count):
                env.step(action)

        tool = arms[0]._compute_frame_pose()[0] + robot.data.root_pos_w
        pose = box.data.root_pose_w.clone()
        pose[:, :3] = tool + torch.tensor([
            [-TEST_BOX_WIDTH / 2, 0., -(TEST_BOX_HEIGHT + TEST_BOX_FLAP - .005)]
        ])
        pose[:, 3:] = torch.tensor([[1., 0., 0., 0.]])
        steps(3)
        assert abs(float(platform.data.root_pos_w[0, 2]) - support_z) < 1e-5
        box.write_root_pose_to_sim(pose)
        box.write_root_velocity_to_sim(torch.zeros((1, 6)))
        box.write_joint_state_to_sim(box.data.default_joint_pos, torch.zeros_like(box.data.joint_vel))
        steps(60)
        action[:, 16] = -1
        steps(120)
        before = float(box.data.root_pos_w[0, 2])
        sensor = env.scene["contacts"]
        finger_ids = [sensor.body_names.index(n) for n in ("l_f_finger", "l_b_finger")]
        forces = sensor.data.net_forces_w[0, finger_ids].norm(dim=-1)
        print(f"[VERIFY] Closed finger contact force N={forces.tolist()}; box mass="
              f"{float(box.data.default_mass.sum()):.3f} kg", flush=True)
        assert float(forces.min()) > 1.
        body = TeleopBodyMapper(ASSET_DIR / "kuavo_s200062/urdf/biped_s200062.urdf")
        packet = np.zeros((2, 7)); packet[0, 3] = 1; packet[1, 1] = 1
        for _ in range(30):
            action[0, 18:] = torch.from_numpy(body.advance(None, packet, env.step_dt, enabled=True))
            env.step(action)
        action[0, 18:21] = 0
        steps(90)
        lift = float(box.data.root_pos_w[0, 2]) - before
        held_clearance = body_bottom(box) - (support_z + .02)
        print(f"[VERIFY] 2mm flap grasp: body lift={lift * 1000:.1f}mm; "
              f"bottom clearance={held_clearance * 1000:.1f}mm", flush=True)
        assert lift > .06 and held_clearance > .01
        action[:, 16] = 1
        steps(90)
        released_clearance = body_bottom(box) - (support_z + .02)
        print(f"[VERIFY] Released onto support: bottom clearance={released_clearance * 1000:.1f}mm", flush=True)
        assert abs(released_clearance) < .005

        button, probe = env.scene["button_station"], env.scene["probe"]
        idx = button.find_bodies("Plunger")[0][0]
        center = button.data.body_pos_w[:, idx].clone()
        rotation = button.data.body_quat_w[:, idx].clone()
        normal = quat_apply(rotation, torch.tensor([[0., -1., 0.]]))
        travel = []
        for distance, count in ((.08, 30), (.028, 120), (.08, 90)):
            start = probe.data.root_pos_w.clone()
            target = center + normal * distance
            for i in range(count):
                probe_pose = probe.data.root_pose_w.clone()
                probe_pose[:, :3] = start + (target - start) * min(1., (i + 1) / 30)
                probe_pose[:, 3:] = rotation
                probe.write_root_pose_to_sim(probe_pose)
                env.step(action)
                travel.append(float(button.data.joint_pos[0, 0]))
        print(f"[VERIFY] Contact button stroke={max(travel) * 1000:.1f}mm; "
              f"released={travel[-1] * 1000:.3f}mm", flush=True)
        assert max(travel) > .006 and abs(travel[-1]) < .001
        print("[VERIFY] Contact grasp/release and button PASS", flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
