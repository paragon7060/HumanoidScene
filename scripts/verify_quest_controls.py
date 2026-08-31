"""Bounded, headless physics checks for Quest arm/base/waist/gripper controls."""
import os
import sys
from isaaclab.app import AppLauncher

launcher = AppLauncher(headless=True, enable_cameras=False, device="cpu")
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from kuavo_isaaclab_scene.teleop_env import KuavoQuestTeleopEnvCfg, set_domain_randomization
from kuavo_isaaclab_scene.teleop_body import TeleopBodyMapper
from kuavo_isaaclab_scene.paths import ASSET_DIR
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
import trimesh


def finger_gap_mm(robot, term):
    # Actual simulated joint angles, projected pad meshes in gripper-base X.
    # The imported S200062 is a tree approximation, not a closed four-bar loop.
    q = dict(zip(robot.joint_names, robot.data.joint_pos[0].tolist()))
    side = robot.joint_names[term._joint_ids[0]][0]
    edges = []
    for finger, sign in (("f", 1), ("b", -1)):
        a, b = (q[f"{side}_{finger}_bar_{i}_joint"] for i in (1, 3))
        def ry(angle):
            return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        vertices = trimesh.load(ASSET_DIR / f"kuavo_s200062/meshes/{side}_{finger}_finger.STL").vertices
        points = (vertices @ ry(a + b).T + [sign * .0125, 0, -.063137]
                  + ry(a) @ np.array([sign * .027533, 0, -.0085573])
                  + ry(a + b) @ np.array([-sign * .0081591, 0, -.047301]))
        edges.append(points[:, 0].min() if finger == "f" else points[:, 0].max())
    return (edges[0] - edges[1]) * 1000



def main():
    cfg = KuavoQuestTeleopEnvCfg()
    cfg.sim.device = "cpu"
    cfg.scene.robot.init_state.pos = (0., -5., 0.)
    cfg.scene.robustness_camera = cfg.scene.left_wrist_camera = cfg.scene.right_wrist_camera = None
    cfg.actions.left_arm.controller.use_relative_mode = False
    cfg.actions.right_arm.controller.use_relative_mode = False
    set_domain_randomization(cfg, False)
    from kuavo_isaaclab_scene.teleop_scene import configure_scene_detail
    configure_scene_detail(cfg, "compact")
    env = ManagerBasedRLEnv(cfg)
    robot = env.scene["robot"]
    terms = [env.action_manager.get_term(n) for n in ("left_arm", "right_arm")]
    action = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
    action[:, 16:18] = 1

    def hold():
        for i, term in enumerate(terms):
            p, q = term._compute_frame_pose()
            action[:, i * 7:(i + 1) * 7] = torch.cat((p, q), -1)

    def steps(count):
        for _ in range(count):
            env.step(action)

    try:
        env.reset(seed=42)
        # The source's missing hand inertials must not become 1kg per tiny link.
        for side in "lr":
            ids = [i for i, name in enumerate(robot.body_names)
                   if name.startswith(f"{side}_") or name.startswith(f"zarm_{side}7_end_effector")]
            mass = float(robot.data.default_mass[0, ids].sum())
            print(f"[VERIFY] {side} hand/frame mass={mass:.3f} kg", flush=True)
            assert abs(mass - .743) < 1e-5
        # Obtain a known reachable, forward-facing grasp pose from real FK.
        # Then reset and require the controller to reach it from the ready pose.
        q = robot.data.default_joint_pos.clone()
        for t in terms:
            q[:, t._joint_ids] = torch.tensor([-.15, 0., 0., -1., 0., 0., -.42])
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        env.sim.forward(); env.scene.update(env.step_dt)
        hold()
        env.reset(seed=42)
        steps(360)
        pe = [float(t.target_position_error()[0]) for t in terms]
        re = [float(t.target_orientation_error()[0]) for t in terms]
        print(f"[VERIFY] forward-grasp pose error m={pe}, rad={re}", flush=True)
        assert max(pe) < .015 and max(re) < .04
        action[:, [2, 9]] += .10
        steps(180)
        pe = [float(t.target_position_error()[0]) for t in terms]
        print(f"[VERIFY] orientation-held +10cm error={pe}", flush=True)
        assert max(pe) < .02
        for t in terms:
            t.set_following(False)
        held = [t._held_joints.clone() for t in terms]
        steps(60)
        for t, q in zip(terms, held):
            assert float((robot.data.joint_pos[:, t._joint_ids] - q).abs().max()) < .08
        start = robot.data.root_pos_w.clone()
        action[:, 18] = .1
        steps(30)
        torch.testing.assert_close(robot.data.root_pos_w - start, torch.tensor([[.1, 0., 0.]]), atol=.003, rtol=0)
        action[:, 18] = 0
        for rate in (1.2, -1.2):
            action[:, 20] = rate
            steps(30)
            q = robot.data.root_quat_w[0]
            yaw = float(2 * torch.atan2(q[3], q[0]))
            expected = 1.2 if rate > 0 else 0.
            print(f"[VERIFY] base yaw={yaw:.3f}, expected={expected:.3f}; wheel velocity="
                  f"{robot.data.joint_vel[:, env.action_manager.get_term('body')._wheel_ids].tolist()}", flush=True)
            assert abs(yaw - expected) < .015
            assert robot.data.joint_vel[:, env.action_manager.get_term("body")._wheel_ids].abs().mean() > 1
        action[:, 20] = 0
        body = TeleopBodyMapper(ASSET_DIR / "kuavo_s200062/urdf/biped_s200062.urdf")
        packet = np.zeros((2, 7)); packet[0, 3] = 1
        waist = robot.find_bodies("waist_yaw_link")[0][0]
        before = float(robot.data.body_pos_w[0, waist, 2])
        for axis in (1., -1.):
            packet[1, 1] = axis
            for _ in range(60):
                action[0, 18:] = torch.from_numpy(body.advance(None, packet, env.step_dt, enabled=True))
                env.step(action)
            steps(90)
            lift = float(robot.data.body_pos_w[0, waist, 2]) - before
            print(f"[VERIFY] lift stick={axis}, goal={body.height:.3f}, actual={lift:.3f}", flush=True)
            assert lift > .15 if axis > 0 else abs(lift) < .04
        head_ids = robot.find_joints(["zhead_1_joint", "zhead_2_joint"], preserve_order=True)[0]
        for yaw, pitch in ((.3, -.2), (-.3, .2)):
            action[0, 14:16] = torch.tensor([yaw, pitch]); steps(90)
            actual = robot.data.joint_pos[0, head_ids]
            print(f"[VERIFY] head target left/up (+,-)={yaw,pitch}, actual={actual.tolist()}", flush=True)
            torch.testing.assert_close(actual, torch.tensor([yaw, pitch]), atol=.04, rtol=0)
        for label, value in (("open", 1.), ("closed", -1.)):
            action[:, 16:18] = value; steps(90)
            for name in ("left_gripper", "right_gripper"):
                t = env.action_manager.get_term(name)
                gap = finger_gap_mm(robot, t)
                print(f"[VERIFY] {name} {label} pad gap={gap:.2f} mm", flush=True)
                assert (75 < gap < 110) if value > 0 else (-.5 < gap < 5)
        env.reset()
        for name in ("left_gripper", "right_gripper"):
            t = env.action_manager.get_term(name)
            assert 75 < finger_gap_mm(robot, t) < 110
        print("[VERIFY] PASS", flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
