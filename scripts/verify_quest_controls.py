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


def main():
    cfg = KuavoQuestTeleopEnvCfg()
    cfg.sim.device = "cpu"
    # Isolate servo/reach from rack contacts; collision behavior is exercised
    # separately in the live workcell and must not masquerade as IK error.
    cfg.scene.robot.init_state.pos = (0., -5., 0.)
    cfg.scene.robustness_camera = cfg.scene.left_wrist_camera = cfg.scene.right_wrist_camera = None
    cfg.actions.left_arm.controller.use_relative_mode = False
    cfg.actions.right_arm.controller.use_relative_mode = False
    set_domain_randomization(cfg, False)
    env = ManagerBasedRLEnv(cfg)
    robot = env.scene["robot"]
    assert "moving_human" not in env.scene.keys() and "moving_robot" not in env.scene.keys()
    terms = [env.action_manager.get_term(n) for n in ("left_arm", "right_arm")]
    action = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
    action[:, 16:18] = 1

    def hold():
        for i, term in enumerate(terms):
            p, q = term._compute_frame_pose()
            action[:, i * 7:(i + 1) * 7] = torch.cat((p, q), -1)

    try:
        errors = {}
        for label, kp, kd in (("previous", 220., 22.), ("tuned", 800., 50.)):
            env.reset(seed=42)
            ids = terms[0]._joint_ids + terms[1]._joint_ids
            robot.write_joint_stiffness_to_sim(kp, joint_ids=ids)
            robot.write_joint_damping_to_sim(kd, joint_ids=ids)
            action[:, 18:] = 0
            hold()
            for _ in range(30):
                env.step(action)
            hold()
            starts = [term._compute_frame_pose()[0].clone() for term in terms]
            action[:, 2] += .15
            action[:, 9] += .15
            for _ in range(180):
                env.step(action)
            errors[label] = [float(t.target_position_error()[0]) for t in terms]
            lift = [float(t._compute_frame_pose()[0][0, 2] - s[0, 2]) for t, s in zip(terms, starts)]
            print(f"[VERIFY] vertical +15cm {label}: lift={lift}, error={errors[label]}", flush=True)
            for term, s in zip(terms, starts):
                p, _ = term._compute_frame_pose()
                print(f"[VERIFY] start={s.tolist()} target={term._target_position.tolist()} actual={p.tolist()} "
                      f"joints={robot.data.joint_pos[:, term._joint_ids].tolist()} "
                      f"limits={robot.data.joint_pos_limits[:, term._joint_ids].tolist()}", flush=True)
        assert sum(errors["tuned"]) < sum(errors["previous"]), errors
        # A raised hand requires orientation compromise near the wrist limits.
        hold()
        action[0, :3] = torch.tensor([.35, .25, 1.20])
        action[0, 7:10] = torch.tensor([.35, -.25, 1.20])
        for _ in range(300):
            env.step(action)
        raised_errors = [float(t.target_position_error()[0]) for t in terms]
        print(f"[VERIFY] raised-hand target errors={raised_errors}", flush=True)
        for term in terms:
            print(f"[VERIFY] raised actual={term._compute_frame_pose()[0].tolist()} joints={robot.data.joint_pos[:, term._joint_ids].tolist()}", flush=True)
        assert max(raised_errors) < .08, raised_errors
        hold()
        start = robot.data.root_pos_w.clone()
        action[:, 18] = .1
        for _ in range(30):
            env.step(action)
        displacement = robot.data.root_pos_w - start
        print(f"[VERIFY] base displacement={displacement.tolist()}", flush=True)
        torch.testing.assert_close(displacement[0], torch.tensor([.1, 0., 0.]), atol=.002, rtol=0)
        action[:, 18:20] = 0
        body = TeleopBodyMapper(ASSET_DIR / "kuavo_s200062/urdf/biped_s200062.urdf")
        packet = np.zeros((2, 7)); packet[0, 3] = 1; packet[1, 1] = 1
        head_id = robot.find_bodies("waist_yaw_link")[0][0]
        before = float(robot.data.body_pos_w[0, head_id, 2])
        for _ in range(60):
            action[0, 18:] = torch.from_numpy(body.advance(None, packet, env.step_dt, enabled=True))
            env.step(action)
        for _ in range(90):
            env.step(action)
        height = float(robot.data.body_pos_w[0, head_id, 2]) - before
        print(f"[VERIFY] waist target lift={body.height}, actual lift={height}", flush=True)
        body_term = env.action_manager.get_term("body")
        print(f"[VERIFY] waist goal={action[:, 20:].tolist()} actual={robot.data.joint_pos[:, body_term._joint_ids].tolist()}", flush=True)
        assert height > .15
        for name in ("left_gripper", "right_gripper"):
            t = env.action_manager.get_term(name)
            q = robot.data.joint_pos[:, t._joint_ids]
            print(f"[VERIFY] {name} open joints={q.tolist()}", flush=True)
            assert float(q.abs().max()) < .08
        for term in terms:
            term.set_following(False)
        held = [t._held_joints.clone() for t in terms]
        action[:, [2, 9]] += .2
        for _ in range(60):
            env.step(action)
        for t, q in zip(terms, held):
            drift = float((robot.data.joint_pos[:, t._joint_ids] - q).abs().max())
            print(f"[VERIFY] explicit pause joint drift={drift}", flush=True)
            assert drift < .08
        action[:, 16:18] = -1
        for _ in range(60):
            env.step(action)
        for name in ("left_gripper", "right_gripper"):
            t = env.action_manager.get_term(name)
            assert float(robot.data.joint_pos[:, t._joint_ids].abs().max()) > .2
        env.reset()
        for name in ("left_gripper", "right_gripper"):
            t = env.action_manager.get_term(name)
            assert float(robot.data.joint_pos[:, t._joint_ids].abs().max()) < .001
        print("[VERIFY] pause holds joints; trigger close and reset-open passed", flush=True)
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
