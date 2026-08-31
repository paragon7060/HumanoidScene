"""Bounded XR camera restart/save check; synthetic data stays in artifacts/."""
import argparse
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--scene-detail", choices=("full", "compact"), default="compact")
args = parser.parse_args()
from isaaclab.app import AppLauncher

app = AppLauncher(headless=False, xr=True, enable_cameras=True, device="cpu").app
import h5py
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from omni.kit.viewport.utility import get_active_viewport
from kuavo_isaaclab_scene.teleop_env import KuavoQuestTeleopEnvCfg, set_domain_randomization
from kuavo_isaaclab_scene.teleop_camera import camera_rgb
from kuavo_isaaclab_scene.teleop_scene import configure_scene_detail
from kuavo_isaaclab_scene.teleop_recorder import TeleopHdf5EpisodeRecorder, new_session_path
from kuavo_isaaclab_scene.quest_openxr import start_quest_xr_session


def verify_anchor_motion():
    from types import SimpleNamespace
    from pxr import Gf
    from kuavo_isaaclab_scene.quest_openxr import RawQuestOpenXRDevice
    state = {"anchor": Gf.Matrix4d(1.), "head": Gf.Matrix4d(1.)}
    state["head"].SetTranslateOnly(Gf.Vec3d(.2, .1, 1.4))
    core = SimpleNamespace(
        get_input_device=lambda path: SimpleNamespace(get_virtual_world_pose=lambda: state["head"]),
        get_world_transform_matrix=lambda path: state["anchor"],
        set_world_transform_matrix=lambda path, value: state.update(anchor=value),
    )
    device = SimpleNamespace(_xr_core=core, _xr_anchor_headset_path="/World/TestAnchor")
    RawQuestOpenXRDevice.pin_view_position(device, [.3, .1, 1.4])
    np.testing.assert_allclose(np.array(state["anchor"])[:3, :3], np.eye(3), atol=1e-8)
    np.testing.assert_allclose(state["anchor"].ExtractTranslation(), [.1, 0, 0], atol=1e-8)
    # A fresh physical head turn must never be copied back into the anchor.
    state["head"].SetRotateOnly(Gf.Rotation(Gf.Vec3d(0, 0, 1), 25))
    RawQuestOpenXRDevice.pin_view_position(device, [.3, .1, 1.4])
    np.testing.assert_allclose(np.array(state["anchor"])[:3, :3], np.eye(3), atol=1e-8)
    RawQuestOpenXRDevice.pin_view_position(device, [.3, .1, 1.4], .2)
    np.testing.assert_allclose(float(state["anchor"].ExtractRotation().GetAngle()), np.degrees(.2), atol=1e-5)
    print("[VERIFY] Anchor translation preserves head turn; base yaw rotates anchor positively.", flush=True)


def main():
    verify_anchor_motion()
    cfg = KuavoQuestTeleopEnvCfg()
    cfg.sim.device = "cpu"
    cfg.scene.left_wrist_camera = cfg.scene.right_wrist_camera = None
    cfg.scene.robustness_camera.data_types = ["rgb"]
    cfg.scene.robustness_camera.width, cfg.scene.robustness_camera.height = 640, 360
    cfg.sim.render.dlss_mode = 0
    cfg.sim.render.enable_reflections = False
    cfg.sim.render.enable_translucency = False
    cfg.sim.render.enable_global_illumination = False
    cfg.sim.render.enable_ambient_occlusion = False
    cfg.sim.render.samples_per_pixel = 1
    set_domain_randomization(cfg, False)
    configure_scene_detail(cfg, args.scene_detail)
    env = ManagerBasedRLEnv(cfg)
    recorder = TeleopHdf5EpisodeRecorder(new_session_path("artifacts/recording-verification"))
    files = []
    try:
        env.reset(seed=42)
        start_quest_xr_session(app, enable_ui=False, resolution_scale=1.0)
        viewport = get_active_viewport()
        viewport.fill_frame = False
        viewport.resolution = (160, 90)
        action = torch.zeros((1, env.action_manager.total_action_dim))
        action[:, 14:16] = 1
        camera = env.scene["robustness_camera"]
        for term in ("left_arm", "right_arm"):
            env.action_manager.get_term(term).set_following(False)
        for phase in ("warmup", "idle_1", "record_1", "idle_2", "record_2"):
            recording = phase.startswith("record")
            viewport.updates_enabled = phase == "warmup" or recording
            if recording:
                recorder.start_episode({"purpose": "synthetic verification, not a demonstration",
                                        "scene_detail": args.scene_detail})
                files.append(recorder.path)
            valid, start = 0, time.perf_counter()
            for step in range(90):
                action[:, 12] = .15 * np.sin(step * np.pi / 45)
                env.step(action)
                frame = camera_rgb(camera)
                if frame is not None:
                    valid += 1
                    if recording:
                        recorder.append({"head_rgb": frame,
                                         "robot_joint_position": env.scene["robot"].data.joint_pos[0].numpy(),
                                         "sim_time_s": np.float64(env.common_step_counter * env.step_dt)})
            print(f"[VERIFY] {args.scene_detail} {phase}: RGB={valid}/90; "
                  f"loop={90 / (time.perf_counter() - start):.2f} Hz", flush=True)
            if recording:
                recorder.finish_episode(success=False, reason="verification_only")
                assert valid >= 80, f"Camera did not recover: {valid}/90 frames"
        assert files[0] != files[1]
        for path in files:
            with h5py.File(path, "r") as f:
                samples = f["data/demo_00000/samples"]
                frames = samples["head_rgb"]
                assert frames.shape[1:] == (360, 640, 3)
                assert frames[0].std() > 5 and frames[-1].std() > 5
                difference = np.abs(frames[10].astype(float) - frames[50].astype(float)).mean()
                assert difference > 1, "Camera must update when head joints move"
                print(f"[VERIFY] HDF5 {path}: {frames.shape}, image difference={difference:.2f}", flush=True)
        print("[VERIFY] Recording stop/start PASS; this is not a headset latency benchmark.", flush=True)
    finally:
        recorder.close()
        env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
