"""Bounded real collector/RTX/HDF5 run with SYNTHETIC inputs, not a Quest benchmark.

Default: no OpenXR display, synthetic status sink. With --xr-display, start the
CloudXR runtime, connect a headset, and source quest-session.env first. This
script never alters real datasets. All results go to artifacts/.
"""
import argparse
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from types import SimpleNamespace

parser = argparse.ArgumentParser()
parser.add_argument("--xr-display", action="store_true")
verification_args = parser.parse_args()
output = Path(tempfile.mkdtemp(prefix="hand-switch-", dir="artifacts")).resolve()
sys.argv = [sys.argv[0], "--robot-model", "s200062", "--hand-switch", "--no-auto-start",
            "--control-hz", "30", "--no-wrist-cameras", "--no-record-depth",
            "--dataset", str(output / "first.hdf5")]
import kuavo_isaaclab_scene.teleop.collect_quest_teleop as collector
import h5py
import numpy as np

real_env_type = collector.ManagerBasedRLEnv
env_box = {}


def make_env(*args, **kwargs):
    env_box["env"] = real_env_type(*args, **kwargs)
    return env_box["env"]


class SyntheticXR:
    TrackingTarget = collector.RawQuestOpenXRDevice.TrackingTarget
    tick = 0

    def __init__(self, *args, **kwargs):
        self.callbacks = {}
        robot = env_box["env"].scene["robot"]
        self.tools = {}
        for side, link in (("left", "zarm_l7_end_effector"), ("right", "zarm_r7_end_effector")):
            idx = robot.find_bodies(link)[0][0]
            self.tools[side] = np.concatenate((robot.data.body_pos_w[0, idx].numpy(),
                                               robot.data.body_quat_w[0, idx].numpy()))
        self.head = np.array([0., 0., 1.5, 1., 0., 0., 0.])

    def reset(self):
        pass

    reset_head_reference = reset

    def recenter_view(self, *args):
        pass

    def pin_view_position(self, *args):
        pass

    def motion_head_pose(self, pose):
        return pose

    def add_callback(self, *args):
        pass

    def bind_button(self, side, button, callback):
        self.callbacks[side, button] = callback

    def controller_aim_pose(self, side):
        return None

    def switch_head_tracked(self):
        return True

    def switch_controller_packet(self, side):
        t = self.tick * .05
        if 2.5 <= t < 17.:
            return None
        packet = np.stack((self.tools[side], np.zeros(7)))
        if side == "right" and (1. <= t < 2.45 or 17. <= t < 18.45):
            packet[1, 3] = 1.
        return packet

    def advance(self):
        SyntheticXR.tick += 1
        t = self.tick * .05
        if self.tick in (6, 460, 480):
            self.callbacks["right", "b"]()
        if self.tick == 160 and verification_args.xr_display:
            from omni.kit.xr.core import XRCore
            XRCore.get_singleton().schedule_capture_display_frame(str(output / "status"))
        if self.tick >= 505:
            os.kill(os.getpid(), signal.SIGTERM)
        result = {self.TrackingTarget.HEAD: self.head.copy()}
        for side, hand_key, ctrl_key in (
            ("left", self.TrackingTarget.HAND_LEFT, self.TrackingTarget.CONTROLLER_LEFT),
            ("right", self.TrackingTarget.HAND_RIGHT, self.TrackingTarget.CONTROLLER_RIGHT),
        ):
            packet = self.switch_controller_packet(side)
            if packet is not None:
                result[ctrl_key] = packet
            if 2.5 <= t < 14. or 16.3 <= t < 17.:
                wrist = self.tools[side].copy()
                thumb, index, middle = wrist.copy(), wrist.copy(), wrist.copy()
                index[0] += .09
                middle[0] += (.01 if side == "right" and (10. <= t < 11.3 or 12. <= t < 13.3) else .1)
                result[hand_key] = {"wrist": wrist, "thumb_tip": thumb, "index_tip": index, "middle_tip": middle}
        return result


def verify_fresh_hand_adapter():
    from pxr import Gf
    from omni.kit.xr.core import XRPoseValidityFlags as F
    from kuavo_isaaclab_scene.teleop.quest_openxr import RawQuestOpenXRDevice
    pose = Gf.Matrix4d(1.)
    pose.SetTranslateOnly(Gf.Vec3d(.2, .3, 1.))
    flags = F.POSITION_VALID | F.ORIENTATION_VALID | F.POSITION_TRACKED | F.ORIENTATION_TRACKED
    desc = SimpleNamespace(pose_matrix=pose, validity_flags=flags)
    source = {"name": "hand"}
    device = SimpleNamespace(get_hand_tracking_data_source=lambda: source["name"],
                             get_all_virtual_world_poses=lambda: {"wrist": desc})
    query = RawQuestOpenXRDevice._calculate_joint_poses
    stale = {"index_tip": np.ones(7)}
    assert set(query(None, device, stale)) == {"wrist"}
    desc.validity_flags = F.POSITION_VALID | F.ORIENTATION_VALID
    assert query(None, device, stale) == {}
    desc.validity_flags = flags
    source["name"] = "controller"
    assert query(None, device, stale) == {}
    assert query(None, None, stale) == {}
    print("[VERIFY] Stale, inferred and controller-derived hand bones rejected.", flush=True)


try:
    verify_fresh_hand_adapter()
    collector.ManagerBasedRLEnv = make_env
    collector.RawQuestOpenXRDevice = SyntheticXR
    collector.time = SimpleNamespace(monotonic=lambda: SyntheticXR.tick * .05, perf_counter=time.perf_counter)
    if not verification_args.xr_display:
        from kuavo_isaaclab_scene.display import xr_control_status
        statuses = []

        class StatusSink:
            def update(self, text):
                statuses.append(text)

            def close(self):
                pass

        xr_control_status.QuestControlStatus = StatusSink
        collector.start_quest_xr_session = lambda *args, **kwargs: None
        print("[VERIFY] OpenXR display disabled; real RTX cameras, synthetic input/status only.", flush=True)
    collector.main()
    if not verification_args.xr_display:
        assert any("HANDS IN" in text for text in statuses)
        assert any("CONTROLLERS | FOLLOW OFF | REC OFF" in text for text in statuses[-100:])
    summaries = []
    for path in output.glob("*.hdf5"):
        with h5py.File(path, "r") as f:
            for episode in f["data"].values():
                samples = episode["samples"]
                mode = episode.attrs["input_mode"]
                assert samples["action"].shape[1:] == (24,)
                assert np.isfinite(samples["action"][:]).all()
                assert samples["head_rgb"].shape[1:] == (360, 640, 3)
                assert samples["head_rgb"][0].std() > 5
                if mode == "hands":
                    np.testing.assert_allclose(samples["action"][:, 18:21], 0.)
                    np.testing.assert_allclose(samples["robot_root_pose_w"][:],
                                               np.broadcast_to(samples["robot_root_pose_w"][0], samples["robot_root_pose_w"].shape), atol=1e-5)
                    assert np.isnan(samples["openxr_left_controller"][:]).all()
                    assert episode.attrs["position_gain"] == 1.1
                else:
                    assert np.isnan(samples["openxr_left_hand"][:]).all()
                summaries.append((mode, episode.attrs["end_reason"], len(samples["action"])))
    assert sorted(m for m, _, _ in summaries) == ["controllers", "controllers", "hands", "hands"], summaries
    assert any(reason == "hand_tracking_lost" for _, reason, _ in summaries), summaries
    print(f"[VERIFY] {summaries}; separate files, stable 24D action, RGB, fixed hand-mode base PASS. Output: {output}", flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush(); os._exit(1)
sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
