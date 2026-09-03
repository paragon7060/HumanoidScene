import json
from dataclasses import replace

import numpy as np
import pytest

from kuavo_isaaclab_scene.browser_teleop_bridge import (
    BrowserTeleopBridge, PROTOCOL_VERSION, parse_tracking_message,
)
from kuavo_isaaclab_scene.browser_teleop_control import browser_body_action, compose_browser_action
from kuavo_isaaclab_scene.paths import ASSET_DIR
from kuavo_isaaclab_scene.teleop_body import TeleopBodyMapper
from kuavo_isaaclab_scene.teleop_safety import TrackingLossGuard


def packet(left=(0.0, 0.0), right=(0.0, 0.0)):
    pose = [0.2, 1.2, -0.5, 1.0, 0.0, 0.0, 0.0]
    return {
        "type": "tracking", "protocol_version": PROTOCOL_VERSION,
        "sequence": 1, "timestamp_ms": 10.0,
        "head": pose, "left_hand": {"wrist": pose}, "right_hand": {"wrist": pose},
        "left_controller": {"grip": pose, "thumbstick": left, "trigger": 0.0},
        "right_controller": {"grip": pose, "thumbstick": right, "trigger": 1.0},
    }


def sample(**kwargs):
    return parse_tracking_message(json.dumps(packet(**kwargs)), received_at=1.0)


def mapper():
    return TeleopBodyMapper(ASSET_DIR / "kuavo_s200062/urdf/biped_s200062.urdf")


def test_browser_controller_axes_become_native_openxr_axes():
    controller = sample(left=(0.5, -1.0)).left_controller
    np.testing.assert_allclose(controller.native_packet()[1, :3], [0.5, 1.0, 0.0])
    np.testing.assert_allclose(controller.grip[:3], [0.5, -0.2, 1.2])
    assert sample().right_controller.trigger == 1.0


@pytest.mark.parametrize("bad", [None, {}, {"grip": "bad"}, {
    "grip": [0, 0, 0, 1, 0, 0, 0], "thumbstick": [0, float("nan")],
}, {
    "grip": [0, 0, 0, 1, 0, 0, 0], "thumbstick": [0, 1, 2],
}])
def test_bad_controller_is_ignored_without_losing_hand_packet(bad):
    message = packet()
    message["left_controller"] = bad
    parsed = parse_tracking_message(json.dumps(message))
    assert parsed.left_controller is None
    assert parsed.left_hand is not None


def test_controller_values_are_clamped():
    message = packet(left=(5, -2))
    message["left_controller"]["trigger"] = 10
    state = parse_tracking_message(json.dumps(message)).left_controller
    assert state.thumbstick == (1.0, -1.0)
    assert state.trigger == 1.0


@pytest.mark.parametrize("left,right,axis,sign", [
    ((0, -1), (0, 0), 0, 1),  # stick up -> forward
    ((0, 1), (0, 0), 0, -1),
    ((-1, 0), (0, 0), 1, 1),  # stick left -> strafe left
    ((1, 0), (0, 0), 1, -1),
    ((0, 0), (-1, 0), 2, 1),  # right stick left -> yaw left
    ((0, 0), (1, 0), 2, -1),
])
def test_browser_base_direction(left, right, axis, sign):
    command = browser_body_action(sample(left=left, right=right), mapper(), 1 / 30, control_allowed=True)
    assert np.sign(command[axis]) == sign
    assert np.linalg.norm(command[:2]) <= .250001
    assert abs(command[2]) <= 1.200001


def test_browser_lift_lower_pause_and_missing_controller():
    body = mapper()
    for _ in range(60):
        raised = browser_body_action(sample(right=(0, -1)), body, 1 / 30, control_allowed=True)
    assert body.height == pytest.approx(.24)
    assert abs(raised[3:].sum()) < 1e-6
    lost = replace(sample(left=(0, -1), right=(1, -1)), left_controller=None)
    stopped = browser_body_action(lost, body, 1 / 30, control_allowed=True)
    paused = browser_body_action(sample(left=(0, -1), right=(1, -1)), body, 1 / 30, control_allowed=False)
    np.testing.assert_allclose(stopped[:3], 0)
    np.testing.assert_allclose(stopped[3:], raised[3:])
    np.testing.assert_allclose(paused, stopped)
    for _ in range(30):
        browser_body_action(sample(right=(0, 1)), body, 1 / 30, control_allowed=True)
    assert body.height == pytest.approx(.12)


def test_stale_bridge_removes_controller_inputs_and_recovery_stops_base():
    bridge = BrowserTeleopBridge()
    bridge._sample = sample(left=(0, -1))  # timestamp is deliberately expired
    stale = bridge.latest()
    assert stale.left_controller is None and stale.right_controller is None
    np.testing.assert_allclose(browser_body_action(stale, mapper(), .1, control_allowed=False), 0)
    guard = TrackingLossGuard(recovery_frames=2)
    guard.advance(False, 1.0)
    assert not guard.advance(True, 1.1).control_allowed
    assert guard.advance(True, 1.2).control_allowed


@pytest.mark.parametrize("grippers", [(), (1.0,), (1.0, -1.0)])
def test_browser_action_has_body_channels_even_without_tracking(grippers):
    body = browser_body_action(BrowserTeleopBridge().latest(), mapper(), .1, control_allowed=False)
    action = compose_browser_action(np.arange(14), grippers, body)
    assert action.shape == (20 + len(grippers),)
    assert action.dtype == np.float32
    np.testing.assert_array_equal(action[:14], np.arange(14))
    np.testing.assert_array_equal(action[14:14 + len(grippers)], grippers)
    np.testing.assert_array_equal(action[-6:], 0)


def test_old_browser_client_and_hand_mode_keep_body_stationary():
    message = packet()
    del message["left_controller"], message["right_controller"]
    legacy = parse_tracking_message(json.dumps(message))
    np.testing.assert_allclose(browser_body_action(legacy, mapper(), .1, control_allowed=True), 0)


def test_s56_keeps_planar_motion_but_has_no_lift_axis():
    body = TeleopBodyMapper(ASSET_DIR / "kuavo_s56/urdf/kuavo_s56.urdf", has_wheel_base=False)
    command = browser_body_action(sample(left=(0, -1), right=(1, -1)), body, .1, control_allowed=True)
    assert command[0] > 0 and command[2] < 0
    np.testing.assert_array_equal(command[3:], 0)
