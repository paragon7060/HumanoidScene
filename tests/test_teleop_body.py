import numpy as np
from kuavo_isaaclab_scene.core.paths import ASSET_DIR
from kuavo_isaaclab_scene.teleop.teleop_body import (
    BASE_YAW_ACCEL_RAD_S2, BASE_YAW_SPEED_RAD_S, TORSO_HEIGHT_SPEED_M_S, TeleopBodyMapper,
)


def packet(x=0., y=0.):
    p = np.zeros((2, 7)); p[0, 3] = 1
    p[1, :2] = [x, y]
    return p


def mapper():
    return TeleopBodyMapper(ASSET_DIR / "kuavo_s200062/urdf/biped_s200062.urdf")


def test_native_openxr_up_moves_forward_and_lifts_torso_without_pitch():
    m = mapper()
    for _ in range(20):
        command = m.advance(packet(y=1), packet(y=1), 1/30, enabled=True)
    assert command[0] == 0.75
    reached_height = min(20 * (1/30) * TORSO_HEIGHT_SPEED_M_S, .40)
    np.testing.assert_allclose(
        m._planar_position(command[3:5]), m.links.sum(axis=0) + [0, reached_height], atol=2e-5,
    )
    assert abs(command[3:6].sum()) < 1e-6


def test_absolute_height_reuses_upright_lift_kinematics():
    m = mapper()
    assert m.set_height(.25)
    assert m.height == .25
    np.testing.assert_allclose(
        m._planar_position(m.joints[:2]), m.links.sum(axis=0) + [0, .25], atol=2e-5
    )
    assert abs(m.joints[:3].sum()) < 1e-6


def test_absolute_height_clamps_to_existing_range_and_rejects_nonfinite():
    m = mapper()
    assert m.set_height(1.0)
    assert m.height == m.MAX_HEIGHT_M
    with np.testing.assert_raises(ValueError):
        m.set_height(float("nan"))


def test_deadzone_loss_and_pause_stop_base_and_hold_waist():
    m = mapper()
    np.testing.assert_allclose(m.advance(packet(.1, -.1), packet(.1, -.1), .1, enabled=True), 0)
    active = m.advance(packet(x=1), packet(x=1, y=1), .1, enabled=True)
    # A single 0.1s step only ramps yaw partway toward the full commanded rate.
    ramped_yaw = -min(BASE_YAW_SPEED_RAD_S, BASE_YAW_ACCEL_RAD_S2 * .1)
    np.testing.assert_allclose(active[1:3], [-0.75, ramped_yaw])
    paused = m.advance(packet(1, 1), packet(1, 1), .1, enabled=False)
    lost = m.advance(None, None, .1, enabled=True)
    np.testing.assert_allclose(paused[:3], 0)
    np.testing.assert_allclose(paused[3:], active[3:])
    np.testing.assert_allclose(lost, paused)


def test_s56_fixed_biped_keeps_height_channels_zero():
    m = TeleopBodyMapper(
        ASSET_DIR / "kuavo_s56/urdf/kuavo_s56.urdf",
        has_wheel_base=False,
    )
    command = m.advance(packet(y=1), packet(x=1, y=1), .1, enabled=True)
    assert command[0] > 0 and command[2] < 0
    np.testing.assert_allclose(command[3:], 0)
    assert m.height == 0.0
    assert not m.set_height(.2)
