import numpy as np
import pytest
from kuavo_isaaclab_scene.core.paths import ASSET_DIR
from kuavo_isaaclab_scene.teleop.teleop_body import (
    BASE_LINEAR_SPEED_M_S, BASE_LINEAR_ACCEL_M_S2,
    BASE_YAW_ACCEL_RAD_S2, BASE_YAW_SPEED_RAD_S,
    TORSO_HEIGHT_SPEED_M_S, TORSO_HEIGHT_ACCEL_M_S2, TeleopBodyMapper,
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
    assert command[0] == pytest.approx(BASE_LINEAR_SPEED_M_S)
    reached_height = sum(min(i / 30 * TORSO_HEIGHT_ACCEL_M_S2, TORSO_HEIGHT_SPEED_M_S) / 30
                         for i in range(1, 21))
    np.testing.assert_allclose(
        m._planar_position(command[3:5]), m.links.sum(axis=0) + [0, reached_height], atol=2e-5,
    )
    assert abs(command[3:6].sum()) < 1e-6


def test_deadzone_loss_and_pause_stop_base_and_hold_waist():
    m = mapper()
    np.testing.assert_allclose(m.advance(packet(.1, -.1), packet(.1, -.1), .1, enabled=True), 0)
    active = m.advance(packet(x=1), packet(x=1, y=1), .1, enabled=True)
    # A single 0.1s step only ramps yaw partway toward the full commanded rate.
    ramped_yaw = -min(BASE_YAW_SPEED_RAD_S, BASE_YAW_ACCEL_RAD_S2 * .1)
    np.testing.assert_allclose(active[1:3], [-BASE_LINEAR_ACCEL_M_S2 * .1, ramped_yaw])
    paused = m.advance(packet(1, 1), packet(1, 1), .1, enabled=False)
    lost = m.advance(None, None, .1, enabled=True)
    np.testing.assert_allclose(paused[:3], 0)
    np.testing.assert_allclose(paused[3:], active[3:])
    np.testing.assert_allclose(lost, paused)


@pytest.mark.parametrize("initial", [[.272, -.580, .329], [.272, -.580, .329, .25]])
def test_reset_synchronizes_nonzero_torso_without_neutral_stick_jump(initial):
    m = mapper()
    m.reset(initial)
    command = m.advance(packet(), packet(), 1/30, enabled=True)
    expected = [*initial, 0.] if len(initial) == 3 else initial
    np.testing.assert_allclose(command[3:], expected)


def test_s56_fixed_biped_keeps_height_channels_zero():
    m = TeleopBodyMapper(
        ASSET_DIR / "kuavo_s56/urdf/kuavo_s56.urdf",
        has_wheel_base=False,
    )
    command = m.advance(packet(y=1), packet(x=1, y=1), .1, enabled=True)
    assert command[0] > 0 and command[2] < 0
    np.testing.assert_allclose(command[3:], 0)
    assert m.height == 0.0


@pytest.mark.parametrize("hz", [30, 60])
def test_start_release_and_reversal_bound_all_motion_rates(hz):
    m = mapper()
    dt = 1 / hz
    previous = np.zeros(3)
    previous_height_rate = 0.0
    for stick in (1., 0., -1., 0.):
        for _ in range(hz):
            command = m.advance(packet(stick, stick), packet(stick, stick), dt, enabled=True)
            assert np.linalg.norm(command[:2]) <= BASE_LINEAR_SPEED_M_S + 1e-7
            assert np.linalg.norm(command[:2] - previous[:2]) <= BASE_LINEAR_ACCEL_M_S2 * dt + 1e-7
            assert abs(command[2] - previous[2]) <= BASE_YAW_ACCEL_RAD_S2 * dt + 1e-7
            assert abs(command[2]) <= BASE_YAW_SPEED_RAD_S + 1e-7
            # A hard height boundary is allowed to stop immediately.
            if 1e-5 < m.height < .40 - 1e-5:
                assert abs(m._height_rate - previous_height_rate) <= TORSO_HEIGHT_ACCEL_M_S2 * dt + 1e-7
            assert abs(m._height_rate) <= TORSO_HEIGHT_SPEED_M_S + 1e-7
            previous = command[:3].copy()
            previous_height_rate = m._height_rate


@pytest.mark.parametrize("bad", [None, np.zeros((1, 7)), np.full((2, 7), np.nan)])
def test_tracking_loss_stops_all_ramps_immediately(bad):
    m = mapper()
    for _ in range(30):
        m.advance(packet(y=1), packet(x=1, y=1), 1/30, enabled=True)
    height = m.height
    stopped = m.advance(bad, packet(), 1/30, enabled=True)
    np.testing.assert_array_equal(stopped[:3], 0)
    assert m._height_rate == 0 and m.height == height
    m.reset()
    np.testing.assert_array_equal(m._linear_velocity, 0)
    assert m._yaw_rate == m._height_rate == m.height == 0
