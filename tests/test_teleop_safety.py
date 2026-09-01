from kuavo_isaaclab_scene.teleop_safety import GripperCommandLatch, TrackingLossGuard


def test_gripper_latch_holds_only_the_lost_hand_command():
    latch = GripperCommandLatch(("left", "right"))
    assert latch.advance((-1.0, -1.0), left_valid=True, right_valid=True) == (-1.0, -1.0)
    assert latch.advance((1.0, 1.0), left_valid=False, right_valid=True) == (-1.0, 1.0)
    assert latch.advance((1.0, -1.0), left_valid=False, right_valid=False) == (-1.0, 1.0)


def test_tracking_guard_pauses_then_requires_stable_recovery():
    guard = TrackingLossGuard(recovery_frames=2, abort_after_s=1.0)
    assert guard.advance(True, 0.0).control_allowed
    lost = guard.advance(False, 1.0)
    assert lost.recording_paused and not lost.abort_episode
    assert not guard.advance(True, 1.1).control_allowed
    recovered = guard.advance(True, 1.2)
    assert recovered.control_allowed and recovered.recovered


def test_tracking_guard_aborts_after_timeout():
    guard = TrackingLossGuard(recovery_frames=2, abort_after_s=0.5)
    guard.advance(False, 2.0)
    assert not guard.advance(False, 2.49).abort_episode
    assert guard.advance(False, 2.5).abort_episode
