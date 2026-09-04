import numpy as np

from kuavo_isaaclab_scene.teleop.teleop_mapping import AbsoluteControllerMapper, ScaledControllerMapper, BimanualTeleopMapper, TeleopMappingCfg


def test_scaled_reach_and_clutch_without_drift_or_recalibration_on_tracking_loss():
    mapper = ScaledControllerMapper()
    root = np.array([0., 0., 0., 1., 0., 0., 0.])
    tool = np.array([.3, .25, .65, 1., 0., 0., 0.])
    packet = np.array([[.2, .25, 1.1, 1., 0., 0., 0.], [0.] * 7])
    first = mapper.target("left", packet, tool, root, following=True)
    np.testing.assert_allclose(first, tool, atol=1e-6)
    packet[0, :3] += [.2, 0., -.2]
    for _ in range(100):
        goal = mapper.target("left", packet, tool, root, following=True)
        np.testing.assert_allclose(goal[:3], [.52, .25, .43], atol=1e-6)
    np.testing.assert_allclose(mapper.target("left", None, tool, root, following=True), goal)
    np.testing.assert_allclose(mapper.target("left", packet, tool, root, following=True), goal)
    mapper.target("left", packet, tool, root, following=False)
    packet[0, :3] = [.1, .2, 1.2]
    resumed = mapper.target("left", packet, tool, root, following=True)
    np.testing.assert_allclose(resumed[:3], tool[:3], atol=1e-6)


def test_scaled_gain_does_not_multiply_base_turn_or_torso_lift():
    from kuavo_isaaclab_scene.teleop.teleop_mapping import _quat_rotate
    mapper = ScaledControllerMapper(position_gain=2.)
    root = np.array([0., 0., 0., 1., 0., 0., 0.])
    torso = np.array([0., 0., .5, 1., 0., 0., 0.])
    tool = np.array([.3, .25, .7, 1., 0., 0., 0.])
    packet = np.array([[.2, .25, 1., 1., 0., 0., 0.], [0.] * 7])
    mapper.target("left", packet, tool, root, following=True, reference_pose_w=torso)
    rotation = np.array([np.sqrt(.5), 0, 0, np.sqrt(.5)])
    torso[:3] = [1., 2., .8]; torso[3:] = rotation
    packet[0, :3] = torso[:3] + _quat_rotate(rotation, [.2, .25, .5])
    packet[0, 3:] = rotation
    goal = mapper.target("left", packet, tool, root, following=True, reference_pose_w=torso)
    expected = torso[:3] + _quat_rotate(rotation, [.3, .25, .2])
    np.testing.assert_allclose(goal[:3], expected, atol=1e-6)
    np.testing.assert_allclose(goal[3:], rotation, atol=1e-6)


def test_scaled_orientation_clutch_uses_actual_pose_and_unscaled_torso_axis_rotation():
    from kuavo_isaaclab_scene.teleop.teleop_mapping import _quat_multiply, _quat_rotate
    mapper = ScaledControllerMapper(position_gain=2.)
    root = np.array([0., 0., 0., 1., 0., 0., 0.])
    # Tool points 90 degrees about Z; the comfortable controller points about X.
    c = np.sqrt(.5)
    tool = np.array([.3, .25, .65, c, 0., 0., c])
    packet = np.array([[.2, .25, 1.1, c, c, 0., 0.], [0.] * 7])
    first = mapper.target("left", packet, tool, root, following=True)
    np.testing.assert_allclose(first, tool, atol=1e-6)
    # A 90 degree turn about torso Y must rotate the original tool axes about Y,
    # not about the original tool's local Y, and must not use position gain=2.
    packet[0, 3:] = _quat_multiply([c, 0., c, 0.], packet[0, 3:])
    for _ in range(100):
        goal = mapper.target("left", packet, tool, root, following=True)
        axes = np.column_stack([_quat_rotate(goal[3:], axis) for axis in np.eye(3)])
        np.testing.assert_allclose(axes, [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]], atol=1e-6)
        np.testing.assert_allclose(goal[:3], tool[:3], atol=1e-6)
    # Quaternion sign and an aim-pose disappearance must not cause a rotation.
    packet[0, 3:] *= -1
    aim = np.array([0., 0., 0., 1., 0., 0., 0.])
    stable = mapper.target("left", packet, tool, root, following=True, aim_pose=aim)
    assert abs(np.dot(stable[3:], goal[3:])) > 1 - 1e-6
    assert abs(np.dot(mapper.target("left", packet, tool, root, following=True)[3:], goal[3:])) > 1 - 1e-6
    np.testing.assert_allclose(mapper.target("left", None, tool, root, following=True), stable)
    # The actual tool lags the commanded orientation. Resume from that actual
    # pose after the operator rotates their wrist back into a comfortable pose.
    tool[3:] = [np.cos(np.pi / 12), 0., np.sin(np.pi / 12), 0.]
    mapper.target("left", packet, tool, root, following=False)
    packet[0, 3:] = [1., 0., 0., 0.]
    resumed = mapper.target("left", packet, tool, root, following=True)
    np.testing.assert_allclose(resumed, tool, atol=1e-6)
    packet[0, 3:] = [np.cos(np.pi / 12), 0., np.sin(np.pi / 12), 0.]
    advanced = mapper.target("left", packet, tool, root, following=True)
    np.testing.assert_allclose(_quat_rotate(advanced[3:], [0., 0., 1.]),
                               [np.sqrt(3) / 2, 0., .5], atol=1e-6)


def test_scaled_orientation_reacquisition_preserves_reference_and_reset_is_per_hand():
    mapper = ScaledControllerMapper()
    root = np.array([1., 2., 0., np.sqrt(.5), 0., 0., np.sqrt(.5)])
    tool = np.array([1., 2., 1., *root[3:]])
    packet = np.array([[1., 2.2, 1., 1., 0., 0., 0.], [0.] * 7])
    for side in ("left", "right"):
        first = mapper.target(side, packet, tool, root, following=True)
        np.testing.assert_allclose(first[3:], [1., 0., 0., 0.], atol=1e-6)
    packet[0, 3:] = [0., 0., 0., 1.]  # 180 degrees; quaternion conversion stays finite.
    goal = mapper.target("right", packet, tool, root, following=True)
    mapper.target("right", None, tool, root, following=True)
    np.testing.assert_allclose(mapper.target("right", packet, tool, root, following=True), goal)
    mapper.target("left", packet, tool, root, following=False)
    left = mapper.target("left", packet, tool, root, following=True)
    np.testing.assert_allclose(left[3:], [1., 0., 0., 0.], atol=1e-6)
    np.testing.assert_allclose(mapper.target("right", packet, tool, root, following=True), goal)
    mapper.reset()
    reset = mapper.target("right", packet, tool, root, following=True)
    np.testing.assert_allclose(reset[3:], [1., 0., 0., 0.], atol=1e-6)


def test_absolute_controller_position_uses_translated_rotated_robot_frame_without_drift():
    mapper = AbsoluteControllerMapper()
    # Root is translated and rotated 90 degrees about world Z.
    root = np.array([2., 3., 1., np.sqrt(.5), 0., 0., np.sqrt(.5)])
    tool = np.array([2., 3.1, 1.2, 1., 0., 0., 0.])
    packet = np.array([[2., 3.5, 1.3, 1., 0., 0., 0.], [0.] * 7])
    for _ in range(100):
        goal = mapper.target("left", packet, tool, root, following=True)
        np.testing.assert_allclose(goal[:3], [.5, 0., .3], atol=1e-6)
    packet[0, :3] = [2., 3.1, 1.2]
    returned = mapper.target("left", packet, tool, root, following=True)
    np.testing.assert_allclose(returned[:3], [.1, 0., .2], atol=1e-6)


def test_absolute_controller_loss_retains_goal_but_explicit_pause_holds_actual_tool():
    mapper = AbsoluteControllerMapper()
    root = np.array([0., 0., 0., 1., 0., 0., 0.])
    tool = np.array([.2, .1, 1., 1., 0., 0., 0.])
    packet = np.array([[.5, .2, 1.2, 1., 0., 0., 0.], [0.] * 7])
    goal = mapper.target("left", packet, tool, root, following=True)
    np.testing.assert_allclose(mapper.target("left", None, tool, root, following=True), goal)
    np.testing.assert_allclose(mapper.target("left", packet, tool, root, following=False), tool)
    packet[0, 0] = .3
    resumed = mapper.target("left", packet, tool, root, following=True)
    np.testing.assert_allclose(resumed[:3], packet[0, :3], atol=1e-6)


def test_absolute_orientation_uses_finger_axes_independent_of_robot_pose_or_calibration():
    from kuavo_isaaclab_scene.teleop.teleop_mapping import _quat_rotate
    mapper = AbsoluteControllerMapper()
    root = np.array([0., 0., 0., 1., 0., 0., 0.])
    tool = np.array([.2, .1, 1., 1., 0., 0., 0.])
    packet = np.array([[.5, .2, 1.2, 1., 0., 0., 0.], [0.] * 7])
    aim = np.array([.5, .2, 1.2, np.sqrt(.5), -np.sqrt(.5), 0., 0.])
    first = mapper.target("left", packet, tool, root, following=True, aim_pose=aim)
    # Aim -Z -> -Y for this synthetic controller; thumb is grip -Z.
    np.testing.assert_allclose(_quat_rotate(first[3:], [0, 0, -1]), [0, -1, 0], atol=1e-6)
    np.testing.assert_allclose(_quat_rotate(first[3:], [1, 0, 0]), [0, 0, -1], atol=1e-6)
    mapper.reset(); tool[3:] = [0., 1., 0., 0.]
    again = mapper.target("right", packet, tool, root, following=True, aim_pose=aim)
    np.testing.assert_allclose(again, first, atol=1e-6)
    s63 = AbsoluteControllerMapper(tool_forward_sign=1).target(
        "left", packet, tool, root, following=True, aim_pose=aim)
    np.testing.assert_allclose(_quat_rotate(s63[3:], [0, 0, 1]), [0, -1, 0], atol=1e-6)
    np.testing.assert_allclose(_quat_rotate(s63[3:], [1, 0, 0]), [0, 0, -1], atol=1e-6)


def _hand(x: float, orientation=(1.0, 0.0, 0.0, 0.0)):
    wrist = np.array([x, 0.2, 1.0, *orientation], dtype=np.float32)
    thumb = wrist.copy()
    index = wrist.copy()
    index[0] += 0.03
    return {"wrist": wrist, "thumb_tip": thumb, "index_tip": index}


def test_first_valid_frame_calibrates_without_motion():
    mapper = BimanualTeleopMapper()
    output = mapper.advance(
        _hand(0.2),
        _hand(-0.2),
        np.array([0.0, 0.0, 1.6, 1.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    assert output.bimanual_valid
    np.testing.assert_allclose(output.action, np.zeros(14), atol=1.0e-8)


def test_bimanual_translation_and_safety_clamp():
    mapper = BimanualTeleopMapper(
        TeleopMappingCfg(position_gain=10.0, position_smoothing=1.0, max_position_step_m=0.02)
    )
    head = np.array([0.0, 0.0, 1.6, 1.0, 0.0, 0.0, 0.0])
    root = np.array([1.0, 0.0, 0.0, 0.0])
    mapper.advance(_hand(0.2), _hand(-0.2), head, root)
    output = mapper.advance(_hand(0.3), _hand(-0.1), head, root)
    assert output.action[0] > 0.0
    assert output.action[6] > 0.0
    assert np.linalg.norm(output.action[:3]) <= 0.020001
    assert np.linalg.norm(output.action[6:9]) <= 0.020001


def test_tracking_loss_recalibrates_instead_of_jumping():
    mapper = BimanualTeleopMapper()
    head = np.array([0.0, 0.0, 1.6, 1.0, 0.0, 0.0, 0.0])
    root = np.array([1.0, 0.0, 0.0, 0.0])
    mapper.advance(_hand(0.2), _hand(-0.2), head, root)
    lost = mapper.advance(None, _hand(-0.2), head, root)
    assert not lost.left_valid
    reacquired = mapper.advance(_hand(0.8), _hand(-0.2), head, root)
    assert reacquired.left_valid
    np.testing.assert_allclose(reacquired.action[:6], np.zeros(6), atol=1.0e-8)


def test_reclutch_hands_preserves_head_reference_and_prevents_arm_jump():
    mapper = BimanualTeleopMapper(TeleopMappingCfg(head_smoothing=1.0))
    root = np.array([1.0, 0.0, 0.0, 0.0])
    head = np.array([0.0, 0.0, 1.6, 1.0, 0.0, 0.0, 0.0])
    mapper.advance(_hand(0.2), _hand(-0.2), head, root)
    head[3:] = [np.cos(0.2), 0.0, np.sin(0.2), 0.0]
    turned = mapper.advance(_hand(0.3), _hand(-0.1), head, root)
    mapper.reset_hands()
    reclutched = mapper.advance(_hand(1.0), _hand(0.9), head, root)
    np.testing.assert_allclose(reclutched.action[:12], 0.0, atol=1e-8)
    np.testing.assert_allclose(reclutched.action[12:], turned.action[12:], atol=1e-8)


def test_recenter_uses_current_head_direction_as_neutral():
    mapper = BimanualTeleopMapper(TeleopMappingCfg(head_smoothing=1.0))
    root = np.array([1.0, 0.0, 0.0, 0.0])
    head = np.array([0.0, 0.0, 1.6, 1.0, 0.0, 0.0, 0.0])
    mapper.advance(_hand(0.2), _hand(-0.2), head, root)
    head[3:] = [np.cos(0.3), 0.0, np.sin(0.3), 0.0]
    assert abs(mapper.advance(_hand(0.2), _hand(-0.2), head, root).action[12]) > 0.1
    mapper.reset()
    centered = mapper.advance(_hand(0.2), _hand(-0.2), head, root)
    np.testing.assert_allclose(centered.action, 0.0, atol=1e-8)


def test_recenter_can_hold_the_robot_head_at_its_current_target():
    mapper = BimanualTeleopMapper(TeleopMappingCfg(head_smoothing=1.0))
    head = np.array([0.0, 0.0, 1.6, np.cos(0.3), 0.0, 0.0, np.sin(0.3)])
    target = np.array([0.2, -0.1])
    mapper.reset(head_target=target)
    centered = mapper.advance(_hand(0.2), _hand(-0.2), head, np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(centered.action[12:], target, atol=1e-8)
    np.testing.assert_allclose(centered.action[:12], 0.0, atol=1e-8)


def test_controller_motion_reclutches_after_tracking_loss_without_fake_fingers():
    mapper = BimanualTeleopMapper(TeleopMappingCfg(position_smoothing=1.0))
    head = np.array([0.0, 0.0, 1.6, 1.0, 0.0, 0.0, 0.0])
    root = np.array([1.0, 0.0, 0.0, 0.0])
    packet = np.array([[0.2, 0.1, 1.0, 1.0, 0.0, 0.0, 0.0], [0, 0, 1, 0, 0, 0, 0]], dtype=float)
    first = mapper.advance_controllers(packet, packet, head, root)
    np.testing.assert_allclose(first.action, 0.0)
    assert first.bimanual_valid and np.isnan(first.left_pinch_m)
    packet[0, 0] += 0.01
    moved = mapper.advance_controllers(packet, packet, head, root)
    assert 0 < moved.action[0] <= mapper.cfg.max_position_step_m
    lost = mapper.advance_controllers(None, packet, head, root)
    assert not lost.left_valid and lost.right_valid
    np.testing.assert_allclose(lost.action[:6], 0.0)
    packet[0, 0] += 1.0
    reacquired = mapper.advance_controllers(packet, packet, head, root)
    np.testing.assert_allclose(reacquired.action[:6], 0.0)


def test_invalid_controller_packets_never_move_arms():
    mapper = BimanualTeleopMapper()
    for packet in (np.zeros((2, 7)), np.full((2, 7), np.nan), np.zeros(7)):
        mapped = mapper.advance_controllers(packet, packet, None, np.array([1., 0., 0., 0.]))
        assert not mapped.bimanual_valid
        np.testing.assert_allclose(mapped.action[:12], 0.0)


def test_head_turn_and_nod_signs_follow_hmd_and_ignore_base_yaw():
    for root in ([1., 0, 0, 0], [np.sqrt(.5), 0, 0, np.sqrt(.5)]):
        mapper = BimanualTeleopMapper(TeleopMappingCfg(head_smoothing=1.0))
        neutral = np.array([0., 0., 1.6, 1., 0., 0., 0.])
        mapper._head_target(neutral, root)
        left = neutral.copy(); left[3:] = [np.cos(.15), 0, np.sin(.15), 0]
        command, _ = mapper._head_target(left, root)
        np.testing.assert_allclose(command, [.3, 0], atol=1e-6)
        up = neutral.copy(); up[3:] = [np.cos(.1), np.sin(.1), 0, 0]
        command, _ = mapper._head_target(up, root)
        np.testing.assert_allclose(command, [0, -.2], atol=1e-6)
