import numpy as np

from kuavo_isaaclab_scene.teleop_mapping import BimanualTeleopMapper, TeleopMappingCfg


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
    head[3:] = [np.cos(0.2), 0.0, 0.0, np.sin(0.2)]
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
    head[3:] = [np.cos(0.3), 0.0, 0.0, np.sin(0.3)]
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
