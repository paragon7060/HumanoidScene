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

