from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from kuavo_isaaclab_scene.groot_lerobot_bridge import (
    CONTROLLED_JOINT_NAMES,
    MANAGER_ACTION_SCALES,
    KuavoLeRobotBridge,
    LeRobotGrootRunner,
    adapt_manager_action,
    adapt_policy_action,
    camera_rgb_to_lerobot,
    parse_camera_map,
)


def _joint_batch(value: float = 0.0) -> torch.Tensor:
    return torch.full((1, len(CONTROLLED_JOINT_NAMES)), value)


def test_joint_schema_is_waist_left_then_right() -> None:
    assert len(CONTROLLED_JOINT_NAMES) == 15
    assert CONTROLLED_JOINT_NAMES[0] == "waist_yaw_joint"
    assert CONTROLLED_JOINT_NAMES[1:8] == tuple(f"zarm_l{i}_joint" for i in range(1, 8))
    assert CONTROLLED_JOINT_NAMES[8:] == tuple(f"zarm_r{i}_joint" for i in range(1, 8))
    assert MANAGER_ACTION_SCALES == (1.0, *([0.45] * 14))


def test_camera_rgb_to_lerobot_drops_alpha_and_normalizes() -> None:
    rgba = torch.zeros((2, 4, 5, 4), dtype=torch.uint8)
    rgba[..., 0] = 255
    result = camera_rgb_to_lerobot(rgba)
    assert result.shape == (2, 3, 4, 5)
    assert result.dtype == torch.float32
    assert torch.all(result[:, 0] == 1.0)
    assert torch.all(result[:, 1:] == 0.0)


def test_adapt_absolute_joint_position_to_manager_coordinates() -> None:
    defaults = _joint_batch(0.1)
    scales = torch.tensor(MANAGER_ACTION_SCALES).unsqueeze(0)
    target = defaults + 0.5 * scales
    result = adapt_policy_action(
        target,
        current_joint_pos=defaults,
        default_joint_pos=defaults,
        action_scales=scales,
        mode="joint_position",
        clip=1.0,
    )
    assert torch.allclose(result.action, _joint_batch(0.5))
    assert result.saturation_fraction == 0.0


def test_adapt_joint_delta_uses_current_position_and_reports_clipping() -> None:
    defaults = _joint_batch(0.0)
    scales = torch.tensor(MANAGER_ACTION_SCALES).unsqueeze(0)
    current = scales * 0.8
    delta = scales * 0.5
    result = adapt_policy_action(
        delta,
        current_joint_pos=current,
        default_joint_pos=defaults,
        action_scales=scales,
        mode="joint_delta",
        clip=1.0,
    )
    assert torch.allclose(result.unclipped_action, _joint_batch(1.3))
    assert torch.allclose(result.action, _joint_batch(1.0))
    assert result.saturation_fraction == 1.0


def test_adapt_rejects_wrong_action_dimension() -> None:
    with pytest.raises(ValueError, match="action dimension"):
        adapt_policy_action(
            torch.zeros((1, 14)),
            current_joint_pos=torch.zeros((1, 15)),
            default_joint_pos=torch.zeros((1, 15)),
            action_scales=torch.ones((1, 15)),
        )


def test_manager_action_supports_configured_gripper_dimensions() -> None:
    result = adapt_manager_action(
        torch.tensor([[0.0] * 15 + [-1.2, 1.2]]),
        expected_dim=17,
        device="cpu",
        clip=1.0,
    )
    assert result.action.shape == (1, 17)
    assert result.action[0, -2:].tolist() == [-1.0, 1.0]
    assert result.saturation_fraction == pytest.approx(2.0 / 17.0)


def test_parse_camera_map_accepts_short_policy_keys() -> None:
    assert parse_camera_map(["front=robustness_camera", "left=left_wrist_camera"]) == {
        "observation.images.front": "robustness_camera",
        "observation.images.left": "left_wrist_camera",
    }
    with pytest.raises(ValueError, match="expected"):
        parse_camera_map(["invalid"])


class _FakeRobot:
    def __init__(self) -> None:
        self.device = "cpu"
        self.data = SimpleNamespace(
            joint_pos=torch.zeros((1, 15)),
            default_joint_pos=torch.zeros((1, 15)),
        )

    def find_joints(self, names, preserve_order=False):
        assert preserve_order
        assert tuple(names) == CONTROLLED_JOINT_NAMES
        return list(range(15)), list(names)


def test_bridge_builds_lerobot_observation_and_applies_action() -> None:
    robot = _FakeRobot()
    camera = SimpleNamespace(
        data=SimpleNamespace(output={"rgb": torch.full((1, 3, 4, 3), 128, dtype=torch.uint8)})
    )
    env = SimpleNamespace(scene={"robot": robot, "head_sensor": camera})
    bridge = KuavoLeRobotBridge(
        env,
        camera_map={"observation.images.front": "head_sensor"},
    )
    observation = bridge.observation("move boxes")
    assert observation["observation.state"].shape == (1, 15)
    assert observation["observation.images.front"].shape == (1, 3, 3, 4)
    assert observation["task"] == ["move boxes"]
    assert torch.equal(bridge.action(torch.zeros((1, 15))).action, torch.zeros((1, 15)))


class _Feature:
    shape = (15,)


class _FakePolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            n_action_steps=2,
            input_features={"observation.state": _Feature()},
            output_features={"action": _Feature()},
        )
        self.calls = 0
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def predict_action_chunk(self, batch):
        self.calls += 1
        batch_size = batch["observation.state"].shape[0]
        first = torch.full((batch_size, 15), float(self.calls))
        second = torch.full((batch_size, 15), float(self.calls) + 0.5)
        return torch.stack((first, second), dim=1)


def test_policy_runner_postprocesses_full_chunk_then_queues_steps() -> None:
    policy = _FakePolicy()
    runner = LeRobotGrootRunner(
        policy,
        preprocessor=lambda batch: batch,
        postprocessor=lambda chunk: chunk + 10.0,
    )
    runner.reset()
    observation = {"observation.state": torch.zeros((1, 15))}
    first = runner.select_action(observation)
    second = runner.select_action(observation)
    third = runner.select_action(observation)
    assert first.inferred_new_chunk
    assert not second.inferred_new_chunk
    assert third.inferred_new_chunk
    assert torch.all(first.action == 11.0)
    assert torch.all(second.action == 11.5)
    assert torch.all(third.action == 12.0)
    assert policy.calls == 2
    assert policy.resets == 1
