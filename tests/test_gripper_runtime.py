import pytest

torch = pytest.importorskip("torch")

from kuavo_isaaclab_scene.robots.gripper_action import interpolate_signed_gripper_action


def test_signed_gripper_interpolation_preserves_endpoints_and_midpoint() -> None:
    opened = torch.tensor([-0.25, -0.25, 0.25, 0.25])
    closed = torch.zeros(4)
    actions = torch.tensor([[1.0], [0.0], [-1.0]])
    result = interpolate_signed_gripper_action(actions, opened, closed)
    assert torch.allclose(result[0], opened)
    assert torch.allclose(result[1], opened * 0.5)
    assert torch.allclose(result[2], closed)


def test_signed_gripper_interpolation_clamps_out_of_range_policy_values() -> None:
    opened = torch.tensor([-0.25, 0.25])
    closed = torch.zeros(2)
    result = interpolate_signed_gripper_action(
        torch.tensor([[2.0], [-2.0]]), opened, closed
    )
    assert torch.allclose(result[0], opened)
    assert torch.allclose(result[1], closed)
