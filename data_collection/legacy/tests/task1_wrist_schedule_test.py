"""CPU checks for the Task1 staged wrist-pitch contract."""

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "task1_pregrasp_smoke.py"
SPEC = importlib.util.spec_from_file_location("task1_pregrasp_smoke", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_task1_config_has_shared_partial_then_full_pitch():
    configs = MODULE._load_configs(ROOT / "data_collection" / "configs")
    wrist = configs["task1"]["wrist_pitch"]
    assert wrist["enabled"] is True
    assert wrist["full_target_rad"] == 0.65
    assert wrist["transit_fraction"] == 0.33
    assert wrist["direct_q7_gain"] > 0.0
    assert wrist["transit_orientation_weight"] == 0.0
    assert 0.0 < wrist["full_target_rad"] * wrist["transit_fraction"] < wrist["full_target_rad"]
    assert wrist["prepare_steps"] >= 0
    assert wrist["rotate_steps"] > 0


def test_axis_angle_quaternion_is_unit_and_uses_wxyz():
    axis = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    angle = torch.tensor([0.0, 3.141592653589793])
    result = MODULE._quat_axis_angle(axis, angle)
    torch.testing.assert_close(result[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(result[1], torch.tensor([0.0, 1.0, 0.0, 0.0]), atol=1.0e-6, rtol=0)
    torch.testing.assert_close(result.norm(dim=-1), torch.ones(2), atol=1.0e-6, rtol=0)
