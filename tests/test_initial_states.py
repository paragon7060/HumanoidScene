"""Initial-state storage and physical reset contract without launching Isaac Sim."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.robots.initial_states import (
    apply_initial_state, capture_initial_state, load_initial_state,
    read_states, save_initial_state, validate_state,
)


def preset():
    return {"robot_model": "s200062", "gripper": "s200062_integrated",
            "coordinate_frame": "env-origin", "assets": {"robot": {
                "joint_positions": {"arm": 0.4},
                "root_pose": [1, 2, 3, 1, 0, 0, 0]}}}


def test_library_preserves_names_and_requires_explicit_overwrite(tmp_path):
    path = tmp_path / "states.json"
    save_initial_state("a", preset(), path)
    b = preset(); b["assets"]["robot"]["joint_positions"]["arm"] = 0.7
    save_initial_state("b", b, path)
    with pytest.raises(FileExistsError):
        save_initial_state("a", b, path)
    assert load_initial_state("a", path) == preset()
    save_initial_state("a", b, path, overwrite=True)
    assert load_initial_state("a", path) == b
    assert set(read_states(path)["states"]) == {"a", "b"}


def test_unknown_name_and_model_mismatch(tmp_path):
    path = tmp_path / "states.json"
    save_initial_state("a", preset(), path)
    with pytest.raises(ValueError, match="available"):
        load_initial_state("other", path)
    for kwargs in ({"robot_model": "s56"}, {"gripper": "robotiq_2f85"}):
        with pytest.raises(ValueError, match="selected"):
            load_initial_state("a", path, **kwargs)


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(coordinate_frame="world"),
    lambda s: s["assets"]["robot"]["joint_positions"].update(arm=float("nan")),
    lambda s: s["assets"]["robot"].update(root_pose=[0]*7),
    lambda s: s["assets"]["robot"].update(root_pose=[0]*6),
])
def test_reject_invalid_state(mutation):
    state = preset(); mutation(state)
    with pytest.raises(ValueError):
        validate_state(state)


class Asset:
    joint_names = ["arm", "head"]

    def __init__(self):
        self.data = SimpleNamespace(
            joint_pos=torch.tensor([[0.1, 0.2], [0.3, 0.6]]),
            joint_pos_limits=torch.tensor([[[-1., 1.], [-1., 1.]]]*2),
            root_pose_w=torch.tensor([[0., 0., 0., 1., 0., 0., 0.], [10., 0., 0., 1., 0., 0., 0.]]),
        )
        self.calls = []

    def write_root_pose_to_sim(self, value, *, env_ids):
        self.calls.append("root"); self.data.root_pose_w[env_ids] = value

    def write_root_velocity_to_sim(self, value, *, env_ids):
        self.calls.append("velocity"); self.root_velocity = value

    def write_joint_state_to_sim(self, q, dq, *, env_ids):
        self.calls.append("joints"); self.data.joint_pos[env_ids] = q; self.velocity = dq

    def set_joint_position_target(self, value, *, env_ids):
        self.calls.append("target"); self.target = value

    def set_joint_velocity_target(self, value, *, env_ids):
        self.velocity_target = value


def environment():
    asset = Asset()
    scene = {"robot": asset}
    class Scene(dict):
        env_origins = torch.tensor([[0., 0., 0.], [10., 0., 0.]])
    env = SimpleNamespace(num_envs=2, device="cpu", scene=Scene(scene))
    env.unwrapped = env
    return env, asset


def test_partial_reset_maps_joint_names_and_env_origin_and_holds_targets():
    env, asset = environment()
    apply_initial_state(env, [1], preset())
    torch.testing.assert_close(asset.data.joint_pos, torch.tensor([[0.1, 0.2], [0.4, 0.6]]))
    torch.testing.assert_close(asset.target, torch.tensor([[0.4, 0.6]]))
    assert asset.data.root_pose_w[1, :3].tolist() == [11., 2., 3.]
    assert not asset.velocity.any() and not asset.root_velocity.any()


def test_joint_only_keeps_reset_root_and_validation_precedes_writes():
    env, asset = environment()
    state = preset(); state["assets"]["robot"].pop("root_pose")
    apply_initial_state(env, [1], state)
    assert "root" not in asset.calls
    asset.calls.clear()
    state["assets"]["robot"]["joint_positions"]["arm"] = 2.0
    with pytest.raises(ValueError, match="limits"):
        apply_initial_state(env, [1], state)
    assert not asset.calls
    state = preset(); state["assets"]["robot"]["joint_positions"]["typo"] = 0
    with pytest.raises(ValueError, match="Unknown joints"):
        apply_initial_state(env, [1], state)
    assert not asset.calls


def test_capture_roundtrip_and_env_relative_root(tmp_path, monkeypatch):
    from kuavo_isaaclab_scene.robots import robot_model, gripper_config
    monkeypatch.setattr(robot_model, "resolve_robot_model", lambda: SimpleNamespace(name="s200062"))
    monkeypatch.setattr(gripper_config, "resolve_gripper_settings", lambda: SimpleNamespace(
        name="s200062_integrated", active_sides=("left", "right"), asset_name_for=lambda side: "robot"))
    env, asset = environment()
    path = tmp_path / "states.json"
    capture_initial_state(env, "vr", path, env_index=1)
    state = load_initial_state("vr", path)
    assert state["assets"]["robot"]["root_pose"][:3] == [0., 0., 0.]
    assert set(state["assets"]) == {"robot"}
    apply_initial_state(env, [0], state)
    torch.testing.assert_close(asset.data.joint_pos[0], asset.data.joint_pos[1])


def test_user_pose_packaged_copy_and_all_36_joints():
    root = Path(__file__).resolve().parents[1]
    source = read_states(root / "configs/initial_states.json")
    packaged = read_states(root / "src/kuavo_isaaclab_scene/configs/initial_states.json")
    assert source == packaged
    saved = source["states"]["quest_ready_01"]["assets"]["robot"]
    assert "root_pose" not in saved
    assert len(saved["joint_positions"]) == 36
    assert saved["joint_positions"]["zhead_1_joint"] == 0.05829557031393051
    ready = source["states"]["quest_ready_02"]
    assert ready["robot_model"] == "s200062"
    assert ready["gripper"] == "s200062_integrated"
    assert len(ready["assets"]["robot"]["root_pose"]) == 7
    assert len(ready["assets"]["robot"]["joint_positions"]) == 36


def test_second_rack_pose_preserves_meta_base_and_sets_user_controls():
    root = Path(__file__).resolve().parents[1]
    state = read_states(root / "configs/initial_states.json")["states"][
        "second_rack_pose"
    ]
    robot = state["assets"]["robot"]
    joints = robot["joint_positions"]

    assert "root_pose" not in robot
    assert len(joints) == 24
    assert joints["zarm_l1_joint"] == 0.25
    assert joints["zarm_r4_joint"] == -0.6499999761581421
    assert joints["knee_joint"] == 0.25423744320869446
    assert joints["leg_joint"] == -0.5588340163230896
    assert joints["waist_pitch_joint"] == 0.30459657311439514
    assert joints["zhead_2_joint"] == 0.5240000486373901
    assert joints["l_f_bar_1_joint"] == -0.25
    assert joints["r_b_bar_1_joint"] == 0.25
