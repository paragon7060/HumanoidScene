"""Test logical hand isolation without requiring Isaac Sim."""

import re
from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from kuavo_isaaclab_scene.evaluation.groot_lerobot_bridge import (
    CONTROLLED_JOINT_NAMES, RWH_KUAVO_V2_CAMERA_MAP, RWH_KUAVO_V2_NAMES, KuavoLeRobotBridge,
)
from kuavo_isaaclab_scene.robots.gripper_config import load_gripper_settings
from kuavo_isaaclab_scene.robots.gripper_io import resolve_gripper_views


class Asset:
    def __init__(self, names):
        self.device = "cpu"
        self.data = SimpleNamespace(
            joint_names=list(names), joint_pos=torch.zeros((2, len(names))),
            default_joint_pos=torch.zeros((2, len(names))),
            soft_joint_pos_limits=torch.tensor([[[-3.0, 3.0]] * len(names)] * 2),
        )

    def find_joints(self, patterns, preserve_order=False):
        assert preserve_order
        ids = []
        for pattern in patterns:
            matched = [i for i, name in enumerate(self.data.joint_names) if re.fullmatch(pattern, name)]
            if not matched:
                raise ValueError(f"Missing joints: {pattern}")
            ids.extend(matched)
        return ids, [self.data.joint_names[i] for i in ids]


def make_env(preset):
    settings = load_gripper_settings(preset)
    names = list(reversed(CONTROLLED_JOINT_NAMES))  # Deliberately not policy order.
    if settings.integrated:
        for side in settings.active_sides:
            names.extend(reversed(settings.joint_names_for(side)))
        names.append("passive_unobserved_joint")
    scene = {"robot": Asset(names)}
    if settings.enabled and not settings.integrated:
        hand_names = [f"{s}_{part}_joint" for s in ("left", "right")
                      for part in ("driver", "coupler", "spring_link", "follower")]
        hand_names.append("extra_passive_joint")
        for side in settings.active_sides:
            scene[f"{side}_gripper"] = Asset(hand_names)
    camera = SimpleNamespace(data=SimpleNamespace(output={"rgb": torch.zeros((2, 2, 3, 3), dtype=torch.uint8)}))
    scene["test_camera"] = camera
    env = SimpleNamespace(scene=scene, action_manager=SimpleNamespace(total_action_dim=15 + len(settings.active_sides)))
    views = resolve_gripper_views(env, settings)
    for view in views.values():
        view.asset.data.default_joint_pos[:, view.command_joint_ids] = view.open_pos
        view.asset.data.joint_pos[:, view.command_joint_ids] = view.open_pos
    return env, settings


def bridge_for(env, settings, profile="default", mode="manager"):
    return KuavoLeRobotBridge(
        env, gripper_settings=settings, policy_profile=profile,
        state_mode=mode, action_mode=mode, action_clip=None,
        camera_map={"observation.images.test": "test_camera"},
    )


@pytest.mark.parametrize("preset", ["s200062_integrated", "s56_twofinger", "s56_qiangnao", "robotiq_2f85"])
def test_same_measured_claw_interface_for_integrated_and_external_hands(preset):
    env, settings = make_env(preset)
    bridge = bridge_for(env, settings, "kuavo-arm-claw", "joint_position")
    assert bridge.state_names == bridge.action_names == RWH_KUAVO_V2_NAMES
    assert bridge.state().shape == (2, 16)
    for side, fractions in (("left", [0.2, 0.8]), ("right", [0.7, 0.1])):
        view = bridge.grippers[side]
        values = torch.tensor(fractions).unsqueeze(1)
        view.asset.data.joint_pos[:, view.command_joint_ids] = view.open_pos + values * (view.close_pos - view.open_pos)
        assert view.claw_state().flatten().tolist() == pytest.approx(fractions)
        assert view.asset_name == ("robot" if settings.integrated else f"{side}_gripper")
        assert view.metadata()["command_joint_names"]
    measured = bridge.state()
    assert measured[:, 7].tolist() == pytest.approx([0.2, 0.8])
    assert measured[:, 15].tolist() == pytest.approx([0.7, 0.1])
    action = bridge.action(measured).action
    assert action.shape == (2, 17)
    assert action[:, 15].tolist() == pytest.approx([0.6, -0.6])
    assert action[:, 16].tolist() == pytest.approx([-0.4, 0.8])
    assert torch.allclose(bridge.hold_action().action, action)
    # Read live joint state after a reset, not cached commands or prior observations.
    for view in bridge.grippers.values():
        view.asset.data.joint_pos.copy_(view.asset.data.default_joint_pos)
    assert bridge.state()[:, [7, 15]].tolist() == [[0.0, 0.0], [0.0, 0.0]]


@pytest.mark.parametrize(("preset", "state_dim"), [
    ("s200062_integrated", 19), ("s56_twofinger", 19),
    ("s56_qiangnao", 35), ("robotiq_2f85", 33), ("none", 15),
])
def test_default_state_includes_hand_joints_and_names_match_dimensions(preset, state_dim):
    env, settings = make_env(preset)
    bridge = bridge_for(env, settings)
    assert bridge.state().shape == (2, state_dim)
    assert len(bridge.state_names) == state_dim
    assert len(set(bridge.state_names)) == state_dim
    assert len(bridge.action_names) == bridge.action_dim == env.action_manager.total_action_dim
    assert torch.count_nonzero(bridge.state()) == 0
    offset = 15
    for view in bridge.grippers.values():
        view.asset.data.joint_pos[:, view.joint_ids] += 0.1
        assert torch.allclose(bridge.state()[:, offset:offset + len(view.joint_ids)], torch.full((2, len(view.joint_ids)), 0.1))
        offset += len(view.joint_ids)
    raw = bridge_for(env, settings, mode="joint_position")
    assert raw.action_dim == len(raw.action_names) == 15
    assert raw.action(torch.zeros((2, 15))).action.shape[-1] == env.action_manager.total_action_dim
    offset = 15
    for view in raw.grippers.values():
        assert torch.equal(raw.state()[:, offset:offset + len(view.joint_ids)], view.joint_state())
        offset += len(view.joint_ids)


def test_s56_profile_remains_equivalent_to_common_profile():
    env, settings = make_env("s56_twofinger")
    common = bridge_for(env, settings, "kuavo-arm-claw", "joint_position")
    legacy = bridge_for(env, settings, "rwh-kuavo-v2-s56", "joint_position")
    assert torch.equal(common.state(), legacy.state())
    assert torch.equal(common.hold_action().action, legacy.hold_action().action)


def test_enabled_missing_hand_fails_instead_of_silently_dropping_state():
    env, settings = make_env("robotiq_2f85")
    del env.scene["right_gripper"]
    with pytest.raises(ValueError, match="Enabled right gripper"):
        bridge_for(env, settings)


def test_integrated_missing_joint_fails_instead_of_silently_dropping_state():
    env, settings = make_env("s200062_integrated")
    env.scene["robot"].data.joint_names[15] = "wrong_name"
    with pytest.raises(ValueError, match="Missing joints"):
        bridge_for(env, settings)


def test_action_manager_dimension_mismatch_fails_early():
    env, settings = make_env("s200062_integrated")
    env.action_manager.total_action_dim = 15
    with pytest.raises(RuntimeError, match="Manager action dimension"):
        bridge_for(env, settings)


def test_single_hand_default_has_one_action_but_uniform_profile_rejects_it():
    env, settings = make_env("s200062_integrated")
    settings = replace(settings, sides={
        **settings.sides, "right": replace(settings.sides["right"], enabled=False),
    })
    env.action_manager.total_action_dim = 16
    bridge = bridge_for(env, settings)
    assert bridge.state().shape == (2, 17)
    assert len(bridge.action_names) == 16
    assert bridge.action_names[-1] == "left_gripper"
    with pytest.raises(RuntimeError, match="17-D Kuavo manager"):
        bridge_for(env, settings, "kuavo-arm-claw", "joint_position")


@pytest.mark.parametrize("profile", ["kuavo-arm-claw", "rwh-kuavo-v2-s56"])
def test_uniform_profile_uses_three_camera_keys_without_explicit_map(profile):
    env, settings = make_env("s56_twofinger")
    for key in RWH_KUAVO_V2_CAMERA_MAP.values():
        env.scene[key] = env.scene["test_camera"]
    bridge = KuavoLeRobotBridge(
        env, gripper_settings=settings, policy_profile=profile,
        state_mode="joint_position", action_mode="joint_position",
    )
    assert bridge.camera_map == RWH_KUAVO_V2_CAMERA_MAP
    assert len(bridge.observation("pick up the box")) == 5  # state, task, 3 cameras


def test_explicit_empty_camera_map_is_rejected():
    env, settings = make_env("s200062_integrated")
    with pytest.raises(ValueError, match="At least one camera"):
        KuavoLeRobotBridge(env, gripper_settings=settings, camera_map={})
