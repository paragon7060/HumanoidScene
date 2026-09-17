"""CPU checks for privileged/deployable state separation."""

from dataclasses import fields, replace

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.hierarchy import SkillStateMachine
from kuavo_isaaclab_scene.rl.multi_box.observations import build_observations
from kuavo_isaaclab_scene.rl.multi_box.state import (
    DeployableBoxState,
    DeployableRobotState,
    DeployableTaskState,
    MultiBoxState,
    PlacementSetTracker,
    PrivilegedTaskState,
)


def state(num_envs=2):
    def poses(*shape):
        value = torch.zeros(*shape, 7)
        value[..., 3] = 1.0
        return value

    active = torch.zeros(num_envs, 12, dtype=torch.bool)
    active[:, 0] = True
    placement = PlacementSetTracker(num_envs, "cpu").update(
        active, torch.zeros_like(active), 0.1)
    deployable = DeployableTaskState(
        boxes=DeployableBoxState(
            active=active,
            box_type_id=torch.zeros(num_envs, 12, dtype=torch.long),
            rack_region_id=torch.zeros(num_envs, 12, dtype=torch.long),
            size_m=torch.ones(num_envs, 12, 3),
            pose_world=poses(num_envs, 12),
            pose_confidence=torch.ones(num_envs, 12),
        ),
        robot=DeployableRobotState(
            joint_pos=torch.zeros(num_envs, 25),
            joint_vel=torch.zeros(num_envs, 25),
            base_pose_world=poses(num_envs),
            base_twist_world=torch.zeros(num_envs, 6),
            tcp_pose_world=poses(num_envs, 2),
            gripper_position=torch.zeros(num_envs, 2),
            gripper_command=torch.zeros(num_envs, 2),
        ),
        rack_pose_world=poses(num_envs),
        conveyor_pose_world=poses(num_envs),
        placement=placement,
        control=SkillStateMachine(num_envs, "cpu").state(),
        transition_confidence=torch.zeros(num_envs, 3),
    )
    privileged = PrivilegedTaskState(
        box_pose_world=poses(num_envs, 12),
        box_pose_rack=poses(num_envs, 12),
        box_pose_conveyor=poses(num_envs, 12),
        box_linear_velocity=torch.zeros(num_envs, 12, 3),
        box_angular_velocity=torch.zeros(num_envs, 12, 3),
        hand_pinching=torch.zeros(num_envs, 12, 2, dtype=torch.bool),
        hand_flap_index=torch.zeros(num_envs, 12, 2, dtype=torch.long),
        hand_box_pose_stable=torch.zeros(num_envs, 12, 2, dtype=torch.bool),
        finger_contact_force_n=torch.zeros(num_envs, 12, 2, 2),
        rack_clearance_m=torch.zeros(num_envs, 12),
        belt_support=torch.zeros(num_envs, 12, dtype=torch.bool),
        footprint_corners_belt=torch.zeros(num_envs, 12, 4, 2),
        overlaps_placed_box=torch.zeros(num_envs, 12, dtype=torch.bool),
        valid_place=torch.zeros(num_envs, 12, dtype=torch.bool),
        rack_contact_force_n=torch.zeros(num_envs),
        self_contact_force_n=torch.zeros(num_envs),
        obstacle_contact_force_n=torch.zeros(num_envs),
        base_distance_m=torch.zeros(num_envs),
    )
    return MultiBoxState(deployable, privileged)


def test_approved_state_schema_validates_and_keeps_privileged_signals_separate():
    value = state()
    value.validate(2)
    deployable_fields = {field.name for field in fields(DeployableBoxState)} \
        | {field.name for field in fields(DeployableRobotState)}
    assert "box_linear_velocity" not in deployable_fields
    assert "finger_contact_force_n" not in deployable_fields
    privileged_fields = {field.name for field in fields(PrivilegedTaskState)}
    assert {"box_linear_velocity", "finger_contact_force_n"} <= privileged_fields


def test_state_schema_rejects_wrong_box_or_contact_tensor_shapes():
    value = state()
    wrong_boxes = replace(value.deployable.boxes, pose_world=torch.zeros(2, 12, 6))
    with pytest.raises(ValueError, match="pose_world"):
        replace(value, deployable=replace(value.deployable, boxes=wrong_boxes)).validate(2)
    wrong_privileged = replace(value.privileged, finger_contact_force_n=torch.zeros(2, 12, 2))
    with pytest.raises(ValueError, match="finger_contact_force"):
        replace(value, privileged=wrong_privileged).validate(2)


def test_actor_observation_uses_base_frame_and_excludes_privileged_signals():
    value = state()
    first = build_observations(value, torch.zeros(2, 25))
    assert first.actor.box_tokens.shape == (2, 12, 22)
    assert first.actor.anchor_poses.shape == (2, 2, 9)
    assert first.critic.privileged_box_tokens.shape[-1] > first.actor.box_tokens.shape[-1]

    changed_privileged = replace(
        value.privileged,
        box_linear_velocity=torch.full((2, 12, 3), 7.0),
        finger_contact_force_n=torch.full((2, 12, 2, 2), 11.0),
    )
    second = build_observations(replace(value, privileged=changed_privileged), torch.zeros(2, 25))
    assert torch.equal(first.actor.box_tokens, second.actor.box_tokens)
    assert not torch.equal(first.critic.privileged_box_tokens, second.critic.privileged_box_tokens)


def test_common_world_translation_does_not_change_base_relative_pose_observations():
    value = state()
    delta = torch.tensor([2.0, -3.0, 0.7])

    def translated(pose):
        result = pose.clone()
        result[..., :3] += delta
        return result

    boxes = replace(value.deployable.boxes, pose_world=translated(value.deployable.boxes.pose_world))
    robot = replace(
        value.deployable.robot,
        base_pose_world=translated(value.deployable.robot.base_pose_world),
        tcp_pose_world=translated(value.deployable.robot.tcp_pose_world),
    )
    deployable = replace(
        value.deployable,
        boxes=boxes,
        robot=robot,
        rack_pose_world=translated(value.deployable.rack_pose_world),
        conveyor_pose_world=translated(value.deployable.conveyor_pose_world),
    )
    privileged = replace(
        value.privileged,
        box_pose_world=translated(value.privileged.box_pose_world),
    )
    shifted = build_observations(MultiBoxState(deployable, privileged), torch.zeros(2, 25))
    original = build_observations(value, torch.zeros(2, 25))
    torch.testing.assert_close(original.actor.anchor_poses, shifted.actor.anchor_poses)
    torch.testing.assert_close(original.actor.box_tokens, shifted.actor.box_tokens)
    torch.testing.assert_close(
        original.critic.privileged_box_tokens, shifted.critic.privileged_box_tokens)
