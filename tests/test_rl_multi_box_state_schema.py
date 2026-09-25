"""CPU checks for privileged/deployable state separation."""

from dataclasses import fields, replace

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.hierarchy import SkillStateMachine
from kuavo_isaaclab_scene.rl.multi_box.demo_replay import convert_legacy_actor_observation
from kuavo_isaaclab_scene.rl.multi_box.observations import (
    build_observations,
    flat_actor_observation_dim,
    flatten_actor_observation,
)
from kuavo_isaaclab_scene.rl.multi_box.state import (
    DeployableBoxState,
    DeployablePlacementEstimateState,
    DeployableRobotState,
    DeployableTaskState,
    MultiBoxState,
    PerceptionFrame,
    PlacementSetTracker,
    PrivilegedTaskState,
    assemble_deployable_task_state,
)


def state(num_envs=2):
    def poses(*shape):
        value = torch.zeros(*shape, 7)
        value[..., 3] = 1.0
        return value

    active = torch.zeros(num_envs, 12, dtype=torch.bool)
    active[:, 0] = True
    placement = DeployablePlacementEstimateState(
        active=active, placed=torch.zeros_like(active),
        selectable=active.clone(), hold_time_s=torch.zeros(num_envs, 12))
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
            joint_pos=torch.zeros(num_envs, 20),
            joint_vel=torch.zeros(num_envs, 20),
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


def test_actor_rejects_privileged_placement_tracker_output():
    value = state()
    active = value.deployable.boxes.active
    privileged_placement = PlacementSetTracker(2, "cpu").update(
        active, torch.zeros_like(active), 0.1)
    with pytest.raises(TypeError, match="pose-based deployable estimate"):
        build_observations(replace(value, deployable=replace(
            value.deployable, placement=privileged_placement)), torch.zeros(2, 25))


def test_deployable_assembler_uses_only_perception_proprio_and_estimated_placement():
    deployable = state().deployable
    frame = PerceptionFrame(
        boxes=deployable.boxes, rack_pose_world=deployable.rack_pose_world,
        conveyor_pose_world=deployable.conveyor_pose_world)
    rebuilt = assemble_deployable_task_state(
        perception=frame, robot=deployable.robot,
        placement=deployable.placement, control=deployable.control,
        transition_confidence=deployable.transition_confidence)
    assert rebuilt.boxes is frame.boxes
    assert rebuilt.placement is deployable.placement
    privileged_placement = PlacementSetTracker(2, "cpu").update(
        deployable.boxes.active, torch.zeros_like(deployable.boxes.active), 0.1)
    with pytest.raises(TypeError, match="pose-based placement estimate"):
        assemble_deployable_task_state(
            perception=frame, robot=deployable.robot,
            placement=privileged_placement, control=deployable.control,
            transition_confidence=deployable.transition_confidence)


def test_actor_observation_uses_base_frame_and_excludes_privileged_signals():
    value = state()
    value = replace(value, deployable=replace(
        value.deployable, control=replace(
            value.deployable.control,
            target_box=torch.zeros(2, dtype=torch.long),
        ),
    ))
    first = build_observations(value, torch.zeros(2, 25))
    assert first.actor.box_tokens.shape == (2, 12, 22)
    assert first.actor.anchor_poses.shape == (2, 2, 9)
    assert first.actor.hand_flap_relations.shape == (2, 2, 2, 9)
    assert first.actor.opposing_flap_assignment.shape == (2, 2)
    assert first.critic.privileged_box_tokens.shape[-1] > first.actor.box_tokens.shape[-1]

    changed_privileged = replace(
        value.privileged,
        box_linear_velocity=torch.full((2, 12, 3), 7.0),
        finger_contact_force_n=torch.full((2, 12, 2, 2), 11.0),
    )
    second = build_observations(replace(value, privileged=changed_privileged), torch.zeros(2, 25))
    assert torch.equal(first.actor.box_tokens, second.actor.box_tokens)
    assert torch.equal(first.actor.hand_flap_relations, second.actor.hand_flap_relations)
    assert torch.equal(first.actor.opposing_flap_assignment,
                       second.actor.opposing_flap_assignment)
    assert not torch.equal(first.critic.privileged_box_tokens, second.critic.privileged_box_tokens)


def test_actor_observation_has_stable_flat_manager_layout_without_ordinal_target():
    observation = build_observations(state(), torch.zeros(2, 25)).actor
    flat = flatten_actor_observation(observation)
    expected = (
        observation.robot_proprio.shape[1]
        + observation.anchor_poses[0].numel()
        + observation.box_tokens[0].numel()
        + observation.hand_flap_relations[0].numel()
        + observation.opposing_flap_assignment.shape[1]
        + observation.box_mask.shape[1]
        + observation.target_one_hot.shape[1]
        + observation.current_skill_one_hot.shape[1]
        + 1
        + observation.previous_action.shape[1]
    )
    assert flat.shape == (2, expected)
    assert expected == flat_actor_observation_dim(25)
    assert expected == 441


def test_hand_flap_goal_uses_fixed_centers_and_masks_missing_target():
    value = state()
    boxes = value.deployable.boxes
    size = boxes.size_m.clone()
    size[:, 0] = torch.tensor((0.266, 0.185, 0.130))
    pose = boxes.pose_world.clone()
    pose[0, 0, :3] = torch.tensor((1.0, 2.0, 0.4))
    tcp = value.deployable.robot.tcp_pose_world.clone()
    tcp[0, 0, :3] = torch.tensor((1.3, 2.0, 0.58065))
    tcp[0, 1, :3] = torch.tensor((0.7, 2.0, 0.58065))
    target = value.deployable.control.target_box.clone()
    target[0] = 0
    deployable = replace(
        value.deployable,
        boxes=replace(boxes, size_m=size, pose_world=pose),
        robot=replace(value.deployable.robot, tcp_pose_world=tcp),
        control=replace(value.deployable.control, target_box=target),
    )
    actor = build_observations(replace(value, deployable=deployable), torch.zeros(2, 25)).actor
    assert actor.opposing_flap_assignment[0].tolist() == [1.0, 0.0]
    torch.testing.assert_close(
        actor.hand_flap_relations[0, 0, 0, :3],
        torch.tensor((0.266 / 1.01 * 0.5 - 0.3, 0.0, 0.0)), atol=1e-6, rtol=0,
    )
    torch.testing.assert_close(
        actor.hand_flap_relations[0, 1, 1, :3],
        torch.tensor((0.3 - 0.266 / 1.01 * 0.5, 0.0, 0.0)), atol=1e-6, rtol=0,
    )
    shifted_tcp = tcp.clone()
    shifted_tcp[0, 0, 1] += 0.03
    shifted_actor = build_observations(replace(
        value, deployable=replace(
            deployable, robot=replace(deployable.robot, tcp_pose_world=shifted_tcp)),
    ), torch.zeros(2, 25)).actor
    torch.testing.assert_close(
        shifted_actor.hand_flap_relations[0, 0, 0, :3],
        actor.hand_flap_relations[0, 0, 0, :3] - torch.tensor((0.0, 0.03, 0.0)),
        atol=1e-6, rtol=0,
    )
    assert torch.count_nonzero(actor.hand_flap_relations[1]) == 0
    assert torch.count_nonzero(actor.opposing_flap_assignment[1]) == 0

    no_confidence = boxes.pose_confidence.clone()
    no_confidence[0, 0] = 0
    hidden = build_observations(replace(
        value, deployable=replace(
            deployable, boxes=replace(deployable.boxes, pose_confidence=no_confidence)),
    ), torch.zeros(2, 25)).actor
    assert torch.count_nonzero(hidden.hand_flap_relations[0]) == 0
    assert torch.count_nonzero(hidden.opposing_flap_assignment[0]) == 0


def test_legacy_demo_conversion_reconstructs_current_observation_from_recorded_poses():
    value = state()
    boxes = value.deployable.boxes
    pose = boxes.pose_world.clone()
    pose[0, 0, :3] = torch.tensor((0.7, -0.2, 0.3))
    pose[0, 0, 3:] = torch.tensor((0.9238795, 0.0, 0.0, 0.3826834))
    size = boxes.size_m.clone()
    size[0, 0] = torch.tensor((0.266, 0.185, 0.130))
    tcp = value.deployable.robot.tcp_pose_world.clone()
    tcp[0, 0, :3] = torch.tensor((0.9, -0.05, 0.5))
    tcp[0, 1, :3] = torch.tensor((0.55, -0.4, 0.5))
    tcp[0, 1, 3:] = torch.tensor((0.9659258, 0.0, 0.0, -0.2588190))
    target = value.deployable.control.target_box.clone()
    target[0] = 0
    deployable = replace(
        value.deployable,
        boxes=replace(boxes, pose_world=pose, size_m=size),
        robot=replace(value.deployable.robot, tcp_pose_world=tcp),
        control=replace(value.deployable.control, target_box=target),
    )
    current = flatten_actor_observation(build_observations(
        replace(value, deployable=deployable), torch.zeros(2, 25)).actor)
    legacy = torch.cat((current[:, :350], current[:, 388:]), dim=-1)
    converted = convert_legacy_actor_observation(legacy)
    torch.testing.assert_close(converted, current, atol=2e-6, rtol=1e-6)


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
