"""Pure CPU checks for real-obtainable v2 skill transition evidence."""

from dataclasses import replace

import torch

from kuavo_isaaclab_scene.rl.multi_box.hierarchy import (
    HighLevelSelection, SKILL_IDS, SkillStateMachine,
)
from kuavo_isaaclab_scene.rl.multi_box.hierarchy.transition_estimator import (
    DeployableTransitionEstimator,
)
from kuavo_isaaclab_scene.rl.multi_box.state.perception import PerceptionFrame
from kuavo_isaaclab_scene.rl.multi_box.state.pose_placement import PosePlacementEstimateTracker
from kuavo_isaaclab_scene.rl.multi_box.state.schema import DeployableBoxState, DeployableRobotState


def _poses(*shape):
    result = torch.zeros(*shape, 7)
    result[..., 3] = 1
    return result


def _inputs():
    active = torch.zeros(1, 12, dtype=torch.bool)
    active[0, 0] = True
    size = torch.zeros(1, 12, 3)
    size[0, 0] = torch.tensor([0.266, 0.185, 0.130])
    box_pose = _poses(1, 12)
    box_pose[0, 0, 2] = 0.30
    perception = PerceptionFrame(
        boxes=DeployableBoxState(
            active=active, box_type_id=torch.zeros(1, 12, dtype=torch.long),
            rack_region_id=torch.zeros(1, 12, dtype=torch.long),
            size_m=size, pose_world=box_pose,
            pose_confidence=active.float()),
        rack_pose_world=_poses(1), conveyor_pose_world=_poses(1))
    robot = DeployableRobotState(
        joint_pos=torch.zeros(1, 20), joint_vel=torch.zeros(1, 20),
        base_pose_world=_poses(1), base_twist_world=torch.zeros(1, 6),
        tcp_pose_world=box_pose[:, :1].expand(-1, 2, -1).clone(),
        gripper_position=torch.ones(1, 2), gripper_command=torch.ones(1, 2))
    machine = SkillStateMachine(1, "cpu")
    machine.assign(HighLevelSelection(torch.tensor([0])), active)
    return perception, robot, machine


def _move_box(perception, robot, z):
    pose = perception.boxes.pose_world.clone()
    pose[0, 0, 2] = z
    return (replace(perception, boxes=replace(perception.boxes, pose_world=pose)),
            replace(robot, tcp_pose_world=pose[:, :1].expand(-1, 2, -1).clone()))


def _update(estimator, placement_tracker, perception, robot, machine):
    placement, placement_evidence = placement_tracker.update(perception, robot, 0.1)
    return estimator.update(
        perception=perception, robot=robot, placement=placement,
        placement_evidence=placement_evidence,
        control=machine.state(), dt=0.1)


def test_pose_only_transition_sequence_and_explicit_skill_state():
    perception, robot, machine = _inputs()
    placement_tracker = PosePlacementEstimateTracker(1, "cpu")
    estimator = DeployableTransitionEstimator(1, "cpu")
    first = _update(estimator, placement_tracker, perception, robot, machine)
    assert not first.evidence.grasp_complete[0]
    perception, robot = _move_box(perception, robot, 0.309)
    for _ in range(3):
        grasp = _update(estimator, placement_tracker, perception, robot, machine)
    assert grasp.grippers_closed.all()
    assert grasp.hand_box_pose_stable.all()
    assert grasp.evidence.grasp_complete[0]
    assert grasp.progress[0, 0] == 1
    machine.advance(grasp.evidence)
    assert machine.state().current_skill.item() == SKILL_IDS["carry"]

    perception, robot = _move_box(perception, robot, 0.115 + 0.005 * 0.130)
    carry = _update(estimator, placement_tracker, perception, robot, machine)
    assert carry.pre_place_height[0]
    assert carry.evidence.carry_complete[0]
    machine.advance(carry.evidence)
    assert machine.state().current_skill.item() == SKILL_IDS["place"]

    perception, robot = _move_box(perception, robot, 0.015 + 0.005 * 0.130)
    robot = replace(robot, gripper_position=torch.zeros(1, 2))
    for _ in range(5):
        place = _update(estimator, placement_tracker, perception, robot, machine)
    assert place.evidence.place_complete[0]
    completed = machine.advance(place.evidence)
    assert machine.state().needs_target[0]
    assert completed.completed_box.item() == 0


def test_grasp_requires_box_lift_and_resets_on_target_loss():
    perception, robot, machine = _inputs()
    placement = PosePlacementEstimateTracker(1, "cpu")
    estimator = DeployableTransitionEstimator(1, "cpu")
    for _ in range(5):
        result = _update(estimator, placement, perception, robot, machine)
    assert not result.evidence.grasp_complete[0]
    assert not result.proof_lift[0]
    machine.reset()
    result = _update(estimator, placement, perception, robot, machine)
    assert not result.evidence.grasp_complete[0]
    assert result.progress.sum() == 0


def test_deployable_stability_stays_armed_when_box_is_lowered_below_rack():
    perception, robot, machine = _inputs()
    placement = PosePlacementEstimateTracker(1, "cpu")
    estimator = DeployableTransitionEstimator(1, "cpu")

    _update(estimator, placement, perception, robot, machine)
    perception, robot = _move_box(perception, robot, 0.309)
    for _ in range(3):
        grasp = _update(estimator, placement, perception, robot, machine)
    assert grasp.evidence.grasp_complete[0]
    machine.advance(grasp.evidence)

    perception, robot = _move_box(perception, robot, 0.115 + 0.005 * 0.130)
    carry = _update(estimator, placement, perception, robot, machine)
    assert carry.hand_box_pose_stable.all()
    assert carry.pre_place_height[0]
    assert carry.evidence.carry_complete[0]


def test_deployable_carry_does_not_reuse_grasp_pose_tolerance():
    perception, robot, machine = _inputs()
    placement = PosePlacementEstimateTracker(1, "cpu")
    estimator = DeployableTransitionEstimator(1, "cpu")

    _update(estimator, placement, perception, robot, machine)
    perception, robot = _move_box(perception, robot, 0.309)
    for _ in range(3):
        grasp = _update(estimator, placement, perception, robot, machine)
    assert grasp.evidence.grasp_complete[0]
    machine.advance(grasp.evidence)

    perception, robot = _move_box(perception, robot, 0.115 + 0.005 * 0.130)
    tcp = robot.tcp_pose_world.clone()
    tcp[:, :, 0] += 0.03
    robot = replace(robot, tcp_pose_world=tcp)
    carry = _update(estimator, placement, perception, robot, machine)
    assert not carry.hand_box_pose_stable.all()
    assert carry.evidence.carry_complete[0]


def test_deployable_carry_rejects_box_tilt_above_twenty_degrees():
    perception, robot, machine = _inputs()
    placement = PosePlacementEstimateTracker(1, "cpu")
    estimator = DeployableTransitionEstimator(1, "cpu")

    _update(estimator, placement, perception, robot, machine)
    perception, robot = _move_box(perception, robot, 0.309)
    for _ in range(3):
        grasp = _update(estimator, placement, perception, robot, machine)
    machine.advance(grasp.evidence)

    perception, robot = _move_box(perception, robot, 0.115 + 0.005 * 0.130)
    pose = perception.boxes.pose_world.clone()
    angle = torch.deg2rad(torch.tensor(20.1))
    pose[0, 0, 3] = torch.cos(angle / 2)
    pose[0, 0, 4] = torch.sin(angle / 2)
    perception = replace(
        perception,
        boxes=replace(perception.boxes, pose_world=pose),
    )
    carry = _update(estimator, placement, perception, robot, machine)
    assert not carry.box_tilt_ok[0]
    assert not carry.evidence.carry_complete[0]
