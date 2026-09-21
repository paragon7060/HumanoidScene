"""Policy-side placement hints use only pose and measured gripper opening."""

from dataclasses import replace
import math

import torch

from kuavo_isaaclab_scene.rl.multi_box.state.perception import PerceptionFrame
from kuavo_isaaclab_scene.rl.multi_box.state.pose_placement import (
    PosePlacementEstimateTracker,
    pose_placement_evidence,
)
from kuavo_isaaclab_scene.rl.multi_box.state.schema import DeployableBoxState, DeployableRobotState


def _poses(*shape):
    result = torch.zeros(*shape, 7)
    result[..., 3] = 1.0
    return result


def _input():
    active = torch.zeros(1, 12, dtype=torch.bool)
    active[0, 0] = True
    size = torch.zeros(1, 12, 3)
    size[0, 0] = torch.tensor([0.266, 0.185, 0.130])
    pose = _poses(1, 12)
    pose[0, 0, 2] = 0.015 + 0.005 * size[0, 0, 2]
    confidence = active.float()
    frame = PerceptionFrame(
        boxes=DeployableBoxState(
            active=active, box_type_id=torch.zeros(1, 12, dtype=torch.long),
            rack_region_id=torch.zeros(1, 12, dtype=torch.long),
            size_m=size, pose_world=pose, pose_confidence=confidence),
        rack_pose_world=_poses(1), conveyor_pose_world=_poses(1))
    robot = DeployableRobotState(
        joint_pos=torch.zeros(1, 20), joint_vel=torch.zeros(1, 20),
        base_pose_world=_poses(1), base_twist_world=torch.zeros(1, 6),
        # TCP proximity is deliberately irrelevant to this deployable hint.
        tcp_pose_world=pose[:, :1].expand(-1, 2, -1).clone(),
        gripper_position=torch.zeros(1, 2),
        gripper_command=torch.zeros(1, 2))
    return frame, robot


def test_release_and_stable_belt_height_are_needed_but_tcp_clearance_is_not():
    frame, robot = _input()
    tracker = PosePlacementEstimateTracker(1, "cpu")
    closed = replace(robot, gripper_position=torch.ones(1, 2))
    for _ in range(6):
        placed, _ = tracker.update(frame, closed, 0.1)
    assert not placed.placed[0, 0]

    for _ in range(5):
        placed, _ = tracker.update(frame, robot, 0.1)
    assert placed.placed[0, 0]
    # Another box can be grasped without revoking this box's estimate.
    placed, _ = tracker.update(frame, closed, 0.1)
    assert placed.placed[0, 0]
    assert not placed.selectable[0, 0]


def test_height_drift_revokes_placed_then_new_release_can_restore_it():
    frame, robot = _input()
    tracker = PosePlacementEstimateTracker(1, "cpu")
    for _ in range(5):
        placed, _ = tracker.update(frame, robot, 0.1)
    assert placed.placed[0, 0]
    moved_pose = frame.boxes.pose_world.clone()
    moved_pose[0, 0, 2] += 0.008
    moved = replace(frame, boxes=replace(frame.boxes, pose_world=moved_pose))
    placed, evidence = tracker.update(moved, robot, 0.1)
    assert not evidence.height_stable[0, 0]
    assert not placed.placed[0, 0]
    assert placed.selectable[0, 0]


def test_box_must_fit_belt_and_be_parallel_without_overlapping_peer():
    frame, robot = _input()
    config = PosePlacementEstimateTracker(1, "cpu").config
    evidence, _ = pose_placement_evidence(frame, robot, config)
    assert evidence.geometry_candidate[0, 0]

    tilted = frame.boxes.pose_world.clone()
    angle = math.radians(15)
    tilted[0, 0, 3] = math.cos(angle / 2)
    tilted[0, 0, 6] = math.sin(angle / 2)
    evidence, _ = pose_placement_evidence(
        replace(frame, boxes=replace(frame.boxes, pose_world=tilted)), robot, config)
    assert not evidence.axis_parallel[0, 0]

    active = frame.boxes.active.clone()
    active[0, 1] = True
    pose = frame.boxes.pose_world.clone()
    pose[0, 1] = pose[0, 0]
    size = frame.boxes.size_m.clone()
    size[0, 1] = size[0, 0]
    confidence = frame.boxes.pose_confidence.clone()
    confidence[0, 1] = 1
    evidence, _ = pose_placement_evidence(replace(frame, boxes=replace(
        frame.boxes, active=active, pose_world=pose, size_m=size,
        pose_confidence=confidence)), robot, config)
    assert not evidence.no_overlap[0, 0]
    assert not evidence.geometry_candidate[0, 0]
