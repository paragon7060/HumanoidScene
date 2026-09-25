"""Pure tensor observation construction; no Isaac or reward dependency."""

from __future__ import annotations

import torch

from ..geometry import pose_to_position_rotation_6d, relative_pose
from ..geometry.grasp import (
    GRASP_APPROACH_REWARD_SCALE_M,
    estimated_flap_center_poses,
    opposing_flap_reach_assignment,
)
from ..spec import BOX_TYPES, MAX_BOXES
from ..state import DeployableTaskState, MultiBoxState
from .schema import ActorObservation, CriticObservation, ObservationBundle


NUM_REGIONS = 4
BOX_TOKEN_DIM = 22
ROBOT_PROPRIO_DIM = 68
HAND_FLAP_RELATION_DIM = 2 * 2 * 9
OPPOSING_FLAP_ASSIGNMENT_DIM = 2


def flat_actor_observation_dim(action_dim: int) -> int:
    """Feature count of :func:`flatten_actor_observation`."""
    if not isinstance(action_dim, int) or isinstance(action_dim, bool) or action_dim < 1:
        raise ValueError("action_dim must be a positive integer.")
    return (
        ROBOT_PROPRIO_DIM
        + 2 * 9
        + MAX_BOXES * BOX_TOKEN_DIM
        + HAND_FLAP_RELATION_DIM
        + OPPOSING_FLAP_ASSIGNMENT_DIM
        + MAX_BOXES
        + MAX_BOXES
        + 3
        + 1
        + action_dim
    )


def _masked_one_hot(indices: torch.Tensor, classes: int, valid: torch.Tensor) -> torch.Tensor:
    encoded = torch.nn.functional.one_hot(indices.clamp(0, classes - 1), classes).to(torch.float32)
    return encoded * valid[..., None]


def _relative_pose9(base_pose: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
    return pose_to_position_rotation_6d(relative_pose(base_pose, target_pose))


def gather_target_box_token(observation: ActorObservation) -> torch.Tensor:
    """Gather the locked target token; rows without a target return zeros."""
    valid = observation.target_box >= 0
    ids = observation.target_box.clamp(0, MAX_BOXES - 1)
    env_ids = torch.arange(len(ids), device=ids.device)
    token = observation.box_tokens[env_ids, ids]
    return token * valid[:, None]


def flatten_actor_observation(observation: ActorObservation) -> torch.Tensor:
    """Flatten the deployable actor contract in one stable feature order.

    Routing fields use masks and one-hot tensors.  The raw target index is
    omitted so an MLP cannot infer an ordinal relationship between logical box
    identifiers.
    """
    num_envs = observation.robot_proprio.shape[0]
    values = (
        observation.robot_proprio,
        observation.anchor_poses.flatten(1),
        observation.box_tokens.flatten(1),
        observation.hand_flap_relations.flatten(1),
        observation.opposing_flap_assignment,
        observation.box_mask.to(torch.float32),
        observation.target_one_hot,
        observation.current_skill_one_hot,
        observation.needs_target.to(torch.float32).reshape(num_envs, 1),
        observation.previous_action,
    )
    if any(value.ndim != 2 or value.shape[0] != num_envs for value in values):
        raise ValueError("Actor observation fields must flatten to [num_envs, features].")
    return torch.cat(values, dim=-1)


def build_actor_observation(
    deployable: DeployableTaskState, previous_action: torch.Tensor,
) -> ActorObservation:
    """Build policy inputs without accepting privileged simulator state."""
    num_envs = len(deployable.boxes.active)
    deployable.validate(num_envs)
    if previous_action.ndim != 2 or previous_action.shape[0] != num_envs \
            or not previous_action.is_floating_point():
        raise ValueError("previous_action must be floating point [num_envs, action_dim].")

    boxes = deployable.boxes
    robot = deployable.robot
    base = robot.base_pose_world
    rack_pose9 = _relative_pose9(base, deployable.rack_pose_world)
    conveyor_pose9 = _relative_pose9(base, deployable.conveyor_pose_world)
    tcp_pose9 = _relative_pose9(base, robot.tcp_pose_world).flatten(1)
    box_pose9 = _relative_pose9(base, boxes.pose_world)

    active_float = boxes.active.to(torch.float32)
    type_one_hot = _masked_one_hot(boxes.box_type_id, len(BOX_TYPES), boxes.active)
    region_one_hot = _masked_one_hot(boxes.rack_region_id, NUM_REGIONS, boxes.active)
    box_tokens = torch.cat((
        active_float[..., None],
        deployable.placement.selectable.to(torch.float32)[..., None],
        deployable.placement.placed.to(torch.float32)[..., None],
        type_one_hot,
        boxes.size_m,
        region_one_hot,
        box_pose9,
        boxes.pose_confidence[..., None],
    ), dim=-1) * active_float[..., None]

    robot_proprio = torch.cat((
        robot.joint_pos,
        robot.joint_vel,
        robot.base_twist_world,
        robot.gripper_position,
        robot.gripper_command,
        tcp_pose9,
    ), dim=-1)
    target = deployable.control.target_box
    target_valid = target >= 0
    target_one_hot = _masked_one_hot(target, MAX_BOXES, target_valid)
    rows = torch.arange(num_envs, device=target.device)
    target_ids = target.clamp(0, MAX_BOXES - 1)
    goal_valid = (target_valid & (target < MAX_BOXES)
                  & boxes.active[rows, target_ids]
                  & (boxes.pose_confidence[rows, target_ids] > 0))
    center_poses, candidate_distances = estimated_flap_center_poses(
        boxes.pose_world[rows, target_ids],
        boxes.size_m[rows, target_ids],
        boxes.box_type_id[rows, target_ids],
        robot.tcp_pose_world,
    )
    relations = pose_to_position_rotation_6d(relative_pose(
        robot.tcp_pose_world[:, :, None], center_poses,
    )) * goal_valid[:, None, None, None]
    _, assignment = opposing_flap_reach_assignment(
        candidate_distances, GRASP_APPROACH_REWARD_SCALE_M)
    # Encode the first hand's flap; the second hand always takes the opposite.
    assignment_one_hot = torch.nn.functional.one_hot(assignment[:, 0], 2).to(
        relations.dtype) * goal_valid[:, None]
    actor = ActorObservation(
        robot_proprio=robot_proprio,
        anchor_poses=torch.stack((rack_pose9, conveyor_pose9), dim=1),
        box_tokens=box_tokens,
        hand_flap_relations=relations,
        opposing_flap_assignment=assignment_one_hot,
        box_mask=boxes.active.clone(),
        target_box=target.clone(),
        target_one_hot=target_one_hot,
        current_skill_one_hot=deployable.control.current_skill_one_hot.clone(),
        needs_target=deployable.control.needs_target.clone(),
        previous_action=previous_action.clone(),
    )
    return actor


def build_observations(state: MultiBoxState, previous_action: torch.Tensor) -> ObservationBundle:
    """Build deployable actor inputs and an asymmetric privileged critic view."""
    num_envs = len(state.deployable.boxes.active)
    state.validate(num_envs)
    actor = build_actor_observation(state.deployable, previous_action)
    base = state.deployable.robot.base_pose_world
    active_float = state.deployable.boxes.active.to(torch.float32)

    privileged = state.privileged
    exact_pose9 = _relative_pose9(base, privileged.box_pose_world)
    flap_valid = (privileged.hand_flap_index >= 0) & (privileged.hand_flap_index < 2)
    flap_one_hot = _masked_one_hot(privileged.hand_flap_index, 2, flap_valid).flatten(-2)
    privileged_box = torch.cat((
        exact_pose9,
        privileged.box_linear_velocity,
        privileged.box_angular_velocity,
        privileged.hand_pinching.to(torch.float32),
        flap_one_hot,
        privileged.hand_box_pose_stable.to(torch.float32),
        privileged.finger_contact_force_n.flatten(-2),
        privileged.rack_clearance_m[..., None],
        privileged.belt_support.to(torch.float32)[..., None],
        privileged.footprint_corners_belt.flatten(-2),
        privileged.overlaps_placed_box.to(torch.float32)[..., None],
        privileged.valid_place.to(torch.float32)[..., None],
    ), dim=-1) * active_float[..., None]
    privileged_global = torch.stack((
        privileged.rack_contact_force_n,
        privileged.self_contact_force_n,
        privileged.obstacle_contact_force_n,
        privileged.base_distance_m,
    ), dim=-1)
    critic = CriticObservation(actor, privileged_box, privileged_global)
    return ObservationBundle(actor, critic)
