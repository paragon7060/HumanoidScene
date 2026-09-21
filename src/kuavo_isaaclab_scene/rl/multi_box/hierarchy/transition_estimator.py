"""Real-obtainable evidence for grasp -> carry -> place skill transitions.

This estimator is deliberately separate from the privileged success tests.
It consumes box perception and robot proprioception, never PhysX contacts,
forces, box velocity, or simulator success flags.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..geometry.pose import quat_apply, relative_pose
from ..spec import MAX_BOXES
from ..state.perception import PerceptionFrame
from ..state.placement import DeployablePlacementEstimateState
from ..state.pose_placement import PosePlacementEvidence
from ..state.schema import DeployableRobotState
from ..success.stability import RelativePoseStabilityTracker
from .state_machine import DeployableTransitionEvidence, NO_TARGET, SkillControlState


@dataclass(frozen=True)
class DeployableTransitionConfig:
    min_gripper_closed_fraction: float = 0.80
    proof_lift_m: float = 0.008
    grasp_hold_seconds: float = 0.25
    place_hold_seconds: float = 0.50
    min_pre_place_height_m: float = 0.05
    max_pre_place_height_m: float = 0.15
    max_box_tilt_rad: float = math.radians(20.0)

    def validate(self) -> None:
        values = vars(self)
        if not all(math.isfinite(value) and value > 0 for value in values.values()):
            raise ValueError("Transition thresholds must be finite and positive.")
        if self.min_gripper_closed_fraction >= 1:
            raise ValueError("Gripper close fraction must be below one.")
        if self.min_pre_place_height_m >= self.max_pre_place_height_m:
            raise ValueError("Pre-place height interval must be ordered.")
        if self.max_box_tilt_rad >= math.pi / 2:
            raise ValueError("Carry box tilt limit must be below 90 degrees.")


@dataclass(frozen=True)
class DeployableTransitionEstimate:
    evidence: DeployableTransitionEvidence
    # Progress diagnostics in [0,1], not calibrated success probabilities.
    progress: torch.Tensor
    grippers_closed: torch.Tensor
    hand_box_pose_stable: torch.Tensor
    proof_lift: torch.Tensor
    grasp_hold_time_s: torch.Tensor
    pre_place_height: torch.Tensor
    box_tilt_ok: torch.Tensor


class DeployableTransitionEstimator:
    """Tracks one selected logical box per environment without truth leakage."""

    def __init__(self, num_envs: int, device: str | torch.device, config=None):
        if num_envs < 1:
            raise ValueError("num_envs must be positive.")
        self.config = config or DeployableTransitionConfig()
        self.config.validate()
        self.device = torch.device(device)
        self.selected_target = torch.full((num_envs,), NO_TARGET,
                                          dtype=torch.long, device=self.device)
        self.initial_box_z = torch.zeros(num_envs, device=self.device)
        self.grasp_hold_time_s = torch.zeros_like(self.initial_box_z)
        self.stability_armed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.relative_pose = RelativePoseStabilityTracker(num_envs, self.device)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.selected_target[ids] = NO_TARGET
        self.initial_box_z[ids] = 0.0
        self.grasp_hold_time_s[ids] = 0.0
        self.stability_armed[ids] = False
        self.relative_pose.reset(env_ids)

    def update(
        self, *, perception: PerceptionFrame, robot: DeployableRobotState,
        placement: DeployablePlacementEstimateState,
        placement_evidence: PosePlacementEvidence,
        control: SkillControlState, dt: float,
    ) -> DeployableTransitionEstimate:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")
        n = len(self.selected_target)
        perception.validate(n)
        robot.validate(n)
        if control.target_box.shape != (n,) or control.target_box.dtype != torch.long:
            raise ValueError("Target box must be torch.long [num_envs].")
        if bool(((control.target_box < NO_TARGET) | (control.target_box >= MAX_BOXES)).any()):
            raise ValueError("Target box must be -1 or a valid logical box index.")
        if placement.active.shape != (n, MAX_BOXES) \
                or placement.placed.shape != (n, MAX_BOXES) \
                or placement.hold_time_s.shape != (n, MAX_BOXES):
            raise ValueError("Placement estimate must be [num_envs, 12].")
        if placement.active.dtype != torch.bool or placement.placed.dtype != torch.bool \
                or not torch.equal(placement.active, perception.boxes.active):
            raise ValueError("Placement active/placed masks must match perception.")
        for name in ("bottom_height_m", "footprint_inside", "no_overlap"):
            if getattr(placement_evidence, name).shape != (n, MAX_BOXES):
                raise ValueError(f"Belt {name} evidence must be [num_envs, 12].")
        devices = {self.device, perception.boxes.active.device, robot.joint_pos.device,
                   placement.active.device, placement_evidence.bottom_height_m.device,
                   control.target_box.device}
        if len(devices) != 1:
            raise ValueError("Transition inputs and tracker must share one device.")

        target = control.target_box
        valid = (target >= 0) & (target < MAX_BOXES)
        ids = target.clamp(0, MAX_BOXES - 1)
        rows = torch.arange(n, device=self.device)
        active_target = valid & perception.boxes.active[rows, ids]
        changed = target != self.selected_target
        if bool(changed.any()):
            self.relative_pose.reset(changed)
            self.grasp_hold_time_s[changed] = 0.0
            self.stability_armed[changed] = False
            self.initial_box_z[changed] = perception.boxes.pose_world[rows[changed], ids[changed], 2]
            self.selected_target[changed] = target[changed]

        box_pose = perception.boxes.pose_world[rows, ids]
        hand_to_box = relative_pose(box_pose[:, None], robot.tcp_pose_world)
        observed = active_target & (perception.boxes.pose_confidence[rows, ids] >= 0.5)
        closed = robot.gripper_position >= self.config.min_gripper_closed_fraction
        lifted = observed & (
            box_pose[:, 2] - self.initial_box_z >= self.config.proof_lift_m)
        # Arm the reference only after the selected box has completed its
        # proof lift.  Once armed, keep checking hand-box drift while lowering
        # toward the conveyor; its world height is intentionally below the
        # rack and must not disable stability tracking.
        self.stability_armed |= observed & closed.all(-1) & lifted
        track_stability = observed & closed.all(-1) & self.stability_armed
        stable = self.relative_pose.update(
            hand_to_box, closed & track_stability[:, None])
        grasp_condition = observed & closed.all(-1) & stable.all(-1) & lifted
        self.grasp_hold_time_s = torch.where(
            grasp_condition, self.grasp_hold_time_s + dt,
            torch.zeros_like(self.grasp_hold_time_s))
        grasp_complete = grasp_condition & (
            self.grasp_hold_time_s >= self.config.grasp_hold_seconds)

        height = placement_evidence.bottom_height_m[rows, ids]
        pre_place_height = observed & torch.isfinite(height) \
            & (height >= self.config.min_pre_place_height_m) \
            & (height <= self.config.max_pre_place_height_m)
        box_up = quat_apply(
            box_pose[:, 3:],
            box_pose.new_tensor([0.0, 0.0, 1.0]).expand(n, -1),
        )
        box_tilt = torch.acos(box_up[:, 2].clamp(-1.0, 1.0))
        box_tilt_ok = observed & torch.isfinite(box_tilt) \
            & (box_tilt <= self.config.max_box_tilt_rad + 1e-6)
        # Relative-pose stability proves the initial grasp only.  Carry
        # completion uses measured closure plus perceived box/belt geometry;
        # normal in-hand settling must not block the transition forever.
        carry_complete = observed & closed.all(-1) & box_tilt_ok \
            & placement_evidence.footprint_inside[rows, ids] \
            & placement_evidence.no_overlap[rows, ids] & pre_place_height
        place_complete = observed & placement.placed[rows, ids]
        evidence = DeployableTransitionEvidence(
            grasp_complete=grasp_complete,
            carry_complete=carry_complete,
            place_complete=place_complete,
        )
        progress = torch.stack((
            (self.grasp_hold_time_s / self.config.grasp_hold_seconds).clamp(0, 1),
            carry_complete.float(),
            (placement.hold_time_s[rows, ids] / self.config.place_hold_seconds).clamp(0, 1),
        ), dim=-1) * active_target[:, None]
        return DeployableTransitionEstimate(
            evidence=evidence, progress=progress,
            grippers_closed=closed, hand_box_pose_stable=stable,
            proof_lift=lifted, grasp_hold_time_s=self.grasp_hold_time_s.clone(),
            pre_place_height=pre_place_height, box_tilt_ok=box_tilt_ok,
        )
