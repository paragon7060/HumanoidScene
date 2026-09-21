"""Track hand-to-box pose drift while a physical pinch is maintained."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class RelativePoseStabilityConfig:
    max_translation_m: float = 0.010
    max_rotation_rad: float = math.radians(10.0)

    def validate(self) -> None:
        if not math.isfinite(self.max_translation_m) or self.max_translation_m <= 0:
            raise ValueError("Maximum hand-to-box translation drift must be positive.")
        if not math.isfinite(self.max_rotation_rad) or not 0 < self.max_rotation_rad < math.pi:
            raise ValueError("Maximum hand-to-box rotation drift must be in (0, pi).")


class RelativePoseStabilityTracker:
    """Use the pose at first valid pinch as the reference for each hand."""

    def __init__(self, num_envs: int, device: str | torch.device, config=None):
        if num_envs < 1:
            raise ValueError("num_envs must be positive.")
        self.config = config or RelativePoseStabilityConfig()
        self.config.validate()
        self.reference = torch.zeros(num_envs, 2, 7, device=device)
        self.latched = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.latched.zero_()
        else:
            self.latched[env_ids] = False

    def update(self, hand_to_box_pose: torch.Tensor, pinching: torch.Tensor) -> torch.Tensor:
        if hand_to_box_pose.shape != self.reference.shape or not hand_to_box_pose.is_floating_point():
            raise ValueError("hand_to_box_pose must be floating point [num_envs, 2, 7].")
        if pinching.shape != self.latched.shape or pinching.dtype != torch.bool:
            raise ValueError("pinching must be boolean [num_envs, 2].")
        if hand_to_box_pose.device != self.reference.device or pinching.device != self.reference.device:
            raise ValueError("Pose stability measurements must share one device.")
        finite = torch.isfinite(hand_to_box_pose).all(dim=-1)
        valid_quaternion = torch.linalg.vector_norm(hand_to_box_pose[..., 3:], dim=-1) > 1e-8
        active = pinching & finite & valid_quaternion
        newly_pinching = active & ~self.latched
        self.reference[newly_pinching] = hand_to_box_pose[newly_pinching]
        self.latched = active.clone()
        translation = torch.linalg.vector_norm(
            hand_to_box_pose[..., :3] - self.reference[..., :3], dim=-1)
        safe_pose = torch.where(active[..., None], hand_to_box_pose, self.reference)
        current_q = torch.nn.functional.normalize(safe_pose[..., 3:], dim=-1)
        reference_q = torch.nn.functional.normalize(self.reference[..., 3:], dim=-1)
        rotation = 2.0 * torch.acos((current_q * reference_q).sum(-1).abs().clamp(0, 1))
        return active & (translation <= self.config.max_translation_m) \
            & (rotation <= self.config.max_rotation_rad)
