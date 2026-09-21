"""Privileged whole-robot self-collision measurement for vectorized training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ....robots.gripper_config import resolve_gripper_settings
from ....robots.robot_model import resolve_robot_model
from ....robots.self_collision import RobotCollisionModel, resolve_self_collision_policy


@dataclass(frozen=True)
class IsaacSelfCollisionStep:
    collision: torch.Tensor
    minimum_distance_m: torch.Tensor


class IsaacSelfCollisionAdapter:
    """Apply the collector's reviewed URDF/FCL policy to every RL environment.

    This is a privileged state predicate.  It does not alter policy actions or
    enable PhysX articulation self-contact, whose mechanical-interface pairs
    do not match the reviewed collector exclusions.
    """

    def __init__(self, env):
        self.robot = env.scene["robot"]
        robot = resolve_robot_model()
        gripper = resolve_gripper_settings()
        policy = resolve_self_collision_policy(
            robot.name, gripper.name, gripper.integrated)
        self.model = RobotCollisionModel(robot.urdf_path, policy)
        self.clearance = float(env.cfg.multi_box.self_collision_clearance)

        self.joint_ids = []
        for name in self.model.names:
            ids, _ = self.robot.find_joints(name)
            if len(ids) != 1:
                raise ValueError(
                    f"Self-collision model joint missing/ambiguous in RL USD: {name}")
            self.joint_ids.append(ids[0])
        if set(self.robot.joint_names) != set(self.model.names):
            raise ValueError(
                "RL USD joints do not match the reviewed whole-robot self-collision URDF")

    def measure(self) -> IsaacSelfCollisionStep:
        joint_pos = self.robot.data.joint_pos[:, self.joint_ids]
        finite = torch.isfinite(joint_pos).all(-1)
        positions = joint_pos.detach().to("cpu", dtype=torch.float64).numpy()
        minimum = np.full(len(positions), -np.inf, dtype=np.float32)
        for env_id in np.flatnonzero(finite.detach().cpu().numpy()):
            distances, _ = self.model.distances(
                positions[env_id], threshold=self.clearance)
            if distances.size:
                minimum[env_id] = float(distances.min())
        minimum_distance = torch.as_tensor(
            minimum, dtype=torch.float32, device=joint_pos.device)
        collision = (~finite) | (minimum_distance < self.clearance)
        return IsaacSelfCollisionStep(collision, minimum_distance)
