"""Physical joint limits for an arms-only experiment; no per-step pose teleporting.

Torch-only helper kept separate from ActionTerm so reset/index behavior can be
checked without launching Isaac Sim. The articulation root must also be fixed.
"""

import math
import re
import torch


BODY_JOINT_PATTERNS = (
    r"wheel_.*_joint", r"knee_joint", r"leg_joint", r"leg_[lr][0-9]+_joint",
    r"waist_.*_joint", r"zhead_.*_joint",
)
ARM_JOINT_NAMES = tuple(f"zarm_{side}{i}_joint" for side in "lr" for i in range(1, 8))


class FixedBody:
    """Latch measured post-reset joints and constrain only the selected environments."""

    def __init__(self, asset, *, tolerance=1e-4):
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Body lock tolerance must be finite and positive (rad).")
        if not asset.is_fixed_base:
            raise ValueError("arms-only requires a fixed articulation root; set fix_root_link=True.")
        self.asset = asset
        self.tolerance = tolerance
        self.joint_ids = [i for i, name in enumerate(asset.joint_names)
                          if any(re.fullmatch(pattern, name) for pattern in BODY_JOINT_PATTERNS)]
        self.joint_names = [asset.joint_names[i] for i in self.joint_ids]
        required = {"waist_yaw_joint", "zhead_1_joint", "zhead_2_joint"}
        if not required.issubset(self.joint_names):
            raise ValueError(f"arms-only is missing fixed body joints: {sorted(required - set(self.joint_names))}")
        self.original_limits = asset.data.joint_pos_limits[:, self.joint_ids].clone()
        self.targets = asset.data.joint_pos[:, self.joint_ids].clone()
        self.ready = torch.zeros(self.targets.shape[0], dtype=torch.bool, device=self.targets.device)

    def reset(self, env_ids=None):
        ids = (torch.arange(self.targets.shape[0], device=self.targets.device) if env_ids is None
               else torch.as_tensor(env_ids, dtype=torch.long, device=self.targets.device))
        if not len(ids):
            return
        values = self.asset.data.joint_pos[ids][:, self.joint_ids].clone()
        bounds = self.original_limits[ids]
        if not torch.isfinite(values).all() or ((values < bounds[..., 0] - 1e-4)
                                               | (values > bounds[..., 1] + 1e-4)).any():
            raise ValueError("Fixed body initial position is invalid or outside physical joint limits.")
        values = values.clamp(bounds[..., 0], bounds[..., 1])
        self.targets[ids] = values
        limits = torch.stack(((values - self.tolerance).clamp_min(bounds[..., 0]),
                              (values + self.tolerance).clamp_max(bounds[..., 1])), dim=-1)
        # Isaac Lab's limit setter clamps default_joint_pos as a side effect.
        # Keep the policy's observation/default-offset reference unchanged.
        defaults = self.asset.data.default_joint_pos.clone()
        try:
            self.asset.write_joint_position_limit_to_sim(
                limits, joint_ids=self.joint_ids, env_ids=ids, warn_limit_violation=False)
        finally:
            self.asset.data.default_joint_pos[:] = defaults
        self.asset.write_joint_state_to_sim(values, torch.zeros_like(values),
                                           joint_ids=self.joint_ids, env_ids=ids)
        self.ready[ids] = True
        self._apply(ids)

    def _apply(self, ids):
        self.asset.set_joint_position_target(self.targets[ids], joint_ids=self.joint_ids, env_ids=ids)
        self.asset.set_joint_velocity_target(torch.zeros_like(self.targets[ids]),
                                             joint_ids=self.joint_ids, env_ids=ids)

    def apply(self):
        ids = self.ready.nonzero().flatten()
        if len(ids):
            self._apply(ids)
