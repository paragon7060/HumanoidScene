"""Per-environment validation of randomized box resets."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..geometry.pose import quat_apply, quat_conjugate, normalize_quaternion
from ..geometry.rack import box_shelf_clearance_m
from .spawn import BOX_TYPE_IDS, physical_asset_names
from ....workcell.rack_box_layout import BOX_DIMENSIONS_M
from ....workcell.workcell_layout import (
    RACK_RAW_BOUNDS_M,
    RACK_SHELF_CENTER_LOCAL_X_RAW,
    RACK_SHELF_WIDTH_RAW,
    scale as workcell_scale,
)


@dataclass(frozen=True)
class ResetSettlingStep:
    ready: torch.Tensor
    invalid: torch.Tensor
    settling: torch.Tensor
    just_ready: torch.Tensor
    in_assigned_region: torch.Tensor
    stable: torch.Tensor
    elapsed_s: torch.Tensor
    ready_steps: torch.Tensor


class IsaacResetSettling:
    """Accept a reset only after its box settles on the assigned rack region."""

    def __init__(self, env):
        self.env = env
        self.device = torch.device(env.device)
        self.names = physical_asset_names()
        n = env.num_envs
        self.ready = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.invalid = torch.zeros_like(self.ready)
        self.elapsed = torch.zeros(n, device=self.device)
        self.stable_time = torch.zeros_like(self.elapsed)
        self.in_assigned_region = torch.zeros_like(self.ready)
        self.footprint_in_region = torch.zeros_like(self.ready)
        self.on_assigned_shelf = torch.zeros_like(self.ready)
        self.stable = torch.zeros_like(self.ready)
        self.just_ready = torch.zeros_like(self.ready)
        self.ready_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.invalid_count = torch.zeros(n, dtype=torch.long, device=self.device)
        self.region_invalid_count = torch.zeros_like(self.invalid_count)
        self.footprint_invalid_count = torch.zeros_like(self.invalid_count)
        self.shelf_invalid_count = torch.zeros_like(self.invalid_count)
        self.timeout_invalid_count = torch.zeros_like(self.invalid_count)
        self.last_invalid_logical_id = torch.full(
            (n,), -1, dtype=torch.long, device=self.device)
        self.last_invalid_type_id = torch.full_like(self.last_invalid_logical_id, -1)
        self.last_invalid_region_id = torch.full_like(self.last_invalid_logical_id, -1)
        self.last_invalid_spawn_depth_m = torch.full(
            (n,), float("nan"), device=self.device)
        self._last_counter = int(env.common_step_counter)

    def reset(self, env_ids) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.ready[ids] = False
        self.invalid[ids] = False
        self.elapsed[ids] = 0.0
        self.stable_time[ids] = 0.0
        self.in_assigned_region[ids] = False
        self.footprint_in_region[ids] = False
        self.on_assigned_shelf[ids] = False
        self.stable[ids] = False
        self.just_ready[ids] = False
        self.ready_steps[ids] = 0
        # A post-reset observation uses the current global counter.  Do not
        # count it as a physics/control step.
        self._last_counter = int(self.env.common_step_counter)

    def _selected(self):
        active = self.env._multi_box_active
        logical = active.to(torch.long).argmax(-1)
        rows = torch.arange(self.env.num_envs, device=self.device)
        pool = self.env._multi_box_pool_ids[rows, logical]
        poses = torch.stack(
            [self.env.scene[name].data.root_pose_w for name in self.names], dim=1)
        velocities = torch.stack(
            [self.env.scene[name].data.root_vel_w for name in self.names], dim=1)
        return (
            logical,
            poses[rows, pool],
            velocities[rows, pool],
            self.env._multi_box_box_type_ids[rows, logical],
            self.env._multi_box_region_ids[rows, logical],
        )

    def _footprint_in_region(self, pose, type_id, region_id):
        rack_pose = self.env.scene["rack"].data.root_pose_w
        rack_scale = workcell_scale("rack")
        corners = torch.zeros(self.env.num_envs, 4, 3, device=self.device)
        type_names = {value: key for key, value in BOX_TYPE_IDS.items()}
        for candidate, name in type_names.items():
            mask = type_id == candidate
            half_x = BOX_DIMENSIONS_M[name][0] / 2.0
            half_y = BOX_DIMENSIONS_M[name][1] / 2.0
            corners[mask, :, :2] = pose.new_tensor((
                (-half_x, -half_y), (-half_x, half_y),
                (half_x, -half_y), (half_x, half_y),
            ))
        world = pose[:, None, :3] + quat_apply(
            normalize_quaternion(pose[:, None, 3:]), corners)
        local = quat_apply(
            quat_conjugate(normalize_quaternion(rack_pose[:, None, 3:])),
            world - rack_pose[:, None, :3],
        )
        x_min = local[..., 0].amin(-1)
        x_max = local[..., 0].amax(-1)
        y_min = local[..., 1].amin(-1)
        y_max = local[..., 1].amax(-1)

        center_x = RACK_SHELF_CENTER_LOCAL_X_RAW * rack_scale[0]
        shelf_min_x = center_x - RACK_SHELF_WIDTH_RAW * rack_scale[0] / 2.0
        shelf_max_x = center_x + RACK_SHELF_WIDTH_RAW * rack_scale[0] / 2.0
        side_is_right = region_id.remainder(2) == 0
        side_ok = torch.where(
            side_is_right,
            (x_min >= shelf_min_x) & (x_max <= center_x),
            (x_min >= center_x) & (x_max <= shelf_max_x),
        )
        raw_min_y, raw_max_y = RACK_RAW_BOUNDS_M[0][1], RACK_RAW_BOUNDS_M[1][1]
        depth_ok = (
            (y_min >= raw_min_y * rack_scale[1])
            & (y_max <= raw_max_y * rack_scale[1])
        )
        return side_ok & depth_ok

    def _on_assigned_shelf(self, pose, type_id, region_id):
        rack_pose = self.env.scene["rack"].data.root_pose_w
        rack_scale = workcell_scale("rack")
        shelves = torch.tensor(
            [region.shelf for region in self.env.cfg.multi_box.rack_regions],
            dtype=torch.long, device=self.device,
        )[region_id]
        clearance = torch.full(
            (self.env.num_envs,), float("nan"), device=self.device)
        type_names = {value: key for key, value in BOX_TYPE_IDS.items()}
        for candidate, name in type_names.items():
            for shelf in (2, 3):
                mask = (type_id == candidate) & (shelves == shelf)
                if bool(mask.any()):
                    clearance[mask] = box_shelf_clearance_m(
                        pose[mask], rack_pose[mask], BOX_DIMENSIONS_M[name],
                        shelf=shelf, rack_scale=rack_scale,
                    )
        low, high = self.env.cfg.multi_box.reset_shelf_clearance_range
        return torch.isfinite(clearance) & (clearance >= low) & (clearance <= high)

    def measure(self, dt: float) -> ResetSettlingStep:
        counter = int(self.env.common_step_counter)
        self.just_ready.zero_()
        if counter != self._last_counter:
            self._last_counter = counter
            _logical, pose, velocity, type_id, region_id = self._selected()
            pending = ~self.ready & ~self.invalid
            finite = torch.isfinite(pose).all(-1) & torch.isfinite(velocity).all(-1)
            self.footprint_in_region = finite & self._footprint_in_region(
                pose, type_id, region_id)
            self.on_assigned_shelf = finite & self._on_assigned_shelf(
                pose, type_id, region_id)
            self.in_assigned_region = (
                self.footprint_in_region & self.on_assigned_shelf)
            self.stable = finite & (
                velocity[:, :3].norm(dim=-1)
                <= self.env.cfg.multi_box.reset_settle_linear_speed
            ) & (
                velocity[:, 3:].norm(dim=-1)
                <= self.env.cfg.multi_box.reset_settle_angular_speed
            )
            self.elapsed[pending] += dt
            eligible = pending & self.in_assigned_region & self.stable & (
                self.elapsed >= self.env.cfg.multi_box.reset_settle_min_seconds)
            self.stable_time[pending] = torch.where(
                eligible[pending], self.stable_time[pending] + dt, 0.0)
            self.just_ready = pending & (
                self.stable_time >= self.env.cfg.multi_box.reset_settle_hold_seconds)
            self.ready |= self.just_ready
            self.ready_steps[self.ready & ~self.just_ready] += 1
            left_region = pending & ~self.in_assigned_region
            timed_out = pending & (
                self.elapsed >= self.env.cfg.multi_box.reset_settle_timeout_seconds
            ) & ~self.just_ready
            # Permit the first control step to resolve initial contact.  From
            # the second step onward, leaving the assigned shelf/region makes
            # this a reset failure and requests a partial respawn.
            left_region &= self.elapsed > dt + 1e-8
            newly_invalid = ~self.invalid & (left_region | timed_out)
            self.invalid |= newly_invalid
            self.invalid_count += newly_invalid.to(torch.long)
            self.region_invalid_count += (newly_invalid & left_region).to(torch.long)
            self.footprint_invalid_count += (
                newly_invalid & ~self.footprint_in_region).to(torch.long)
            self.shelf_invalid_count += (
                newly_invalid & ~self.on_assigned_shelf).to(torch.long)
            self.timeout_invalid_count += (newly_invalid & timed_out).to(torch.long)
            if bool(newly_invalid.any()):
                ids = newly_invalid.nonzero(as_tuple=False).flatten()
                self.last_invalid_logical_id[ids] = logical[ids]
                self.last_invalid_type_id[ids] = type_id[ids]
                self.last_invalid_region_id[ids] = region_id[ids]
                self.last_invalid_spawn_depth_m[ids] = -self.env._multi_box_rack_local_positions[
                    ids, logical[ids], 1]
        return ResetSettlingStep(
            ready=self.ready.clone(), invalid=self.invalid.clone(),
            settling=(~self.ready & ~self.invalid).clone(),
            just_ready=self.just_ready.clone(),
            in_assigned_region=self.in_assigned_region.clone(),
            stable=self.stable.clone(), elapsed_s=self.elapsed.clone(),
            ready_steps=self.ready_steps.clone(),
        )


def reset_settling_step(env) -> ResetSettlingStep:
    tracker = getattr(env, "_multi_box_reset_settling", None)
    if tracker is None:
        tracker = IsaacResetSettling(env)
        env._multi_box_reset_settling = tracker
    return tracker.measure(env.step_dt)
