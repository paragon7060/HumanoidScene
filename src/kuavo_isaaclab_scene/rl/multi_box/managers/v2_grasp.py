"""Manager terms for the staged multi-box v2 grasp skill."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewardTerm
from isaaclab.managers import TerminationTermCfg as Done
from isaaclab.utils import configclass

from ..debug.contact_sensors import (
    V2_OBSTACLE_SENSOR_NAME,
    V2_RACK_SENSOR_NAMES,
)
from ..debug.contact_force import maximum_filtered_force
from ..rewards import CommonRewardInput, GraspRewardInput, MultiBoxRewardModel
from ..success import SkillTerminationInput, low_level_termination
from ..state.isaac_privileged_grasp import (
    IsaacPrivilegedGraspAdapter,
    IsaacPrivilegedGraspStep,
)


@dataclass(frozen=True)
class V2GraspSafetyStep:
    robot_rack_collision: torch.Tensor
    self_collision: torch.Tensor
    obstacle_collision: torch.Tensor
    workspace_limit: torch.Tensor
    box_drop: torch.Tensor
    box_lift_limit: torch.Tensor
    box_speed_limit: torch.Tensor
    base_distance_m: torch.Tensor
    rack_force_n: torch.Tensor
    obstacle_force_n: torch.Tensor
    self_collision_distance_m: torch.Tensor

    @property
    def unsafe(self) -> torch.Tensor:
        return (
            self.robot_rack_collision
            | self.self_collision
            | self.obstacle_collision
            | self.workspace_limit
            | self.box_drop
            | self.box_lift_limit
            | self.box_speed_limit
        )


def privileged_grasp_step(env) -> IsaacPrivilegedGraspStep:
    """Compute exact grasp truth once per control step for all managers."""
    counter = int(env.common_step_counter)
    if getattr(env, "_multi_box_privileged_grasp", None) is None:
        env._multi_box_privileged_grasp = IsaacPrivilegedGraspAdapter(env)
    if getattr(env, "_multi_box_privileged_grasp_counter", -1) != counter:
        env._multi_box_privileged_grasp_step = env._multi_box_privileged_grasp.measure(
            env.step_dt)
        env._multi_box_privileged_grasp_counter = counter
    return env._multi_box_privileged_grasp_step


def grasp_safety_step(env) -> V2GraspSafetyStep:
    counter = int(env.common_step_counter)
    if getattr(env, "_multi_box_grasp_safety_counter", -1) == counter:
        return env._multi_box_grasp_safety_step

    grasp = privileged_grasp_step(env)
    force = env.scene[V2_OBSTACLE_SENSOR_NAME].data.net_forces_w
    if force is None or force.shape[0] != env.num_envs:
        raise RuntimeError("V2 obstacle contact forces are unavailable.")
    obstacle_force = force.norm(dim=-1).amax(dim=-1)
    rack_force = maximum_filtered_force(env, V2_RACK_SENSOR_NAMES)
    # Ignore import/reset snap impulses for three control steps. Subsequent
    # rack, conveyor, box, or floor contact by arm/torso links is terminal.
    grace_over = env.episode_length_buf > 3
    threshold = float(env.cfg.task.obstacle_contact_force)
    robot_rack_collision = (rack_force > threshold) & grace_over

    if getattr(env, "_multi_box_self_collision", None) is None:
        from ..state.isaac_self_collision import IsaacSelfCollisionAdapter
        env._multi_box_self_collision = IsaacSelfCollisionAdapter(env)
    self_collision_step = env._multi_box_self_collision.measure()
    self_collision = self_collision_step.collision & grace_over
    # The aggregate sensor also contains rack contacts. Attribute a step to
    # the more specific rack event first so one physical collision does not
    # receive both common penalties.
    obstacle_collision = (
        (obstacle_force > threshold) & grace_over & ~robot_rack_collision)

    base_offset = env.scene["robot"].data.root_pos_w - env.scene.env_origins
    base_distance = base_offset[:, :2].norm(dim=-1)
    workspace_limit = base_distance > float(env.cfg.multi_box.workspace_radius)

    origin_z = env.scene.env_origins[:, 2]
    finite = torch.isfinite(grasp.box_pose_world).all(-1) \
        & torch.isfinite(grasp.box_velocity_world).all(-1)
    box_drop = (grasp.box_pose_world[:, 2] - origin_z < 0.12) | ~finite
    box_lift_limit = grasp.lift_from_reset_m > float(env.cfg.multi_box.max_box_lift_height)
    box_speed_limit = (
        grasp.box_velocity_world[:, :3].norm(dim=-1)
        > float(env.cfg.multi_box.max_box_linear_speed)
    ) | (
        grasp.box_velocity_world[:, 3:].norm(dim=-1)
        > float(env.cfg.multi_box.max_box_angular_speed)
    )
    result = V2GraspSafetyStep(
        robot_rack_collision=robot_rack_collision,
        self_collision=self_collision,
        obstacle_collision=obstacle_collision,
        workspace_limit=workspace_limit,
        box_drop=box_drop,
        box_lift_limit=box_lift_limit,
        box_speed_limit=box_speed_limit,
        base_distance_m=base_distance,
        rack_force_n=rack_force,
        obstacle_force_n=obstacle_force,
        self_collision_distance_m=self_collision_step.minimum_distance_m,
    )
    env._multi_box_grasp_safety_step = result
    env._multi_box_grasp_safety_counter = counter
    return result


def grasp_terminal_step(env):
    grasp = privileged_grasp_step(env)
    safety = grasp_safety_step(env)
    false = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    true = torch.ones_like(false)
    return low_level_termination("grasp", SkillTerminationInput(
        success=grasp.success.success,
        unsafe=safety.unsafe,
        phase_armed=true,
        grasp_maintained=true,
        box_tilt_ok=true,
        released=false,
        placement_region=false,
    ))


def grasp_success(env) -> torch.Tensor:
    return grasp_terminal_step(env).success


def grasp_unsafe(env) -> torch.Tensor:
    return grasp_terminal_step(env).failure


def _base_motion(env) -> torch.Tensor:
    base = env.action_manager.get_term("base")
    velocity = base.processed_actions
    limits = base._scale.abs().clamp_min(1e-6)
    translation = (velocity[:, :2] / limits[:2]).square().sum(-1)
    yaw = 0.25 * (velocity[:, 2] / limits[2]).square()
    return (translation + yaw).clamp(0, 1)


class V2GraspReward(ManagerTermBase):
    """Approved potential/event grasp reward with no legacy reward dependency."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.model = MultiBoxRewardModel()
        self.previous = {
            name: torch.zeros(env.num_envs, device=env.device)
            for name in ("approach", "alignment", "capture", "proof_lift")
        }
        self.initialized = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.initialized[ids] = False
        for value in self.previous.values():
            value[ids] = 0.0

    def __call__(self, env) -> torch.Tensor:
        grasp = privileged_grasp_step(env)
        safety = grasp_safety_step(env)
        current = grasp.potentials
        # Potential shaping pays no artificial reset bonus. Choosing gamma*Phi
        # as the first previous value makes the first delta exactly zero.
        previous = {
            name: torch.where(
                self.initialized,
                self.previous[name],
                self.model.weights.discount * current[name],
            )
            for name in self.previous
        }
        action_rate = (
            env.action_manager.action - env.action_manager.prev_action
        ).square().mean(-1).clamp(0, 1)
        joint_limit = mdp.joint_pos_limits(env).clamp(0, 1)
        box_failure = safety.box_drop | safety.box_lift_limit | safety.box_speed_limit
        breakdown = self.model.grasp(GraspRewardInput(
            previous_approach=previous["approach"],
            approach=current["approach"],
            previous_alignment=previous["alignment"],
            alignment=current["alignment"],
            previous_capture=previous["capture"],
            capture=current["capture"],
            previous_proof_lift=previous["proof_lift"],
            proof_lift=current["proof_lift"],
            bilateral_pinch_event=grasp.bilateral_pinch_event,
            # Failure wins if success and a hard safety predicate arrive on
            # the same physics step; do not pay success on a failed terminal.
            success_event=grasp.success_event & ~safety.unsafe,
            common=CommonRewardInput(
                robot_rack_collision_event=safety.robot_rack_collision,
                self_collision_event=safety.self_collision,
                box_drop_event=box_failure,
                obstacle_collision_event=safety.obstacle_collision,
                workspace_limit_event=safety.workspace_limit,
                normalized_base_motion=_base_motion(env),
                normalized_action_rate=action_rate,
                normalized_joint_limit=joint_limit,
            ),
        ))
        for name, value in current.items():
            self.previous[name].copy_(value)
        self.initialized.fill_(True)
        env._multi_box_grasp_reward_breakdown = breakdown
        # RewardManager multiplies every term by dt. The v2 model defines one
        # potential/event reward per control step, so undo that outer scaling.
        return breakdown.total / env.step_dt


@configclass
class V2GraspRewardsCfg:
    grasp = RewardTerm(func=V2GraspReward, weight=1.0)


@configclass
class V2GraspTerminationsCfg:
    success = Done(func=grasp_success)
    unsafe = Done(func=grasp_unsafe)
    time_out = Done(func=mdp.time_out, time_out=True)
