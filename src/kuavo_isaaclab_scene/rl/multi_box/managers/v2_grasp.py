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
    V2_RACK_SENSOR_NAMES,
    V2_OBSTACLE_FILTER_SENSOR_NAMES,
)
from ..debug.contact_force import filtered_force_by_body, filtered_force_by_target
from ..metrics import grasp_gated_lift_inputs
from ..rewards import (
    CommonRewardInput,
    GraspRewardInput,
    MultiBoxRewardModel,
    RewardBreakdown,
)
from ..scene.reset_settling import reset_settling_step
from ..success import SkillTerminationInput, low_level_termination
from ..state.isaac_privileged_grasp import (
    IsaacPrivilegedGraspAdapter,
    IsaacPrivilegedGraspStep,
)


@dataclass(frozen=True)
class V2GraspSafetyStep:
    invalid_box_pose: torch.Tensor
    invalid_flap_pose: torch.Tensor
    robot_rack_collision: torch.Tensor
    self_collision: torch.Tensor
    obstacle_collision: torch.Tensor
    workspace_limit: torch.Tensor
    base_projection_invalid: torch.Tensor
    box_drop: torch.Tensor
    box_lift_limit: torch.Tensor
    box_speed_limit: torch.Tensor
    contact_eligible: torch.Tensor
    base_distance_m: torch.Tensor
    rack_force_n: torch.Tensor
    rack_body_force_n: torch.Tensor
    obstacle_force_n: torch.Tensor
    obstacle_target_force_n: torch.Tensor
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
    obstacle_targets = filtered_force_by_target(env, V2_OBSTACLE_FILTER_SENSOR_NAMES)
    obstacle_force = obstacle_targets.amax(-1)
    rack_body_force = filtered_force_by_body(env, V2_RACK_SENSOR_NAMES)
    rack_force = rack_body_force.amax(-1)
    settling = reset_settling_step(env)
    # Ignore import/reset snap impulses and the first three task control steps.
    # Subsequent contacts are checked against their respective force limits.
    grace_over = settling.ready & (settling.ready_steps > 3)
    obstacle_threshold = float(env.cfg.task.obstacle_contact_force)
    rack_threshold = float(env.cfg.multi_box.rack_contact_force)
    robot_rack_collision = (rack_force > rack_threshold) & grace_over

    if env.cfg.multi_box.self_collision_enabled:
        if getattr(env, "_multi_box_self_collision", None) is None:
            from ..state.isaac_self_collision import IsaacSelfCollisionAdapter
            env._multi_box_self_collision = IsaacSelfCollisionAdapter(env)
        self_collision_step = env._multi_box_self_collision.measure()
        self_collision = self_collision_step.collision & grace_over
        self_collision_distance = self_collision_step.minimum_distance_m
    else:
        self_collision = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device)
        self_collision_distance = torch.full(
            (env.num_envs,),
            4.0 * float(env.cfg.multi_box.self_collision_clearance),
            device=env.device,
        )
    # Independent pair filters keep simultaneous rack and conveyor contacts
    # detectable without treating box handling or floor contact as obstacles.
    obstacle_collision = (obstacle_force > obstacle_threshold) & grace_over

    base_offset = env.scene["robot"].data.root_pos_w - env.scene.env_origins
    base_distance = base_offset[:, :2].norm(dim=-1)
    minimum = getattr(env, '_base_plane_min_abs_determinant', None)
    if minimum is None:
        base_projection_invalid = torch.zeros_like(settling.ready)
    else:
        from ..geometry.projected_base import planar_projection_determinant, outside_projected_base_domain
        determinant = planar_projection_determinant(
            env.scene['robot'].data.root_quat_w, env.scene['rack'].data.root_quat_w)
        base_projection_invalid = outside_projected_base_domain(determinant,minimum)
    workspace_limit = settling.ready & (
        (base_distance > float(env.cfg.multi_box.workspace_radius)) | base_projection_invalid)

    origin_z = env.scene.env_origins[:, 2]
    finite = torch.isfinite(grasp.box_pose_world).all(-1) \
        & torch.isfinite(grasp.box_velocity_world).all(-1)
    box_drop = settling.ready & (
        (grasp.box_pose_world[:, 2] - origin_z < 0.12) | ~finite)
    box_lift_limit = settling.ready & (
        grasp.lift_from_reset_m > float(env.cfg.multi_box.max_box_lift_height))
    box_speed_limit = settling.ready & ((
        grasp.box_velocity_world[:, :3].norm(dim=-1)
        > float(env.cfg.multi_box.max_box_linear_speed)
    ) | (
        grasp.box_velocity_world[:, 3:].norm(dim=-1)
        > float(env.cfg.multi_box.max_box_angular_speed)
    ))
    result = V2GraspSafetyStep(
        invalid_box_pose=grasp.invalid_box_pose,
        invalid_flap_pose=grasp.invalid_flap_pose,
        robot_rack_collision=robot_rack_collision,
        self_collision=self_collision,
        obstacle_collision=obstacle_collision,
        workspace_limit=workspace_limit,
        base_projection_invalid=base_projection_invalid,
        box_drop=box_drop,
        box_lift_limit=box_lift_limit,
        box_speed_limit=box_speed_limit,
        contact_eligible=grace_over,
        base_distance_m=base_distance,
        rack_force_n=rack_force,
        rack_body_force_n=rack_body_force,
        obstacle_force_n=obstacle_force,
        obstacle_target_force_n=obstacle_targets,
        self_collision_distance_m=self_collision_distance,
    )
    env._multi_box_grasp_safety_step = result
    env._multi_box_grasp_safety_counter = counter
    return result


def grasp_terminal_step(env):
    grasp = privileged_grasp_step(env)
    safety = grasp_safety_step(env)
    settling = reset_settling_step(env)
    false = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    true = torch.ones_like(false)
    return low_level_termination("grasp", SkillTerminationInput(
        success=grasp.success.success & settling.ready,
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


def invalid_reset(env) -> torch.Tensor:
    """Request a partial respawn without labeling reset physics as task failure."""
    grasp = privileged_grasp_step(env)
    invalid = (reset_settling_step(env).invalid | grasp.invalid_box_pose
               | grasp.invalid_flap_pose)
    numerical = getattr(env, "_numerical_failure", None)
    return invalid if numerical is None else invalid | numerical


def task_time_out(env) -> torch.Tensor:
    """Count the episode horizon only after reset acceptance."""
    return reset_settling_step(env).ready_steps >= env.max_episode_length


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
        from ..rewards.contact_profile import configured_reward_weights,ContactProgressReward
        profile=cfg.params.get('reward_profile')
        from ..rewards.absorbing_geometry import configured_geometry_shaping
        self.model = MultiBoxRewardModel(configured_reward_weights(profile) if profile else None,
            geometry_shaping=configured_geometry_shaping(profile))
        self.contact_progress=(ContactProgressReward(env.num_envs,env.device,
            discount=self.model.weights.discount,config=profile['contact_shaping'])
            if profile and 'contact_shaping' in profile else None)
        self.previous = {
            name: torch.zeros(env.num_envs, device=env.device)
            for name in ("approach", "front_staging", "alignment", "capture", "jaw_gap", "proof_lift")
        }
        self.initialized = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device)
        self.previous_assignment = torch.full(
            (env.num_envs, 2), -1, dtype=torch.long, device=env.device)
        self.lift_armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.previous_gated_alignment = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.initialized[ids] = False
        self.previous_assignment[ids] = -1
        self.lift_armed[ids] = False
        self.previous_gated_alignment[ids] = 0.0
        for value in self.previous.values():
            value[ids] = 0.0
        if self.contact_progress is not None:self.contact_progress.reset(env_ids)

    def __call__(self, env, reward_profile=None) -> torch.Tensor:
        grasp = privileged_grasp_step(env)
        safety = grasp_safety_step(env)
        settling = reset_settling_step(env)
        current = dict(grasp.potentials)
        bilateral_eligible = grasp.success.bilateral_pinch & grasp.success.opposing_flaps
        lift_previous, current["proof_lift"] = grasp_gated_lift_inputs(
            current["proof_lift"], self.previous["proof_lift"],
            bilateral_eligible, self.lift_armed, self.model.weights.discount)
        assignment_changed = (
            grasp.assigned_flap_index != self.previous_assignment).any(-1)
        # Rebase after reset or a flap-assignment switch so geometry changes
        # cannot create artificial progress on the boundary step.
        previous = {
            name: torch.where(
                self.initialized
                & (~assignment_changed if name in (
                    "front_staging", "alignment", "capture", "jaw_gap") else True),
                self.previous[name],
                (current[name] if name == "alignment"
                 else self.model.weights.discount * current[name]),
            )
            for name in self.previous
        }
        previous["proof_lift"] = lift_previous
        geometry_terminal = (grasp.success.success | safety.unsafe | task_time_out(env)
            if self.model.geometry_shaping is not None or self.contact_progress is not None
            else torch.zeros_like(safety.unsafe))
        previous_gated_alignment = None
        if self.model.geometry_shaping is not None:
            # The absorbing endpoint uses the last stored potentials, even
            # when the final contact or flap assignment changes eligibility.
            # Rebasing to the terminal pose would leave an endpoint bias.
            previous = {
                name: torch.where(self.initialized & geometry_terminal, self.previous[name], value)
                for name, value in previous.items()
            }
            gated_alignment = current['alignment'] * current['alignment_proximity']
            previous_gated_alignment = torch.where(
                self.initialized & (~assignment_changed | geometry_terminal),
                self.previous_gated_alignment, self.model.weights.discount * gated_alignment)
        action_rate = (
            env.action_manager.action - env.action_manager.prev_action
        ).square().mean(-1).clamp(0, 1)
        joint_limit = mdp.joint_pos_limits(env).clamp(0, 1)
        box_failure = safety.box_drop | safety.box_lift_limit | safety.box_speed_limit
        breakdown = self.model.grasp(GraspRewardInput(
            previous_approach=previous["approach"],
            approach=current["approach"],
            previous_front_staging=previous["front_staging"],
            front_staging=current["front_staging"],
            previous_alignment=previous["alignment"],
            alignment=current["alignment"],
            alignment_proximity=current["alignment_proximity"],
            previous_capture=previous["capture"],
            capture=current["capture"],
            previous_jaw_gap=previous["jaw_gap"],
            jaw_gap=current["jaw_gap"],
            premature_close=current["premature_close"],
            previous_proof_lift=previous["proof_lift"],
            proof_lift=current["proof_lift"],
            one_hand_pinch_event=grasp.one_hand_pinch_event,
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
            previous_gated_alignment=previous_gated_alignment,
            geometry_terminated=geometry_terminal,
        ))
        for name, value in current.items():
            if name in self.previous:
                self.previous[name].copy_(value)
        self.previous_assignment.copy_(grasp.assigned_flap_index)
        self.lift_armed.copy_(bilateral_eligible)
        if self.model.geometry_shaping is not None:
            self.previous_gated_alignment.copy_(gated_alignment)
        self.initialized |= settling.ready
        trainable = (settling.ready & ~settling.just_ready
                     & ~grasp.invalid_box_pose & ~grasp.invalid_flap_pose)
        if self.contact_progress is not None:
            contact_reward=self.contact_progress.step(grasp.contacts,trainable=trainable,terminated=geometry_terminal)
            terms=dict(breakdown.terms,contact_quality_progress=contact_reward)
            breakdown=RewardBreakdown(terms,torch.stack(tuple(terms.values())).sum(0))
            env._multi_box_contact_quality=self.contact_progress.last_quality
        if not bool(trainable.all()):
            terms = {
                name: torch.where(trainable, value, torch.zeros_like(value))
                for name, value in breakdown.terms.items()
            }
            breakdown = RewardBreakdown(
                terms=terms, total=torch.stack(tuple(terms.values())).sum(0))
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
    invalid_reset = Done(func=invalid_reset)
    time_out = Done(func=task_time_out, time_out=True)
