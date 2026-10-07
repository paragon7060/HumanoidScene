"""Manager observation term backed by the deployable multi-box v2 runtime."""

from __future__ import annotations

from isaaclab.managers import ManagerTermBase, ObservationGroupCfg
from isaaclab.managers import ObservationTermCfg as Term
from isaaclab.utils import configclass

import torch

from ..observations import flat_actor_observation_dim, flatten_actor_observation
from ..geometry import pose_to_position_rotation_6d, relative_pose
from ..geometry.pose import quat_apply, quat_conjugate
from ..state.isaac_deployable import IsaacDeployableStateAdapter


GRASP_PRIVILEGED_DIM = 66


class DeployableActorObservation(ManagerTermBase):
    """Expose the real-obtainable actor contract as one flat policy tensor."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Observation terms are constructed while Isaac is still assembling
        # the other managers.  The robot source reads action terms, so defer
        # its construction until the first observation after manager setup.
        self.adapter = None

    def reset(self, env_ids=None) -> None:
        if self.adapter is not None:
            self.adapter.reset(env_ids)

    def __call__(self, env):
        # ObservationManager evaluates every term once during construction to
        # discover its shape.  Scene reset has not produced the logical-box
        # buffers yet, so do not touch live perception during that sizing call.
        if not hasattr(env, "_multi_box_active"):
            return torch.zeros(
                env.num_envs,
                flat_actor_observation_dim(env.action_manager.total_action_dim),
                device=env.device,
            )
        if self.adapter is None:
            self.adapter = IsaacDeployableStateAdapter(env)
        step = self.adapter.step(
            previous_action=env.action_manager.action,
            dt=env.step_dt,
        )
        # Preserve the structured result for later privileged/reward manager
        # terms without inserting it into the actor observation.
        env._multi_box_deployable_step = step
        return flatten_actor_observation(step.actor_observation)


class PrivilegedGraspObservation(ManagerTermBase):
    """Exact simulator-only grasp and safety features for the critic."""

    def __call__(self, env):
        if not hasattr(env, "_multi_box_active"):
            return torch.zeros(
                env.num_envs, GRASP_PRIVILEGED_DIM, device=env.device)
        # Import lazily so observation configuration remains importable before
        # Isaac has created contact sensors and manager state.
        from .v2_grasp import grasp_safety_step, privileged_grasp_step

        grasp = privileged_grasp_step(env)
        safety = grasp_safety_step(env)
        base_pose = env.scene["robot"].data.root_pose_w
        exact_box_pose = pose_to_position_rotation_6d(
            relative_pose(base_pose, grasp.box_pose_world))
        inverse_base = quat_conjugate(base_pose[:, 3:])
        linear_velocity = quat_apply(inverse_base, grasp.box_velocity_world[:, :3])
        angular_velocity = quat_apply(inverse_base, grasp.box_velocity_world[:, 3:])
        flap_valid = (grasp.pinch.hand_flap_index >= 0) & (
            grasp.pinch.hand_flap_index < 2)
        flap_one_hot = torch.nn.functional.one_hot(
            grasp.pinch.hand_flap_index.clamp(0, 1), 2).to(torch.float32)
        flap_one_hot *= flap_valid[..., None]
        success = grasp.success
        values = (
            exact_box_pose,
            linear_velocity,
            angular_velocity,
            (grasp.contacts.force_n / 50.0).clamp(0, 4).flatten(1),
            grasp.contacts.in_region.to(torch.float32).flatten(1),
            grasp.contacts.opposed.to(torch.float32).flatten(1),
            grasp.pinch.hand_pinching.to(torch.float32),
            flap_one_hot.flatten(1),
            grasp.stable_hands.to(torch.float32),
            (grasp.rack_clearance_m / 0.008).clamp(-4, 4)[:, None],
            (success.hold_time_s / 0.25).clamp(0, 1)[:, None],
            # Keep the critic observation contract stable when reward-only
            # shaping diagnostics gain additional keys.
            torch.stack(tuple(grasp.potentials[name] for name in (
                "approach", "alignment", "capture", "proof_lift")), dim=-1),
            torch.stack((
                success.bilateral_pinch,
                success.opposing_flaps,
                success.stable,
                success.proof_lift,
                success.instantaneous,
                success.success,
            ), dim=-1).to(torch.float32),
            torch.stack((
                (safety.rack_force_n / float(env.cfg.task.obstacle_contact_force)).clamp(0, 4),
                (safety.obstacle_force_n / float(env.cfg.task.obstacle_contact_force)).clamp(0, 4),
                (safety.base_distance_m / float(env.cfg.multi_box.workspace_radius)).clamp(0, 2),
                (safety.self_collision_distance_m
                 / float(env.cfg.multi_box.self_collision_clearance)).clamp(-1, 4),
            ), dim=-1),
            torch.stack((
                safety.robot_rack_collision,
                safety.self_collision,
                safety.obstacle_collision,
                safety.workspace_limit,
                safety.box_drop,
                safety.box_lift_limit,
                safety.box_speed_limit,
            ), dim=-1).to(torch.float32),
        )
        result = torch.cat(values, dim=-1)
        if result.shape != (env.num_envs, GRASP_PRIVILEGED_DIM):
            raise RuntimeError(
                f"V2 grasp critic contract changed: expected {GRASP_PRIVILEGED_DIM}, "
                f"got {result.shape[-1]} features")
        return result


class ActualFlapRelationsObservation(ManagerTermBase):
    """Optional perception only; do not step the nominal target-selection runtime."""
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.perception=self.robot=None

    def __call__(self,env):
        from ..observations.flap_supplement import actual_flap_relations,SUPPLEMENTAL_DIM
        if not hasattr(env,'_multi_box_active') or not hasattr(env,'_multi_box_deployable_step'):
            return torch.zeros(env.num_envs,SUPPLEMENTAL_DIM,device=env.device)
        if self.perception is None:
            from ..state.isaac_perception import IsaacScenePerceptionAdapter
            from ..state.isaac_robot_proprio import IsaacRobotProprioAdapter
            self.perception=IsaacScenePerceptionAdapter(env,flap_pose_source='articulated')
            self.robot=IsaacRobotProprioAdapter(env)
        boxes=self.perception.read().boxes
        # This is the nominal runtime's locked logical target, not a second
        # selector or the privileged reward assignment.
        target=env._multi_box_deployable_step.actor_observation.target_box
        rows=torch.arange(env.num_envs,device=target.device);ids=target.clamp(0,11)
        valid=(target>=0)&(target<12)&boxes.active[rows,ids] \
            &(boxes.pose_confidence[rows,ids]>0)&(boxes.flap_pose_confidence[rows,ids]>0).all(-1)
        return actual_flap_relations(boxes.flap_pose_world[rows,ids],boxes.size_m[rows,ids],
            boxes.box_type_id[rows,ids],self.robot.read().tcp_pose_world,valid)


@configclass
class V2PolicyCfg(ObservationGroupCfg):
    actor = Term(func=DeployableActorObservation)

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False


@configclass
class V2CriticCfg(ObservationGroupCfg):
    privileged = Term(func=PrivilegedGraspObservation)

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False


@configclass
class ActualFlapRelationsCfg(ObservationGroupCfg):
    relations=Term(func=ActualFlapRelationsObservation)

    def __post_init__(self):
        self.concatenate_terms=True
        self.enable_corruption=False


@configclass
class V2ObservationsCfg:
    policy: V2PolicyCfg = V2PolicyCfg()
    critic: V2CriticCfg = V2CriticCfg()


def task_time_remaining(env):
    from ..scene.reset_settling import reset_settling_step
    from ..observations.task_timing import task_remaining_fraction
    return task_remaining_fraction(reset_settling_step(env).ready_steps, env.max_episode_length)


@configclass
class TaskTimeRemainingCfg(ObservationGroupCfg):
    remaining = Term(func=task_time_remaining)

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False
