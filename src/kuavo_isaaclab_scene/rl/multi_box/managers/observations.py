"""State-only actor/critic input, identical ordering across strategies."""
import torch
from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg as Term
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_mul, quat_conjugate
from ...mdp.geometry import unrotate
from .._legacy_state import state


def objects(env):
    t = state(env)
    q = t.robot.data.root_quat_w[:, None].expand(-1, 4, -1)
    return torch.cat((unrotate(q, t.centers - t.robot.data.root_pos_w[:, None]),
        quat_mul(quat_conjugate(q), t.poses[..., 3:]), unrotate(q, t.velocities[..., :3]),
        unrotate(q, t.velocities[..., 3:]), t.half[None].expand(env.num_envs, -1, -1),
        unrotate(q, t.destinations - t.robot.data.root_pos_w[:, None]),
        unrotate(q, t.centers - env.scene.env_origins[:, None] - t.initial_centers)), -1).flatten(1)


def task_features(env):
    t = state(env)
    context = t.command if t.spec.strategy == "staged" else torch.zeros(env.num_envs, 8, device=env.device)
    return torch.cat((context, t.hand_box_grasp.flatten(1).float(), t.distances.flatten(1),
        t.alignment.flatten(1), t.forces.flatten(1).clamp(0, 50) / 50,
        t.paid.float(), t.complete.float(), t.free_slots.flatten(1).float(),
        t.placement_time.clamp_max(t.spec.placement_hold),
        t.elapsed[:, None] / t.spec.episode_seconds, t.ready[:, None].float(),
        t.skill_time[:, None].clamp_max(t.spec.skill_hold)), -1)


def grasp_memory(env):
    t = state(env)
    values = []
    for g in t.grasps:
        latch = g.latch
        values.extend((latch.active.float(), latch.missing_s, latch.reference_gap,
                       latch.reference_midpoint.flatten(1)))
    return torch.cat(values, -1)


def integrators(env):
    terms = [env.action_manager.get_term(n) for n in env.action_manager.active_terms]
    return torch.cat([t.processed_actions for t in terms] +
        [t._signed_target for t in terms if hasattr(t, "_signed_target")], -1)


def root(env):
    t = state(env)
    return torch.cat((t.robot.data.root_pos_w - env.scene.env_origins,
                      t.robot.data.root_quat_w, t.robot.data.root_vel_w), -1)


@configclass
class PolicyCfg(ObservationGroupCfg):
    joint_pos = Term(func=mdp.joint_pos_rel)
    joint_vel = Term(func=mdp.joint_vel_rel, scale=.1)
    root = Term(func=root)
    objects = Term(func=objects)
    task = Term(func=task_features)
    grasp_memory = Term(func=grasp_memory)
    targets = Term(func=integrators)
    last_action = Term(func=mdp.last_action)

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False


@configclass
class ObservationsCfg:
    policy: PolicyCfg = PolicyCfg()
