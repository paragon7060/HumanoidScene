"""Dense shaping is phase-gated; success still requires the physical predicates."""

import math
import torch
from .commands import task
from .settling import ready
from .grasp_stability import grasp_stability_scores


def navigation(env):
    t = task(env)
    return torch.exp(-4 * t.nav_distance) * torch.exp(-t.heading_error) * ((t.reward_phase == 0) | (t.reward_phase == 2))


def reaching(env):
    t = task(env)
    return torch.exp(-6 * t.reach_distance) * (t.reward_phase == 1)


def approach_reaching(env):
    """TCP proximity supplements base navigation; does not replace safe base/yaw goals."""
    t = task(env)
    distance = (t.tools[:, t.spec.grasp_hand_indices] - t.grips[:, t.spec.grasp_hand_indices]).norm(dim=-1).mean(-1)
    return torch.exp(-6 * distance) * (t.reward_phase == 0) * ready(t)


def lift(env):
    t = task(env)
    height = t.centers[t.ids, t.reward_box, 2] - env.scene.env_origins[:, 2] - t.initial_z[t.ids, t.reward_box]
    return (height / t.spec.lift_height).clamp(0, 1) * t.grasped * (t.reward_phase == 1) * ready(t)


def carrying(env):
    t = task(env)
    return t.grasped.float() * t.upright[t.ids, t.reward_box].clamp(0, 1) * (t.reward_phase == 2)


def placement(env):
    t = task(env)
    desired = t.slot_goal.clone()
    desired[:, 2] += t.belt_half[t.ids, t.reward_box, 2]
    distance = (t.centers[t.ids, t.reward_box] - desired).norm(dim=-1)
    return (torch.exp(-6 * distance) + t.supported[t.ids, t.reward_box].float() * t.released) * (t.reward_phase == 3)


def button_reach(env):
    t = task(env)
    distance = (t.tools - t.button_point[:, None]).norm(dim=-1).amin(-1)
    return torch.exp(-6 * distance) * (t.reward_phase == 4) * t.supported.all(-1)


def stability(env):
    t = task(env)
    return ((1 - t.upright).clamp_min(0) + 0.02 * t.velocities[..., 3:].square().sum(-1)).mean(-1) * ready(t)


def stage_completed(env):
    # RewardManager multiplies every term by dt. These are discrete bonuses,
    # not rates: keep their configured magnitude independent of control Hz.
    return task(env).transition.float() / env.step_dt


def success(env):
    return task(env).success.float() / env.step_dt


def failure(env):
    return task(env).failure.float() / env.step_dt


def flap_reaching(env):
    t = task(env)
    # A new best proximity earns once. dt cancellation makes a fixed amount of
    # progress worth the same at different control rates (RewardManager applies dt).
    return t.reach_progress.delta[:, t.spec.grasp_hand_indices].mean(-1) * ready(t) / env.step_dt


def flap_contact(env):
    t = task(env)
    # Grasp/contact shaping, like reaching, follows required hands only.
    selected = t.spec.grasp_hand_indices
    return (0.25 * t.finger_grasp_contacts[:, selected].float().mean((1, 2))
            + t.hand_grasp_flags[:, selected].float().mean(-1)) * ready(t)


def flap_orientation(env, distance_threshold: float = 0.10):
    """Weak additive pre-grasp alignment to the same nearest flap as reaching."""
    if not math.isfinite(distance_threshold) or distance_threshold <= 0:
        raise ValueError("Orientation distance_threshold must be finite and positive.")
    t = task(env)
    selected = t.spec.grasp_hand_indices
    proximity = (1 - t.hand_target_distance[:, selected] / distance_threshold).clamp(0, 1)
    alignment = t.grasp_alignment[:, selected].clamp(0, 1).square()
    before_grasp = ~t.hand_grasp_flags[:, selected]
    return (proximity * alignment * before_grasp).mean(-1) * ready(t) * (t.reward_phase == 1)


def flap_hold(env):
    t = task(env)
    return (t.dwell / t.spec.hold_seconds).clamp(0, 1)


def unwanted_contact(env):
    t = task(env)
    other_finger = (t.unexpected_finger_force / t.spec.unexpected_contact_limit).clamp(0, 5).mean(-1)
    obstacle = t.obstacle_forces.amax(-1)
    return other_finger * ready(t) + (obstacle / 20).clamp(0, 5)


def settled_action_rate(env):
    return ready(task(env)) * (env.action_manager.action - env.action_manager.prev_action).square().sum(-1)


def settled_joint_speed(env):
    t = task(env)
    return ready(t) * t.robot.data.joint_vel.square().sum(-1)


def settled_time(env):
    return ready(task(env)) * (~env.termination_manager.terminated).float()


def _grasp_stability(env):
    t = task(env)
    box = t.reward_box
    delta = t.centers[t.ids, box] - env.scene.env_origins - t.initial_centers[t.ids, box]
    height = t.centers[t.ids, box, 2] - env.scene.env_origins[:, 2] - t.initial_z[t.ids, box]
    scores = grasp_stability_scores(delta, t.poses[t.ids, box, 3:], t.initial_quats[t.ids, box],
                                   t.velocities[t.ids, box], t.grasped, height, t.spec)
    return tuple(score * ready(t) for score in scores)


def prelift_disturbance(env):
    return _grasp_stability(env)[0]


def stable_flap_grasp(env):
    return _grasp_stability(env)[1]
