"""Dense shaping is phase-gated; success still requires the physical predicates."""

import torch
from .commands import task


def navigation(env):
    t = task(env)
    return torch.exp(-4 * t.nav_distance) * torch.exp(-t.heading_error) * ((t.reward_phase == 0) | (t.reward_phase == 2))


def reaching(env):
    t = task(env)
    return torch.exp(-6 * t.reach_distance) * (t.reward_phase == 1)


def lift(env):
    t = task(env)
    height = t.centers[t.ids, t.reward_box, 2] - env.scene.env_origins[:, 2] - t.initial_z[t.ids, t.reward_box]
    return (height / t.spec.lift_height).clamp(0, 1) * t.grasped * (t.reward_phase == 1)


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
    return ((1 - t.upright).clamp_min(0) + 0.02 * t.velocities[..., 3:].square().sum(-1)).mean(-1)


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
    # Both tools must approach their assigned flap; align closing axes to plate normals.
    return (torch.exp(-12 * t.hand_target_distance) * (0.25 + 0.75 * t.grasp_alignment)).mean(-1)


def flap_contact(env):
    t = task(env)
    # Partial shaping leads from first valid upper-band contact to two opposed jaws.
    return 0.25 * t.finger_grasp_contacts.float().mean((1, 2)) + t.hand_grasp_flags.float().mean(-1)


def flap_hold(env):
    t = task(env)
    return (t.dwell / t.spec.hold_seconds).clamp(0, 1)


def unwanted_contact(env):
    t = task(env)
    other_finger = (t.unexpected_finger_force / t.spec.unexpected_contact_limit).clamp(0, 5).mean(-1)
    arm_contact = env.scene["robot_contact"].data.net_forces_w.norm(dim=-1).amax(-1)
    return other_finger + (arm_contact / 20).clamp(0, 5)
