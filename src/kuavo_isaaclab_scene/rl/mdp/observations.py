"""State observations for PPO; all positions relative to the robot or env origin."""

import torch
from .commands import task
from .geometry import unrotate


def objects(env):
    t = task(env)
    q = t.robot.data.root_quat_w[:, None].expand(-1, t.n, -1)
    relative = unrotate(q, t.centers - t.robot.data.root_pos_w[:, None])
    velocity = unrotate(q, t.velocities[..., :3])
    return torch.cat((relative, t.poses[..., 3:], velocity, t.velocities[..., 3:]), -1).flatten(1)


def task_state(env):
    t = task(env)
    button = unrotate(t.robot.data.root_quat_w, t.button_point - t.robot.data.root_pos_w)
    return torch.cat((t.cargo_ok.float(), t.supported.float(), t.done_boxes.float(),
        t.contact_force.clamp(0, 50) / 50, button,
        t.button_pressed[:, None].float(), t.belt_running[:, None].float(),
        t.dwell[:, None], t.belt_time[:, None]), -1)


def actuator_state(env):
    # Include every action integrator state: persistent targets are history dependent.
    return torch.cat([env.action_manager.get_term(n).processed_actions
                      for n in env.action_manager.active_terms], -1)


def cargo_state(env):
    t = task(env)
    values = []
    for i, name in enumerate(t.spec.box_names):
        for item in range(t.spec.cargo_per_box):
            cargo = env.scene[f"cargo_{name}_{item}"]
            values.append(unrotate(t.poses[:, i, 3:], cargo.data.root_pos_w - t.poses[:, i, :3]))
            values.append(unrotate(t.poses[:, i, 3:], cargo.data.root_lin_vel_w - t.velocities[:, i, :3]))
    return torch.cat(values, -1) if values else torch.zeros(env.num_envs, 0, device=env.device)
