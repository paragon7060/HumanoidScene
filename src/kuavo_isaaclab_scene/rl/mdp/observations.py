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


def box_pose_only(env):
    """Object 6-DoF pose as xyz + quaternion wxyz in robot frame, no box velocities."""
    from isaaclab.utils.math import quat_mul, quat_conjugate
    t = task(env)
    q = t.robot.data.root_quat_w[:, None].expand(-1, t.n, -1)
    relative = unrotate(q, t.centers - t.robot.data.root_pos_w[:, None])
    orientation = quat_mul(quat_conjugate(q), t.poses[..., 3:])
    return torch.cat((relative, orientation), dim=-1).flatten(1)


def box_velocity(env):
    """Expose the motion used by settle predicates to the feed-forward policy."""
    t = task(env)
    q = t.robot.data.root_quat_w[:, None].expand(-1, t.n, -1)
    return torch.cat((unrotate(q, t.velocities[..., :3]),
                      unrotate(q, t.velocities[..., 3:])), dim=-1).flatten(1)


def hand_flap_relation(env):
    from isaaclab.utils.math import quat_mul, quat_conjugate
    t = task(env)
    q = t.robot.data.body_link_quat_w[:, t.tool_ids]
    delta = unrotate(q, t.grips - t.tools)
    relative = quat_mul(quat_conjugate(q), t.flap_quat)
    return torch.cat((delta, relative, t.grasp_alignment[..., None]), dim=-1).flatten(1)


def reaching_history(env):
    """Latest active score, used as the next step's baseline; zero while held/disabled."""
    return task(env).reach_progress.previous.clone()


def flap_progress_history(env):
    return task(env).flap_progress.observation()


def box_rest_relation(env):
    """Expose the latched resting pose used by the pre-lift disturbance cost."""
    from isaaclab.utils.math import quat_mul, quat_conjugate
    t = task(env)
    box = t.active_box
    delta = t.centers[t.ids, box] - env.scene.env_origins - t.initial_centers[t.ids, box]
    delta = unrotate(t.robot.data.root_quat_w, delta)
    rotation = quat_mul(quat_conjugate(t.initial_quats[t.ids, box]), t.poses[t.ids, box, 3:])
    return torch.cat((delta, rotation), -1)


def flap_pick_state(env):
    t = task(env)
    robot_contact = t.obstacle_forces.clamp(0, 100) / 100
    height = (t.centers[t.ids, t.active_box, 2] - env.scene.env_origins[:, 2]
              - t.initial_z[t.ids, t.active_box])
    settle_duration = t.spec.reset_settle_timeout or max(t.spec.reset_settle_seconds, 1e-6)
    settling = (torch.stack((t.settling.ready.float(), t.settling.elapsed / settle_duration), -1)
                if t.settling is not None else torch.tensor((1., 0.), device=env.device).expand(env.num_envs, -1))
    return torch.cat((t.contact_force.clamp(0, 50) / 50, t.hand_grasp_flags.float(),
                      t.raw_hand_grasp_flags.float(), t.grasp_contact_missing_s,
                      torch.nn.functional.one_hot(t.contact_flap_index, 2).float().flatten(1),
                      t.unexpected_finger_force.clamp(0, 50) / 50, robot_contact,
                      t.half_size[t.active_box], height[:, None], t.dwell[:, None], settling), dim=-1)
