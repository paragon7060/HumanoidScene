"""Reset and conveyor events. Objects are only pose-reset at episode boundaries."""

import torch
from isaaclab.envs import mdp as base_mdp
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
from .geometry import rotate, slot_offsets
from .reset_bank import restore_bank


def _env_ids(env, env_ids, device=None):
    """Resolve only the manager-selected environments, including all/empty resets."""
    device = env.device if device is None else device
    if env_ids is None:
        return torch.arange(env.num_envs, device=device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


def reset_episode(env, env_ids):
    ids = _env_ids(env, env_ids)
    if not len(ids):
        return
    base_mdp.reset_scene_to_default(env, ids)
    spec = env.cfg.task
    if spec.reset_bank:
        restore_bank(env, ids)
        return
    command = env.command_manager.get_term("workcell")
    command.active_box[ids] = 0
    command.phase[ids] = 0
    command._measure()
    command._goals()
    robot = env.scene["robot"]
    pose = robot.data.default_root_state[ids, :7].clone()
    pose[:, :3] += env.scene.env_origins[ids]
    if spec.name == "pick" and spec.control_mode != "arms-only":
        pose[:, :2] = command.goal[ids, :2]
        angle = command.goal[ids, 2]
        pose[:, 3:] = quat_from_euler_xyz(torch.zeros_like(angle), torch.zeros_like(angle), angle)
    difficulty = (0.1 + 0.9 * float(getattr(env, "_rl_difficulty", 0.0))
                  if spec.randomization and spec.control_mode != "arms-only" else 0.0)
    # Rack/box poses are deliberately not jittered independently through shelves.
    pose[:, :2] += torch.empty(len(ids), 2, device=env.device).uniform_(-1, 1) * spec.reset_xy_jitter * difficulty
    angle = torch.empty(len(ids), device=env.device).uniform_(-1, 1) * spec.reset_yaw_jitter * difficulty
    pose[:, 3:] = quat_mul(quat_from_euler_xyz(torch.zeros_like(angle), torch.zeros_like(angle), angle), pose[:, 3:])
    robot.write_root_pose_to_sim(pose, env_ids=ids)
    # Start with open hands. Snapshot resets instead retain their recorded grasp.
    from ...robots.gripper_config import resolve_gripper_settings
    from isaaclab.utils.string import resolve_matching_names_values
    hand = resolve_gripper_settings()
    for side in hand.active_sides:
        indices, _, values = resolve_matching_names_values(hand.command_for(side, hand.open_command), robot.joint_names)
        positions = torch.tensor(values, device=env.device).expand(len(ids), -1)
        robot.write_joint_state_to_sim(positions, torch.zeros_like(positions), joint_ids=indices, env_ids=ids)
        robot.set_joint_position_target(positions, joint_ids=indices, env_ids=ids)
    for box_id, name in enumerate(spec.box_names):
        box = env.scene[name]
        geometry = env.cfg.commands.workcell.geometry[name]
        for item in range(spec.cargo_per_box):
            cargo = env.scene[f"cargo_{name}_{item}"]
            pose = cargo.data.default_root_state[ids, :7].clone()
            local = torch.tensor(geometry.center, device=env.device).expand(len(ids), -1).clone()
            local[:, 0] += (-0.25 if item == 0 else 0.25) * geometry.half_size[0]
            local[:, 2] -= geometry.half_size[2] - spec.cargo_radius - 0.012
            pose[:, :3] = box.data.root_pos_w[ids] + rotate(box.data.root_quat_w[ids], local)
            cargo.write_root_pose_to_sim(pose, env_ids=ids)
            cargo.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), env_ids=ids)
    belt = env.scene["conveyor_surface"]
    offsets = slot_offsets(spec.slot_count, spec.slot_pitch, env.device)
    for index in range(spec.prefill_count):
        item = env.scene[f"prefill_{index}"]
        pose = item.data.default_root_state[ids, :7].clone()
        local = offsets[-1-index].expand(len(ids), -1).clone()
        local[:, 2] += 0.09
        pose[:, :3] = belt.data.root_pos_w[ids] + rotate(belt.data.root_quat_w[ids], local)
        pose[:, 3:] = belt.data.root_quat_w[ids]
        item.write_root_pose_to_sim(pose, env_ids=ids)


def randomize_box_flap_joint_friction(
    env,
    env_ids: torch.Tensor | None,
    asset_names: tuple[str, ...],
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
) -> None:
    """Randomize flap DOF friction independently in the selected RL environments."""
    for asset_name in asset_names:
        box = env.scene[asset_name]
        ids = _env_ids(env, env_ids, box.device)
        if not len(ids):
            continue
        joint_ids, _ = box.find_joints("joint_(front|back|left|right)")
        shape = (len(ids), len(joint_ids))
        static = torch.empty(shape, device=box.device).uniform_(*static_friction_range)
        dynamic = torch.empty(shape, device=box.device).uniform_(*dynamic_friction_range)
        # PhysX requires static friction >= dynamic friction for every DOF.
        dynamic = torch.minimum(dynamic, static)
        box.write_joint_friction_coefficient_to_sim(static, joint_ids=joint_ids, env_ids=ids)
        box.write_joint_dynamic_friction_coefficient_to_sim(dynamic, joint_ids=joint_ids, env_ids=ids)


def conveyor_motion(env, env_ids):
    """Approximate belt drive via velocity on supported objects, only after valid press.

    This is an explicit simulation conveyor abstraction, not a material surface
    velocity extension. It never transports a grasped/airborne box by pose writes.
    """
    selected_ids = _env_ids(env, env_ids)
    if not len(selected_ids):
        return
    command = env.command_manager.get_term("workcell")
    spec = env.cfg.task
    direction = torch.zeros(env.num_envs, 3, device=env.device); direction[:, 0] = spec.conveyor_speed
    velocity = rotate(env.scene["conveyor_surface"].data.root_quat_w, direction)
    for box_id, name in enumerate(spec.box_names):
        ids = selected_ids[(command.belt_running & command.supported[:, box_id])[selected_ids]]
        if not len(ids):
            continue
        box = env.scene[name]
        state = box.data.root_vel_w[ids].clone()
        state[:, :3] = velocity[ids]
        box.write_root_velocity_to_sim(state, env_ids=ids)
        for item in range(spec.cargo_per_box):
            cargo = env.scene[f"cargo_{name}_{item}"]
            state = cargo.data.root_vel_w[ids].clone(); state[:, :3] = velocity[ids]
            cargo.write_root_velocity_to_sim(state, env_ids=ids)
    for index in range(spec.prefill_count):
        ids = selected_ids[command.belt_running[selected_ids]]
        if len(ids):
            foreign = env.scene[f"prefill_{index}"]
            state = foreign.data.root_vel_w[ids].clone(); state[:, :3] = velocity[ids]
            foreign.write_root_velocity_to_sim(state, env_ids=ids)
