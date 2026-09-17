"""Opt VR collectors/debug into force-close without changing policy actions."""

import math


def _is_contact_box_asset(name):
    # The v2 pool uses stable mb_* physical keys because type/region belong to
    # per-environment logical slots, while legacy scenes use *box* keys.
    lower = name.lower()
    return lower == "box" or "_box_" in lower or lower.startswith("mb_")


def _incremental_delta_scale(action_cfg, fallback=0.12):
    """Restore a usable scale when replacing a binary action in Quest debug.

    BinaryGripperCfg keeps ``delta_scale=0`` only for old config compatibility;
    incremental controller tracking must never inherit that sentinel value.
    """
    value = getattr(action_cfg, "delta_scale", None)
    if value is None:
        return fallback
    value = float(value)
    return value if math.isfinite(value) and value > 0 else fallback


def configure_vr_gripper_force(cfg, force_n=None, *, incremental=False):
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg
    from isaaclab.sim import JointDrivePropertiesCfg
    from ..gripper_config import resolve_gripper_settings
    from ..gripper_runtime import ForceBinaryGripperActionCfg
    from .linkage import TWO_FINGER_PRESETS
    from .package import default_close_force_n
    from ...rl.scenes.asset_geometry import robot_rigid_body_paths

    force_n = default_close_force_n() if force_n is None else float(force_n)
    if not math.isfinite(force_n) or force_n < 0:
        raise ValueError("VR gripper close force must be finite and nonnegative")
    hand = resolve_gripper_settings()
    if force_n == 0 or hand.name not in TWO_FINGER_PRESETS:
        return 0
    targets = []
    for name, asset in vars(cfg.scene).items():
        if not _is_contact_box_asset(name) or not isinstance(asset, (ArticulationCfg, RigidObjectCfg)):
            continue
        path = getattr(asset.spawn, "usd_path", None)
        if path is not None:
            targets.extend(asset.prim_path + ("/" + body if body != "." else "")
                           for body in robot_rigid_body_paths(path))
    if not targets:
        raise ValueError("VR gripper force control requires contact-enabled box bodies")
    # Include feedforward torque in TGS position iterations as well as
    # velocity updates, so contact and joint limits resolve the same load.
    cfg.sim.physx.enable_external_forces_every_iteration = True
    count = 0
    for side in hand.active_sides:
        name = side + "_gripper"
        if getattr(cfg.actions, name, None) is None:
            continue
        sensors = tuple(f"gripper_force_{side}_{jaw}" for jaw in "fb")
        asset = getattr(cfg.scene, hand.asset_name_for(side))
        asset.spawn.activate_contact_sensors = True
        if asset.spawn.joint_drive_props is None:
            asset.spawn.joint_drive_props = JointDrivePropertiesCfg(drive_type="force")
        else:
            asset.spawn.joint_drive_props.drive_type = "force"
        for jaw, sensor in zip("fb", sensors):
            setattr(cfg.scene, sensor, ContactSensorCfg(
                prim_path=asset.prim_path + f"/{side[0]}_{jaw}_finger",
                update_period=0., history_length=1, filter_prim_paths_expr=list(targets)))
        if incremental:
            # RL now defaults to binary grippers. Quest manual control still
            # needs a gradual signed target, so restore that action explicitly.
            from ...rl.mdp.actions import IncrementalGripperCfg
            previous = getattr(cfg.actions, name)
            setattr(cfg.actions, name, IncrementalGripperCfg(
                asset_name=hand.asset_name_for(side), joint_names=list(hand.joint_names_for(side)),
                open_command_expr=hand.command_for(side, hand.open_command),
                close_command_expr=hand.command_for(side, hand.close_command),
                position_mapping=hand.sides[side].position_mapping,
                target_filter=hand.sides[side].target_filter,
                delta_scale=_incremental_delta_scale(previous),
                close_force_n=float(force_n), force_side=side, force_sensor_names=sensors))
        else:
            setattr(cfg.actions, name, ForceBinaryGripperActionCfg(
                asset_name=hand.asset_name_for(side), joint_names=list(hand.joint_names_for(side)),
                open_command_expr=hand.command_for(side, hand.open_command),
                close_command_expr=hand.command_for(side, hand.close_command),
                position_mapping=hand.sides[side].position_mapping,
                target_filter=hand.sides[side].target_filter,
                close_force_n=float(force_n), force_side=side, force_sensor_names=sensors))
        count += 1
    return count


def configure_rl_gripper_force(cfg, force_n=None):
    """Override the RL binary gripper's sensor-free PD force assist.

    Reward inspection must exercise the same action term and actuator path as
    training.  It may still expose a CLI force override, but it must not add
    the VR-only contact servo or replace the binary action with an incremental
    controller.
    """
    from ..gripper_config import resolve_gripper_settings
    from .linkage import TWO_FINGER_PRESETS
    from .package import default_close_force_n

    force_n = default_close_force_n() if force_n is None else float(force_n)
    if not math.isfinite(force_n) or force_n < 0:
        raise ValueError("RL gripper close force must be finite and nonnegative")
    hand = resolve_gripper_settings()
    if hand.name not in TWO_FINGER_PRESETS:
        return 0
    count = 0
    for side in hand.active_sides:
        action = getattr(cfg.actions, side + "_gripper", None)
        if action is None:
            continue
        action.close_force_n = force_n or None
        action.force_side = side if force_n else None
        action.force_sensor_names = None
        count += 1
    return count
