"""Opt VR collectors/debug into force-close without changing policy actions."""

import math


def configure_vr_gripper_force(cfg, force_n=50., *, incremental=False):
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg
    from isaaclab.sim import JointDrivePropertiesCfg
    from .gripper_config import resolve_gripper_settings
    from .gripper_runtime import ForceBinaryGripperActionCfg
    from .twofinger_linkage import TWO_FINGER_PRESETS
    from ..rl.scenes.asset_geometry import robot_rigid_body_paths

    if not math.isfinite(force_n) or force_n < 0:
        raise ValueError("VR gripper close force must be finite and nonnegative")
    hand = resolve_gripper_settings()
    if force_n == 0 or hand.name not in TWO_FINGER_PRESETS:
        return 0
    targets = []
    for name, asset in vars(cfg.scene).items():
        if "box" not in name.lower() or not isinstance(asset, (ArticulationCfg, RigidObjectCfg)):
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
            action = getattr(cfg.actions, name)
            action.close_force_n = float(force_n)
            action.force_side = side
            action.force_sensor_names = sensors
        else:
            setattr(cfg.actions, name, ForceBinaryGripperActionCfg(
                asset_name=hand.asset_name_for(side), joint_names=list(hand.joint_names_for(side)),
                open_command_expr=hand.command_for(side, hand.open_command),
                close_command_expr=hand.command_for(side, hand.close_command),
                target_filter=hand.sides[side].target_filter,
                close_force_n=float(force_n), force_side=side, force_sensor_names=sensors))
        count += 1
    return count
