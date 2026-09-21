"""One binary gripper-control setup for Quest, RL debug, and training configs."""

import math


def _is_contact_box_asset(name):
    # The v2 pool uses stable mb_* physical keys because type/region belong to
    # per-environment logical slots, while legacy scenes use *box* keys.
    lower = name.lower()
    return lower == "box" or "_box_" in lower or lower.startswith("mb_")


def build_binary_gripper_action_cfg(
    hand,
    side,
    *,
    force_n=None,
    force_sensor_names=None,
    command_gate=None,
):
    """Build the package-owned 0=open/1=close action for one hand."""
    from ..gripper_runtime import BinaryGripperActionCfg

    if command_gate not in (None, "settling", "ready"):
        raise ValueError(f"Unknown gripper command gate: {command_gate}")
    return BinaryGripperActionCfg(
        asset_name=hand.asset_name_for(side),
        joint_names=list(hand.joint_names_for(side)),
        open_command_expr=hand.command_for(side, hand.open_command),
        close_command_expr=hand.command_for(side, hand.close_command),
        position_mapping=hand.sides[side].position_mapping,
        target_filter=hand.sides[side].target_filter,
        close_force_n=force_n,
        force_side=side if force_n is not None else None,
        force_sensor_names=force_sensor_names,
        command_gate=command_gate,
    )


def configure_binary_gripper_control(
    cfg,
    force_n=None,
    *,
    contact_feedback=False,
    command_gate=None,
):
    """Install the same package binary action in every supported execution mode.

    ``contact_feedback`` only selects the force backend. It never changes the
    public command, calibrated position mapping, target filter, or close force.
    """
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.assets import ArticulationCfg, RigidObjectCfg
    from isaaclab.sim import JointDrivePropertiesCfg
    from ..gripper_config import resolve_gripper_settings
    from .linkage import TWO_FINGER_PRESETS
    from .package import default_close_force_n

    force_n = default_close_force_n() if force_n is None else float(force_n)
    if not math.isfinite(force_n) or force_n < 0:
        raise ValueError("Gripper close force must be finite and nonnegative")
    hand = resolve_gripper_settings()
    force_enabled = force_n > 0 and hand.name in TWO_FINGER_PRESETS
    feedback_enabled = bool(contact_feedback and force_enabled)
    targets = []
    if feedback_enabled:
        from ...rl.scenes.asset_geometry import robot_rigid_body_paths
        for name, asset in vars(cfg.scene).items():
            if not _is_contact_box_asset(name) or not isinstance(
                    asset, (ArticulationCfg, RigidObjectCfg)):
                continue
            path = getattr(asset.spawn, "usd_path", None)
            if path is not None:
                targets.extend(
                    asset.prim_path + ("/" + body if body != "." else "")
                    for body in robot_rigid_body_paths(path)
                )
        if not targets:
            raise ValueError(
                "Contact-feedback gripper control requires contact-enabled box bodies")
    if force_enabled:
        # Include gripper feedforward torque in every TGS position iteration.
        cfg.sim.physx.enable_external_forces_every_iteration = True
    count = 0
    for side in hand.active_sides:
        name = side + "_gripper"
        if getattr(cfg.actions, name, None) is None:
            continue
        sensors = None
        if feedback_enabled:
            sensors = tuple(f"gripper_force_{side}_{jaw}" for jaw in "fb")
            asset = getattr(cfg.scene, hand.asset_name_for(side))
            asset.spawn.activate_contact_sensors = True
            if asset.spawn.joint_drive_props is None:
                asset.spawn.joint_drive_props = JointDrivePropertiesCfg(
                    drive_type="force")
            else:
                asset.spawn.joint_drive_props.drive_type = "force"
            for jaw, sensor in zip("fb", sensors, strict=True):
                setattr(cfg.scene, sensor, ContactSensorCfg(
                    prim_path=asset.prim_path + f"/{side[0]}_{jaw}_finger",
                    update_period=0.0,
                    history_length=1,
                    filter_prim_paths_expr=list(targets),
                ))
        setattr(cfg.actions, name, build_binary_gripper_action_cfg(
            hand,
            side,
            force_n=float(force_n) if force_enabled else None,
            force_sensor_names=sensors,
            command_gate=command_gate,
        ))
        count += 1
    return count


def configure_vr_gripper_force(cfg, force_n=None):
    """Compatibility wrapper for direct Quest teleoperation."""
    return configure_binary_gripper_control(
        cfg, force_n, contact_feedback=True, command_gate=None)


def configure_rl_gripper_force(cfg, force_n=None, *, command_gate="settling"):
    """Compatibility wrapper for RL and RL-reward inspection."""
    return configure_binary_gripper_control(
        cfg, force_n, contact_feedback=False, command_gate=command_gate)
