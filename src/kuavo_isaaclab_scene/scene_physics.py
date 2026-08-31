"""Shared asset physics for standalone, manager-based and Quest scenes.

Input devices, servo gains and kinematic base control remain environment-specific.
"""

from .teleop_contacts import spawn_contact_box
from .teleop_inertials import spawn_s200062_robot


def build_box_flap_actuator(settings):
    from isaaclab.actuators import ImplicitActuatorCfg

    return ImplicitActuatorCfg(
        joint_names_expr=["joint_front", "joint_back", "joint_left", "joint_right"],
        effort_limit_sim=5.0,
        velocity_limit_sim=10.0,
        stiffness=0.0,
        damping=0.05,
        friction=settings.static,
        dynamic_friction=settings.dynamic,
    )


def build_contact_box_spawn(usd_path, scale):
    import isaaclab.sim as sim_utils

    return sim_utils.UsdFileCfg(
        func=spawn_contact_box,
        usd_path=str(usd_path),
        scale=scale,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=8,
        ),
    )


def configure_robot_asset_physics(cfg, model, gripper_settings):
    """Apply hand contacts/inertials without changing wheel contacts or arm control."""
    if model.name != "s200062":
        return
    cfg.spawn.func = spawn_s200062_robot
    cfg.spawn.articulation_props.solver_position_iteration_count = 32
    cfg.spawn.articulation_props.solver_velocity_iteration_count = 8
    cfg.actuators["integrated_grippers"].armature = .001
    if gripper_settings.integrated:
        cfg.init_state.joint_pos.pop("[lr]_[fb]_bar_[13]_joint", None)
        for side in gripper_settings.active_sides:
            cfg.init_state.joint_pos.update(
                gripper_settings.command_for(side, gripper_settings.open_command)
            )
