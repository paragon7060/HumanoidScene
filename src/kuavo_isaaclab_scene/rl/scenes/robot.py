"""Build only the controlled robot for the lightweight RL scene.

This module shares asset-level physics with the interactive workcell, but never
imports its scene or teleoperation/evaluation configuration. Model selection is
resolved when the factory is called, after the runner has exported CLI options.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from ...envs.scene_physics import configure_robot_asset_physics
from ...robots.gripper_config import load_gripper_settings, resolve_gripper_settings
from ...robots.robot_inertials import spawn_s56_twofinger_robot, spawn_teleop_robot
from ...robots.robot_model import resolve_robot_model
from ...workcell.workcell_layout import load_layout


def build_robot_cfg() -> ArticulationCfg:
    """Create an independent robot configuration for each RL scene assembly.

    The two supported presets retain their original arm/body gains, measured
    inertials, closed finger linkages, passive-joint initialization, and S56
    motor limits. The saved named initial state is applied by the reset manager;
    these are only the model/layout defaults used before that reset.
    """
    model = resolve_robot_model()
    hand = resolve_gripper_settings()
    if hand.name not in ("s200062_integrated", "s56_twofinger"):
        raise ValueError(
            "RL contact presets support s200062_integrated/s56_twofinger only. "
            "Configure finger/tool sensors and actions before using another gripper."
        )
    integrated_hand = load_gripper_settings(model.integrated_gripper_preset)
    robot_pose = load_layout()["robot"]

    if model.has_wheel_base:
        base_positions = {
            "wheel_.*_joint": 0.0,
            "knee_joint": 0.0,
            "leg_joint": 0.0,
            "waist_pitch_joint": 0.0,
        }
        body_actuators = {
            "height_axis": ImplicitActuatorCfg(
                joint_names_expr=["knee_joint", "leg_joint", "waist_pitch_joint"],
                effort_limit_sim=700.0,
                velocity_limit_sim=25.0,
                stiffness=400.0,
                damping=40.0,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["wheel_.*_joint"],
                effort_limit_sim=100.0,
                velocity_limit_sim=30.0,
                stiffness=0.0,
                damping=10.0,
            ),
        }
    else:
        base_positions = {"leg_.*_joint": 0.0}
        body_actuators = {
            "lower_body": ImplicitActuatorCfg(
                joint_names_expr=["leg_.*_joint"],
                effort_limit_sim=300.0,
                velocity_limit_sim=20.0,
                stiffness=400.0,
                damping=40.0,
            ),
        }

    cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Kuavo",
        spawn=sim_utils.UsdFileCfg(
            usd_path=model.usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=2.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=model.spawn_position(robot_pose.pos),
            rot=robot_pose.rot,
            joint_pos={
                **base_positions,
                "zarm_.*_joint": 0.0,
                "waist_yaw_joint": 0.0,
                "zhead_.*_joint": 0.0,
                **integrated_hand.command_for_all_sides(integrated_hand.default_joint_pos),
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            **body_actuators,
            "arms": ImplicitActuatorCfg(
                joint_names_expr=["zarm_.*_joint"],
                effort_limit_sim=100.0,
                velocity_limit_sim=20.0,
                stiffness=220.0,
                damping=22.0,
            ),
            "upper_body": ImplicitActuatorCfg(
                joint_names_expr=["waist_yaw_joint", "zhead_.*_joint"],
                effort_limit_sim=100.0,
                velocity_limit_sim=10.0,
                stiffness=120.0,
                damping=15.0,
            ),
            "integrated_grippers": ImplicitActuatorCfg(
                joint_names_expr=list(integrated_hand.joint_name_exprs_for_robot),
                effort_limit_sim=integrated_hand.actuator.effort_limit_sim,
                velocity_limit_sim=5.0,
                stiffness=integrated_hand.actuator.stiffness,
                damping=integrated_hand.actuator.damping,
                friction=integrated_hand.actuator.friction,
            ),
        },
    )
    configure_robot_asset_physics(cfg, model, hand)
    # Despite its historical name, this shared asset spawner only corrects
    # inertials/colliders; it imports no Quest code. Omit wheel-ground contacts
    # for the fixed-root planar controller, as in the existing RL path.
    cfg.spawn.func = spawn_teleop_robot if model.has_wheel_base else spawn_s56_twofinger_robot
    return cfg
