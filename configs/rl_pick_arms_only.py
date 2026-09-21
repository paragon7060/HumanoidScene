"""Stage 1: right arm only; pinch either flap surface, lift 6 cm and hold."""

from dataclasses import replace
import math

from kuavo_isaaclab_scene.robots.robot_model import resolve_robot_model

# Keep model-specific prepared poses separate from the measured S200062 state.
INITIAL_STATE = {"leju-twofinger": "s63_leju_ready_01",
                 "s56_twofinger": "s56_twofinger_ready_01"}.get(
                     resolve_robot_model().integrated_gripper_preset, "quest_ready_02")


def configure_task(spec):
    return replace(spec,
        control_mode="arms-only",
        active_arm="right",  # "both" restores two-arm actions; does not change grasp_hand.
        required_grasp_hands=1,
        grasp_hand="right",
        grasp_mode="flap_top",
        # Shared candidates, not hand assignments; only the active right hand trains.
        grasp_flaps=("flap_right", "flap_left"),
        flap_contact_region="surface",  # "top_band": require contacts in the upper 3 cm.
        flap_grasp_depth=0.015,
        flap_top_band=0.030,
        # Acquire with allowed-region contacts; tolerate only bounded losses after grasping.
        grasp_contact_grace_s=0.10,
        grasp_hold_slip_m=0.020,
        grasp_open_tolerance_m=0.008,
        grasp_force=0.20,  # N per finger on the SAME candidate flap.
        prelift_position_scale=0.05,
        prelift_speed_scale=0.20,
        prelift_angular_scale=1.0,
        prelift_rotation_scale=math.radians(30),
        prelift_position_deadband=0.005,
        prelift_speed_deadband=0.02,
        prelift_angular_deadband=0.10,
        prelift_rotation_deadband=math.radians(5),
        prelift_grasp_scale=0.25,
        prelift_penalty_cap=1.0,
        flap_lock_degrees=0.5,
        reset_settle_seconds=0.5,
        reset_settle_timeout=0.0,  # fixed 0.5s initial delay; never fail/wait indefinitely on settling
        obstacle_contact_force=0.1,
        collision_constraints_enabled=True,
        lift_height=0.06,
        max_tilt=math.radians(40),
        hold_seconds=0.5,
        cargo_per_box=0,
        prefill_count=0,
        randomization=False,
    )


def configure(env_cfg, agent_cfg):
    if (env_cfg.task.control_mode == "whole-body"
            and not getattr(getattr(env_cfg.scene.robot, "class_type", None),
                            "gravity_compensation_enabled", False)):
        # Released torso joints must support the upper body's gravity load.
        # The arms-only experiment instead locks these joints physically.
        actuators = env_cfg.scene.robot.actuators
        if "height_axis" in actuators:
            actuators["height_axis"].stiffness = 8000.0
            actuators["height_axis"].damping = 200.0
        actuators["upper_body"].stiffness = 800.0
        actuators["upper_body"].damping = 50.0
    # At 30 Hz, disturbance contributes at worst -0.25/30 per step.
    env_cfg.rewards.prelift_disturbance.weight = -0.25
    # Signed alignment improvement only within 10 cm and before grasp acquisition.
    env_cfg.rewards.orientation.weight = 0.5
    env_cfg.rewards.orientation.params["distance_threshold"] = 0.10
    # Arm actions are normalized incremental joint targets. Grippers use the
    # shared 0=open, 1=close binary action with 25 N assist per jaw.
    env_cfg.actions.upper_body.scale = 0.02
    env_cfg.actions.upper_body.body_lock_tolerance = 1e-4
    if getattr(agent_cfg, "_skip_training_setup", False) or agent_cfg.__class__.__module__ == "types":
        # Config/reward inspection needs the final scalar values but must not
        # import the optional rsl_rl implementation classes.
        agent_cfg.policy.init_noise_std = 0.15
        agent_cfg.policy.noise_std_type = "log"
        agent_cfg.algorithm.entropy_coef = 0.001
        agent_cfg.num_steps_per_env = 64
        agent_cfg.algorithm.num_mini_batches = 32
        agent_cfg.algorithm.num_learning_epochs = 4
    else:
        from kuavo_isaaclab_scene.rl.agents.flap_ppo import configure_flap_ppo
        configure_flap_ppo(agent_cfg)
