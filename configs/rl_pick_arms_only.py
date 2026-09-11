"""Stage 1: right arm only; pinch either flap surface, lift 6 cm and hold."""

from dataclasses import replace
import math

# This experiment always restores this measured robot root + 36 joint pose.
# Change this name deliberately when starting a new initial-pose experiment.
INITIAL_STATE = "quest_ready_02"


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
        obstacle_contact_force=20.0,
        collision_constraints_enabled=False,  # learn grasp first; keep physics/sensors, no collision failure/cost
        lift_height=0.06,
        max_tilt=math.radians(40),
        hold_seconds=0.5,
        cargo_per_box=0,
        prefill_count=0,
        randomization=False,
    )


def configure(env_cfg, agent_cfg):
    # At 30 Hz, disturbance contributes at worst -0.25/30 per step.
    env_cfg.rewards.prelift_disturbance.weight = -0.25
    # Additive orientation shaping only within 10 cm and before grasp acquisition.
    env_cfg.rewards.orientation.weight = 0.5
    env_cfg.rewards.orientation.params["distance_threshold"] = 0.10
    # Actions are normalized incremental joint targets in radians/control step.
    # Zero holds the previous target. Use a small range for the first experiments.
    env_cfg.actions.upper_body.scale = 0.02
    env_cfg.actions.upper_body.body_lock_tolerance = 1e-4
    for side in ("left", "right"):
        gripper = getattr(env_cfg.actions, side + "_gripper")
        if gripper is not None:
            gripper.delta_scale = 0.08
    agent_cfg.policy.init_noise_std = 0.15
