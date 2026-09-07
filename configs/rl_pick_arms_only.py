"""Stage 1: pinch two flap upper edges, lift 6 cm and hold. No rack extraction."""

from dataclasses import replace
import math

# This experiment always restores this measured robot root + 36 joint pose.
# Change this name deliberately when starting a new initial-pose experiment.
INITIAL_STATE = "quest_ready_02"


def configure_task(spec):
    return replace(spec,
        control_mode="arms-only",
        required_grasp_hands=2,
        grasp_mode="flap_top",
        # Robot left/right hand assignment; change if the workcell is rotated.
        grasp_flaps=("flap_right", "flap_left"),
        flap_grasp_depth=0.015,
        flap_top_band=0.030,
        flap_lock_degrees=0.5,
        lift_height=0.06,
        max_tilt=math.radians(40),
        hold_seconds=0.5,
        cargo_per_box=0,
        prefill_count=0,
        randomization=False,
    )


def configure(env_cfg, agent_cfg):
    # Actions are normalized incremental joint targets in radians/control step.
    # Zero holds the previous target. Use a small range for the first experiments.
    env_cfg.actions.upper_body.scale = 0.02
    env_cfg.actions.upper_body.body_lock_tolerance = 1e-4
    env_cfg.actions.left_gripper.delta_scale = 0.08
    env_cfg.actions.right_gripper.delta_scale = 0.08
    agent_cfg.policy.init_noise_std = 0.15
