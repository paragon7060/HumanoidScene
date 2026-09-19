"""Diagnostic-only: pick_place with a much faster per-step arm/waist action scale.

Not used by training. Load explicitly with --rl-config to test whether the RL
action space's per-control-step joint-delta cap (env_cfg.actions.upper_body.scale)
is the bottleneck behind arm/gripper lag during base motion in Quest RL reward
debug, separate from the base-tracking and root-velocity fixes already applied.

Why this cap exists at all: RL reward debug intentionally reuses the exact
same action term/scale a trained policy would be bound by (see
docs/RL_QUEST_REWARD_DEBUG.md), so manual VR probing reflects what a policy
could actually achieve. Normal (non-RL) Quest collection does NOT go through
this action term at all -- teleop_ik.PersistentTeleopIKAction writes a joint
VELOCITY target every physics step directly from the arm response profile
(RESPONSIVE: max_velocity=2.5 rad/s), uncapped by any control-step scale.

RL reward debug instead converts the same IK solution into a normalized delta
clamped to +/-1 and multiplied by this scale once per 30 Hz control step
(rl/debug/quest_control.py: normalized_delta(...) -> JointDeltaTargets).
Current whole-body scale: zarm_*_joint=0.02 rad/step (~0.6 rad/s implied),
waist_yaw_joint=0.01 rad/step (~0.3 rad/s implied) -- roughly 4x-8x slower
than the arm servo's own 2.5 rad/s cap. This file raises both to test whether
that gap (not grip force or base tracking) is why the arm still can't keep up
with base motion.

If slip disappears with this file but not with the default pick_place config,
the actual fix is a deliberate scale change (a training-affecting decision,
not just a VR comfort setting) rather than more physics/mapper tuning.
"""
from kuavo_isaaclab_scene.configs.rl_pick_place import INITIAL_STATE  # noqa: F401
from kuavo_isaaclab_scene.configs.rl_pick_place import configure_task as pick_place_task
from kuavo_isaaclab_scene.configs.rl_pick_place import configure as pick_place_configure


def configure_task(spec):
    return pick_place_task(spec)


def configure(env_cfg, agent_cfg):
    pick_place_configure(env_cfg, agent_cfg)
    if env_cfg.task.control_mode == "whole-body":
        # ~4x the default per-step cap, approaching the arm servo's own
        # 2.5 rad/s "responsive" ceiling (2.5/30 ~= 0.083 rad/step at 30 Hz).
        env_cfg.actions.upper_body.scale = {"waist_yaw_joint": .04, "zarm_.*_joint": .08}

