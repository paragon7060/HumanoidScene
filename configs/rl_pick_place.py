"""Flap grasp -> lift -> extract -> carry -> release on a stopped conveyor."""
from dataclasses import replace
from kuavo_isaaclab_scene.configs.rl_pick_whole_body import INITIAL_STATE
from kuavo_isaaclab_scene.configs.rl_pick_whole_body import configure_task as pick_task
from kuavo_isaaclab_scene.configs.rl_pick_whole_body import configure as pick_configure


def configure_task(spec):
    return replace(pick_task(spec), name="pick_place", episode_length_s=120.,
                   transfer_target_tolerance=.15, rack_extract_clearance=.025,
                   placement_contact_force=.20, settle_speed=.08,
                   settle_angular_speed=.35, hold_seconds=.5,
                   collision_constraints_enabled=True, workspace_radius=1.5)


def configure(env_cfg, agent_cfg):
    pick_configure(env_cfg, agent_cfg)
