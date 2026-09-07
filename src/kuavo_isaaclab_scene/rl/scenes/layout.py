"""Shared capture inputs, resolved without importing a general environment."""

import math
from ...workcell.rack_box_layout import (
    build_box_spawn_plan, rack_instance_names, resolve_rack_box_layout, resolve_rack_box_pose_path,
)

# Same measured shelf slope as the authored Rack.usd used by the general scene.
RACK_SLOPE_RAD = math.radians(5.114147010769473)


def box_spawn_plan():
    return build_box_spawn_plan(resolve_rack_box_layout(), RACK_SLOPE_RAD,
                                resolve_rack_box_pose_path())


def active_rack_box_scene_keys():
    plans = box_spawn_plan()
    return tuple(plans[name].scene_key for name in rack_instance_names(plans))
