"""Pure geometry predicates and coordinate-frame transforms."""

from .belt import footprint_inside_rectangle, unsigned_axis_angle_error
from .pose import pose_to_position_rotation_6d, relative_pose

__all__ = (
    "footprint_inside_rectangle",
    "pose_to_position_rotation_6d",
    "relative_pose",
    "unsigned_axis_angle_error",
)
