"""Keep termination predicates separate from shaping rewards."""

from .commands import task
from .robot_safety import robot_motion_unsafe


def success(env):
    return task(env).success


def unsafe(env):
    t = task(env)
    return t.failure | robot_motion_unsafe(t.robot)
