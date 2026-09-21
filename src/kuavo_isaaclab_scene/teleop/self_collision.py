"""Compatibility imports for the shared robot self-collision package."""

from ..robots.self_collision import (  # noqa: F401
    ClearanceViolation,
    CollisionStop,
    RobotCollisionModel,
    SELF_COLLISION_POLICY_FILES,
    SelfCollisionFilter,
    load_fcl,
    resolve_self_collision_policy,
)


__all__ = (
    "ClearanceViolation",
    "CollisionStop",
    "RobotCollisionModel",
    "SELF_COLLISION_POLICY_FILES",
    "SelfCollisionFilter",
    "load_fcl",
    "resolve_self_collision_policy",
)
