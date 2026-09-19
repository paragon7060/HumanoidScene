"""Selection of the mobile base model shared by teleop, RL and evaluation.

The default base is a kinematic abstraction: the root pose is overwritten every
physics step, so the chassis never accelerates anything. That is cheap and exact
for reaching tasks, but a carried box only stays in the jaws because the gripper
teleports with it, and any base motion looks like a slip event at the grasp
contact.

The dynamic base instead frees the articulation root and tracks the same planar
velocity command with a PD wrench, so robot, gripper and payload accelerate in
one PhysX solve. It is the default, and resolves identically in every entry
point, so a policy trains against the chassis dynamics its demonstrations were
recorded with. --no-dynamic-base (or KUAVO_DYNAMIC_BASE=0) restores the
kinematic base for checkpoints and experiments that were built against it.

Configurations without a moving base need no decision here: arms-only control,
the evaluation body lock and models without a wheeled chassis keep the fixed
root, and only an explicit request reports the conflict instead. The controller
lives in base_drive_control; this module stays import-safe before Isaac starts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os

DYNAMIC_BASE_ENV = "KUAVO_DYNAMIC_BASE"
DEFAULT_DYNAMIC = True

# Four radial omni wheels. Only their spin is synchronized with the planar
# command; neither base model drives the robot through wheel contact.
WHEEL_JOINTS = (
    "wheel_left_front_joint",
    "wheel_right_front_joint",
    "wheel_left_behind_joint",
    "wheel_right_behind_joint",
)
WHEEL_ANGLES_RAD = (0.785398163, -0.785398163, 2.356194372, -2.356194372)
WHEEL_RADIUS_M = 0.13035
WHEEL_OFFSET_M = 0.32879


@dataclass(frozen=True)
class BaseDriveSettings:
    """Which base model every entry point should build."""

    dynamic: bool
    # Whether a CLI flag or environment variable asked for this model. A
    # configuration that cannot move its base falls back to the fixed root
    # silently on the default, and reports the conflict when asked explicitly.
    explicit: bool = False

    @property
    def fix_root_link(self) -> bool:
        """A wrench-driven chassis needs the imported world joint disabled."""
        return not self.dynamic


def add_base_drive_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dynamic-base",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Drive the mobile base as a floating rigid body tracked by a PD wrench "
            "instead of overwriting the root pose. A carried box then stays coupled "
            "to the gripper through real contact while the base moves. On by default; "
            "pass --no-dynamic-base (or set KUAVO_DYNAMIC_BASE=0) for the kinematic "
            "base that earlier scenes and checkpoints were built against."
        ),
    )


def export_base_drive_cli(args: argparse.Namespace) -> None:
    value = getattr(args, "dynamic_base", None)
    if value is None:
        os.environ.pop(DYNAMIC_BASE_ENV, None)
    else:
        os.environ[DYNAMIC_BASE_ENV] = "1" if value else "0"


def resolve_base_drive_settings(*, dynamic: bool | None = None) -> BaseDriveSettings:
    """Resolve the explicit argument, then the environment, then the default."""
    explicit = dynamic is not None
    if dynamic is None:
        value = os.environ.get(DYNAMIC_BASE_ENV)
        if value is None:
            dynamic = DEFAULT_DYNAMIC
        elif value in ("0", "1"):
            dynamic, explicit = value == "1", True
        else:
            raise ValueError(f"{DYNAMIC_BASE_ENV} must be 0 or 1, got {value!r}")
    return BaseDriveSettings(dynamic=bool(dynamic), explicit=explicit)


def apply_base_drive(robot_cfg, actions_cfg, action_name, *, settings=None) -> bool:
    """Point one robot/action pair at the resolved base model.

    Returns whether the dynamic base was applied. A configuration that cannot
    move its base keeps the fixed root, because the base model only decides how
    base motion is produced: with no base action, or no wheeled chassis, there
    is nothing to decide. Asking for it explicitly there is reported instead of
    being ignored.
    """
    from .robot_model import resolve_robot_model

    settings = resolve_base_drive_settings() if settings is None else settings
    if not settings.dynamic:
        return False
    if not resolve_robot_model().has_wheel_base:
        if settings.explicit:
            raise ValueError(
                "--dynamic-base drives a wheeled chassis; this robot model has none."
            )
        return False
    action = getattr(actions_cfg, action_name, None)
    if action is None:
        if settings.explicit:
            raise ValueError(
                f"--dynamic-base needs the '{action_name}' base action; this "
                "configuration locks the base instead."
            )
        return False
    if robot_cfg.spawn.articulation_props is None:
        raise ValueError("--dynamic-base needs articulation properties on the robot spawn.")
    robot_cfg.spawn.articulation_props.fix_root_link = False
    action.dynamic = True
    return True
