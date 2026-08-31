"""Pure-Python selection for the packaged Kuavo robot variants."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os

from .paths import ASSET_DIR
from .wrist_camera_mount import (
    CAMERA_BODY_TO_ROS_OPTICAL_ROT,
    S63_ROBOTIQ_D405_MOUNTS,
    WristCameraMount,
)


ROBOT_MODEL_ENV = "KUAVO_ROBOT_MODEL"
DEFAULT_ROBOT_MODEL = "s200062"
ROBOT_MODEL_NAMES = ("s200062", "s63")


@dataclass(frozen=True)
class RobotModelSettings:
    name: str
    usd_path: str
    integrated_gripper_preset: str | None
    head_camera_body: str
    head_camera_mount: WristCameraMount
    wrist_camera_bodies: dict[str, str]
    wrist_camera_mounts: dict[str, WristCameraMount]

    @property
    def has_integrated_grippers(self) -> bool:
        return self.integrated_gripper_preset is not None


_MODELS = {
    "s200062": RobotModelSettings(
        name="s200062",
        usd_path=str(ASSET_DIR / "kuavo_s200062" / "usd" / "kuavo_s200062_fixed.usd"),
        integrated_gripper_preset="s200062_integrated",
        # These links are the source URDF's physical camera reference frames.
        head_camera_body="camera",
        head_camera_mount=WristCameraMount(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        wrist_camera_bodies={
            "left": "l_d405_camera",
            "right": "r_d405_camera",
        },
        wrist_camera_mounts={
            # Physical D405 links are camera body frames (+X forward).
            # Rotate the sensor's ROS optical +Z into that direction.
            "left": WristCameraMount((0.0, 0.0, 0.0), CAMERA_BODY_TO_ROS_OPTICAL_ROT),
            "right": WristCameraMount((0.0, 0.0, 0.0), CAMERA_BODY_TO_ROS_OPTICAL_ROT),
        },
    ),
    "s63": RobotModelSettings(
        name="s63",
        usd_path=str(ASSET_DIR / "kuavo_s63" / "usd" / "kuavo_s63_fixed.usd"),
        integrated_gripper_preset=None,
        head_camera_body="head_camera_base",
        head_camera_mount=WristCameraMount(
            pos=(0.08, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        wrist_camera_bodies={
            "left": "zarm_l7_end_effector",
            "right": "zarm_r7_end_effector",
        },
        wrist_camera_mounts=dict(S63_ROBOTIQ_D405_MOUNTS),
    ),
}


def add_robot_model_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--robot-model",
        choices=ROBOT_MODEL_NAMES,
        default=os.environ.get(ROBOT_MODEL_ENV, DEFAULT_ROBOT_MODEL),
        help="Packaged Kuavo model to run (default: s200062; use s63 for comparison).",
    )


def export_robot_model_cli(args: argparse.Namespace) -> None:
    os.environ[ROBOT_MODEL_ENV] = str(args.robot_model)


def resolve_robot_model(name: str | None = None) -> RobotModelSettings:
    selected = name or os.environ.get(ROBOT_MODEL_ENV, DEFAULT_ROBOT_MODEL)
    try:
        return _MODELS[selected]
    except KeyError as exc:
        choices = ", ".join(ROBOT_MODEL_NAMES)
        raise ValueError(f"Unknown robot model {selected!r}; available models: {choices}.") from exc


def default_gripper_for_model(model: RobotModelSettings) -> str | None:
    """Return a forced integrated preset, or ``None`` for config-file default."""
    return model.integrated_gripper_preset


def validate_robot_gripper(model: RobotModelSettings, gripper_name: str) -> None:
    integrated = model.integrated_gripper_preset
    if integrated is not None and gripper_name not in (integrated, "none"):
        raise ValueError(
            f"Robot {model.name!r} already contains its grippers; use "
            f"{integrated!r} (default) or 'none', not {gripper_name!r}."
        )
    if integrated is None and gripper_name == "s200062_integrated":
        raise ValueError("The s200062_integrated gripper is part of the S200062 robot USD.")
