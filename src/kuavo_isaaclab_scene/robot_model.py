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
ROBOT_MODEL_NAMES = ("s200062", "s63", "s56")

WHEEL_BODY_JOINT_NAMES = (
    "knee_joint",
    "leg_joint",
    "waist_pitch_joint",
    "waist_yaw_joint",
)
S56_LEG_JOINT_NAMES = tuple(
    f"leg_{side}{index}_joint" for side in ("l", "r") for index in range(1, 7)
)


@dataclass(frozen=True)
class RobotModelSettings:
    name: str
    usd_path: str
    urdf_path: str
    integrated_gripper_preset: str | None
    default_gripper_preset: str
    spawn_height_m: float
    has_wheel_base: bool
    tool_forward_sign: int
    head_camera_body: str
    head_camera_mount: WristCameraMount
    wrist_camera_bodies: dict[str, str]
    wrist_camera_mounts: dict[str, WristCameraMount]

    @property
    def has_integrated_grippers(self) -> bool:
        return self.integrated_gripper_preset is not None

    @property
    def teleop_body_joint_names(self) -> tuple[str, ...]:
        if self.has_wheel_base:
            return WHEEL_BODY_JOINT_NAMES
        return (*S56_LEG_JOINT_NAMES, "waist_yaw_joint")

    def spawn_position(self, layout_position: tuple[float, float, float]) -> tuple[float, float, float]:
        return (
            layout_position[0],
            layout_position[1],
            layout_position[2] + self.spawn_height_m,
        )


_MODELS = {
    "s200062": RobotModelSettings(
        name="s200062",
        usd_path=str(ASSET_DIR / "kuavo_s200062" / "usd" / "kuavo_s200062_fixed.usd"),
        urdf_path=str(ASSET_DIR / "kuavo_s200062" / "urdf" / "biped_s200062.urdf"),
        integrated_gripper_preset="s200062_integrated",
        default_gripper_preset="s200062_integrated",
        spawn_height_m=0.0,
        has_wheel_base=True,
        tool_forward_sign=-1,
        # These links are the source URDF's physical camera reference frames.
        head_camera_body="camera",
        head_camera_mount=WristCameraMount(
            pos=(0.0, 0.0, 0.0),
            # URDF camera is a body frame (+X forward), not ROS optical +Z.
            rot=CAMERA_BODY_TO_ROS_OPTICAL_ROT,
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
        urdf_path=str(ASSET_DIR / "kuavo_s63" / "urdf" / "kuavo_s63.urdf"),
        integrated_gripper_preset=None,
        default_gripper_preset="robotiq_2f85",
        spawn_height_m=0.0,
        has_wheel_base=True,
        tool_forward_sign=1,
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
    "s56": RobotModelSettings(
        name="s56",
        usd_path=str(ASSET_DIR / "kuavo_s56" / "usd" / "kuavo_s56_fixed.usd"),
        urdf_path=str(ASSET_DIR / "kuavo_s56" / "urdf" / "kuavo_s56.urdf"),
        integrated_gripper_preset=None,
        default_gripper_preset="robotiq_2f85",
        # The source is a biped whose root is the torso rather than a
        # ground-aligned wheel chassis. Its MuJoCo home pose uses z=0.98 m.
        spawn_height_m=0.98,
        has_wheel_base=False,
        # The runtime's empty end-effector frames are adapted so mounted
        # Robotiq +Z points outward, matching the S63 policy convention.
        tool_forward_sign=1,
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
        help="Packaged Kuavo model to run (default: s200062; comparison: s63 or s56).",
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
    """Return the model-specific gripper preset used when CLI selection is omitted."""
    return model.default_gripper_preset


def validate_robot_gripper(model: RobotModelSettings, gripper_name: str) -> None:
    integrated = model.integrated_gripper_preset
    if integrated is not None and gripper_name not in (integrated, "none"):
        raise ValueError(
            f"Robot {model.name!r} already contains its grippers; use "
            f"{integrated!r} (default) or 'none', not {gripper_name!r}."
        )
    if integrated is None and gripper_name == "s200062_integrated":
        raise ValueError("The s200062_integrated gripper is part of the S200062 robot USD.")
