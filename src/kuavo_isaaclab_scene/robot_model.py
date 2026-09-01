"""Pure-Python selection for the packaged Kuavo robot variants."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os

from .paths import ASSET_DIR
from .wrist_camera_mount import (
    CAMERA_BODY_TO_ROS_OPTICAL_ROT,
    S56_QIANGNAO_D405_MOUNTS,
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

# biped_s56 actuator data from the upstream kuavo-ros-opensource model.
# The URDF carries the per-motor limits while the accompanying MuJoCo XML
# applies the armature and Coulomb loss to every movable joint.  Keep these
# values independent of the generic scene gains: S56's light distal wrist
# links become numerically unstable when driven with the generic 100 Nm limit
# and zero reflected rotor inertia.
S56_MUJOCO_ARMATURE = 0.05
S56_MUJOCO_FRICTIONLOSS = 0.02
S56_ACTUATOR_LIMITS = {
    "lower_body": {
        "effort_limit_sim": {
            "leg_[lr]1_joint": 127.0,
            "leg_[lr]2_joint": 71.0,
            "leg_[lr]3_joint": 132.0,
            "leg_[lr]4_joint": 280.0,
            "leg_[lr]5_joint": 91.6,
            "leg_[lr]6_joint": 68.4,
        },
        "velocity_limit_sim": {
            "leg_[lr]1_joint": 10.4,
            "leg_[lr]2_joint": 8.7,
            "leg_[lr]3_joint": 12.7,
            "leg_[lr]4_joint": 10.4,
            "leg_[lr][56]_joint": 17.8,
        },
    },
    "arms": {
        "effort_limit_sim": {
            "zarm_[lr]1_joint": 66.0,
            "zarm_[lr]2_joint": 75.0,
            "zarm_[lr]3_joint": 57.0,
            "zarm_[lr]4_joint": 75.0,
            "zarm_[lr][567]_joint": 14.1,
        },
        "velocity_limit_sim": {
            "zarm_[lr]1_joint": 18.8,
            "zarm_[lr]2_joint": 8.0,
            "zarm_[lr]3_joint": 7.5,
            "zarm_[lr]4_joint": 8.0,
            "zarm_[lr][567]_joint": 17.5,
        },
    },
    "upper_body": {
        "effort_limit_sim": {
            "waist_yaw_joint": 102.0,
            "zhead_1_joint": 1.5,
            "zhead_2_joint": 12.0,
        },
        "velocity_limit_sim": {
            "waist_yaw_joint": 8.7,
            "zhead_[12]_joint": 5.23,
        },
    },
}


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
        integrated_gripper_preset="s56_qiangnao",
        default_gripper_preset="s56_qiangnao",
        # The source is a biped whose root is the torso rather than a
        # ground-aligned wheel chassis. Its MuJoCo home pose uses z=0.98 m.
        spawn_height_m=0.98,
        has_wheel_base=False,
        # The QiangNao fingers extend along the source wrist's -Z axis.
        tool_forward_sign=-1,
        head_camera_body="head_camera_base",
        head_camera_mount=WristCameraMount(
            pos=(0.08, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        wrist_camera_bodies={
            "left": "zarm_l7_end_effector",
            "right": "zarm_r7_end_effector",
        },
        wrist_camera_mounts=dict(S56_QIANGNAO_D405_MOUNTS),
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
    if integrated is None and gripper_name in ("s200062_integrated", "s56_qiangnao"):
        raise ValueError(f"The {gripper_name} gripper is part of its matching robot USD.")
