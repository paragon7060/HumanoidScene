"""Gravity feedforward for fixed-root Kuavo models, independent of input/task code."""

import json
import math
import os
from pathlib import Path
import re

import torch


GRAVITY_JOINT_PATTERN = r"(?:knee_joint|leg_joint|leg_[lr][1-6]_joint|waist_(?:pitch|yaw)_joint|zarm_[lr][1-7]_joint)"


def wbc_acceleration_profile(joint_names, device=None, dtype=torch.float32):
    """Return the active S63 wheel-WBC acceleration-task gains and limits.

    These are acceleration gains from task.info, not low-level motor gains.
    Unsupported joints stay at zero so they cannot inject dynamic feedforward.
    """
    kp = torch.zeros(len(joint_names), device=device, dtype=dtype)
    kd = torch.zeros_like(kp)
    limit = torch.zeros_like(kp)
    arm_kp = (300., 64., 64., 300., 70., 70., 70.)
    arm_kd = (18., 12., 12., 40., 30., 30., 30.)
    for index, name in enumerate(joint_names):
        if name in ("knee_joint", "leg_joint", "waist_pitch_joint", "waist_yaw_joint"):
            kp[index], kd[index], limit[index] = 30., 6.2, 20.
        else:
            match = re.fullmatch(r"zarm_[lr]([1-7])_joint", name)
            if match:
                arm_index = int(match.group(1)) - 1
                kp[index], kd[index], limit[index] = arm_kp[arm_index], arm_kd[arm_index], 300.
    return kp, kd, limit


def feedforward_joint_ids(joint_names, profile, kp):
    """Joints whose feedforward is the acceleration task instead of plain gravity.

    The selection follows the same profile that defines the gains, so a joint
    cannot carry WBC gains that nothing applies. A fixed joint PD holds the
    torso at a damping ratio that drops as the arms extend and as they pick a
    load; the inertia-normalized task keeps it constant. ``s63-arm-id`` keeps
    the torso on gravity-PD so the earlier behaviour stays reproducible.
    """
    ids = [index for index, value in enumerate(kp) if value > 0]
    arms = [index for index in ids if re.fullmatch(r"zarm_[lr][1-7]_joint", joint_names[index])]
    if len(arms) != 14:
        raise ValueError(f"{profile} requires both seven-joint S63 arms")
    return arms if profile == "s63-arm-id" else ids


def gravity_drive_bias(gravity, stiffness, limits, joint_ids):
    """Return a solver-only target bias, without changing commanded posture.

    For an implicit force drive, Kp*(q_cmd + g/Kp - q) adds gravity torque
    inside the existing drive force cap. Direct external torques would bypass
    that cap. Locked joints and zero-stiffness drives receive no bias.
    """
    if gravity.shape != stiffness.shape or limits.shape != (*stiffness.shape, 2):
        raise ValueError("Gravity compensation needs joint-aligned force/stiffness/limit tensors")
    if not torch.isfinite(gravity).all() or not torch.isfinite(stiffness).all():
        raise ValueError("Non-finite robot gravity compensation or stiffness")
    selected = torch.zeros_like(stiffness, dtype=torch.bool)
    selected[:, joint_ids] = True
    selected &= (stiffness > 0) & ((limits[..., 1] - limits[..., 0]) > 1e-3)
    return torch.where(selected, gravity / stiffness.clamp_min(torch.finfo(stiffness.dtype).tiny), 0.)


def gravity_joint_ids(joint_names):
    return [i for i, name in enumerate(joint_names) if re.fullmatch(GRAVITY_JOINT_PATTERN, name)]


def configure_gravity_compensation(cfg, model):
    """Select the common writer while leaving all servo gains configurable."""
    from .gravity_articulation import GravityCompensatedArticulation
    from isaaclab.sim import JointDrivePropertiesCfg

    cfg.class_type = GravityCompensatedArticulation
    # g/Kp has torque units only for force drives, not acceleration drives.
    drive = cfg.spawn.joint_drive_props
    cfg.spawn.joint_drive_props = drive.copy() if drive is not None else JointDrivePropertiesCfg()
    cfg.spawn.joint_drive_props.drive_type = "force"
    # Other models retain their configured gains and per-motor limits. The
    # S63 profile belongs to the host, regardless of its gripper selection.
    if model.name == "s63":
        for name, gains in load_s63_servo_gains().items():
            cfg.actuators[name].stiffness = gains["stiffness"]
            cfg.actuators[name].damping = gains["damping"]


def load_s63_servo_gains():
    """Mutable simulation gains, not raw EtherCAT register values."""
    from ..core.paths import CONFIG_DIR, PACKAGE_CONFIG_DIR

    override = os.environ.get("KUAVO_S63_SERVO_CONFIG")
    path = Path(override).expanduser() if override else CONFIG_DIR / "s63_servo.json"
    if not override and not path.is_file():
        path = PACKAGE_CONFIG_DIR / "s63_servo.json"
    profile = json.loads(path.read_text())
    gains = profile["actuators"]
    if set(gains) != {"height_axis", "arms", "upper_body"}:
        raise ValueError("S63 servo config requires height_axis, arms and upper_body")
    for group in gains.values():
        if set(group) != {"stiffness", "damping"}:
            raise ValueError("S63 servo config requires stiffness and damping")
        for key, value in group.items():
            values = list(value.values()) if isinstance(value, dict) else [value]
            if not values or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                                 or v < 0 or (key == "stiffness" and v == 0) for v in values):
                raise ValueError("S63 stiffness must be positive and damping nonnegative finite gains")
    return gains
