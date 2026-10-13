"""Dormant soft geometry primitive for a future reviewed TRAIN actor objective.

No learner hook/default margin/weight is installed. A caller must choose an
activation distance from evidence and pass correctly decoded joint positions.
This neither changes environment rewards nor bypasses any physical collision
termination, controller clipping, or fixed successful-TRAIN guard.
"""
import math

import torch

from .physical_body_actions import servo_raw_actor


def decoded_next_arm_targets(encoder, kinematics, raw, goals):
    """Arm drive targets after the existing real clipped delta command.

    This is a geometric reference, not next physical joint positions. Torso,
    waist and head stay measured in this reference. In particular, the guided
    explorer's separate 80mrad proposal limit is not imposed on general SAC
    actions. Only the servo command and actual URDF joint limits are clipped.
    """
    nominal, _ = encoder.physical_inputs(raw, goals)
    measured = servo_raw_actor(nominal)
    columns = sum(kinematics.columns, [])
    if encoder.coordinates.joints.joint_columns[1:15] != columns:
        raise ValueError('Reviewed S63 arm servo ordering required')
    command = encoder(raw, goals)[:, 1:15]
    scale = raw.new_tensor(encoder.coordinates.joints.scales[1:15])
    pending = measured[:, :20] + measured[:, 416:436]
    targets = pending[:, columns] + command * scale
    targets = targets.maximum(kinematics.lower.flatten()).minimum(kinematics.upper.flatten())
    q = measured[:, :20].clone()
    q[:, columns] = targets
    return q


def soft_rack_clearance_rows(distances, *, activation_m, scale_m):
    """Squared shortfall of the most exposed palm/forearm in each row.

    activation_m is a soft-cost onset, not a minimum-distance constraint or
    a safety classification. In particular, no universal 10mm margin is used.
    All derivatives follow the supplied differentiable geometry/decoder; the
    function uses no straight-through estimator or new state/policy samples.
    """
    if not isinstance(distances, torch.Tensor) or distances.ndim != 2 \
            or distances.shape[1] != 4 or not len(distances) \
            or not distances.is_floating_point() or not torch.isfinite(distances).all():
        raise ValueError('Finite nonempty four-body proxy distances required')
    if isinstance(activation_m, bool) or isinstance(scale_m, bool) \
            or not math.isfinite(activation_m) or not math.isfinite(scale_m) \
            or activation_m < 0 or scale_m <= 0:
        raise ValueError('Explicit finite nonnegative activation and positive scale required')
    nearest = distances.min(-1).values
    return ((activation_m - nearest).clamp_min(0) / scale_m).square()
