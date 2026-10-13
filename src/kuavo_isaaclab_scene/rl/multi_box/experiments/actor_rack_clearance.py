"""Explicit optional soft geometry objective for measured TRAIN actor proposals.

No default margin/weight is installed. A caller must choose an
activation distance from evidence and pass correctly decoded joint positions.
This neither changes environment rewards nor bypasses any physical collision
termination, controller clipping, or fixed successful-TRAIN guard.
"""
import math
from copy import deepcopy

import torch

from .physical_body_actions import servo_raw_actor

OBJECTIVE_NAME = 'sampled_clipped_arm_drive_target_rack_soft_cost_v1'


def actor_clearance_config(*, weight, activation_m, scale_m):
    """No implicit safety margin: all three objective values are explicit."""
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) or v <= 0 for v in (weight, activation_m, scale_m)):
        raise ValueError('Explicit finite positive clearance weight, activation and scale required')
    from .rack_entry_clearance import GEOMETRY_SHA256
    return dict(name=OBJECTIVE_NAME, weight=float(weight), activation_m=float(activation_m),
                scale_m=float(scale_m), geometry_SHA256=GEOMETRY_SHA256, chunk_rows=32)


def validate_actor_clearance_config(config):
    if config is None:
        return None
    if not isinstance(config, dict) or set(config) != {
            'name', 'weight', 'activation_m', 'scale_m', 'geometry_SHA256', 'chunk_rows'}:
        raise ValueError('Unknown actor rack clearance configuration')
    expected = actor_clearance_config(weight=config['weight'], activation_m=config['activation_m'],
                                      scale_m=config['scale_m'])
    if config != expected or type(config['chunk_rows']) is not int:
        raise ValueError('Actor rack geometry/version/chunk contract differs')
    return deepcopy(expected)


def explicit_actor_clearance_config(weight=None, activation_m=None, scale_m=None,
                                    *, training=False, has_train=False):
    """Validate the CLI before scene construction; absent options keep legacy."""
    values = (weight, activation_m, scale_m)
    if all(v is None for v in values):
        return None
    if not training or not has_train or any(v is None for v in values):
        raise ValueError('Actor clearance needs actual TRAIN and all three explicit objective values')
    return actor_clearance_config(weight=weight, activation_m=activation_m, scale_m=scale_m)


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


class ActorRackClearanceObjective:
    """Uses the already sampled/projected TRAIN action; no extra actor sample.

    It is deliberately not a Module: fixed asset geometry is neither another
    learned network nor an optimizer/state_dict participant. Construction is
    lazy, so restoring a frozen policy never creates a geometry graph.
    """
    def __init__(self, encoder, config, raw):
        from .servo_critic import BodyServoCriticEncoder
        from .tensor_arm_kinematics import TensorArmKinematics
        from .rack_entry_clearance import RackEntryClearance
        if type(encoder) is not BodyServoCriticEncoder:
            raise ValueError('Actor clearance requires the reviewed literal servo encoder')
        self.config = validate_actor_clearance_config(config)
        if self.config is None:
            raise ValueError('Disabled clearance must not construct geometry')
        self.encoder = encoder
        self.kinematics = TensorArmKinematics(device=raw.device, dtype=raw.dtype)
        self.geometry = RackEntryClearance(self.kinematics)

    def __call__(self, raw, actions):
        if raw.ndim != 2 or raw.shape[1] != 518 or not len(raw) \
                or actions.shape != (len(raw), 4, 21) or actions.device != raw.device \
                or actions.dtype != raw.dtype or not torch.isfinite(actions).all() \
                or not torch.equal(actions[:, :, :19], actions[:, :1, :19].expand(-1, 4, -1)):
            raise ValueError('Existing four jaw branches must share their actual sampled body')
        costs, distances = [], []
        for start in range(0, len(raw), self.config['chunk_rows']):
            part = raw[start:start+self.config['chunk_rows']].detach()
            goals = actions[start:start+self.config['chunk_rows'], 0]
            q = decoded_next_arm_targets(self.encoder, self.kinematics, part, goals)
            body_distances = self.geometry.differentiable_distances(q, part[:, 68:77])
            costs.append(soft_rack_clearance_rows(body_distances,
                activation_m=self.config['activation_m'], scale_m=self.config['scale_m']))
            distances.append(body_distances.detach().min(-1).values)
        per_row = torch.cat(costs)
        nearest = torch.cat(distances)
        loss = self.config['weight'] * per_row.mean()
        return loss, dict(actor_rack_clearance=dict(
            name=OBJECTIVE_NAME, weight=self.config['weight'],
            activation_m=self.config['activation_m'], scale_m=self.config['scale_m'],
            rows=len(raw), positive_cost_rows=int((per_row.detach() > 0).sum()),
            unweighted_cost_mean=float(per_row.detach().mean()), weighted_loss=float(loss.detach()),
            commanded_arm_target_proxy_distance_min_m=float(nearest.min()),
            sampled_same_body_four_jaw_branches=True, extra_policy_samples=0,
            reference='actual_clipped_arm_drive_goal_at_measured_torso_waist_head',
            physical_force_or_future_safety_certificate=False))
