"""Absolute bounded upright X/Z and arm goals, with matched SAC densities.

The source is an actor-only prior. New Q/replay remain fresh. Software torso
travel bounds are intersected with the existing goal affine range; no extra
physical travel, pitch, base action, reward or success relaxation is introduced.
"""
import math

import torch
from torch.nn import functional as F

from .urdf_full_arm_sac import ANCHOR_MARGIN
from .urdf_servo_guard_sac import ServoGuardCorrectionSAC, validate_urdf_servo_guard_state
from .urdf_strong_success_sac import STRONG_SERVO_COEFFICIENT
from .urdf_perceived_contact_sac import URDFMotionFeedbackSACPilot
from .body_behavior_exploration import GENTLE_GREEDY_REST_VARIANT, body_behavior_config
from .perceived_contact_exploration import perceived_contact_contract, contact_statistics

ABSOLUTE_COLUMNS = (*range(1, 15), 17, 18)
TORSO_ANCHOR_MARGIN = .01


def upright_support_affine(center, scale, links, height_range=(0., .46)):
    center = torch.as_tensor(center)
    scale, links = [torch.as_tensor(x).to(center) for x in (scale, links)]
    if center.shape != (21,) or scale.shape != (21,) or links.shape != (2, 2) \
            or not all(torch.isfinite(x).all() for x in (center, scale, links)) \
            or (scale <= 0).any() or tuple(height_range) != (0., .46):
        raise ValueError('Reviewed upright travel and finite goal normalization required')
    low, high = center[17:19] - scale[17:19], center[17:19] + scale[17:19]
    low, high = low.clone(), high.clone()
    nominal_z = links[:, 1].sum()
    low[1] = torch.maximum(low[1], nominal_z + height_range[0])
    high[1] = torch.minimum(high[1], nominal_z + height_range[1])
    if not (low < high).all():
        raise ValueError('Empty physical upright goal support')
    lower, upper = (low - center[17:19]) / scale[17:19], (high - center[17:19]) / scale[17:19]
    return (lower + upper) / 2, (upper - lower) / 2


def upright_support_contract():
    return dict(name='source_centered_absolute_URDF_arms_and_bounded_upright_XZ_v5',
        absolute_body_columns=list(ABSOLUTE_COLUMNS), upright_goal_columns=[17, 18],
        torso_bounds='intersection_of_existing_goal_affine_and_reviewed_absolute_software_Z',
        reviewed_relative_height_range_m=[0., .46], extra_physical_travel=False,
        torso_source_interior_margin=TORSO_ANCHOR_MARGIN,
        source_outside_software_support_projected=True,
        torso_margin_displacement='one_percent_of_attainable_half_range_at_boundary_only',
        other_body_columns_original_symmetric_correction=True,
        mean='atanh_source_in_attainable_interval_plus_local_residual',
        local_scale='original_radius_in_interval_coordinates_over_one_minus_origin_squared',
        density_and_entropy='tanh_plus_physical_support_affine_Jacobian',
        raw_source_required=True, collection_actor_target_Q_success_and_eval_same_distribution=True,
        actual_absolute_goal_servo_Q_decoder_unchanged=True,
        reset_pitch_base_head_DR_reward_success_safety_preserved=True,
        fresh_Q_replay_success_banks=True, curriculum=False)


class UprightServoGuardSAC(ServoGuardCorrectionSAC):
    success_servo_interval_coefficient = STRONG_SERVO_COEFFICIENT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('upright_support_anchor', torch.zeros_like(self.body_std_calibration[:2]))
        self.register_buffer('upright_support_scale', torch.ones_like(self.body_std_calibration[:2]))

    def configure_upright_support(self, center, scale, links):
        anchor, radius = upright_support_affine(center, scale, links)
        self.upright_support_anchor.copy_(anchor.to(self.upright_support_anchor))
        self.upright_support_scale.copy_(radius.to(self.upright_support_scale))

    def absolute_origin_and_local_scale(self, raw):
        if raw is None:
            raise ValueError('Absolute upright distribution requires measured raw source')
        with torch.no_grad():
            source = self.executed_body_anchor(raw)[:, ABSOLUTE_COLUMNS]
            if not torch.isfinite(source).all() or (source.abs() > 1).any():
                raise ValueError('Finite original normalized source goals required')
            affine_anchor = torch.cat((source.new_zeros(14), self.upright_support_anchor.to(source)))
            affine_scale = torch.cat((source.new_ones(14), self.upright_support_scale.to(source)))
            margin = torch.cat((source.new_full((14,), ANCHOR_MARGIN),
                                source.new_full((2,), TORSO_ANCHOR_MARGIN)))
            origin = ((source - affine_anchor) / affine_scale).clamp(-1 + margin, 1 - margin)
            available = torch.minimum(1 - origin.abs(), self.correction_radius / affine_scale)
            local = available / (1 - origin.square())
        return origin, local

    def continuous_parameters(self, normalized, raw=None):
        mean, log_std, logits = self.parameters_at(normalized)
        origin, local = self.absolute_origin_and_local_scale(raw)
        mean, log_std = mean.clone(), log_std.clone()
        mean[:, ABSOLUTE_COLUMNS] = origin.atanh() + local * mean[:, ABSOLUTE_COLUMNS]
        log_std[:, ABSOLUTE_COLUMNS] += local.log()
        return mean, log_std, logits

    def continuous_sample(self, normalized, *, deterministic=False, noise=None,
                          body_latent_offset=None, raw=None):
        if body_latent_offset is not None:
            _, local = self.absolute_origin_and_local_scale(raw)
            body_latent_offset = body_latent_offset.clone()
            body_latent_offset[:, ABSOLUTE_COLUMNS] *= local
        return super().continuous_sample(normalized, deterministic=deterministic,
            noise=noise, body_latent_offset=body_latent_offset, raw=raw)

    def anchor_and_scale(self, raw):
        anchor, scale = super().anchor_and_scale(raw)
        anchor, scale = anchor.clone(), scale.clone()
        anchor[:, 1:15], scale[:, 1:15] = 0., 1.
        anchor[:, 17:19], scale[:, 17:19] = self.upright_support_anchor, self.upright_support_scale
        return anchor, scale

    def continuous_entropy_target(self, normalized, raw=None):
        mean, _, _ = self.continuous_parameters(normalized, raw)
        _, local = self.absolute_origin_and_local_scale(raw)
        calibration = self.body_std_calibration.to(mean).expand_as(mean).clone()
        calibration[:, ABSOLUTE_COLUMNS] *= local
        cap = self.config.max_policy_std * calibration
        per_dim = torch.minimum(cap.log() + .5 * math.log(2 * math.pi * math.e) - .5,
                                torch.full_like(cap, -1.))
        per_dim += 2 * (math.log(2) - mean - F.softplus(-2 * mean)) - cap.square()
        return self.body_entropy_target(normalized, per_dim, raw)

    @property
    def hybrid_contract(self):
        return super().hybrid_contract | dict(URDF_upright_support=upright_support_contract(),
            body_sample_coordinates='absolute_URDF_arms_upright_XZ_and_other_bounded_corrections_v5',
            entropy_jacobian='tanh_and_attainable_torso_affine_and_other_available_radius')


class URDFUprightContactSACPilot(URDFMotionFeedbackSACPilot):
    artifact_type = 'staged_actual_flap_URDF_upright_contact_exploration_sac_v5'
    agent_class = UprightServoGuardSAC
    upright_feedback = True

    def configure_controller(self, saved):
        super().configure_controller(saved)
        if self.physical_contract.get('action_contract') != 's63_upright_torso_xz_fixed_pitch_diagnostic_up_0.0600m':
            raise ValueError('Upright support requires the existing reviewed six-centimeter extension')
        self.agent.configure_upright_support(self.center, self.scale, self.coordinates.links)

    def validate_saved_state(self, state):
        validate_upright_contact_state(state)
        self._contact_statistics = contact_statistics(state['perceived_contact_statistics'])

    @property
    def contract(self):
        result = super().contract
        if self._URDF_configuring_source:
            return result
        result = {k: v for k, v in result.items() if k != 'URDF_full_arm_support'}
        return result | dict(URDF_upright_support=upright_support_contract(),
            body_controller='source_centered_absolute_URDF_arms_and_software_bounded_upright_XZ_v5',
            initial_body_mean='source_projected_into_physical_support_with_declared_interior_margin')

    def report(self):
        result = super().report()
        result.pop('URDF_full_arm_support', None)
        return result | dict(URDF_upright_support=upright_support_contract(),
                            effective_torso_std_includes_local_scale=True)


class URDFInteriorContactSACPilot(URDFUprightContactSACPilot):
    """Same v5 SAC distribution; TRAIN attempts target a 20mm inset point."""
    artifact_type = 'staged_actual_flap_URDF_upright_interior_contact_exploration_sac_v6'
    interior_contact = True

    def validate_saved_state(self, state):
        validate_interior_contact_state(state)
        self._contact_statistics = contact_statistics(state['perceived_contact_statistics'])


def validate_upright_contact_state(state, *, interior_contact=False):
    if type(interior_contact) is not bool:
        raise ValueError('Explicit interior-contact variant required')
    pilot_class = URDFInteriorContactSACPilot if interior_contact else URDFUprightContactSACPilot
    validate_urdf_servo_guard_state(state, artifact_type=pilot_class.artifact_type,
                                   servo_coefficient=STRONG_SERVO_COEFFICIENT)
    contract = state['goal_contract']
    expected = perceived_contact_contract(settled_close=True, precise_feedback=True,
                                          motion_feedback=True, upright_feedback=True, interior_contact=interior_contact)
    if contract.get('URDF_upright_support') != upright_support_contract() \
            or state['hybrid_contract'].get('URDF_upright_support') != upright_support_contract() \
            or contract.get('body_controller') != 'source_centered_absolute_URDF_arms_and_software_bounded_upright_XZ_v5' \
            or contract.get('TRAIN_perceived_contact_exploration') != expected \
            or state.get('body_behavior') != body_behavior_config(GENTLE_GREEDY_REST_VARIANT):
        raise ValueError('Saved upright distribution, selection or contact contract differs')
    from ....robots.robot_model import resolve_robot_model
    from ..geometry.upright_torso import torso_links_from_urdf
    links = torso_links_from_urdf(resolve_robot_model('s63', 'leju-twofinger').urdf_path)
    anchor, radius = upright_support_affine(contract['goal_center'], contract['goal_scale'], links)
    if contract['physical_contract'].get('action_contract') != 's63_upright_torso_xz_fixed_pitch_diagnostic_up_0.0600m' \
            or not torch.equal(state['model']['upright_support_anchor'].cpu(), anchor) \
            or not torch.equal(state['model']['upright_support_scale'].cpu(), radius):
        raise ValueError('Saved upright software bounds or affine density buffers differ')
    contact_statistics(state['perceived_contact_statistics'])


def validate_interior_contact_state(state):
    validate_upright_contact_state(state, interior_contact=True)
