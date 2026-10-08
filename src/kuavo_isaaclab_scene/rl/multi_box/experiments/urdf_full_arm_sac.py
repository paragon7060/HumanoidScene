"""Full arm goal support with source-preserving, locally calibrated SAC means.

Fourteen arms use absolute tanh goals, centered at the frozen source's atanh
goal. The source's symmetric radius calibrates local gradients/noise only;
it no longer removes valid arm goals. Other body coordinates retain their
original bounded correction and the physical servo decoder stays unchanged.
"""
import math

import torch
from torch.nn import functional as F

from .urdf_servo_guard_sac import (
    ServoGuardCorrectionSAC, URDFServoGuardSACPilot, validate_urdf_servo_guard_state,
)

ARM_COLUMNS = slice(1, 15)
ANCHOR_MARGIN = 1e-6


def full_arm_contract():
    return dict(name='source_centered_full_URDF_arm_tanh_support_v1',
        arm_columns=list(range(1, 15)), arm_goals='absolute_normalized_URDF_tanh',
        mean='atanh_frozen_source_plus_locally_scaled_learned_residual',
        local_scale='original_symmetric_available_radius_over_one_minus_anchor_squared',
        source_anchor_interior_margin=ANCHOR_MARGIN,
        local_scale_controls_noise_and_gradient_NOT_arm_goal_support=True,
        raw_measured_state_required_not_reconstructed_from_clipped_normalizer=True,
        same_raw_distribution_in_collection_actor_success_and_target_Q=True,
        tanh_probability_Jacobian_and_attainable_entropy_target_preserved=True,
        non_arm_symmetric_correction_and_binary_jaws_unchanged=True,
        actual_absolute_goal_and_servo_Q_coordinates_unchanged=True,
        physical_limits_reward_DR_success_safety_unchanged=True,
        fresh_initial_Q_replay_success_return_banks=True,
        teacher_or_evaluation_training_import=False, curriculum=False)


class FullArmServoGuardSAC(ServoGuardCorrectionSAC):
    def arm_origin_and_local_scale(self, raw):
        if raw is None:
            raise ValueError('Full arm distribution requires the measured raw actor state')
        with torch.no_grad():
            source = self.executed_body_anchor(raw)[:, ARM_COLUMNS]
            if not torch.isfinite(source).all() or (source.abs() > 1).any():
                raise ValueError('Frozen arm source must be finite bounded URDF goals')
            origin = source.clamp(-1 + ANCHOR_MARGIN, 1 - ANCHOR_MARGIN)
            available = (1 - origin.abs()).clamp(max=self.correction_radius)
            local = available / (1 - origin.square())
        return origin, local

    def continuous_parameters(self, normalized, raw=None):
        mean, log_std, logits = self.parameters_at(normalized)
        origin, local = self.arm_origin_and_local_scale(raw)
        mean = torch.cat((mean[:, :1], origin.atanh() + local * mean[:, ARM_COLUMNS],
                          mean[:, 15:]), -1)
        log_std = torch.cat((log_std[:, :1], log_std[:, ARM_COLUMNS] + local.log(),
                             log_std[:, 15:]), -1)
        return mean, log_std, logits

    def continuous_sample(self, normalized, *, deterministic=False, noise=None,
                          body_latent_offset=None, raw=None):
        if body_latent_offset is not None:
            # Coherent TRAIN offsets use the same local physical units as
            # the learned residual, while preserving full asymptotic support.
            _, local = self.arm_origin_and_local_scale(raw)
            body_latent_offset = torch.cat((body_latent_offset[:, :1],
                local * body_latent_offset[:, ARM_COLUMNS], body_latent_offset[:, 15:]), -1)
        return super().continuous_sample(normalized, deterministic=deterministic,
            noise=noise, body_latent_offset=body_latent_offset, raw=raw)

    def anchor_and_scale(self, raw):
        anchor, scale = super().anchor_and_scale(raw)
        # Affine change after tanh is identity for arms. Their source centering
        # is already included in the Gaussian mean, before tanh.
        return (torch.cat((anchor[:, :1], torch.zeros_like(anchor[:, ARM_COLUMNS]),
                           anchor[:, 15:]), -1),
                torch.cat((scale[:, :1], torch.ones_like(scale[:, ARM_COLUMNS]),
                           scale[:, 15:]), -1))

    def continuous_entropy_target(self, normalized, raw=None):
        mean, _, _ = self.continuous_parameters(normalized, raw)
        _, local = self.arm_origin_and_local_scale(raw)
        calibration = self.body_std_calibration.to(mean).expand_as(mean)
        calibration = torch.cat((calibration[:, :1], calibration[:, ARM_COLUMNS] * local,
                                 calibration[:, 15:]), -1)
        cap = self.config.max_policy_std * calibration
        per_dim = torch.minimum(cap.log() + .5 * math.log(2 * math.pi * math.e) - .5,
                                torch.full_like(cap, -1.))
        per_dim = per_dim + 2 * (math.log(2) - mean - F.softplus(-2 * mean)) - cap.square()
        return self.body_entropy_target(normalized, per_dim, raw)

    @property
    def hybrid_contract(self):
        return super().hybrid_contract | dict(URDF_full_arm_support=full_arm_contract(),
            body_sample_coordinates='absolute_URDF_arms_and_bounded_non_arm_corrections_v1',
            entropy_jacobian='arm_tanh_and_non_arm_available_affine_radius')


class URDFFullArmSACPilot(URDFServoGuardSACPilot):
    artifact_type = 'staged_actual_flap_URDF_full_arm_servo_guard_hybrid_sac_v1'
    agent_class = FullArmServoGuardSAC

    def validate_saved_state(self, state):
        validate_full_arm_state(state)

    @property
    def contract(self):
        result = super().contract
        if self._URDF_configuring_source:
            return result
        return result | dict(URDF_full_arm_support=full_arm_contract(),
            body_controller='source_centered_full_URDF_arms_plus_bounded_non_arm_correction_v1',
            initial_body_mean='source_arm_goals_with_interior_margin_and_original_non_arm_mean')

    def report(self):
        # Scalar/config noise ranges are base values; actual arm Gaussian
        # also includes the declared measured-state local Jacobian factor.
        result = super().report()
        result['base_calibrated_continuous_std_min'] = result.pop('effective_continuous_std_min')
        result['base_calibrated_continuous_std_max'] = result.pop('effective_continuous_std_max')
        return result | dict(URDF_full_arm_support=full_arm_contract(),
            effective_arm_std_includes_state_local_scale=True,
            arm_state_local_scale_bounds=[self.correction_radius, 1 / (2 - self.correction_radius)])


def validate_full_arm_state(state):
    validate_urdf_servo_guard_state(state, artifact_type=URDFFullArmSACPilot.artifact_type)
    contract = state['goal_contract']
    if contract.get('URDF_full_arm_support') != full_arm_contract() \
            or state.get('hybrid_contract', {}).get('URDF_full_arm_support') != full_arm_contract() \
            or contract.get('body_controller') != 'source_centered_full_URDF_arms_plus_bounded_non_arm_correction_v1':
        raise ValueError('Saved full-arm source centering, support or density contract differs')
