"""Opt-in physical noise calibration and stronger actual-success retention.

Wide deterministic URDF goals and coherent TRAIN arm biases remain available.
Small Gaussian arm perturbations retain the source goal's physical scale;
success labels always remain the exact bounded goals actually executed.
"""
import math

import torch
from torch.nn import functional as F

from .actual_flap_residual_sac import BoundedCorrectionHybridSAC
from .servo_success_retention import (
    ServoRetainedCorrectionSAC, servo_interval_loss, servo_success_retention_contract,
)
from .staged_train_success import retention_config
from .urdf_regional_goal_sac import URDFRegionalGoalSACPilot, validate_urdf_regional_state

SERVO_GUARD_COEFFICIENT = .1


def calibrated_body_noise_scales(source_scale, urdf_scale):
    target = torch.as_tensor(urdf_scale)
    if not target.is_floating_point():
        target = target.float()
    source = torch.as_tensor(source_scale, dtype=target.dtype, device=target.device)
    if source.shape != (21,) or target.shape != (21,) or not torch.isfinite(source).all() \
            or not torch.isfinite(target).all() or (source <= 0).any() or (target <= 0).any():
        raise ValueError('Finite matching original and URDF21 goal spans required')
    result = torch.ones_like(target[:19])
    result[1:15] = (source[1:15] / target[1:15]).clamp(max=1.)
    return result


def servo_guard_retention_contract():
    return servo_success_retention_contract() | dict(
        name='measured_servo_equivalent_success_retention_guard_v1',
        servo_interval_Huber_coefficient=SERVO_GUARD_COEFFICIENT)


def servo_guard_contract():
    return dict(name='URDF_arm_physical_Gaussian_scale_and_success_servo_guard_v1',
        quarter_std_config_is_base_before_arm_calibration=True,
        Gaussian_arm_scales='min_source_goal_halfspan_over_URDF_halfspan_and_one',
        continuous_noise_dimensions=19, calibrated_arm_columns=list(range(1, 15)),
        non_arm_noise_scales_unchanged=True,
        same_calibration_in_collection_actor_objective_and_all_target_Q_branches=True,
        Gaussian_log_probability_and_attainable_entropy_target_calibrated=True,
        initial_greedy_body_and_binary_jaws_unchanged=True,
        wide_mean_goal_bounds_and_coherent_TRAIN_arm_bias_unchanged=True,
        success_servo_interval_coefficient=SERVO_GUARD_COEFFICIENT,
        success_actor_sampling='tail64-half', success_Q_sampling_unchanged=True,
        actual_completed_TRAIN_labels_only=True, curriculum=False,
        physical_decoder_reward_DR_success_safety_unchanged=True)


class ServoGuardCorrectionSAC(ServoRetainedCorrectionSAC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('body_std_calibration', torch.ones(19, device=self.log_alpha.device))

    def parameters_at(self, normalized):
        mean, log_std, logits = super().parameters_at(normalized)
        return mean, log_std + self.body_std_calibration.log(), logits

    def continuous_entropy_target(self, normalized, raw=None):
        mean, _, _ = self.parameters_at(normalized)
        cap = self.config.max_policy_std * self.body_std_calibration.to(mean)
        per_dim = torch.minimum(cap.log() + .5 * math.log(2 * math.pi * math.e) - .5,
                                torch.full_like(cap, -1.)).expand_as(mean)
        per_dim = per_dim + 2 * (math.log(2) - mean - F.softplus(-2 * mean)) - cap.square()
        return self.body_entropy_target(normalized, per_dim, raw)

    def success_body_loss(self, raw, requested_body, labels):
        original = BoundedCorrectionHybridSAC.success_body_loss(self, raw, requested_body, labels)
        body = self.body_from_latent(self.actor_normalizer(self.actor_features(raw)), requested_body, raw)
        goals = torch.cat((body, labels[:, 19:]), -1)
        interval = servo_interval_loss(self.goal_servo_critic_encoder, raw, goals, labels)
        self._success_servo_statistics = dict(success_absolute_goal_MSE=original.detach().item(),
            success_servo_interval_Huber=interval.detach().item(),
            success_servo_interval_coefficient=SERVO_GUARD_COEFFICIENT)
        return original + SERVO_GUARD_COEFFICIENT * interval

    def update(self, *args, **kwargs):
        report = super().update(*args, **kwargs)
        if self._success_servo_statistics:
            report['success_servo_interval_weight'] = report['success_goal_weight'] * SERVO_GUARD_COEFFICIENT
        return report

    @property
    def hybrid_contract(self):
        return super().hybrid_contract | dict(success_body_retention=servo_guard_retention_contract(),
            URDF_servo_guard=servo_guard_contract())


class URDFServoGuardSACPilot(URDFRegionalGoalSACPilot):
    artifact_type = 'staged_actual_flap_URDF_regional_goal_servo_guard_tail_hybrid_sac_v1'
    agent_class = ServoGuardCorrectionSAC

    def validate_saved_state(self, state):
        validate_urdf_servo_guard_state(state)

    def configure_controller(self, saved):
        super().configure_controller(saved)
        with torch.no_grad():
            self.agent.body_std_calibration.copy_(
                calibrated_body_noise_scales(self._source_goal_scale, self.scale))
        if not saved:
            self.success_bank.config = retention_config('tail64-half')
        elif self.success_bank.config != retention_config('tail64-half'):
            raise ValueError('Servo guard continuation needs its own real-success sampling contract')

    @property
    def contract(self):
        result = super().contract
        if self._URDF_configuring_source:
            return result
        return result | dict(URDF_servo_guard=servo_guard_contract(),
            calibrated_body_noise_scales=self.agent.body_std_calibration.tolist(),
            success_body_retention=servo_guard_retention_contract())

    def report(self):
        config = self.agent.config
        return super().report() | dict(URDF_servo_guard=servo_guard_contract(),
            calibrated_body_noise_scales=self.agent.body_std_calibration.tolist(),
            effective_continuous_std_min=(config.min_policy_std * self.agent.body_std_calibration).tolist(),
            effective_continuous_std_max=(config.max_policy_std * self.agent.body_std_calibration).tolist())


def validate_urdf_servo_guard_state(state, *, artifact_type=None):
    validate_urdf_regional_state(state, artifact_type=artifact_type or URDFServoGuardSACPilot.artifact_type)
    contract = state['goal_contract']
    scales = calibrated_body_noise_scales(contract['source_goal_scale'], contract['goal_scale'])
    saved_scales = state['model'].get('body_std_calibration', torch.empty(0))
    if contract.get('URDF_servo_guard') != servo_guard_contract() \
            or contract.get('success_body_retention') != servo_guard_retention_contract() \
            or contract.get('calibrated_body_noise_scales') != scales.tolist() \
            or not torch.equal(saved_scales, scales.to(saved_scales)) \
            or contract.get('train_success_retention') != retention_config('tail64-half') \
            or state.get('successful_train_transitions', {}).get('config') != retention_config('tail64-half') \
            or state.get('hybrid_contract', {}).get('URDF_servo_guard') != servo_guard_contract() \
            or state.get('hybrid_contract', {}).get('success_body_retention') != servo_guard_retention_contract():
        raise ValueError('Saved URDF servo guard noise, entropy or real-success objective differs')
