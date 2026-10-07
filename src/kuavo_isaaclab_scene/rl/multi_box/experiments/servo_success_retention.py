"""Retain real successful drive commands without inverting clipped goals."""
import torch
from torch.nn import functional as F

from .actual_flap_residual_sac import BoundedCorrectionHybridSAC
from .gentle_servo_critic_sac import GentleServoCriticSACPilot

SERVO_COEFFICIENT=.01
HUBER_BETA=1.


def servo_success_retention_contract():
    return dict(name='measured_servo_equivalent_success_retention_v1',
        goal_MSE_preserved=True,servo_interval_Huber_coefficient=SERVO_COEFFICIENT,Huber_beta=HUBER_BETA,
        continuous_dimensions=19,measured_pending_targets_and_production_step_scales=True,
        saturated_label_defines_feasible_half_line=True,
        wrong_clipped_goal_has_nonzero_retention_gradient=True,
        source='same_completed_safe_TRAIN_success_bank_only',
        actual_reward_Q_targets_action_sampler_and_controller_unchanged=True,
        evaluation_or_VR_teacher_rows_imported=False)


def servo_interval_loss(encoder,raw,predicted_goals,recorded_goals):
    predicted=encoder.unclipped_body(raw,predicted_goals)
    recorded=encoder(raw,recorded_goals)[:,:19].detach()
    # A +1 command means any pre-clip delta >=1, and -1 means <=-1.
    # Unlike comparing clipped commands, this still has an escape gradient
    # when the predicted command saturates on the wrong side of that set.
    error=torch.where(recorded>=1,(1-predicted).clamp_min(0),
        torch.where(recorded<=-1,(predicted+1).clamp_min(0),predicted-recorded))
    return F.smooth_l1_loss(error,torch.zeros_like(error),beta=HUBER_BETA)


class ServoRetainedCorrectionSAC(BoundedCorrectionHybridSAC):
    def success_body_loss(self,raw,requested_body,labels):
        original=super().success_body_loss(raw,requested_body,labels)
        body=self.body_from_latent(self.actor_normalizer(self.actor_features(raw)),requested_body,raw)
        goals=torch.cat((body,labels[:,19:]),-1)
        interval=servo_interval_loss(self.goal_servo_critic_encoder,raw,goals,labels)
        self._success_servo_statistics=dict(success_absolute_goal_MSE=original.detach().item(),
            success_servo_interval_Huber=interval.detach().item(),
            success_servo_interval_coefficient=SERVO_COEFFICIENT)
        return original+SERVO_COEFFICIENT*interval

    def update(self,*args,**kwargs):
        self._success_servo_statistics={}
        report=super().update(*args,**kwargs)
        if self._success_servo_statistics:
            report.update(self._success_servo_statistics,
                success_servo_interval_weight=report['success_goal_weight']*SERVO_COEFFICIENT)
        return report

    @property
    def hybrid_contract(self):
        return super().hybrid_contract|dict(success_body_retention=servo_success_retention_contract())


class ServoRetentionGentleSACPilot(GentleServoCriticSACPilot):
    artifact_type='staged_actual_flap_reanchored_gentle_servo_retention_hybrid_sac_v1'
    agent_class=ServoRetainedCorrectionSAC

    @property
    def contract(self):
        return super().contract|dict(
            success_body_labels='measured_absolute_goal_MSE_plus_servo_equivalent_interval_Huber_v1',
            success_body_retention=servo_success_retention_contract())
