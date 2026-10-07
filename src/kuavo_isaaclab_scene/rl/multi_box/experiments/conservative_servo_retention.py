"""An explicit retained-servo SAC variant with a smaller shared actor step."""
from dataclasses import replace

from .servo_success_retention import ServoRetentionGentleSACPilot

BASE_ACTOR_LR = 1e-5
ACTOR_LR = 1e-6
CRITIC_ALPHA_LR = 3e-4


def conservative_actor_contract():
    return dict(name='retained_servo_actor_lr_tenth_v1',
        source_actor_learning_rate=BASE_ACTOR_LR, actor_learning_rate=ACTOR_LR,
        critic_and_entropy_learning_rate=CRITIC_ALPHA_LR,
        body_and_jaw_share_actor_optimizer=True,
        actor_Q_entropy_and_success_retention_objectives_unchanged=True,
        initial_policy_Gaussian_control_reward_and_randomization_unchanged=True,
        learned_optimizer_or_reward_data_imported=False)


def validate_conservative_actor_state(state, *, artifact_type=None):
    config = state.get('config', {})
    expected = artifact_type or ConservativeServoRetentionSACPilot.artifact_type
    if (state.get('artifact_type') != expected
            or state.get('goal_contract', {}).get('actor_update_step') != conservative_actor_contract()
            or (config.get('actor_lr'), config.get('lr')) != (ACTOR_LR, CRITIC_ALPHA_LR)):
        raise ValueError('Conservative actor checkpoint configuration differs')
    optimizers = state.get('optimizers', [])
    if len(optimizers) != 4 or any(
            not optimizer.get('param_groups') or any(group.get('lr') != rate
                for group in optimizer['param_groups'])
            for optimizer, rate in zip(optimizers, (ACTOR_LR, *([CRITIC_ALPHA_LR] * 3)))):
        raise ValueError('Conservative actor optimizer learning rates differ')


class ConservativeServoRetentionSACPilot(ServoRetentionGentleSACPilot):
    artifact_type = 'staged_actual_flap_conservative_servo_retention_hybrid_sac_v1'

    def __init__(self, *args, checkpoint=None, device='cpu', **kwargs):
        if checkpoint is not None:
            import torch
            validate_conservative_actor_state(torch.load(checkpoint, map_location=device, weights_only=True),
                                              artifact_type=self.artifact_type)
        super().__init__(*args, checkpoint=checkpoint, device=device, **kwargs)

    def learning_config(self, config):
        config = super().learning_config(config)
        if (config.actor_lr, config.lr) != (BASE_ACTOR_LR, CRITIC_ALPHA_LR):
            raise ValueError('Conservative actor variant requires the reviewed source learning rates')
        return replace(config, actor_lr=ACTOR_LR)

    @property
    def contract(self):
        return super().contract | dict(actor_update_step=conservative_actor_contract())

    def report(self):
        return super().report() | dict(actor_update_step=conservative_actor_contract(),
            actual_actor_optimizer_learning_rates=[g['lr'] for g in self.agent.actor_optimizer.param_groups],
            actual_critic_optimizer_learning_rates=[g['lr'] for g in self.agent.q_optimizer.param_groups])
