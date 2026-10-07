"""Penalize optimistic counterfactual Q relative to actual online TRAIN actions.

This finite-mixture CQL-inspired penalty is an explicitly separate experiment.
Candidates are critic queries, never fabricated simulator/replay transitions.
"""
import math
from copy import deepcopy

import torch

from .regional_actor_servo import RegionalActorMemorySACPilot
from .servo_success_retention import ServoRetainedCorrectionSAC

WEIGHT=1.
TEMPERATURE=1.
DRAWS_PER_SOURCE=4


def support_conservative_contract():
    return dict(name='finite_command_support_conservative_critic_v1',weight=WEIGHT,
        temperature=TEMPERATURE,uniform_body_draws=DRAWS_PER_SOURCE,
        current_policy_body_draws=DRAWS_PER_SOURCE,physical_binary_jaw_branches=4,
        source='same_actual_online_TRAIN_Q_batch_observations_and_executed_actions',
        penalty='sum_two_Q_temperature_logmeanexp_candidate_Q_minus_actual_data_Q',
        candidates='existing_bounded_body_correction_and_production_near_flap_jaw_gate',
        critic_uses_existing_servo_command_encoding=True,
        candidates_not_replay_or_reward_labels=True,no_policy_gradient_in_critic_penalty=True,
        actor_entropy_retention_targets_reward_and_randomization_unchanged=True,
        offline_CQL_theoretical_lower_bound_NOT_claimed=True)


class SupportConservativeHybridSAC(ServoRetainedCorrectionSAC):
    @torch.no_grad()
    def support_candidate_actions(self, raw):
        n=len(raw);draws=DRAWS_PER_SOURCE
        repeated=raw[:,None].expand(-1,draws,-1).reshape(n*draws,-1)
        normalized=self.actor_normalizer(self.actor_features(repeated))
        uniform=self.body_from_latent(normalized,torch.rand(n*draws,19,device=raw.device)*2-1,repeated)
        policy=self.continuous_sample(normalized,raw=repeated)[0]
        body=torch.cat((uniform.reshape(n,draws,19),policy.reshape(n,draws,19)),1)
        states=raw[:,None].expand(-1,2*draws,-1).reshape(n*2*draws,-1)
        # Only the physically projected commands are queried. Enumeration
        # duplicates the sole open command for far jaws; its normalized mean
        # leaves those duplicate values unchanged.
        commands=self.enumerate_jaws(states,body.reshape(n*2*draws,19),
                                    raw.new_zeros(n*2*draws,2))[0]
        return commands.reshape(n,2*draws*4,21).detach()

    def critic_regularization(self, batch, normalized_critic, replay_values):
        raw=batch['actor_obs'];commands=self.support_candidate_actions(raw)
        n,count,_=commands.shape
        repeated=raw[:,None].expand(-1,count,-1).reshape(n*count,-1)
        encoded=self.critic_action_features(repeated,commands.reshape(n*count,21))
        features=torch.cat((normalized_critic[:,None].expand(-1,count,-1).reshape(n*count,-1),encoded),-1)
        candidate=[q(features).reshape(n,count) for q in (self.q1,self.q2)]
        gaps=[TEMPERATURE*(torch.logsumexp(values/TEMPERATURE,-1)-math.log(count))-observed
              for values,observed in zip(candidate,replay_values)]
        loss=WEIGHT*sum(gap.mean() for gap in gaps)
        return loss,dict(critic_support_loss=float(loss.detach()),critic_support_weight=WEIGHT,
            critic_support_candidates_per_state=count,critic_support_actual_TRAIN_rows=n,
            critic_support_Q1_gap=float(gaps[0].detach().mean()),
            critic_support_Q2_gap=float(gaps[1].detach().mean()),
            critic_support_synthetic_replay_rows=0)

    @property
    def hybrid_contract(self):
        return super().hybrid_contract|dict(critic_support_regularization=support_conservative_contract())


class SupportConservativeRegionalSACPilot(RegionalActorMemorySACPilot):
    artifact_type='staged_actual_flap_regional_support_conservative_sac_v1'
    agent_class=SupportConservativeHybridSAC

    def configure_controller(self, saved):
        super().configure_controller(saved)
        if saved['goal_contract'].get('critic_support_regularization')!=support_conservative_contract():
            raise ValueError('Conservative support must be explicit in the saved learner contract')
        self.support_initialization=deepcopy(saved.get('support_conservative_initialization'))
        if (not isinstance(self.support_initialization,dict)
            or self.support_initialization.get('kind')!='pristine_regional_critic_support_only_v1'):
            raise ValueError('Conservative support requires its pristine initialization provenance')

    def checkpoint_extras(self):
        return super().checkpoint_extras()|dict(support_conservative_initialization=deepcopy(self.support_initialization))

    def experience_extras(self):
        return super().experience_extras()|dict(support_conservative_initialization=deepcopy(self.support_initialization))

    def restore_experience_extras(self, state):
        super().restore_experience_extras(state)
        if state.get('support_conservative_initialization')!=self.support_initialization:
            raise ValueError('Conservative support checkpoint and replay origins differ')

    @property
    def contract(self):
        return super().contract|dict(critic_support_regularization=support_conservative_contract())

    def report(self):
        return super().report()|dict(critic_support_regularization=support_conservative_contract())
