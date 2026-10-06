"""Fresh SAC corrections around a frozen, learned actual-flap body policy.

The source's executed body goals are preserved at zero correction. Its Q,
entropy, optimizers and transitions are never part of the frozen anchor.
"""
from copy import deepcopy

import torch

from ...algorithms.common import ObservationNormalizer
from ...algorithms.sac import SquashedActor
from .actual_flap_residual_sac import ActualFlapResidualSACPilot
from .staged_goal_sac import CONTEXT_DIM


def actual_actor_anchor_state(source):
    """Extract only the actual-flap actor and its actor-only nested anchor."""
    if source.get('artifact_type') != ActualFlapResidualSACPilot.artifact_type:
        raise ValueError('Reanchoring requires the original actual-flap correction actor')
    return dict(format_version=2, source_artifact_type=source['artifact_type'],
        source_actor_updates=source['actor_updates'],
        source_critic_updates_not_imported=source['critic_updates'],
        source_prior_schedule_actor_origin=source.get('prior_schedule_actor_origin', 0.),
        source_goal_contract=deepcopy(source['goal_contract']),
        source_hidden=source['config']['hidden'],
        model={k: v.detach().clone() for k, v in source['model'].items()
            if k.startswith(('actor.', 'actor_normalizer.'))},
        nominal_body_anchor=deepcopy(source['body_anchor_state']),
        frozen_actor_prior=deepcopy(source['frozen_actor_prior']),
        source_Q_replay_entropy_and_optimizers_imported=False)


def actual_actor_snapshot(snapshot, device):
    """Strictly load the full-perception actor, without loading source critics."""
    source = snapshot['source_goal_contract']
    model = torch.nn.ModuleDict(dict(
        actor=SquashedActor(source['actor_dim'], 21, snapshot['source_hidden']),
        actor_normalizer=ObservationNormalizer(source['actor_dim']))).to(device)
    expected = model.state_dict()
    if set(snapshot['model']) != set(expected) or any(
            v.shape != expected[k].shape or not torch.isfinite(v).all()
            for k, v in snapshot['model'].items()):
        raise ValueError('Malformed frozen actual-flap actor-only snapshot')
    model.load_state_dict(snapshot['model'])
    model.requires_grad_(False)
    return model


@torch.no_grad()
def source_actual_body_goal(raw, nominal_goal, actual_actor, source_radius):
    """Reproduce the source's single affine mapping, including bound shrinkage."""
    source_raw = raw.clone()
    source_raw[:, -1] = source_radius
    mean = actual_actor['actor'].network(
        actual_actor['actor_normalizer'](source_raw)).chunk(2, -1)[0][:, :19]
    scale = (1 - nominal_goal.abs()).clamp(min=0, max=source_radius)
    return nominal_goal + scale * mean.tanh()


class ReanchoredActualFlapSACPilot(ActualFlapResidualSACPilot):
    artifact_type = 'staged_actual_flap_reanchored_body_correction_hybrid_sac_v2'
    correction_radius = .30
    maximum_fixed_prior_radius = .30

    def nominal_view(self, raw):
        if raw.shape[1] != self.actor_dim:
            raise ValueError('Measured augmented actor width differs')
        result = torch.cat((raw[:, :self.warm_start.actor_dim], raw[:, -CONTEXT_DIM:]), -1).clone()
        result[:, -1] = self.body_anchor_state['nominal_body_anchor']['source_goal_contract']['fixed_prior_radius']
        return result

    def configure_controller(self, saved):
        snapshot = self.body_anchor_state
        required = {'format_version', 'source_artifact_type', 'source_actor_updates',
            'source_critic_updates_not_imported', 'source_prior_schedule_actor_origin',
            'source_goal_contract', 'source_hidden', 'model', 'nominal_body_anchor',
            'frozen_actor_prior', 'source_Q_replay_entropy_and_optimizers_imported'}
        source = snapshot.get('source_goal_contract', {})
        from ..rewards.precision_capture import frozen_capture_actor_contract
        from .staged_physics import frozen_cpu_actor_contract
        if set(snapshot) != required or snapshot.get('format_version') != 2 \
                or snapshot['source_artifact_type'] != ActualFlapResidualSACPilot.artifact_type \
                or snapshot['source_Q_replay_entropy_and_optimizers_imported'] is not False \
                or source.get('body_controller') != 'frozen_executed_nominal_actor_plus_learned_bounded_correction_v1' \
                or source.get('body_correction_radius') != .15 or source.get('fixed_prior_radius') != .15 \
                or source.get('actor_dim') != self.actor_dim or source.get('critic_dim') != self.critic_dim \
                or source.get('source_warm_start') != self.warm_start.contract \
                or source.get('goal_center') != self.center.tolist() or source.get('goal_scale') != self.scale.tolist() \
                or source.get('action_coordinates') != self.coordinates.name \
                or source.get('shelf_templates') != self.stage.templates \
                or source.get('validated_jaw_prior_confidence') != self.validated_jaw_prior_confidence \
                or source.get('jaw_prior_residual_gain') != self.jaw_prior_residual_gain \
                or snapshot['source_hidden'] != self.agent.config.hidden \
                or frozen_cpu_actor_contract(frozen_capture_actor_contract(source['physical_contract'])) != \
                    frozen_cpu_actor_contract(frozen_capture_actor_contract(self.physical_contract)):
            raise ValueError('Reanchored source must match actual-flap goals, .15 source radius, perception and safety')
        # Reuse the original nominal anchor's strict physical/coordinate checks
        # and auxiliary learner setup. Its reference remains the original jaw
        # reference; only the body controller gets another actor-only anchor.
        self.body_anchor_state = snapshot['nominal_body_anchor']
        try:
            super().configure_controller(saved)
        finally:
            self.body_anchor_state = snapshot
        self.actual_body_anchor = actual_actor_snapshot(snapshot, self.device)
        with torch.no_grad():
            self.agent.actor.load_state_dict(self.actual_body_anchor['actor'].state_dict())
            self.agent.actor_normalizer.load_state_dict(self.actual_body_anchor['actor_normalizer'].state_dict())
            self.agent.actor_normalizer.mean[-1].add_(self.correction_radius - source['fixed_prior_radius'])
            output = self.agent.actor.network[-1]
            output.weight[:19].zero_()
            output.bias[:19].zero_()

    @torch.no_grad()
    def executed_body_anchor(self, raw):
        nominal = self.nominal_view(raw)
        command = self.body_anchor['actor'](
            self.body_anchor['actor_normalizer'](nominal), deterministic=True)[0]
        nominal_goal = self.source_projector(nominal, command)[:, :19]
        return source_actual_body_goal(raw, nominal_goal, self.actual_body_anchor,
            self.body_anchor_state['source_goal_contract']['fixed_prior_radius'])

    @property
    def contract(self):
        return super().contract | dict(
            body_controller='frozen_learned_actual_flap_actor_plus_bounded_correction_v2',
            source_actual_flap_correction_radius=.15,
            body_initialization='zero_body_mean_preserve_source_trunk_jaw_logits_logstd_and_actor_normalizer',
            source_actual_Q_replay_entropy_and_optimizers_imported=False)
