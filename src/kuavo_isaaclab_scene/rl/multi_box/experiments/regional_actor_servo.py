"""Independent SAC actors selected by the existing perceived rack-region token.

The physical task, action coordinates and critic are unchanged. A head's Adam
momentum cannot update another region when that region is absent from a batch.
"""
from copy import deepcopy

import torch
from torch import nn

from .actor_memory_servo_retention import ActorMemoryServoRetentionSACPilot
from .actor_train_memory import structure_sha256
from .staged_train_success import REGIONS


def regional_actor_contract():
    return dict(name='independent_perceived_rack_region_SAC_actors_v1',
        regions=list(REGIONS), perceived_region_columns=[94, 98],
        actor_observation_and_action_coordinates_unchanged=True,
        shared_Q_and_entropy_and_success_losses_unchanged=True,
        actor_normalizer_frozen=True, actor_only_component_import=True,
        absent_region_heads_receive_no_gradient_or_Adam_update=True,
        whole_original_randomized_DEV_and_independent_FINAL_required=True)


class RegionRoutedNetwork(nn.Module):
    def __init__(self, network, normalizer):
        super().__init__()
        self.heads = nn.ModuleList([deepcopy(network) for _ in REGIONS])
        # Undo the fixed observation normalization only for routing. This also
        # handles a source with very unequal region frequencies. The supported
        # one-hot tokens must remain distinguishable after the existing clip.
        self.register_buffer('region_mean', normalizer.mean[94:98].detach().clone())
        self.register_buffer('region_scale', normalizer.var[94:98].clamp_min(1e-4).sqrt().detach().clone())
        tokens = torch.eye(4, device=self.region_mean.device)
        normalized = ((tokens-self.region_mean)/self.region_scale).clamp(-10, 10)
        recovered = normalized*self.region_scale+self.region_mean
        if not torch.equal(recovered.argmax(-1), torch.arange(4, device=tokens.device)):
            raise ValueError('Frozen normalization no longer distinguishes all rack regions')

    def region_indices(self, normalized):
        if normalized.ndim != 2 or normalized.shape[1] != 518:
            raise ValueError('Regional SAC requires the unchanged 518-D flat actor features')
        return (normalized[:, 94:98]*self.region_scale+self.region_mean).argmax(-1)

    def forward(self, normalized):
        indices = self.region_indices(normalized)
        if not len(normalized):
            return normalized.new_empty((0, 42))
        output = None
        for region in indices.unique().tolist():
            # Use the same full batch as the component actor so matrix rounding
            # does not change its initial actions. Do not evaluate absent heads:
            # grad=None prevents their existing Adam moments from moving them.
            prediction = self.heads[region](normalized)
            if output is None:
                output = torch.zeros_like(prediction)
            output = torch.where((indices == region)[:, None], prediction, output)
        return output


def install_regional_actor(agent):
    if (agent.config.actor_feature_mode != 'flat'
            or not agent.config.freeze_actor_normalizer
            or isinstance(agent.actor.network, RegionRoutedNetwork)):
        raise ValueError('Regional actors require a fresh flat actor with frozen normalization')
    agent.actor.network = RegionRoutedNetwork(agent.actor.network, agent.actor_normalizer)
    agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=agent.config.actor_lr)


def validate_regional_actor_state(state):
    contract = state.get('goal_contract', {})
    origin = state.get('regional_actor_initialization')
    if (contract.get('regional_actor') != regional_actor_contract()
            or state.get('config', {}).get('freeze_actor_normalizer') is not True
            or state.get('config', {}).get('actor_feature_mode') != 'flat'
            or not isinstance(origin, dict)
            or origin.get('kind') != 'compatible_actor_only_region_components_v1'
            or origin.get('frozen_body_anchor_SHA256') != structure_sha256(state['body_anchor_state'])
            or origin.get('actor_normalizer_SHA256') != structure_sha256(
                {k:v for k,v in state['model'].items() if k.startswith('actor_normalizer.')})
            or set(origin.get('components', {})) != set(REGIONS)
            or state.get('actor_memory_initialization') is not None):
        raise ValueError('Regional actor contract, anchor or initialization provenance differs')
    for index, region in enumerate(REGIONS):
        component = origin['components'][region]
        if not isinstance(component, dict) or len(component.get('checkpoint_SHA256', '')) != 64:
            raise ValueError('Regional actor component checkpoint provenance is missing')
        if state['actor_updates'] == 0:
            prefix = f'actor.network.heads.{index}.'
            parameters = {k.removeprefix(prefix):v for k,v in state['model'].items()
                          if k.startswith(prefix)}
            if not parameters or structure_sha256(parameters) != component.get('network_SHA256'):
                raise ValueError('Unstarted regional actor differs from its component')
    # Routing buffers must match the unchanged frozen actor normalizer even
    # after SAC updates. They cannot silently drift to select another head.
    for name, expected in (
            ('region_mean', state['model']['actor_normalizer.mean'][94:98]),
            ('region_scale', state['model']['actor_normalizer.var'][94:98].clamp_min(1e-4).sqrt())):
        actual = state['model'].get('actor.network.'+name)
        if not isinstance(actual, torch.Tensor) or not torch.equal(actual, expected):
            raise ValueError('Regional routing differs from frozen actor normalization')


class RegionalActorMemorySACPilot(ActorMemoryServoRetentionSACPilot):
    artifact_type = 'staged_actual_flap_regional_actor_memory_servo_retention_sac_v1'

    def configure_controller(self, saved):
        super().configure_controller(saved)
        validate_regional_actor_state(saved)
        self.regional_actor_initialization = deepcopy(saved['regional_actor_initialization'])
        install_regional_actor(self.agent)

    @property
    def contract(self):
        return super().contract | dict(regional_actor=regional_actor_contract())

    def checkpoint_extras(self):
        return super().checkpoint_extras() | dict(
            regional_actor_initialization=deepcopy(self.regional_actor_initialization))

    def experience_extras(self):
        return super().experience_extras() | dict(
            regional_actor_initialization=deepcopy(self.regional_actor_initialization))

    def restore_experience_extras(self, state):
        super().restore_experience_extras(state)
        if state.get('regional_actor_initialization') != self.regional_actor_initialization:
            raise ValueError('Checkpoint and replay regional actor origins differ')

    def report(self):
        return super().report() | dict(regional_actor=regional_actor_contract(),
            regional_actor_initialization=deepcopy(self.regional_actor_initialization))
