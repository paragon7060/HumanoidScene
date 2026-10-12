"""Prepared six-scope actor and independent Adam migration.

Clone existing region networks before separating supported box sizes. The
normalizer, observations, physical action mapping and existing heads' outputs
stay unchanged. The whole-cohort guard supports these independent heads.
Production resume still requires an explicit checkpoint/replay contract
migration and closed CUDA checks; this helper does not start a learner.
"""
from copy import deepcopy

import torch
from torch import nn

from .regional_actor_servo import RegionRoutedNetwork
from .staged_train_success import REGIONS


SCOPES = (('shelf_2_right', 'medium'), ('shelf_2_left', 'small'),
          ('shelf_3_right', 'small'), ('shelf_3_left', 'small'),
          ('shelf_2_right', 'small'), ('shelf_2_left', 'medium'))
SOURCE_HEADS = (0, 1, 2, 3, 0, 1)


class RegionSizeRoutedNetwork(nn.Module):
    def __init__(self, source, normalizer):
        super().__init__()
        if type(source) is not RegionRoutedNetwork or len(source.heads) != 4:
            raise ValueError('Size split requires the original independent four region heads')
        mean, scale = normalizer.mean, normalizer.var.clamp_min(1e-4).sqrt()
        if mean.shape != (518,) or not torch.equal(source.region_mean, mean[94:98]) \
                or not torch.equal(source.region_scale, scale[94:98]):
            raise ValueError('Size and region routing require the same frozen normalizer')
        self.heads = nn.ModuleList([deepcopy(source.heads[i]) for i in SOURCE_HEADS])
        self.register_buffer('region_mean', source.region_mean.detach().clone())
        self.register_buffer('region_scale', source.region_scale.detach().clone())
        types = torch.eye(2, device=mean.device, dtype=mean.dtype)
        patterns = ((types-mean[89:91])/scale[89:91]).clamp(-10, 10)
        if not torch.isfinite(patterns).all() or torch.equal(patterns[0], patterns[1]):
            raise ValueError('Frozen normalization no longer distinguishes small and medium')
        # Inverse-normalization followed by argmax is unsafe after clipping:
        # e.g. source mean[1,0],var[0,0] recovers medium as[.9,.1]. Known
        # normalized one-hot patterns are distinct and keep routing correct.
        self.register_buffer('type_patterns', patterns.detach().clone())
        self.register_buffer('head_lookup', torch.tensor(
            [[4, 0], [1, 5], [2, -1], [3, -1]], device=mean.device))

    def scope_indices(self, normalized):
        if normalized.ndim != 2 or normalized.shape[1] != 518:
            raise ValueError('Size routing requires the unchanged 518-D actor features')
        regions = (normalized[:, 94:98]*self.region_scale+self.region_mean).argmax(-1)
        types = (normalized[:, None, 89:91]-self.type_patterns[None]).square().sum(-1).argmin(-1)
        heads = self.head_lookup[regions, types]
        if (heads < 0).any():
            raise ValueError('Upper-shelf medium is outside the supported task scopes')
        return heads

    def forward(self, normalized):
        indices = self.scope_indices(normalized)
        if not len(normalized):
            return normalized.new_empty((0, 42))
        output = None
        for scope in indices.unique().tolist():
            # Preserve the original full batch shape for forward rounding.
            # Absent heads are never called and cannot accrue Adam momentum.
            prediction = self.heads[scope](normalized)
            if output is None:
                output = torch.zeros_like(prediction)
            output = torch.where((indices == scope)[:, None], prediction, output)
        return output


def split_region_size_actor_and_Adam(agent):
    """Prepare a disposable or next learner; do not alter any source tensors.

    Clone all six parameter groups' moments, including CPU Adam step tensors.
    Critic, entropy, frozen anchors, successful TRAIN bank/cohort and normalizer
    are untouched. This is not a checkpoint/replay contract migration.
    """
    source, old = agent.actor.network, agent.actor_optimizer
    if not agent.config.freeze_actor_normalizer or agent.config.actor_feature_mode != 'flat' \
            or type(source) is not RegionRoutedNetwork or type(old) is not torch.optim.Adam:
        raise ValueError('Size split requires the original frozen-normalizer regional Adam actor')
    source_parameters = list(source.parameters())
    if len(old.param_groups) != 1 or [id(p) for p in old.param_groups[0]['params']] \
            != [id(p) for p in source_parameters] \
            or [id(p) for p in agent.actor.parameters()] != [id(p) for p in source_parameters]:
        raise ValueError('Size split requires one complete original actor Adam group')
    if any(not torch.isfinite(p).all() for p in source_parameters) or any(
            isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
            for state in old.state.values() for v in state.values()):
        raise ValueError('Size split requires finite original actor and Adam state')
    network = RegionSizeRoutedNetwork(source, agent.actor_normalizer)
    optimizer = torch.optim.Adam(network.parameters(), **deepcopy(old.defaults))
    optimizer.param_groups[0].update(deepcopy({k: v for k, v in old.param_groups[0].items()
                                             if k != 'params'}))
    for destination, source_index in zip(network.heads, SOURCE_HEADS):
        for new_parameter, original in zip(destination.parameters(), source.heads[source_index].parameters()):
            if original in old.state:
                optimizer.state[new_parameter] = deepcopy(old.state[original])
    # Install only after every validation and clone completed successfully.
    agent.actor.network, agent.actor_optimizer = network, optimizer
    return dict(scopes=[list(s) for s in SCOPES], source_head_indices=list(SOURCE_HEADS),
                cloned_independent_Adam_moments=True, critic_entropy_cohort_unchanged=True,
                checkpoint_replay_contract_migration_required=True)
