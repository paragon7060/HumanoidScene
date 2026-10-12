"""Exact independent-head partitions used by the original whole-cohort guard."""
import torch


def head_labels(network):
    from .regional_actor_servo import RegionRoutedNetwork
    from .region_size_actor import RegionSizeRoutedNetwork, SCOPES
    from .staged_train_success import REGIONS
    if type(network) is RegionRoutedNetwork:
        labels = tuple(REGIONS)
    elif type(network) is RegionSizeRoutedNetwork:
        labels = tuple('/'.join(scope) for scope in SCOPES)
    else:
        raise ValueError('Guard requires exact independent regional or region-size heads')
    if len(network.heads) != len(labels):
        raise ValueError('Independent actor head count differs from its routing')
    return labels


def parameter_partitions(actor):
    labels = head_labels(actor.network)
    named = list(actor.named_parameters())
    partitions = {label: [j for j, (name, _) in enumerate(named)
                         if name.startswith(f'network.heads.{i}.')]
                  for i, label in enumerate(labels)}
    covered = [j for indices in partitions.values() for j in indices]
    if sorted(covered) != list(range(len(named))) or any(not v for v in partitions.values()):
        raise ValueError('Backtracking requires complete independent actor parameter partitions')
    return partitions


@torch.no_grad()
def raw_head_indices(agent, raw):
    from .regional_actor_servo import RegionRoutedNetwork
    head_labels(agent.actor.network)
    if type(agent.actor.network) is RegionRoutedNetwork:
        # Keep the original four-head cache routing unchanged.
        return raw[:, 94:98].argmax(-1)
    normalized = agent.actor_normalizer(agent.actor_features(raw))
    return agent.actor.network.scope_indices(normalized)
