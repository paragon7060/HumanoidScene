"""Jaw-only correction for a rejected fixed-cohort candidate.

The optional rounded-jaw actor fallback uses this component. A caller must
retain the original projected Adam norm, fixed ceilings, every protected jaw,
rejected Adam restoration and final uncached whole-cohort acceptance test.
"""
import torch

from ...algorithms import success_cohort_guard as cohort
from .independent_actor_heads import head_labels


def jaw_output_correction(agent, batch, groups, parameters):
    """Return a pure correction through final jaw mean rows19:21 only.

Actual body violations forbid this fallback. Hidden layers, body means and
Gaussian std rows are untouched. No optimizer, parameter or budget is updated.
"""
    if not cohort.tube(agent):
        raise ValueError('Jaw output repair requires the fixed servo-tube contract')
    head_labels(agent.actor.network)
    identity = {id(p) for p in parameters}
    heads = [h for h in agent.actor.network.heads
             if {id(p) for p in h.parameters()} == identity]
    if len(heads) != 1 or len(identity) != len(parameters):
        raise ValueError('Jaw output repair requires one complete independent head')
    final = heads[0][-1]
    if not isinstance(final, torch.nn.Linear) or final.out_features != 42:
        raise ValueError('Jaw output repair requires the unchanged21D/42-output actor')
    values, correct = cohort.metrics(agent, batch, groups)
    kinds = dict(body=0, jaw_loss=0, jaw_label=0)
    protected = torch.cat([e['protected_correct_jaws']
                           for e in agent.success_guard_memory['entries']]).cpu()
    for key, terms in values.items():
        if key not in agent.success_guard_memory['ceilings']:
            raise ValueError('Jaw repair requires original fixed TRAIN phase groups')
        for name, value in terms.items():
            limit = agent.success_guard_memory['ceilings'][key][name]
            config = agent.actor_success_guard_config
            tolerance = config['absolute_loss_tolerance'] + config['relative_loss_tolerance']*abs(limit)
            if not torch.isfinite(torch.tensor(value)) or value > limit+tolerance:
                kinds['body' if name in cohort.BODY_GROUPS else 'jaw_loss'] += 1
        rows = groups[key]
        kinds['jaw_label'] += int((protected[rows] & ~correct[rows]).sum())
    if kinds['body'] or not (kinds['jaw_loss'] or kinds['jaw_label']):
        return None, kinds
    selected = [final.weight, final.bias]
    restricted, derivative_kinds = cohort._violation_correction(
        agent, batch, groups, selected)
    if restricted is None or derivative_kinds['body']:
        return None, kinds
    parts = {}; offset = 0
    for p in selected:
        count = p.numel()
        value = restricted[offset:offset+count].reshape_as(p)
        offset += count
        if bool((value[:19] != 0).any()) or bool((value[21:] != 0).any()):
            raise ValueError('Jaw output repair touched body or Gaussian std output rows')
        parts[id(p)] = value
    correction = torch.cat([parts.get(id(p), torch.zeros_like(p)).flatten()
                            for p in parameters])
    if not torch.isfinite(correction).all():
        raise FloatingPointError('Nonfinite jaw-only feasibility correction')
    return correction, kinds
