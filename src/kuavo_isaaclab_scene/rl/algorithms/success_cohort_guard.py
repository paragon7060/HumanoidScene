"""Persistent checks on first safe TRAIN paths, independent of replay eviction.

Only actor observations and literal executed commands are retained. These
checks constrain updates on old states; they do not prove future rollout success.
"""
from copy import deepcopy
import torch
from torch.nn import functional as F

VARIANT = 'train-success-cohort-Adam-backtrack'
BODY_GROUPS = {'left_arm': list(range(1, 8)), 'right_arm': list(range(8, 15)),
               'rest': [0, 15, 16, 17, 18]}


def empty_memory():
    return dict(format='persistent_safe_TRAIN_success_cohort_v1', entries=[], ceilings={})


def enabled(agent):
    return (getattr(agent, 'actor_success_guard_config', None) or {}).get('name') == VARIANT


def cohort_batch(agent):
    memory = agent.success_guard_memory
    device = next(agent.actor.parameters()).device
    raw, actions, groups = [], [], {}
    offset = 0
    for entry in memory['entries']:
        n = len(entry['action'])
        raw.append(entry['actor_obs']); actions.append(entry['action'])
        start_tail = max(0, n-64)
        for name, first, last in (('approach', 0, start_tail), ('tail64', start_tail, n)):
            if first < last:
                groups[entry['identity']+'/'+name] = slice(offset+first, offset+last)
        offset += n
    if not raw:
        raise ValueError('Persistent guard has no completed safe TRAIN paths')
    return dict(actor_obs=torch.cat(raw).to(device), action=torch.cat(actions).to(device)), groups


def losses_and_jaws(agent, batch, groups):
    raw, labels = batch['actor_obs'], batch['action']
    normalized = agent.actor_normalizer(agent.actor_features(raw))
    mean, _, logits = agent.continuous_parameters(normalized, raw)
    body = agent.body_from_latent(normalized, mean.tanh(), raw)
    goals = torch.cat((body, labels[:, 19:]), -1)
    encoder = agent.goal_servo_critic_encoder
    predicted = encoder.unclipped_body(raw, goals)
    recorded = encoder(raw, labels)[:, :19].detach()
    error = torch.where(recorded >= 1, (1-predicted).clamp_min(0),
        torch.where(recorded <= -1, (predicted+1).clamp_min(0), predicted-recorded))
    goal_error = (body-labels[:, :19]).square()
    servo_error = F.smooth_l1_loss(error, torch.zeros_like(error), beta=1., reduction='none')
    near = agent.action_projector.entropy_mask(raw)[:, 19:].bool()
    executed = agent.projected_command(raw, body, logits > 0)
    correct = executed[:, 19:] == labels[:, 19:]
    losses = {}
    for key, rows in groups.items():
        terms = {name: (goal_error[rows][:, columns]+servo_error[rows][:, columns]).mean()
                 for name, columns in BODY_GROUPS.items()}
        for hand in range(2):
            active = near[rows, hand]
            terms['jaw'+str(hand)] = F.binary_cross_entropy_with_logits(
                logits[rows, hand][active], (labels[rows, 19+hand][active]+1)/2) \
                if bool(active.any()) else logits.new_zeros(())
        losses[key] = terms
    return losses, correct


@torch.no_grad()
def metrics(agent, batch, groups):
    losses, correct = losses_and_jaws(agent, batch, groups)
    return {k: {name: float(value) for name, value in terms.items()} for k, terms in losses.items()}, correct.cpu()


def within_ceilings(values, correct, memory, config):
    if values.keys() != memory['ceilings'].keys():
        return False
    offset = 0
    for e in memory['entries']:
        n = len(e['action'])
        if bool((e['protected_correct_jaws'] & ~correct[offset:offset+n]).any()):
            return False
        offset += n
    for key, terms in values.items():
        for name, value in terms.items():
            limit = memory['ceilings'][key][name]
            tolerance = config['absolute_loss_tolerance'] + config['relative_loss_tolerance'] * abs(limit)
            if not torch.isfinite(torch.tensor(value)) or value > limit+tolerance:
                return False
    return True


@torch.no_grad()
def tighten(memory, values, correct):
    for key, terms in values.items():
        memory['ceilings'][key] = {name: min(value, memory['ceilings'].get(key, {}).get(name, value))
                                   for name, value in terms.items()}
    offset = 0
    for e in memory['entries']:
        n = len(e['action'])
        e['protected_correct_jaws'] |= correct[offset:offset+n]
        offset += n


def synchronize(agent, bank):
    if not enabled(agent):
        return
    from ..multi_box.experiments.staged_train_success import validate_success_outcome
    memory = getattr(agent, 'success_guard_memory', None)
    if memory is None:
        memory = agent.success_guard_memory = empty_memory()
    counts = {}
    known = {e['identity'] for e in memory['entries']}
    for e in memory['entries']:
        key = (e['outcome']['layout']['target_region'], e['outcome']['layout']['target_box_type'])
        counts[key] = counts.get(key, 0)+1
    added = False
    for episodes in bank.episodes.values():
        for episode in episodes:
            if episode['identity'] in known:
                continue
            o = episode['outcome']; key = (o['layout']['target_region'], o['layout']['target_box_type'])
            if counts.get(key, 0) >= 2:
                continue
            validate_success_outcome('train', o)
            rows = episode['rows']; raw, action = rows['actor_obs'], rows['action']
            n = len(action)
            if not 1 <= n <= 900 or raw.shape != (n, 518) or action.shape != (n, 21) \
                    or not torch.isfinite(raw).all() or not torch.isfinite(action).all():
                raise ValueError('Malformed actual TRAIN guard path')
            memory['entries'].append(dict(identity=episode['identity'], outcome=deepcopy(o),
                actor_obs=raw.detach().cpu().clone(), action=action.detach().cpu().clone(),
                protected_correct_jaws=torch.zeros(n, 2, dtype=torch.bool)))
            counts[key] = counts.get(key, 0)+1; known.add(episode['identity']); added = True
    if added:
        batch, groups = cohort_batch(agent)
        values, correct = metrics(agent, batch, groups)
        # New groups get their present policy ceiling. Existing ceilings are
        # never relaxed when another path arrives or replay evicts an old path.
        for key, terms in values.items():
            memory['ceilings'].setdefault(key, terms)
        tighten(memory, values, correct)


def gradient_constraints(agent, batch, groups, parameters):
    losses, _ = losses_and_jaws(agent, batch, groups)
    gradients = []
    for terms in losses.values():
        for loss in terms.values():
            if not loss.requires_grad:
                continue
            parts = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
            g = torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten()
                           for p, v in zip(parameters, parts)])
            if not torch.isfinite(g).all():
                raise FloatingPointError('Non-finite persistent TRAIN retention gradient')
            if g.square().sum() > 1e-20:
                gradients.append(g)
    return gradients


def report(agent):
    memory = getattr(agent, 'success_guard_memory', None) or empty_memory()
    return dict(episodes=len(memory['entries']), rows=sum(len(e['action']) for e in memory['entries']),
        groups=len(memory['ceilings']), grouping='episode_region_size_approach_or_tail64',
        first_paths_survive_success_replay_eviction=True, TRAIN_only=True,
        evaluation_rows=0, rollout_success_guarantee=False)


def restore_memory(agent, state):
    if not enabled(agent):
        if state.get('success_guard_memory') is not None:
            raise ValueError('Persistent guard memory requires its own actor contract')
        return
    memory = deepcopy(state.get('success_guard_memory'))
    if not isinstance(memory, dict) or set(memory) != {'format', 'entries', 'ceilings'} \
            or memory['format'] != empty_memory()['format']:
        raise ValueError('Persistent guard memory checkpoint is missing or incompatible')
    from ..multi_box.experiments.staged_train_success import validate_success_outcome, REGIONS
    counts, identities, expected = {}, set(), set()
    for e in memory['entries']:
        o = e['outcome']; validate_success_outcome('train', o)
        region = o['layout']['target_region']; size = o['layout']['target_box_type']
        key = (region, size); counts[key] = counts.get(key, 0)+1
        suffix = f"/wave{o['wave']}/env{o['environment']}/seed{o['layout']['seed']}"
        if counts[key] > 2 or e['identity'] in identities or not e['identity'].endswith(suffix):
            raise ValueError('Persistent guard path identity differs')
        identities.add(e['identity'])
        for name in ('actor_obs', 'action', 'protected_correct_jaws'):
            e[name] = e[name].detach().cpu().clone()
        raw, action = e['actor_obs'], e['action']; n = len(action)
        if not 1 <= n <= 900 or raw.shape != (n, 518) or action.shape != (n, 21) \
                or e['protected_correct_jaws'].shape != (n, 2) or e['protected_correct_jaws'].dtype != torch.bool \
                or not torch.isfinite(raw).all() or not torch.isfinite(action).all() \
                or (action.abs() > 1.00001).any() or not (action[:, 19:].abs() == 1).all() \
                or not (raw[:, -6] == 1).all() or not (raw[:, 94:98].argmax(-1) == REGIONS.index(region)).all():
            raise ValueError('Malformed persistent guard path coordinates')
        if n > 64:
            expected.add(e['identity']+'/approach')
        expected.add(e['identity']+'/tail64')
    if set(memory['ceilings']) != expected or any(
            set(terms) != set(BODY_GROUPS)|{'jaw0','jaw1'} or any(
                type(v) not in (float, int) or not torch.isfinite(torch.tensor(v)) or v < 0 for v in terms.values())
            for terms in memory['ceilings'].values()):
        raise ValueError('Persistent guard loss ceilings differ')
    agent.success_guard_memory = memory
