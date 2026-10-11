"""Persistent checks on first safe TRAIN paths, independent of replay eviction.

Only actor observations and literal executed commands are retained. These
checks constrain updates on old states; they do not prove future rollout success.
"""
from copy import deepcopy
import torch
from torch.nn import functional as F

VARIANT = 'train-success-cohort-Adam-backtrack'
STABLE_VARIANT = 'train-success-cohort-stable-Adam-backtrack'
TUBE_VARIANT = 'train-success-cohort-servo-tube-Adam-backtrack'
BODY_GROUPS = {'left_arm': list(range(1, 8)), 'right_arm': list(range(8, 15)),
               'rest': [0, 15, 16, 17, 18]}


def empty_memory():
    return dict(format='persistent_safe_TRAIN_success_cohort_v1', entries=[], ceilings={})


def enabled(agent):
    return (getattr(agent, 'actor_success_guard_config', None) or {}).get('name') in (VARIANT, STABLE_VARIANT, TUBE_VARIANT)


def stable(agent):
    return (getattr(agent, 'actor_success_guard_config', None) or {}).get('name') in (STABLE_VARIANT, TUBE_VARIANT)


def tube(agent):
    return (getattr(agent, 'actor_success_guard_config', None) or {}).get('name') == TUBE_VARIANT


def cohort_batch(agent):
    memory = agent.success_guard_memory
    device = next(agent.actor.parameters()).device
    raw, actions, groups = [], [], {}
    goal_limits, servo_limits = [], []
    offset = 0
    for entry in memory['entries']:
        n = len(entry['action'])
        raw.append(entry['actor_obs']); actions.append(entry['action'])
        if tube(agent):
            goal_limits.append(entry['goal_error_limits'])
            servo_limits.append(entry['servo_error_limits'])
        start_tail = max(0, n-64)
        for name, first, last in (('approach', 0, start_tail), ('tail64', start_tail, n)):
            if first < last:
                groups[entry['identity']+'/'+name] = slice(offset+first, offset+last)
        offset += n
    if not raw:
        raise ValueError('Persistent guard has no completed safe TRAIN paths')
    batch = dict(actor_obs=torch.cat(raw).to(device), action=torch.cat(actions).to(device))
    if tube(agent):
        batch.update(goal_error_limits=torch.cat(goal_limits).to(device),
                     servo_error_limits=torch.cat(servo_limits).to(device))
    return batch, groups


def _losses_and_jaws(agent, batch, groups):
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
    goal_difference = body-labels[:, :19]
    if tube(agent):
        goal_difference = body.double()-labels[:, :19].double()
    if stable(agent):
        goal_difference, error = goal_difference.double(), error.double()
    if tube(agent):
        # Check every recorded row/joint against fixed physical command
        # tolerances. An improvement elsewhere cannot pay for this error.
        bounded = encoder(raw, goals)[:, :19].double()
        errors = torch.maximum(goal_difference.abs()/batch['goal_error_limits'],
            (bounded-recorded.double()).abs()/batch['servo_error_limits'])
    goal_error = goal_difference.square()
    servo_error = F.smooth_l1_loss(error, torch.zeros_like(error), beta=1., reduction='none')
    near = agent.action_projector.entropy_mask(raw)[:, 19:].bool()
    executed = agent.projected_command(raw, body, logits > 0)
    correct = executed[:, 19:] == labels[:, 19:]
    losses = {}
    for key, rows in groups.items():
        terms = {name: (errors[rows][:, columns].max() if tube(agent) else
                      (goal_error[rows][:, columns]+servo_error[rows][:, columns]).mean())
                 for name, columns in BODY_GROUPS.items()}
        for hand in range(2):
            active = near[rows, hand]
            jaw_logits, targets = logits[rows, hand][active], (labels[rows, 19+hand][active]+1)/2
            if stable(agent):
                jaw_logits, targets = jaw_logits.double(), targets.double()
            terms['jaw'+str(hand)] = F.binary_cross_entropy_with_logits(
                jaw_logits, targets) \
                if bool(active.any()) else jaw_logits.new_zeros(())
        losses[key] = terms
    return losses, correct


def losses_and_jaws(agent, batch, groups):
    if not stable(agent):
        return _losses_and_jaws(agent, batch, groups)
    # Adding another successful path must not change the GEMM batch shape
    # used to check an existing path's fixed ceiling. Production decoding and
    # model tensors remain float32; only scalar loss reductions use float64.
    result = {}; correct = torch.empty(len(batch['action']), 2, dtype=torch.bool,
                                      device=batch['actor_obs'].device)
    for key, rows in groups.items():
        part = {name: values[rows] for name, values in batch.items()}
        terms, jaws = _losses_and_jaws(agent, part, {key: slice(None)})
        result.update(terms); correct[rows] = jaws
    return result, correct


def violations(values, correct, memory, config):
    exceeded = []
    for key, terms in values.items():
        for name, value in terms.items():
            limit = memory['ceilings'][key][name]
            tolerance = config['absolute_loss_tolerance']+config['relative_loss_tolerance']*abs(limit)
            if not torch.isfinite(torch.tensor(value)) or value > limit+tolerance:
                exceeded.append(dict(group=key,term=name,excess=value-limit-tolerance))
    offset = wrong = 0
    for e in memory['entries']:
        n=len(e['action']); wrong+=int((e['protected_correct_jaws'] & ~correct[offset:offset+n]).sum());offset+=n
    return dict(exceeded_terms=len(exceeded),protected_jaw_regressions=wrong,
        worst_term=max(exceeded,key=lambda x:x['excess']) if exceeded else None)


def restoration_score(values, correct, memory, config):
    """Rank temporary optimizer candidates; acceptance still uses hard checks.

    Prefer preserving every protected jaw, then reducing all normalized
    violations. This is not a task reward or a relaxed acceptance threshold.
    """
    if values.keys() != memory['ceilings'].keys():
        raise ValueError('Restoration requires the same fixed cohort groups')
    excess = []
    for key, terms in values.items():
        for name, value in terms.items():
            limit = memory['ceilings'][key][name]
            tolerance = config['absolute_loss_tolerance']+config['relative_loss_tolerance']*abs(limit)
            excess.append(max(0., (value-limit-tolerance)/max(abs(limit), 1e-8)))
    wrong = violations(values, correct, memory, config)['protected_jaw_regressions']
    return (wrong, sum(excess), max(excess, default=0.))


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
def tighten(memory, values, correct, fixed=False):
    if not fixed:
        for key, terms in values.items():
            memory['ceilings'][key] = {name: min(value, memory['ceilings'].get(key, {}).get(name, value))
                                       for name, value in terms.items()}
    offset = 0
    for e in memory['entries']:
        n = len(e['action'])
        e['protected_correct_jaws'] |= correct[offset:offset+n]
        offset += n


@torch.no_grad()
def initialize_tube_limits(agent, entry):
    device = next(agent.actor.parameters()).device
    raw, labels = entry['actor_obs'].to(device), entry['action'].to(device)
    goals_limits, servo_limits = [], []
    n = len(labels); tail = max(0, n-64)
    config = agent.actor_success_guard_config
    for name, start, end in (('approach', 0, tail), ('tail64', tail, n)):
        if start == end:
            continue
        obs, actions = raw[start:end], labels[start:end]
        normalized = agent.actor_normalizer(agent.actor_features(obs))
        mean, _, _ = agent.continuous_parameters(normalized, obs)
        body = agent.body_from_latent(normalized, mean.tanh(), obs)
        encoder = agent.goal_servo_critic_encoder
        recorded = encoder(obs, actions)[:, :19].double()
        predicted = encoder(obs, torch.cat((body, actions[:, 19:]), -1))[:, :19].double()
        goals_limits.append((body.double()-actions[:, :19].double()).abs()+config['normalized_goal_error_budget'])
        servo_limits.append((predicted-recorded).abs()+config['normalized_servo_error_budget'][name])
    entry['goal_error_limits'] = torch.cat(goals_limits).cpu()
    entry['servo_error_limits'] = torch.cat(servo_limits).cpu()


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
            if tube(agent):
                initialize_tube_limits(agent, memory['entries'][-1])
            counts[key] = counts.get(key, 0)+1; known.add(episode['identity']); added = True
    if added:
        batch, groups = cohort_batch(agent)
        values, correct = metrics(agent, batch, groups)
        # New groups get their present policy ceiling. Existing ceilings are
        # never relaxed when another path arrives or replay evicts an old path.
        for key, terms in values.items():
            limits = ({name: (1. if name in BODY_GROUPS else value+agent.actor_success_guard_config['jaw_loss_budget'])
                       for name, value in terms.items()} if tube(agent) else terms)
            memory['ceilings'].setdefault(key, limits)
        tighten(memory, values, correct, fixed=tube(agent))


def gradient_constraints(agent, batch, groups, parameters):
    losses, _ = losses_and_jaws(agent, batch, groups)
    gradients = []
    for key, terms in losses.items():
        for name, loss in terms.items():
            if not loss.requires_grad:
                continue
            parts = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
            g = torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten()
                           for p, v in zip(parameters, parts)])
            if not torch.isfinite(g).all():
                raise FloatingPointError('Non-finite persistent TRAIN retention gradient')
            if g.square().sum() > 1e-20:
                headroom = max(0., agent.success_guard_memory['ceilings'][key][name]-float(loss.detach()))
                gradients.append((g, headroom) if tube(agent) else g)
    return gradients


def violation_correction(agent, batch, groups, parameters):
    """Repair current violations while keeping the original hard checks.

    Every phase uses its fixed forward shape. The most regressed protected
    jaw per hand/phase adds a decision-boundary derivative; a mean BCE alone
    can improve while that particular jaw changes its greedy label.
    """
    if not tube(agent):
        raise ValueError('Candidate repair requires fixed servo tube bounds')
    protected = torch.cat([e['protected_correct_jaws']
                           for e in agent.success_guard_memory['entries']]).to(batch['actor_obs'].device)
    constraints = []
    kinds = {'body': 0, 'jaw_loss': 0, 'jaw_label': 0}

    def add(loss, excess, kind):
        if not loss.requires_grad:
            return
        parts = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
        gradient = torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten()
                              for p, v in zip(parameters, parts)])
        if not torch.isfinite(gradient).all() or not torch.isfinite(torch.tensor(excess)):
            raise FloatingPointError('Nonfinite candidate feasibility derivative')
        if gradient.square().sum() > 1e-20:
            constraints.append((gradient, excess))
            kinds[kind] += 1

    for key, rows in groups.items():
        part = {name: values[rows] for name, values in batch.items()}
        terms, correct = _losses_and_jaws(agent, part, {key: slice(None)})
        for name, loss in terms[key].items():
            limit = agent.success_guard_memory['ceilings'][key][name]
            config = agent.actor_success_guard_config
            tolerance = config['absolute_loss_tolerance']+config['relative_loss_tolerance']*abs(limit)
            if float(loss.detach()) > limit+tolerance:
                add(loss, float(loss.detach())-limit+tolerance,
                    'body' if name in BODY_GROUPS else 'jaw_loss')
        wrong = protected[rows] & ~correct
        if wrong.any():
            raw, labels = part['actor_obs'], part['action'][:, 19:]
            logits = agent.continuous_parameters(agent.actor_normalizer(agent.actor_features(raw)), raw)[2]
            signed = logits.double()*labels.double()
            for hand in range(2):
                bad = torch.where(wrong[:, hand])[0]
                if not len(bad):
                    continue
                index = bad[signed[bad, hand].detach().argmin()]
                loss = -signed[index, hand]
                # A few float32 ulps of optimizer feasibility reserve. The
                # production predicate remains logits>0, with no deadband.
                reserve = 8*torch.finfo(logits.dtype).eps
                add(loss, float(loss.detach())+reserve, 'jaw_label')
    if not constraints:
        return None, kinds
    correction = torch.zeros_like(constraints[0][0])
    for _ in range(agent.actor_success_guard_config['projection_cycles']):
        for gradient, excess in constraints:
            remaining = torch.dot(gradient, correction)+excess
            if remaining > 0:
                correction -= remaining/gradient.square().sum()*gradient
    if not torch.isfinite(correction).all():
        raise FloatingPointError('Nonfinite candidate feasibility displacement')
    return correction, kinds


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
        if tube(agent):
            for name in ('goal_error_limits', 'servo_error_limits'):
                limits = e.get(name)
                if not isinstance(limits, torch.Tensor) or limits.shape != (n, 19) \
                        or limits.dtype != torch.float64 or not torch.isfinite(limits).all() or (limits <= 0).any():
                    raise ValueError('Malformed fixed physical command tube')
                e[name] = limits.detach().cpu().clone()
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
    if tube(agent) and any(any(terms[name] != 1. for name in BODY_GROUPS)
                          for terms in memory['ceilings'].values()):
        raise ValueError('Fixed physical command tube cannot accumulate slack')
    agent.success_guard_memory = memory
