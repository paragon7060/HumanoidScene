"""Preserve exact, successful TRAIN transitions in held-goal coordinates.

Evaluation, altered-waypoint probes, unsafe attempts and inverse labels of
saturated physical commands cannot supply this bank. It stores real Q rows.
"""
from copy import deepcopy
import hashlib
import math
import torch

REGIONS=('shelf_2_right','shelf_2_left','shelf_3_right','shelf_3_left')
FORMAT='actual_train_success_held_goal_transitions_v1'
KEYS=('actor_obs','critic_obs','action','next_actor_obs','next_critic_obs','reward','terminated')


def retention_config(actor_sampling=None):
    result = dict(format=FORMAT,capacity_per_region=4096,initial_replay_fraction=.2,
        final_replay_fraction=.05,fade_actor_updates=5000,
        actor_goal_mse_weight=1.,actor_jaw_nll_weight=.05,
        source='completed_safe_opposing_bilateral_proof_lift_TRAIN_only',
        evaluation_import_allowed=False)
    if actor_sampling in (None, 'uniform'):
        return result
    if actor_sampling != 'tail64-half':
        raise ValueError('Unknown successful TRAIN actor sampling variant')
    result['actor_sampling'] = dict(variant='tail64-half', tail_fraction=.5, tail_steps=64,
        scope='actor_success_MSE_and_jaw_NLL_only', region_balance_preserved=True,
        Q_replay_sampling_unchanged=True, labels_rewards_and_success_conditions_unchanged=True)
    return result


def validate_retention_config(config):
    sampling = config.get('actor_sampling', {}) if isinstance(config, dict) else {}
    variant = sampling.get('variant') if isinstance(sampling, dict) else None
    if config != retention_config(variant):
        raise ValueError('Unknown actual-success retention contract')


def validate_success_outcome(split,outcome):
    r=outcome.get('result') or {}
    if split!='train' or outcome.get('split')!='train':
        raise ValueError('Successful goal bank accepts TRAIN only; DEV/FINAL/probes are excluded')
    if not outcome.get('initial_layout_valid') or not outcome.get('complete') or not r.get('success'):
        raise ValueError('Successful goal bank needs a completed valid successful attempt')
    if r.get('unsafe') or r.get('numerical_failure') or r.get('invalid_reset') or any(r.get('unsafe_causes',{}).values()):
        raise ValueError('Unsafe, numerical or replaced attempts cannot seed successful replay')
    if len(r.get('pinching',[]))!=2 or not all(r['pinching']) or len(r.get('stable_hands',[]))!=2 or not all(r['stable_hands']) \
            or not r.get('opposing_flaps') or not r.get('proof_lift') or r.get('hold_time_s',0)<.25 or r.get('rack_clearance_m',0)<.008:
        raise ValueError('Successful replay requires current physical bilateral pinch/lift/hold evidence')
    stage=r.get('staged_base',{})
    if stage.get('phase')!='held_grasp' or stage.get('manipulation_start') is None or 'waypoint_probe' in stage:
        raise ValueError('Successful replay must come from the matching held-base controller')
    if outcome['layout']['target_region'] not in REGIONS:raise ValueError('Unknown successful rack region')


class TrainSuccessBank:
    """Keep whole episodes and balance scarce successful regions when sampling."""
    def __init__(self,actor_dim,critic_dim,config=None):
        self.actor_dim,self.critic_dim=actor_dim,critic_dim
        self.config=deepcopy(config or retention_config())
        validate_retention_config(self.config)
        self.episodes={region:[] for region in REGIONS}

    @property
    def size(self):return sum(len(e['rows']['reward']) for es in self.episodes.values() for e in es)

    def add_episode(self,rows,outcome,*,source_run,split):
        validate_success_outcome(split,outcome)
        n=len(rows['reward']) if 'reward' in rows else 0
        if not 1<=n<=900 or set(rows)!=set(KEYS):raise ValueError('Expected one real held-goal episode')
        shapes={'actor_obs':(n,self.actor_dim),'next_actor_obs':(n,self.actor_dim),
            'critic_obs':(n,self.critic_dim),'next_critic_obs':(n,self.critic_dim),
            'action':(n,21),'reward':(n,),'terminated':(n,)}
        for key,value in rows.items():
            if value.shape!=shapes[key] or not bool(torch.isfinite(value).all()):raise ValueError('Malformed successful goal transition')
        if rows['terminated'].dtype!=torch.bool or not bool(rows['terminated'][-1]) or bool(rows['terminated'][:-1].any()):
            raise ValueError('Successful path must end at its one actual terminal transition')
        if bool((rows['action'].abs()>1.00001).any()) or not bool((rows['action'][:,19:21].abs()==1).all()):
            raise ValueError('Success labels must be actual bounded21-D goals and binary jaws')
        if not all(bool((rows[key][:,-6]==1).all()) for key in ('actor_obs','critic_obs','next_actor_obs','next_critic_obs')):
            raise ValueError('Successful replay excludes unconfirmed base approach transitions')
        region=outcome['layout']['target_region'];tokens=rows['actor_obs'][:,94:98];expected=REGIONS.index(region)
        if not bool((tokens.argmax(-1)==expected).all()) or not bool((tokens[:,expected]>.5).all()):
            raise ValueError('Successful layout region differs from actual actor observations')
        identity=f"{source_run}/wave{outcome['wave']}/env{outcome['environment']}/seed{outcome['layout']['seed']}"
        episodes=self.episodes[region]
        if any(e['identity']==identity for e in episodes):raise ValueError('Duplicate successful physical episode')
        evidence={k:deepcopy(outcome[k]) for k in ('wave','split','environment','layout','result','complete','initial_layout_valid')}
        episodes.append(dict(identity=identity,outcome=evidence,rows={k:v.detach().cpu().contiguous().clone() for k,v in rows.items()}))
        while sum(len(e['rows']['reward']) for e in episodes)>self.config['capacity_per_region']:episodes.pop(0)

    def sample(self,count,device,*,tail_fraction=0.,tail_steps=64):
        if not math.isfinite(tail_fraction) or not 0<=tail_fraction<=1 \
                or type(tail_steps) is not int or not 1<=tail_steps<=900:
            raise ValueError('Successful actor tail sampling requires a bounded fraction and horizon')
        available=[region for region in REGIONS if self.episodes[region]]
        if count<1 or not available:raise ValueError('Cannot sample empty successful TRAIN replay')
        selected=[]
        tail_count=round(count*tail_fraction)
        for i in range(count):
            episodes=self.episodes[available[i%len(available)]];episode=episodes[int(torch.randint(len(episodes),()))]
            rows=episode['rows'];n=len(rows['reward'])
            low=max(0,n-tail_steps) if i<tail_count else 0
            j=int(torch.randint(low,n,()))
            selected.append({k:v[j] for k,v in rows.items()})
        order=torch.randperm(count)
        return {k:torch.stack([s[k] for s in selected])[order].to(device) for k in KEYS}

    def sample_actor(self,count,device):
        sampling=self.config.get('actor_sampling')
        if sampling is None:return self.sample(count,device)
        return self.sample(count,device,tail_fraction=sampling['tail_fraction'],tail_steps=sampling['tail_steps'])

    def mix(self,batch,fraction,device):
        if not math.isfinite(fraction) or not 0<=fraction<=1:raise ValueError('Invalid successful replay fraction')
        count=round(len(batch['reward'])*fraction) if self.size else 0
        if not count:return batch,0
        success=self.sample(count,device)
        return {k:torch.cat((v[count:],success[k])) for k,v in batch.items()},count

    def report(self):
        return dict(rows=self.size,by_region={region:dict(episodes=len(es),rows=sum(len(e['rows']['reward']) for e in es)) for region,es in self.episodes.items()},TRAIN_only=True,evaluation_rows=0)

    def state(self):return dict(format=FORMAT,config=self.config,episodes=self.episodes)

    def restore(self,state):
        if state.get('format')!=FORMAT or state.get('config')!=self.config or set(state.get('episodes',{}))!=set(REGIONS):raise ValueError('Saved successful TRAIN replay contract differs')
        for region,episodes in state['episodes'].items():
            for episode in episodes:
                if episode['outcome']['layout']['target_region']!=region:raise ValueError('Saved successful replay region differs')
                self.add_episode(episode['rows'],episode['outcome'],source_run=episode['identity'].split('/wave')[0],split='train')


def add_completed_training_wave(bank,wave,outcomes,measured_batches,*,source_run):
    if wave['split']!='train':
        if measured_batches:raise ValueError('Evaluation cannot contribute successful learner batches')
        return 0
    added=0
    for outcome in outcomes:
        if not (outcome.get('result') or {}).get('success'):continue
        parts=[]
        for ids,batch in measured_batches:
            mask=ids==outcome['environment']
            if bool(mask.any()):parts.append({k:v[mask] for k,v in batch.items()})
        if not parts:raise ValueError('Physical success is missing its measured held-goal actions')
        bank.add_episode({k:torch.cat([p[k] for p in parts]) for k in KEYS},outcome,source_run=source_run,split='train');added+=1
    return added


def match_measured_success_paths(replay,paths,*,raw_critic_dim,clock_horizon):
    """Join closed native episodes to their original recorded21-D Q rows.

    A short hash narrows candidates only. Every raw pre/next critic value,
    reward, termination and elapsed clock must match; ambiguity is rejected.
    Physical delta commands are never inverted into hypothetical goal labels.
    """
    co=replay['critic_obs'].cpu();nc=replay['next_critic_obs'].cpu()
    if co.shape[1]!=raw_critic_dim+9 or nc.shape!=co.shape:raise ValueError('Goal critic context differs')
    def fingerprint(a,b):return hashlib.sha256(a[:86].numpy().tobytes()+b[:86].numpy().tobytes()).digest()
    wanted={};found={}
    for pi,path in enumerate(paths):
        n=len(path['reward'])
        for j in range(n):
            key=fingerprint(path['critic_obs'][j],path['next_critic_obs'][j])
            wanted.setdefault(key,[]).append((pi,j));found[(pi,j)]=[]
    for i in range(len(co)):
        key=fingerprint(co[i],nc[i])
        if key not in wanted:continue
        for pi,j in wanted[key]:
            p=paths[pi]
            if not torch.equal(co[i,:raw_critic_dim],p['critic_obs'][j]) \
                    or not torch.equal(nc[i,:raw_critic_dim],p['next_critic_obs'][j]) \
                    or replay['reward'][i]!=p['reward'][j] or replay['terminated'][i]!=p['terminated'][j]:continue
            clock=min(p['first_held_step']+j,clock_horizon)/clock_horizon
            if abs(float(co[i,raw_critic_dim])-clock)>1e-6:continue
            found[(pi,j)].append(i)
    indices=[]
    for pi,path in enumerate(paths):
        ids=[]
        for j in range(len(path['reward'])):
            matches=found[(pi,j)]
            if len(matches)!=1:raise ValueError(f'Measured successful row has{len(matches)} matching Q rows; cannot invent labels')
            ids.append(matches[0])
        if len(set(ids))!=len(ids):raise ValueError('Successful path matched a Q row more than once')
        indices.append(torch.tensor(ids,dtype=torch.long))
    return indices
