"""Validate literal closed TRAIN successes for a fresh physical-command Q."""
from types import SimpleNamespace
import json

import h5py
import torch

from .physical_body_actions import body_command,full_command
from .physical_body_sac import physical_signature
from .staged_train_success import validate_success_outcome


def seed_physical_training_successes(pilot,dataset,outcomes,*,source_run):
    selected={}
    for outcome in outcomes:
        validate_success_outcome('train',outcome)
        identity=(outcome['wave'],outcome['environment'])
        if identity in selected:raise ValueError('Duplicate literal TRAIN episode')
        selected[identity]=outcome
    if not selected:raise ValueError('No actual TRAIN success to seed physical Q')
    original_stage,original_anchor=pilot.stage,pilot.anchor
    evidence=[];prepared=[]
    try:
        with h5py.File(dataset,'r') as stream:
            meta=json.loads(stream.attrs['manifest_json'])
            declared={(row['wave'],row['environment']):row['layout'] for row in meta.get('episode_layouts',[])}
            if physical_signature(meta['training_contract'])!=pilot.physical_contract \
                    or meta.get('collection_source') not in ('staged_base_hold_remaining_hybrid_sac_v1',pilot.artifact_type) \
                    or meta.get('base_waypoint_probe',{}).get('enabled') \
                    or meta.get('centered_world_probe') or meta.get('contact_stability_probe') \
                    or meta.get('background_placement_probe',{}).get('enabled'):
                raise ValueError('Native physical/waypoint dynamics differ; probes cannot seed Q')
            for episode in stream['episodes'].values():
                identity=(int(episode.attrs['wave']),int(episode.attrs['environment']))
                if identity not in selected:continue
                outcome=selected[identity];result=outcome['result'];stage=result['staged_base']
                if not episode.attrs.get('success') or not episode.attrs.get('initial_layout_guard_valid') \
                        or json.loads(episode.attrs['layout_json'])!=outcome['layout'] \
                        or declared.get(identity)!=outcome['layout'] or declared[identity].get('split')!='train':
                    raise ValueError('Native TRAIN identity or physical success differs')
                source={k:torch.tensor(v[:],device=pilot.device) for k,v in episode['transitions'].items()}
                n=len(source['reward']);first=stage['manipulation_start']
                if n!=outcome['executed_transition_rows'] or not 0<=first<n \
                        or source['actor_obs'].shape!=(n,464) or source['critic_obs'].shape!=(n,530) \
                        or source['action'].shape!=(n,24) or not all(torch.isfinite(v).all() for v in source.values()) \
                        or source['unsafe'].any() or not source['success'][-1] or not source['terminated'][-1] \
                        or source['terminated'][:-1].any():
                    raise ValueError('Native TRAIN path is incomplete, unsafe or malformed')
                last=source['next_critic_obs'][-1,464:530]
                if not (last[35:37]>.5).all() or not (last[41:43]>.5).all() \
                        or not (last[49:55]>.5).all() or last[43]<1 or last[44]<1 \
                        or (source['next_critic_obs'][:,523:530]>.5).any():
                    raise ValueError('Literal terminal state lacks current bilateral/lift/hold/safety evidence')
                if not torch.equal(source['critic_obs'][:,:464],source['actor_obs']) \
                        or not torch.equal(source['next_critic_obs'][:,:464],source['next_actor_obs']):
                    raise ValueError('Native pre/next actor and critic snapshots differ')
                pilot.stage=SimpleNamespace(phase='held_grasp',manipulation_start=first,
                    target_xy=torch.tensor(stage['base_target_xy_rack_m'],device=pilot.device)[None],
                    target_yaw=stage['base_target_yaw_rack_rad'],name=original_stage.name,templates=original_stage.templates)
                pilot.anchor=pilot.coordinates.box_anchor(source['actor_obs'][:1])
                clock=torch.arange(n-first,device=pilot.device)
                ao,co=pilot.observations(source['actor_obs'][first:],source['critic_obs'][first:],clock)
                na,nc=pilot.observations(source['next_actor_obs'][first:],source['next_critic_obs'][first:],clock+1)
                literal=body_command(source['action'][first:])
                physical=full_command(pilot.coordinates,ao,literal)
                # CPU/GPU matrix arithmetic can differ at the final float32
                # bit. This is a feedback-law check, not a goal inverse label.
                base_error=float((physical[:,:3]-source['action'][first:,:3]).abs().max())
                if base_error>1e-5:raise ValueError(f'Native held-base feedback differs by {base_error}')
                if not torch.equal(physical[:,3:],source['action'][first:,3:]):
                    raise ValueError('Physical command subset changed literal measured actions')
                rows=dict(actor_obs=ao,critic_obs=co,action=literal,next_actor_obs=na,next_critic_obs=nc,
                    reward=source['reward'][first:].float(),terminated=source['terminated'][first:].bool())
                prepared.append((rows,outcome))
                evidence.append(dict(wave=outcome['wave'],environment=outcome['environment'],seed=outcome['layout']['seed'],
                    region=outcome['layout']['target_region'],held_rows=n-first,base_feedback_max_error=base_error))
        if len(prepared)!=len(selected):raise ValueError('A declared TRAIN path is missing from its native dataset')
        # Validate every case before mutating the destination bank/replay.
        temporary=type(pilot.success_bank)(480,539)
        temporary.restore(pilot.success_bank.state())
        for rows,outcome in prepared:temporary.add_episode(rows,outcome,source_run=source_run,split='train')
        for rows,outcome in prepared:
            pilot.add_native_rows(rows)
            pilot.success_bank.add_episode(rows,outcome,source_run=source_run,split='train')
    finally:pilot.stage,pilot.anchor=original_stage,original_anchor
    pilot.seed_provenance=dict(literal_native_TRAIN_physical_commands=True,source_run=source_run,
        requested_goals_inverted=False,old_goal_Q_imported=False,DEV_FINAL_imported=False,
        original_physical_and_held_base_feedback_verified=True,episodes=evidence,new_training_updates=0)
    return pilot.seed_provenance
