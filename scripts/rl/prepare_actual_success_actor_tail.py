#!/usr/bin/env python3
"""Keep a fresh actor/Q, seed only matching completed TRAIN episode banks.

Inputs are closed checkpoints, not the source run's writer-open replay/HDF.
No behavior action, reward, success flag or physical contract is relabelled.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_reanchored_sac import ReanchoredActualFlapSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.measured_train_credit import (
    MeasuredTrainCreditBank, VARIANT as CREDIT_VARIANT, measured_credit_config,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import KEYS, TrainSuccessBank, retention_config


def identical(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(identical(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
    return a==b


def prepare(initial,measured):
    if initial.get('artifact_type')!=ReanchoredActualFlapSACPilot.artifact_type \
            or measured.get('artifact_type')!=initial['artifact_type']:
        raise ValueError('Actor-tail initialization requires matching reanchored goal checkpoints')
    if initial['actor_updates'] or initial['critic_updates'] \
            or any(o['state'] for o in initial['optimizers']) \
            or initial['model']['critic_normalizer.count'] \
            or any(initial['successful_train_transitions']['episodes'].values()):
        raise ValueError('Initial models, Q, optimizers and success bank must be fresh')
    if initial['goal_contract']!=measured['goal_contract'] or initial['config']!=measured['config'] \
            or any(not identical(initial[k],measured[k]) for k in
                   ('body_anchor_state','frozen_warm_start','frozen_actor_prior')):
        raise ValueError('TRAIN source has different physical, perception, goal, anchor or reward coordinates')
    goal=initial['goal_contract'];actor_dim=goal['actor_dim'];critic_dim=goal['critic_dim']
    sampling=retention_config('tail64-half')
    success=deepcopy(measured['successful_train_transitions'])
    if success['config']!=retention_config():raise ValueError('Expected the original successful TRAIN sampling contract')
    success['config']=sampling
    bank=TrainSuccessBank(actor_dim,critic_dim,config=sampling);bank.restore(success)
    if not bank.size or any(not bank.episodes[r] for r in bank.episodes):
        raise ValueError('Matching real successful TRAIN data from all four regions is required')
    credit_config=measured_credit_config(CREDIT_VARIANT)
    if measured.get('measured_train_credit')!=credit_config \
            or initial.get('measured_train_credit') not in (None,credit_config):
        raise ValueError('Matching measured TRAIN credit configuration is required')
    credit=MeasuredTrainCreditBank(actor_dim,critic_dim,initial['config']['gamma'],credit_config)
    if 'measured_train_credit_bank' in measured:
        credit.restore(measured['measured_train_credit_bank'])
        credit_seed_source='completed_success_and_failure_TRAIN_episode_bank'
    else:
        # Training checkpoints carry complete successful paths, but keep the
        # full success/failure n-step bank in the writer-open replay artifact.
        # Never read that live artifact or pretend its failures were imported.
        for episodes in bank.episodes.values():
            for episode in episodes:
                credit.add_episode(episode['rows'],episode['outcome'],
                    source_run=episode['identity'].split('/wave')[0])
        credit_seed_source='completed_safe_successful_TRAIN_paths_in_immutable_checkpoint'
    output=deepcopy(initial)
    output['goal_contract']['train_success_retention']=sampling
    output['successful_train_transitions']=bank.state()
    output['measured_train_credit']=deepcopy(credit_config)
    output['measured_train_credit_bank_report']=credit.report()
    rows={k:torch.empty(0,d) for k,d in (('actor_obs',actor_dim),('critic_obs',critic_dim),('action',21),
            ('next_actor_obs',actor_dim),('next_critic_obs',critic_dim))}
    rows.update(reward=torch.empty(0),terminated=torch.empty(0,dtype=torch.bool))
    assert set(rows)==set(KEYS)
    experience=dict(goal_contract=deepcopy(output['goal_contract']),executed_goal_transitions=rows,
        successful_train_transitions=bank.state(),measured_train_credit_bank=credit.state())
    for key in ('body_behavior','body_behavior_origin','body_behavior_statistics','body_saturation','body_saturation_origin',
                'success_jaw_balance','success_jaw_balance_origin','jaw_saturation','jaw_saturation_origin',
                'jaw_behavior','jaw_behavior_origin','jaw_behavior_statistics','measured_train_credit'):
        if key in initial:experience[key]=deepcopy(initial[key])
    experience['measured_train_credit']=deepcopy(credit_config)
    if not identical(output['model'],initial['model']) or not identical(output['optimizers'],initial['optimizers']):
        raise ValueError('Fresh actor/Q/optimizer tensors changed during seed preparation')
    return output,experience,dict(actual_completed_TRAIN_success_bank=bank.report(),
        actual_completed_TRAIN_success_and_failure_credit_bank=credit.report(),
        initial_measured_credit_seed_source=credit_seed_source,
        initial_measured_credit_seed_failed_episodes=sum(r['failures'] for r in credit.report()['by_region'].values()),
        new_online_TRAIN_failures_remain_collected_and_added=True,
        online_replay_initially_empty=True,source_Q_actor_and_optimizers_not_imported=True,
        fresh_initial_actor_and_Q_tensors_bit_identical=True,source_actions_rewards_and_outcomes_unchanged=True,
        actor_sampling=sampling['actor_sampling'],Q_success_sampling20_to5_unchanged=True,
        no_DEV_FINAL_VR_IK_or_new_physics_data=True,initialized_not_trained=True,goal_not_complete=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-checkpoint',type=Path,required=True)
    parser.add_argument('--matching-train-checkpoint',type=Path,required=True)
    parser.add_argument('--matching-train-proof',type=Path,required=True)
    parser.add_argument('--training-manifest',type=Path,required=True)
    parser.add_argument('--waypoints',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    sources=(args.initial_checkpoint,args.matching_train_checkpoint,args.matching_train_proof,
             args.training_manifest,args.waypoints)
    if any(p.is_symlink() or not p.is_file() or p.stat().st_uid!=os.getuid() for p in sources):
        raise ValueError('Owned, closed regular input files are required')
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    proof=json.loads(args.matching_train_proof.read_text())
    if Path(proof['protected_checkpoint']).resolve()!=args.matching_train_checkpoint.resolve() \
            or hashes[str(args.matching_train_checkpoint)]!=proof['checkpoint_SHA256'] \
            or not proof['writer_owner_run_and_CUDA3_verified'] or not proof['active_HDF_and_replay_not_read']:
        raise ValueError('Missing immutable completed TRAIN checkpoint capture proof')
    initial=torch.load(args.initial_checkpoint,map_location='cpu',weights_only=True)
    measured=torch.load(args.matching_train_checkpoint,map_location='cpu',weights_only=True)
    state,experience,verification=prepare(initial,measured)
    physical=json.loads(args.training_manifest.read_text())
    if state['goal_contract']['physical_contract']!={k:physical.get(k) for k in state['goal_contract']['physical_contract']}:
        raise ValueError('Physical training manifest differs')
    # Replay-independent restoration checks exact projected greedy outputs.
    from export_eval_q_videos import restored_agent
    baseline,_=restored_agent(initial);candidate,_=restored_agent(state)
    inputs=torch.cat([e['rows']['actor_obs'][::10] for es in state['successful_train_transitions']['episodes'].values() for e in es])
    with torch.no_grad():assert torch.equal(baseline.act(inputs,True),candidate.act(inputs,True))
    args.output_dir.mkdir(parents=True,exist_ok=False,mode=0o700)
    torch.save(state,args.output_dir/'checkpoint_00000000.pt')
    torch.save(experience,args.output_dir/'staged_goal_experience.pt')
    saved=torch.load(args.output_dir/'checkpoint_00000000.pt',map_location='cpu',weights_only=True)
    assert identical(saved,state)
    saved_experience=torch.load(args.output_dir/'staged_goal_experience.pt',map_location='cpu',weights_only=True)
    assert identical(saved_experience,experience)
    assert all(hashes[str(p)]==hashlib.sha256(p.read_bytes()).hexdigest() for p in sources)
    verification.update(recorded_utc=datetime.now(timezone.utc).isoformat(),source_SHA256=hashes,
        actual_TRAIN_identity_input_rows=len(inputs),greedy_goals_and_jaws_bit_identical=True,
        serialized_checkpoint_and_episode_banks_identical=True,source_files_unchanged=True)
    for name,data in (('training_manifest.json',physical),('waypoints.json',json.loads(args.waypoints.read_text())),
        ('initialization_verification.json',verification),('status.json',dict(status='complete',initialized_not_trained=True)),
        ('manifest.json',dict(artifact_type=state['artifact_type'],goal_contract=state['goal_contract'],
            TRAIN_actual_episode_seed=verification,initialization_verification=verification))):
        (args.output_dir/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output_dir=str(args.output_dir),success_rows=verification['actual_completed_TRAIN_success_bank']['rows'],
        input_rows=len(inputs),actor_sampling=verification['actor_sampling'],initialized_not_trained=True)))


if __name__=='__main__':main()
