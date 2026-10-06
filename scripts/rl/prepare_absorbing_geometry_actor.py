#!/usr/bin/env python3
"""Prepare the recognized absorbing reward from an untrained servo-Q learner.

The actor/controller stay exact. All old reward-bearing episode banks are
emptied; only newly measured TRAIN rows can seed this reward's Q and n-step Q.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_servo_critic_sac import ServoCriticReanchoredSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.servo_critic import servo_critic_contract
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import TrainSuccessBank,KEYS
from kuavo_isaaclab_scene.rl.multi_box.experiments.measured_train_credit import MeasuredTrainCreditBank
from kuavo_isaaclab_scene.rl.multi_box.rewards.absorbing_geometry import with_absorbing_geometry_profile
from prepare_actual_success_actor_tail import identical


def prepare(initial,experience):
    goal=initial.get('goal_contract',{})
    if (initial.get('artifact_type')!=ServoCriticReanchoredSACPilot.artifact_type
            or initial.get('algorithm')!='hybrid_goal_sac'
            or (goal.get('actor_dim'),goal.get('critic_dim'))!=(518,577)
            or goal.get('critic_action_encoding')!=servo_critic_contract()):
        raise ValueError('Absorbing geometry requires the recognized fresh servo critic')
    if (initial['actor_updates'] or initial['critic_updates'] or len(initial.get('optimizers',()))!=4
            or any(o['state'] for o in initial['optimizers']) or initial['model']['critic_normalizer.count']):
        raise ValueError('Trained Q, actor updates, optimizer or critic normalization cannot migrate')
    if any(not torch.isfinite(v).all() for v in initial['model'].values() if isinstance(v,torch.Tensor)):
        raise ValueError('Initial actor and untrained critics must be finite')
    if experience.get('goal_contract')!=goal or set(experience.get('executed_goal_transitions',{}))!=set(KEYS) \
            or any(len(v) for v in experience['executed_goal_transitions'].values()) \
            or not identical(initial.get('successful_train_transitions'),experience.get('successful_train_transitions')):
        raise ValueError('Matching closed initialization and empty online replay are required')
    if experience.get('measured_train_credit')!=initial.get('measured_train_credit') \
            or initial.get('measured_train_credit') is None:
        raise ValueError('Matching actual n-step configuration is required')
    success=TrainSuccessBank(518,577,goal['train_success_retention'])
    credit=MeasuredTrainCreditBank(518,577,initial['config']['gamma'],initial['measured_train_credit'])
    state=deepcopy(initial);replay=deepcopy(experience)
    state['goal_contract']['physical_contract']=with_absorbing_geometry_profile(goal['physical_contract'])
    state['successful_train_transitions']=success.state()
    state['measured_train_credit_bank_report']=credit.report()
    state['latest_actor_metrics']={}
    replay.update(goal_contract=deepcopy(state['goal_contract']),successful_train_transitions=success.state(),
        measured_train_credit_bank=credit.state())
    replay.pop('source_experience_collection_contract',None)
    assert identical(initial['model'],state['model']) and identical(initial['optimizers'],state['optimizers'])
    return state,replay


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-checkpoint',type=Path,required=True)
    parser.add_argument('--training-manifest',type=Path,required=True)
    parser.add_argument('--waypoints',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    exp=args.initial_checkpoint.parent/'staged_goal_experience.pt'
    sources=(args.initial_checkpoint,exp,args.training_manifest,args.waypoints)
    if any(p.is_symlink() or not p.is_file() or p.stat().st_uid!=os.getuid() for p in sources):
        raise ValueError('Owned closed regular initialization inputs are required')
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    initial=torch.load(args.initial_checkpoint,map_location='cpu',weights_only=True)
    experience=torch.load(exp,map_location='cpu',weights_only=True)
    state,replay=prepare(initial,experience)
    physical=json.loads(args.training_manifest.read_text())
    if initial['goal_contract']['physical_contract']!={k:physical.get(k) for k in initial['goal_contract']['physical_contract']}:
        raise ValueError('Manifest differs from the recognized initial reward/controller')
    physical['reward_profile']=deepcopy(state['goal_contract']['physical_contract']['reward_profile'])
    from export_eval_q_videos import restored_agent
    before,_=restored_agent(initial);after,_=restored_agent(state)
    episodes=[e for es in initial['successful_train_transitions']['episodes'].values() for e in es]
    if not episodes:raise ValueError('Actual TRAIN observations are required for initial action identity')
    inputs=torch.cat([e['rows']['actor_obs'][::10] for e in episodes])
    with torch.no_grad():assert torch.equal(before.act(inputs,True),after.act(inputs,True))
    args.output_dir.mkdir(parents=True,exist_ok=False,mode=0o700)
    torch.save(state,args.output_dir/'checkpoint_00000000.pt')
    torch.save(replay,args.output_dir/'staged_goal_experience.pt')
    assert identical(state,torch.load(args.output_dir/'checkpoint_00000000.pt',map_location='cpu',weights_only=True))
    assert identical(replay,torch.load(args.output_dir/'staged_goal_experience.pt',map_location='cpu',weights_only=True))
    assert all(hashes[str(p)]==hashlib.sha256(p.read_bytes()).hexdigest() for p in sources)
    proof=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),source_SHA256=hashes,
        absorbing_geometry=physical['reward_profile']['absorbing_geometry'],
        greedy_goal_and_binary_jaw_identity_actual_TRAIN_rows=len(inputs),
        old_success_reward_rows_removed=sum(len(e['rows']['reward']) for e in episodes),
        actual_actor_and_untrained_Q_tensors_preserved=True,actor_Q_updates0=True,
        all_four_optimizer_states_empty=True,critic_normalizer_count0=True,
        old_online_replay_success_and_nstep_reward_rows_imported0=True,
        reward_banks_empty_new_measured_TRAIN_required=True,
        source_files_unchanged=True,saved_checkpoint_and_experience_verified=True,
        randomization_controller_success_safety_unchanged=True,
        physical_performance_NOT_measured=True,independent_FINAL_unused=True,goal_not_complete=True)
    for name,data in (('initialization_verification.json',proof),('training_manifest.json',physical),
        ('waypoints.json',json.loads(args.waypoints.read_text())),
        ('status.json',dict(status='complete',initialized_not_trained=True)),
        ('manifest.json',dict(artifact_type=state['artifact_type'],goal_contract=state['goal_contract'],initialization_verification=proof))):
        (args.output_dir/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output_dir=str(args.output_dir),identity_rows=len(inputs),new_reward_banks_empty=True,initialized_not_trained=True)))


if __name__=='__main__':main()
