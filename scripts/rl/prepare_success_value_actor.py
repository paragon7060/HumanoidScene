#!/usr/bin/env python3
"""Preserve a measured actor while starting safe-success64 with fresh Q/banks.

The matching untrained checkpoint supplies Q, normalization and empty optimizer
states. Neither the trained source's Q nor any old reward-bearing row is used.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib, json, os
from pathlib import Path
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_reanchored_sac import ReanchoredActualFlapSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.measured_train_credit import MeasuredTrainCreditBank
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import TrainSuccessBank, KEYS
from kuavo_isaaclab_scene.rl.multi_box.rewards.success_value import with_success_value_profile
from prepare_actual_success_actor_tail import identical


ACTOR_PREFIXES=('actor.','jaw_actor.','actor_normalizer.')


def prepare(initial, actor, experience):
    goal=initial.get('goal_contract',{})
    if (initial.get('artifact_type')!=ReanchoredActualFlapSACPilot.artifact_type
            or actor.get('artifact_type')!=initial['artifact_type']
            or (goal.get('actor_dim'),goal.get('critic_dim'))!=(518,577)
            or initial.get('action_dim')!=21):
        raise ValueError('Success64 requires matching reanchored actual-flap checkpoints')
    if (initial['actor_updates'] or initial['critic_updates']
            or len(initial.get('optimizers',()))!=4
            or any(o['state'] for o in initial['optimizers'])
            or initial['model']['critic_normalizer.count']):
        raise ValueError('Fresh Q, critic normalization and all four optimizers are required')
    if (goal!=actor['goal_contract'] or initial['config']!=actor['config']
            or any(not identical(initial[k],actor[k]) for k in
                   ('body_anchor_state','frozen_warm_start','frozen_actor_prior'))):
        raise ValueError('Actor/controller/physics/reward coordinates differ from initialization')
    if (experience.get('goal_contract')!=goal
            or set(experience.get('executed_goal_transitions',{}))!=set(KEYS)
            or any(len(v) for v in experience['executed_goal_transitions'].values())
            or experience.get('measured_train_credit')!=initial.get('measured_train_credit')
            or initial.get('measured_train_credit') is None):
        raise ValueError('Matching empty online replay and measured TRAIN configuration are required')
    if set(initial['model'])!=set(actor['model']) or any(
            not torch.isfinite(v).all() or v.shape!=actor['model'][k].shape
            or not torch.isfinite(actor['model'][k]).all() for k,v in initial['model'].items()):
        raise ValueError('Model shapes or finite values differ')
    state=deepcopy(initial);replay=deepcopy(experience)
    for k in state['model']:
        if k.startswith(ACTOR_PREFIXES):state['model'][k]=actor['model'][k].clone()
    state['goal_contract']['physical_contract']=with_success_value_profile(goal['physical_contract'])
    success=TrainSuccessBank(518,577,goal['train_success_retention'])
    credit=MeasuredTrainCreditBank(518,577,initial['config']['gamma'],initial['measured_train_credit'])
    state['successful_train_transitions']=success.state()
    state['measured_train_credit_bank_report']=credit.report()
    state['latest_actor_metrics']={}
    state['success_value_initialization']=dict(source_actor_updates=actor['actor_updates'],
        source_critic_updates_NOT_imported=actor['critic_updates'],new_Q_and_optimizer_updates=0,
        only_actor_jaw_and_actor_normalizer_imported=True,old_reward_rows_imported=0)
    replay.update(goal_contract=deepcopy(state['goal_contract']),
        successful_train_transitions=success.state(),measured_train_credit_bank=credit.state())
    replay.pop('source_experience_collection_contract',None)
    assert all(torch.equal(state['model'][k],(actor if k.startswith(ACTOR_PREFIXES) else initial)['model'][k]) for k in state['model'])
    assert identical(state['optimizers'],initial['optimizers'])
    return state,replay


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial-checkpoint',type=Path,required=True)
    p.add_argument('--actor-checkpoint',type=Path,required=True)
    p.add_argument('--actor-capture-proof',type=Path,required=True)
    p.add_argument('--training-manifest',type=Path,required=True)
    p.add_argument('--waypoints',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    exp=args.initial_checkpoint.parent/'staged_goal_experience.pt'
    sources=(args.initial_checkpoint,exp,args.actor_checkpoint,args.actor_capture_proof,args.training_manifest,args.waypoints)
    if any(f.is_symlink() or not f.is_file() or f.stat().st_uid!=os.getuid() for f in sources):
        raise ValueError('Owned closed regular inputs and protected actor are required')
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    proof=json.loads(args.actor_capture_proof.read_text())
    if (Path(proof['protected_checkpoint']).resolve()!=args.actor_checkpoint.resolve()
            or proof['checkpoint_SHA256']!=hashes[str(args.actor_checkpoint)]
            or proof.get('split')!='validation' or proof.get('wave')!=16):
        raise ValueError('Matching protected final DEV16 actor proof is required')
    initial=torch.load(args.initial_checkpoint,map_location='cpu',weights_only=True)
    actor=torch.load(args.actor_checkpoint,map_location='cpu',weights_only=True)
    state,replay=prepare(initial,actor,torch.load(exp,map_location='cpu',weights_only=True))
    physical=json.loads(args.training_manifest.read_text())
    if goal_physical:=initial['goal_contract']['physical_contract']:
        if goal_physical!={k:physical.get(k) for k in goal_physical}:raise ValueError('Source physical manifest differs')
    physical['reward_profile']=deepcopy(state['goal_contract']['physical_contract']['reward_profile'])
    from export_eval_q_videos import restored_agent
    old,_=restored_agent(actor);new,_=restored_agent(state)
    episodes=[e for es in actor['successful_train_transitions']['episodes'].values() for e in es]
    if not episodes:raise ValueError('Actual matching TRAIN states are required for actor identity')
    raw=torch.cat([e['rows']['actor_obs'][::10] for e in episodes])
    with torch.no_grad():assert torch.equal(old.act(raw,True),new.act(raw,True))
    args.output_dir.mkdir(parents=True,exist_ok=False,mode=0o700)
    torch.save(state,args.output_dir/'checkpoint_00000000.pt');torch.save(replay,args.output_dir/'staged_goal_experience.pt')
    assert identical(state,torch.load(args.output_dir/'checkpoint_00000000.pt',map_location='cpu',weights_only=True))
    assert identical(replay,torch.load(args.output_dir/'staged_goal_experience.pt',map_location='cpu',weights_only=True))
    assert all(hashes[str(f)]==hashlib.sha256(f.read_bytes()).hexdigest() for f in sources)
    report=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),source_SHA256=hashes,
        success_value=physical['reward_profile']['success_value'],source_actor_updates=actor['actor_updates'],
        source_critic_updates_NOT_imported=actor['critic_updates'],actual_TRAIN_identity_states=len(raw),
        greedy_body_goals_binary_jaws_and_actor_normalizer_exact=True,
        Q_targets_entropy_critic_normalizer_from_matching_fresh_input=True,
        all_four_optimizer_states0_actor_Q_updates0=True,online_success_nstep_reward_rows0=True,
        initial_actor_source_has_safe_upper_right_DEV_successes_but_new_physical_performance_NOT_measured=True,
        original_controller_randomization_success_safety_unchanged=True,
        source_files_and_saved_checkpoint_replay_verified=True,independent_FINAL_unused=True,goal_not_complete=True)
    for name,value in [('initialization_verification.json',report),('training_manifest.json',physical),
        ('waypoints.json',json.loads(args.waypoints.read_text())),
        ('manifest.json',dict(artifact_type=state['artifact_type'],goal_contract=state['goal_contract'],initialization_verification=report)),
        ('status.json',dict(status='complete',initialized_not_trained=True))]:
        (args.output_dir/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output_dir=str(args.output_dir),actual_identity_states=len(raw),fresh_Q_banks0=True,success_event64=True,training_NOT_launched=True)))


if __name__=='__main__':main()
