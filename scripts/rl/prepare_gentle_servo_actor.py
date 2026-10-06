#!/usr/bin/env python3
"""Fresh servo-Q plus absorbing geometry and a fixed quarter Gaussian.

The initial mean/jaws stay exact; only the continuous log-std head and its
matching bounds/entropy target change. All old reward-bearing banks are empty.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib,json,math,os
from pathlib import Path
import torch

from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
from kuavo_isaaclab_scene.rl.multi_box.experiments.body_policy_spread import (
    body_policy_spread_contract,quarter_body_policy_config,shift_body_log_std,validate_quarter_policy_state,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.gentle_servo_critic_sac import GentleServoCriticSACPilot
from prepare_absorbing_geometry_actor import prepare as absorbing_prepare
from prepare_actual_success_actor_tail import identical


def prepare(initial,experience):
    state,replay=absorbing_prepare(initial,experience)
    state['config']=asdict(quarter_body_policy_config(SACConfig(**initial['config'])))
    state['model'],bias=shift_body_log_std(initial['model'])
    state['artifact_type']=GentleServoCriticSACPilot.artifact_type
    state['goal_contract'].update(name=state['artifact_type'],
        exploration_std_initial=state['config']['initial_policy_std'],exploration_std_cap=.03,
        body_policy_spread=body_policy_spread_contract())
    state['entropy_contract']=dict(name='squash_aware_active_dims_v2',
        target_per_dim=min(-1.,math.log(.03)+.5*math.log(2*math.pi*math.e)-.5))
    replay['goal_contract']=deepcopy(state['goal_contract'])
    validate_quarter_policy_state(state)
    return state,replay,bias


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial-checkpoint',type=Path,required=True)
    p.add_argument('--training-manifest',type=Path,required=True)
    p.add_argument('--waypoints',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    experience_path=a.initial_checkpoint.parent/'staged_goal_experience.pt'
    sources=(a.initial_checkpoint,experience_path,a.training_manifest,a.waypoints)
    if any(s.is_symlink() or not s.is_file() or s.stat().st_uid!=os.getuid() for s in sources):
        raise ValueError('Owned closed initialization files are required')
    hashes={str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}
    initial=torch.load(a.initial_checkpoint,map_location='cpu',weights_only=True)
    experience=torch.load(experience_path,map_location='cpu',weights_only=True)
    state,replay,bias=prepare(initial,experience)
    physical=json.loads(a.training_manifest.read_text())
    if initial['goal_contract']['physical_contract']!={k:physical.get(k) for k in initial['goal_contract']['physical_contract']}:
        raise ValueError('Source physical manifest differs')
    physical['reward_profile']=deepcopy(state['goal_contract']['physical_contract']['reward_profile'])
    from export_eval_q_videos import restored_agent
    old,_=restored_agent(initial);new,_=restored_agent(state)
    episodes=[e for es in initial['successful_train_transitions']['episodes'].values() for e in es]
    if not episodes:raise ValueError('Actual TRAIN observations required for initial mean identity')
    raw=torch.cat([e['rows']['actor_obs'][::10] for e in episodes])
    with torch.no_grad():
        assert torch.equal(old.act(raw,True),new.act(raw,True))
        oldmean,oldlog,oldjaw=old.parameters_at(old.actor_normalizer(old.actor_features(raw)))
        newmean,newlog,newjaw=new.parameters_at(new.actor_normalizer(new.actor_features(raw)))
        assert torch.equal(oldmean,newmean) and torch.equal(oldjaw,newjaw)
        torch.testing.assert_close(newlog.exp(),oldlog.exp()*.25,rtol=2e-6,atol=1e-8)
    for k,v in initial['model'].items():
        if k!=bias:assert torch.equal(v,state['model'][k])
    assert torch.equal(initial['model'][bias][:21],state['model'][bias][:21])
    assert torch.equal(initial['model'][bias][40:],state['model'][bias][40:])
    a.output_dir.mkdir(parents=True,exist_ok=False,mode=0o700)
    torch.save(state,a.output_dir/'checkpoint_00000000.pt')
    torch.save(replay,a.output_dir/'staged_goal_experience.pt')
    assert identical(state,torch.load(a.output_dir/'checkpoint_00000000.pt',map_location='cpu',weights_only=True))
    assert identical(replay,torch.load(a.output_dir/'staged_goal_experience.pt',map_location='cpu',weights_only=True))
    assert all(hashes[str(s)]==hashlib.sha256(s.read_bytes()).hexdigest() for s in sources)
    proof=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),source_SHA256=hashes,
        body_policy_spread=body_policy_spread_contract(),
        initial_greedy_goal_and_binary_jaw_identity_actual_TRAIN_rows=len(raw),
        initial_actual_TRAIN_body_std_ratio_verified=.25,only19_body_log_std_biases_changed=True,
        all_mean_jaw_Q_target_normalizer_entropy_tensors_preserved=True,
        all_four_optimizer_states_empty=True,actor_Q_updates0=True,critic_normalizer_count0=True,
        old_reward_success_nstep_online_replay_rows_imported0=True,
        old_success_reward_rows_removed=sum(len(e['rows']['reward']) for e in episodes),
        Gaussian_variance_changed_in_collection_actor_and_targets=True,
        larger_episode_body_bias_and_AR_correlation_unchanged=True,
        source_files_unchanged=True,saved_checkpoint_and_experience_verified=True,
        controller_goal_space_and_randomization_success_safety_unchanged=True,
        physical_performance_NOT_measured=True,independent_FINAL_unused=True,goal_not_complete=True)
    for name,data in (('initialization_verification.json',proof),('training_manifest.json',physical),
        ('waypoints.json',json.loads(a.waypoints.read_text())),
        ('status.json',dict(status='complete',initialized_not_trained=True)),
        ('manifest.json',dict(artifact_type=state['artifact_type'],goal_contract=state['goal_contract'],initialization_verification=proof))):
        (a.output_dir/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output_dir=str(a.output_dir),identity_rows=len(raw),std_ratio=.25,
        all_reward_banks_empty=True,initialized_not_trained=True)))


if __name__=='__main__':main()
