#!/usr/bin/env python3
"""Add successful-servo retention to a fresh quarter-Gaussian initialization.

All models, distributions and physical settings stay unchanged. This is not
a migration of a trained Q/optimizer/replay into another actor objective.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.body_policy_spread import validate_quarter_policy_state
from kuavo_isaaclab_scene.rl.multi_box.experiments.gentle_servo_critic_sac import GentleServoCriticSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.servo_success_retention import (
    ServoRetentionGentleSACPilot,servo_success_retention_contract,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.servo_critic import servo_critic_contract
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import KEYS
from kuavo_isaaclab_scene.rl.multi_box.rewards.absorbing_geometry import absorbing_geometry_contract
from prepare_actual_success_actor_tail import identical


def prepare(initial,experience):
    validate_quarter_policy_state(initial)
    goal=initial['goal_contract']
    if initial.get('artifact_type')!=GentleServoCriticSACPilot.artifact_type \
            or goal.get('critic_action_encoding')!=servo_critic_contract() \
            or goal['physical_contract']['reward_profile'].get('absorbing_geometry')!=absorbing_geometry_contract():
        raise ValueError('Successful-servo retention requires a fresh absorbing quarter-Gaussian servo actor')
    if initial['actor_updates'] or initial['critic_updates'] or len(initial.get('optimizers',()))!=4 \
            or any(o['state'] for o in initial['optimizers']) or initial['model']['critic_normalizer.count'] \
            or any(initial['successful_train_transitions']['episodes'].values()):
        raise ValueError('Trained models, Q, optimizer, normalization or success rows cannot migrate')
    if any(not torch.isfinite(v).all() for v in initial['model'].values()):
        raise ValueError('Initial models must be finite')
    if experience.get('goal_contract')!=goal \
            or set(experience.get('executed_goal_transitions',{}))!=set(KEYS) \
            or any(len(v) for v in experience['executed_goal_transitions'].values()) \
            or not identical(initial['successful_train_transitions'],experience.get('successful_train_transitions')) \
            or experience.get('measured_train_credit')!=initial.get('measured_train_credit') \
            or any(experience['measured_train_credit_bank']['episodes'].values()):
        raise ValueError('Matching closed empty TRAIN replay and success/n-step banks are required')
    state=deepcopy(initial);replay=deepcopy(experience)
    state['artifact_type']=ServoRetentionGentleSACPilot.artifact_type
    state['goal_contract'].update(name=state['artifact_type'],
        success_body_labels='measured_absolute_goal_MSE_plus_servo_equivalent_interval_Huber_v1',
        success_body_retention=servo_success_retention_contract())
    state['hybrid_contract']=state['hybrid_contract']|dict(success_body_retention=servo_success_retention_contract())
    replay['goal_contract']=deepcopy(state['goal_contract'])
    assert identical(state['model'],initial['model']) and identical(state['optimizers'],initial['optimizers'])
    return state,replay


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial-checkpoint',type=Path,required=True)
    p.add_argument('--training-manifest',type=Path,required=True)
    p.add_argument('--waypoints',type=Path,required=True)
    p.add_argument('--identity-checkpoint',type=Path,required=True,
        help='Closed actual TRAIN checkpoint for action identity checks only; its rows/models are not imported')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    exp=args.initial_checkpoint.parent/'staged_goal_experience.pt'
    sources=(args.initial_checkpoint,exp,args.training_manifest,args.waypoints,args.identity_checkpoint)
    if any(s.is_symlink() or not s.is_file() or s.stat().st_uid!=os.getuid() for s in sources):
        raise ValueError('Owned closed regular input files are required')
    hashes={str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}
    initial=torch.load(args.initial_checkpoint,map_location='cpu',weights_only=True)
    state,replay=prepare(initial,torch.load(exp,map_location='cpu',weights_only=True))
    physical=json.loads(args.training_manifest.read_text())
    goal_physical=initial['goal_contract']['physical_contract']
    if goal_physical!={k:physical.get(k) for k in goal_physical}:
        raise ValueError('Source physical manifest differs')
    identity=torch.load(args.identity_checkpoint,map_location='cpu',weights_only=True)
    from summarize_batched_staged_run import supported_success
    episodes=[e for es in identity['successful_train_transitions']['episodes'].values() for e in es]
    if identity['goal_contract']['physical_contract']!=goal_physical \
            or not identical(identity['body_anchor_state'],initial['body_anchor_state']) \
            or not episodes or any(not supported_success(e['outcome']) or e['outcome']['split']!='train' for e in episodes):
        raise ValueError('Actual completed safe TRAIN success states are needed for identity checks')
    raw=torch.cat([e['rows']['actor_obs'][::10] for e in episodes])
    from export_eval_q_videos import restored_agent
    old,_=restored_agent(initial);new,_=restored_agent(state)
    with torch.no_grad():
        assert torch.equal(old.act(raw,True),new.act(raw,True))
        assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    args.output_dir.mkdir(parents=True,exist_ok=False,mode=0o700)
    torch.save(state,args.output_dir/'checkpoint_00000000.pt')
    torch.save(replay,args.output_dir/'staged_goal_experience.pt')
    assert identical(state,torch.load(args.output_dir/'checkpoint_00000000.pt',map_location='cpu',weights_only=True))
    assert identical(replay,torch.load(args.output_dir/'staged_goal_experience.pt',map_location='cpu',weights_only=True))
    assert all(hashes[str(s)]==hashlib.sha256(s.read_bytes()).hexdigest() for s in sources)
    proof=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),source_SHA256=hashes,
        success_body_retention=servo_success_retention_contract(),actual_TRAIN_identity_rows=len(raw),
        initial_models_mean_jaw_Gaussian_Q_targets_and_normalizers_exactly_unchanged=True,
        source_identity_models_and_rows_NOT_imported=True,
        actual_TRAIN_online_success_nstep_rows0=True,actor_Q_updates0_four_optimizer_states0=True,
        reward_success_safety_controller_box_base_background_flap_randomization_unchanged=True,
        saved_inputs_and_source_hashes_verified=True,
        physical_gain_NOT_measured=True,independent_FINAL_unused=True,goal_not_complete=True)
    for name,data in [('initialization_verification.json',proof),('training_manifest.json',physical),
        ('waypoints.json',json.loads(args.waypoints.read_text())),
        ('manifest.json',dict(artifact_type=state['artifact_type'],goal_contract=state['goal_contract'],initialization_verification=proof)),
        ('status.json',dict(status='complete',initialized_not_trained=True))]:
        (args.output_dir/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output_dir=str(args.output_dir),actual_identity_rows=len(raw),
        all_models_and_initial_actions_exact=True,all_reward_banks_empty=True,training_NOT_started=True)))


if __name__=='__main__':main()
