#!/usr/bin/env python3
"""Opt into the exact servo critic only from a closed fresh initial learner.

Keep the same actor/controller, untrained critic tensors and actual success
episode goals. No learned absolute-goal Q or its optimizer can migrate.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_reanchored_sac import ReanchoredActualFlapSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_servo_critic_sac import ServoCriticReanchoredSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.servo_critic import servo_critic_contract
from prepare_actual_success_actor_tail import identical


def prepare(initial,experience):
    if initial.get('artifact_type')!=ReanchoredActualFlapSACPilot.artifact_type \
            or initial.get('algorithm')!='hybrid_goal_sac':
        raise ValueError('Servo critic requires a fresh matching reanchored goal learner')
    if initial['actor_updates'] or initial['critic_updates'] or len(initial.get('optimizers',()))!=4 \
            or any(o['state'] for o in initial['optimizers']) \
            or initial['model']['critic_normalizer.count'] \
            or any(len(v) for v in experience['executed_goal_transitions'].values()):
        raise ValueError('Learned Q/optimizer or nonempty online replay cannot migrate to servo critic')
    if experience['goal_contract']!=initial['goal_contract'] \
            or (initial['goal_contract']['actor_dim'],initial['goal_contract']['critic_dim'])!=(518,577) \
            or initial['goal_contract'].get('critic_action_encoding') is not None \
            or initial['hybrid_contract']['Q_action_coordinates']!='actual_absolute_projected_goals':
        raise ValueError('Initial checkpoint and empty experience have different goal coordinates')
    if not identical(initial.get('successful_train_transitions'),experience.get('successful_train_transitions')):
        raise ValueError('Initial matching actual TRAIN success episode banks differ')
    output=deepcopy(initial);replay=deepcopy(experience);contract=servo_critic_contract()
    output['artifact_type']=ServoCriticReanchoredSACPilot.artifact_type
    output['goal_contract']['name']=ServoCriticReanchoredSACPilot.artifact_type
    output['goal_contract']['critic_action_encoding']=contract
    output['hybrid_contract'].update(Q_action_coordinates=contract['critic_action_coordinates'],
        critic_action_encoding=contract)
    replay['goal_contract']=deepcopy(output['goal_contract'])
    assert identical(initial['model'],output['model']) and identical(initial['optimizers'],output['optimizers'])
    assert identical(initial['successful_train_transitions'],output['successful_train_transitions'])
    assert identical(experience['executed_goal_transitions'],replay['executed_goal_transitions'])
    return output,replay


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
    if state['goal_contract']['physical_contract']!={k:physical.get(k) for k in state['goal_contract']['physical_contract']}:
        raise ValueError('Physical manifest differs from the initial actor/controller')
    from export_eval_q_videos import restored_agent
    baseline,_=restored_agent(initial);candidate,_=restored_agent(state)
    episodes=[e for es in state['successful_train_transitions']['episodes'].values() for e in es]
    if not episodes:raise ValueError('Actual safe completed TRAIN paths are required for this preparation')
    inputs=torch.cat([e['rows']['actor_obs'][::10] for e in episodes])
    with torch.no_grad():
        assert torch.equal(baseline.act(inputs,True),candidate.act(inputs,True))
        for episode in episodes:
            rows=episode['rows'];encoded=candidate.critic_action_features(rows['actor_obs'],rows['action'])
            assert torch.isfinite(encoded).all() and (encoded.abs()<=1).all()
            assert torch.equal(encoded[:,19:],rows['action'][:,19:])
    args.output_dir.mkdir(parents=True,exist_ok=False,mode=0o700)
    torch.save(state,args.output_dir/'checkpoint_00000000.pt')
    torch.save(replay,args.output_dir/'staged_goal_experience.pt')
    assert identical(state,torch.load(args.output_dir/'checkpoint_00000000.pt',map_location='cpu',weights_only=True))
    assert identical(replay,torch.load(args.output_dir/'staged_goal_experience.pt',map_location='cpu',weights_only=True))
    assert all(hashes[str(p)]==hashlib.sha256(p.read_bytes()).hexdigest() for p in sources)
    proof=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),source_SHA256=hashes,
        critic_encoding=servo_critic_contract(),greedy_goal_and_jaw_identity_rows=len(inputs),
        actual_TRAIN_success_episodes=len(episodes),actual_TRAIN_success_rows=sum(len(e['rows']['reward']) for e in episodes),
        actual_goal_labels_rewards_outcomes_and_banks_unchanged=True,
        all_actual_TRAIN_goal_labels_decoded_finite_bounded_with_same_binary_jaws=True,
        initial_actor_and_untrained_Q_tensor_identity=True,actor_Q_updates0=True,
        all_four_optimizer_states_empty=True,online_replay_empty=True,
        learned_absolute_goal_Q_optimizers_or_online_replay_imported=False,
        saved_checkpoint_and_experience_verified=True,source_files_unchanged=True,
        no_physical_success_or_improvement_claim=True,independent_FINAL_unused=True,goal_not_complete=True)
    for name,data in (('initialization_verification.json',proof),('training_manifest.json',physical),
        ('waypoints.json',json.loads(args.waypoints.read_text())),
        ('status.json',dict(status='complete',initialized_not_trained=True)),
        ('manifest.json',dict(artifact_type=state['artifact_type'],goal_contract=state['goal_contract'],initialization_verification=proof))):
        (args.output_dir/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output_dir=str(args.output_dir),identity_rows=len(inputs),success_rows=proof['actual_TRAIN_success_rows'],initialized_not_trained=True)))


if __name__=='__main__':main()
