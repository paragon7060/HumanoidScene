#!/usr/bin/env python3
"""Fork pristine regional SAC with only a future critic support penalty."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,io,json
from pathlib import Path

import torch

from compare_actor_train_memory import owned_stable_bytes
from prepare_actual_success_actor_tail import identical
from prepare_greedy_collection_actor import require_pristine_learning_state
from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_train_memory import structure_sha256
from kuavo_isaaclab_scene.rl.multi_box.experiments.regional_actor_servo import (
    RegionalActorMemorySACPilot,validate_regional_actor_state)
from kuavo_isaaclab_scene.rl.multi_box.experiments.support_conservative_sac import (
    SupportConservativeRegionalSACPilot,support_conservative_contract)


def prepare(initial,experience,*,source_checkpoint_SHA256):
    require_pristine_learning_state(initial,experience,artifact_types=(RegionalActorMemorySACPilot.artifact_type,))
    validate_regional_actor_state(initial)
    if (len(source_checkpoint_SHA256)!=64 or
        any(c not in '0123456789abcdef' for c in source_checkpoint_SHA256)):
        raise ValueError('An immutable source checkpoint SHA256 is required')
    state,replay=deepcopy(initial),deepcopy(experience)
    state['artifact_type']=SupportConservativeRegionalSACPilot.artifact_type
    state['hybrid_contract']['critic_support_regularization']=support_conservative_contract()
    origin=dict(kind='pristine_regional_critic_support_only_v1',source_checkpoint_SHA256=source_checkpoint_SHA256,
        source_artifact_type=initial['artifact_type'],initial_model_SHA256=structure_sha256(initial['model']),
        actor_Q_parameters_normalizers_and_all_optimizer_states_unchanged=True,
        previous_reward_or_Q_training_data_imported=False)
    for value in (state,replay):
        value['goal_contract']['name']=state['artifact_type']
        value['goal_contract']['critic_support_regularization']=support_conservative_contract()
        value['support_conservative_initialization']=deepcopy(origin)
    require_pristine_learning_state(state,replay,artifact_types=(state['artifact_type'],))
    assert identical(initial['model'],state['model']) and identical(initial['optimizers'],state['optimizers'])
    assert identical(initial['actor_training_memory'],state['actor_training_memory'])
    return state,replay


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('initial-checkpoint','training-manifest','waypoints','training-waves','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    paths=(args.initial_checkpoint,args.initial_checkpoint.parent/'staged_goal_experience.pt',
           args.training_manifest,args.waypoints,args.training_waves)
    data={str(p):owned_stable_bytes(p) for p in paths}
    hashes={k:hashlib.sha256(v).hexdigest() for k,v in data.items()}
    initial=torch.load(io.BytesIO(data[str(paths[0])]),map_location='cpu',weights_only=True)
    experience=torch.load(io.BytesIO(data[str(paths[1])]),map_location='cpu',weights_only=True)
    physical=json.loads(data[str(args.training_manifest)]);expected=initial['goal_contract']['physical_contract']
    if {k:physical.get(k) for k in expected}!=expected:raise ValueError('Physical source contract differs')
    waves=json.loads(data[str(args.training_waves)])
    if waves[0]['split']!='validation' or len(waves[0]['layouts'])!=128:
        raise ValueError('Original full DEV128 baseline must be retained')
    state,replay=prepare(initial,experience,source_checkpoint_SHA256=hashes[str(args.initial_checkpoint)])
    args.output_dir.mkdir(mode=0o700,parents=True,exist_ok=False)
    torch.save(state,args.output_dir/'checkpoint_00000000.pt');torch.save(replay,args.output_dir/'staged_goal_experience.pt')
    assert identical(state,torch.load(args.output_dir/'checkpoint_00000000.pt',weights_only=True))
    assert hashes=={str(p):hashlib.sha256(owned_stable_bytes(p)).hexdigest() for p in paths}
    proof=dict(recorded_UTC=datetime.now(timezone.utc).isoformat(),source_SHA256=hashes,
        support_conservative_initialization=state['support_conservative_initialization'],
        all_model_tensors_actor_Q_normalizers_optimizers_and_original_actor_memory_preserved=True,
        actor_Q_counters_online_replay_and_reward_banks0=True,
        original_base_box_flap_randomization_reward_success_and_safety_preserved=True,
        no_synthetic_transitions_or_evaluation_training=True,
        original_whole_DEV_and_TRAIN_requests_preserved=True,full_goal_scope_NOT_reduced=True,
        actual_trainer_restore_and_learning_still_required=True,independent_FINAL_unused=True,goal_not_complete=True)
    for name,value in (('training_manifest.json',physical),('waypoints.json',json.loads(data[str(args.waypoints)])),
        ('training_waves.json',waves),('initialization_verification.json',proof),('manifest.json',dict(
            artifact_type=state['artifact_type'],goal_contract=state['goal_contract'],initialization_verification=proof))):
        (args.output_dir/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output_dir=str(args.output_dir.resolve()),critic_support_weight=1.,
        initial_actor_unchanged=True,all_online_learning_state0=True,learning_NOT_started=True)))


if __name__=='__main__':main()
