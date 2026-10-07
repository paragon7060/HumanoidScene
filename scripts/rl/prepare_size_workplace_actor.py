#!/usr/bin/env python3
"""Attach validated size targets to pristine regional SAC, with no learned Q import."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,io,json
from pathlib import Path

import torch

from compare_actor_train_memory import owned_stable_bytes
from prepare_actual_success_actor_tail import identical
from prepare_greedy_collection_actor import require_pristine_learning_state
from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_train_memory import (
    ActorTrainMemory,compatibility_contract,structure_sha256)
from kuavo_isaaclab_scene.rl.multi_box.experiments.regional_actor_servo import (
    RegionalActorMemorySACPilot,validate_regional_actor_state)
from kuavo_isaaclab_scene.rl.multi_box.experiments.size_workplaces import validate_size_workplaces


def prepare(initial,experience,waypoints,*,source_checkpoint_SHA256):
    require_pristine_learning_state(initial,experience,artifact_types=(RegionalActorMemorySACPilot.artifact_type,))
    validate_regional_actor_state(initial)
    typed=validate_size_workplaces(waypoints['size_workplaces'])
    if (typed['source_checkpoint_SHA256']!=source_checkpoint_SHA256
        or typed['source_region_workplaces']!=initial['goal_contract']['shelf_templates']
        or typed['source_shelf_templates']!=waypoints['shelves']
        or not identical(initial['actor_training_memory'],experience['actor_training_memory'])):
        raise ValueError('Measured size sources must exactly match the pristine policy and original actor memory')
    state,replay=deepcopy(initial),deepcopy(experience)
    for value in (state,replay):value['goal_contract']['shelf_templates']=deepcopy(typed)
    ActorTrainMemory(state['actor_training_memory'],compatibility=compatibility_contract(state['goal_contract']),
        frozen_anchor_SHA256=structure_sha256(state['body_anchor_state']))
    require_pristine_learning_state(state,replay,artifact_types=(RegionalActorMemorySACPilot.artifact_type,))
    validate_regional_actor_state(state)
    assert identical(state['model'],initial['model']) and identical(state['optimizers'],initial['optimizers'])
    assert identical(state['actor_training_memory'],initial['actor_training_memory'])
    return state,replay


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('initial-checkpoint','training-manifest','waypoints','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    paths=(args.initial_checkpoint,args.initial_checkpoint.parent/'staged_goal_experience.pt',
        args.training_manifest,args.waypoints)
    data={str(path):owned_stable_bytes(path) for path in paths}
    hashes={name:hashlib.sha256(blob).hexdigest() for name,blob in data.items()}
    initial=torch.load(io.BytesIO(data[str(paths[0])]),map_location='cpu',weights_only=True)
    experience=torch.load(io.BytesIO(data[str(paths[1])]),map_location='cpu',weights_only=True)
    physical=json.loads(data[str(args.training_manifest)]);waypoints=json.loads(data[str(args.waypoints)])
    expected=initial['goal_contract']['physical_contract']
    if {k:physical.get(k) for k in expected}!=expected:raise ValueError('Original physical task contract differs')
    state,replay=prepare(initial,experience,waypoints,source_checkpoint_SHA256=hashes[str(args.initial_checkpoint)])
    args.output_dir.mkdir(mode=0o700,parents=True,exist_ok=False)
    torch.save(state,args.output_dir/'checkpoint_00000000.pt')
    torch.save(replay,args.output_dir/'staged_goal_experience.pt')
    assert identical(state,torch.load(args.output_dir/'checkpoint_00000000.pt',map_location='cpu',weights_only=True))
    assert identical(replay,torch.load(args.output_dir/'staged_goal_experience.pt',map_location='cpu',weights_only=True))
    assert hashes=={str(path):hashlib.sha256(owned_stable_bytes(path)).hexdigest() for path in paths}
    proof=dict(recorded_UTC=datetime.now(timezone.utc).isoformat(),source_SHA256=hashes,
        all_SAC_actor_Q_counters_optimizers_and_reward_banks0=True,
        untrained_Q_initialization_preserved_no_learned_Q_or_replay_import=True,
        all_regional_actor_and_normalizer_and_controller_tensors_unchanged=True,
        old_actor_only_TRAIN_reference_memory_contract_outcomes_and_rows_unchanged=True,
        old_references_used_only_at_original_observed_held_xy_yaw=True,
        six_supported_region_size_targets_from_two_disjoint_physical_TRAIN_searches=True,
        original_reward_success_safety_and_randomization_preserved=True,
        physical_full_trainer_restore_and_learning_still_required=True,
        independent_FINAL_unused=True,goal_not_complete=True)
    for name,value in (('training_manifest.json',physical),('waypoints.json',waypoints),
        ('initialization_verification.json',proof),('manifest.json',dict(
            artifact_type=state['artifact_type'],goal_contract=state['goal_contract'],initialization_verification=proof))):
        (args.output_dir/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output_dir=str(args.output_dir.resolve()),all_SAC_counters0=True,
        old_actor_memory_not_relabelled=True,learning_NOT_started=True)),flush=True)


if __name__=='__main__':main()
