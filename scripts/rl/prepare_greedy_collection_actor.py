#!/usr/bin/env python3
"""Preserve a pristine SAC model while changing only future TRAIN collection."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_reanchored_sac import ReanchoredActualFlapSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.servo_success_retention import ServoRetentionGentleSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.body_behavior_exploration import (
    VARIANT,GREEDY_REST_VARIANT,body_behavior_config,body_behavior_statistics)
from prepare_actual_success_actor_tail import identical


def prepare(initial, experience, *, source_checkpoint):
    """Fork only empty initializations; a learned actor may already be seeded."""
    if (initial.get('artifact_type') not in (
            ReanchoredActualFlapSACPilot.artifact_type, ServoRetentionGentleSACPilot.artifact_type)
            or initial['actor_updates'] or initial['critic_updates']
            or len(initial['optimizers'])!=4 or any(o['state'] for o in initial['optimizers'])
            or initial['model']['critic_normalizer.count']
            or any(len(v) for v in experience['executed_goal_transitions'].values())
            or any(initial['successful_train_transitions']['episodes'].values())
            or not identical(initial['successful_train_transitions'],experience['successful_train_transitions'])
            or initial.get('measured_train_credit')!=experience.get('measured_train_credit')
            or experience['measured_train_credit_bank']['config']!=initial.get('measured_train_credit')
            or any(experience['measured_train_credit_bank']['episodes'].values())):
        raise ValueError('Fresh Q, all empty optimizers and matching empty reward-bearing banks are required')
    previous=initial.get('body_behavior')
    if previous is None:
        pristine=all(value.get(key) is None for value in (initial,experience)
            for key in ('body_behavior','body_behavior_statistics'))
    else:
        pristine=(previous==body_behavior_config(VARIANT)
            and experience.get('body_behavior')==previous
            and all(value.get('body_behavior_statistics')==body_behavior_statistics()
                for value in (initial,experience)))
    if initial['goal_contract']!=experience['goal_contract'] or not pristine:
        raise ValueError('Matching pristine absent or original20% arm-behavior metadata required')
    if any(not torch.isfinite(v).all() for v in initial['model'].values()):
        raise ValueError('Finite initial models are required')
    state,replay=deepcopy(initial),deepcopy(experience)
    config=body_behavior_config(GREEDY_REST_VARIANT)
    origin=dict(source_checkpoint=str(source_checkpoint),actor_updates_at_activation=0,
        critic_updates_at_activation=0,old_replay_rows_at_activation=0,
        model_Q_normalizers_and_four_optimizer_states_kept=True,
        old_replay_kept_with_original_behavior=True,old_rows_not_relabelled=True,
        scope='future_real_TRAIN_collection',source_checkpoint_behavior=previous)
    for value in (state,replay):
        value.update(body_behavior=deepcopy(config),body_behavior_origin=deepcopy(origin),
            body_behavior_statistics=body_behavior_statistics())
    return state,replay


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('initial-checkpoint','training-manifest','waypoints','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    replay_path=a.initial_checkpoint.parent/'staged_goal_experience.pt'
    sources=(a.initial_checkpoint,replay_path,a.training_manifest,a.waypoints)
    if any(x.is_symlink() or not x.is_file() or x.stat().st_uid!=os.getuid() for x in sources):
        raise ValueError('Owned regular pristine input files are required')
    hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}
    state=torch.load(a.initial_checkpoint,map_location='cpu',weights_only=True)
    replay=torch.load(replay_path,map_location='cpu',weights_only=True)
    physical=json.loads(a.training_manifest.read_text())
    contract=state['goal_contract']['physical_contract']
    if {k:physical.get(k) for k in contract}!=contract:
        raise ValueError('Source physical manifest differs')
    original=deepcopy(state)
    state,replay=prepare(state,replay,source_checkpoint=a.initial_checkpoint)
    config=state['body_behavior']
    assert identical(original['model'],state['model']) and identical(original['optimizers'],state['optimizers'])
    assert original['goal_contract']==state['goal_contract'] and original['config']==state['config']
    assert all(torch.isfinite(v).all() for v in state['model'].values())
    a.output_dir.mkdir(parents=True,exist_ok=False,mode=0o700)
    torch.save(state,a.output_dir/'checkpoint_00000000.pt')
    torch.save(replay,a.output_dir/'staged_goal_experience.pt')
    assert identical(state,torch.load(a.output_dir/'checkpoint_00000000.pt',map_location='cpu',weights_only=True))
    assert identical(replay,torch.load(a.output_dir/'staged_goal_experience.pt',map_location='cpu',weights_only=True))
    assert hashes=={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}
    proof=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),source_SHA256=hashes,
        actual_models_Q_targets_normalizers_four_optimizers_and_learning_config_unchanged=True,
        actual_actor_Q_counters_and_all_reward_banks0=True,physical_reward_success_safety_randomization_contract_unchanged=True,
        new_future_TRAIN_collection=config,existing20percent_wide_arm_episode_selection_preserved=True,
        no_teacher_or_demo_trajectory_replay=True,all_source_files_preserved=True,
        new_physical_training_NOT_started=True,goal_not_complete=True)
    for name,data in [('training_manifest.json',physical),('waypoints.json',json.loads(a.waypoints.read_text())),
            ('initialization_verification.json',proof),('manifest.json',dict(artifact_type=state['artifact_type'],
                goal_contract=state['goal_contract'],initialization_verification=proof))]:
        (a.output_dir/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output_dir=str(a.output_dir),all_models_optimizers_Q_and_physical_contract_unchanged=True,
        fresh_Q_banks0=True,TRAIN20percent_arm_explore_rest_greedy=True,training_NOT_started=True)))


if __name__=='__main__':main()
