#!/usr/bin/env python3
"""Add the grasp drop guard to a pristine actor-only/fresh-Q initialization."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import torch

from kuavo_isaaclab_scene.rl.multi_box.geometry.box_drop import with_reset_drop_profile
from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_reanchored_sac import ReanchoredActualFlapSACPilot
from prepare_actual_success_actor_tail import identical


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('initial-checkpoint','training-manifest','waypoints','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    experience=args.initial_checkpoint.parent/'staged_goal_experience.pt'
    inputs=(args.initial_checkpoint,experience,args.training_manifest,args.waypoints)
    if any(p.is_symlink() or not p.is_file() or p.stat().st_uid!=os.getuid() for p in inputs):
        raise ValueError('Owned regular untrained initialization files are required')
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    initial=torch.load(args.initial_checkpoint,map_location='cpu',weights_only=True)
    replay=torch.load(experience,map_location='cpu',weights_only=True)
    if (initial.get('artifact_type')!=ReanchoredActualFlapSACPilot.artifact_type
            or initial['actor_updates'] or initial['critic_updates']
            or len(initial['optimizers'])!=4 or any(x['state'] for x in initial['optimizers'])
            or initial['model']['critic_normalizer.count']
            or any(len(v) for v in replay['executed_goal_transitions'].values())
            or any(initial['successful_train_transitions']['episodes'].values())
            or any(replay['measured_train_credit_bank']['episodes'].values())):
        raise ValueError('Only fresh Q, all empty optimizers and empty reward-bearing banks may migrate')
    if replay['goal_contract']!=initial['goal_contract']:
        raise ValueError('Initialization checkpoint and experience contracts differ')
    physical=json.loads(args.training_manifest.read_text())
    old=initial['goal_contract']['physical_contract']
    if {k:physical.get(k) for k in old}!=old:
        raise ValueError('Training manifest differs from source physical contract')
    state=deepcopy(initial)
    state['goal_contract']['physical_contract']=with_reset_drop_profile(old)
    replay=deepcopy(replay)
    replay['goal_contract']=deepcopy(state['goal_contract'])
    physical['terminal_contract']=deepcopy(state['goal_contract']['physical_contract']['terminal_contract'])
    assert identical(state['model'],initial['model']) and identical(state['optimizers'],initial['optimizers'])
    assert all(torch.isfinite(v).all() for v in state['model'].values())
    args.output_dir.mkdir(parents=True,exist_ok=False,mode=0o700)
    torch.save(state,args.output_dir/'checkpoint_00000000.pt')
    torch.save(replay,args.output_dir/'staged_goal_experience.pt')
    assert identical(state,torch.load(args.output_dir/'checkpoint_00000000.pt',map_location='cpu',weights_only=True))
    assert identical(replay,torch.load(args.output_dir/'staged_goal_experience.pt',map_location='cpu',weights_only=True))
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    proof=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),source_SHA256=hashes,
        reset_relative_drop_height_m=.10,all_actor_jaw_normalizer_Q_target_and_optimizer_tensors_unchanged=True,
        actor_Q_updates_and_online_success_nstep_reward_banks0=True,source_reward_weights_geometry_controller_exploration_and_randomization_unchanged=True,
        old_reward_data_not_relabelled_or_imported=True,new_physical_training_NOT_started=True,goal_not_complete=True)
    for name,data in [('training_manifest.json',physical),('waypoints.json',json.loads(args.waypoints.read_text())),
            ('initialization_verification.json',proof),('manifest.json',dict(artifact_type=state['artifact_type'],
                goal_contract=state['goal_contract'],initialization_verification=proof))]:
        (args.output_dir/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output_dir=str(args.output_dir),new_drop_guard=.10,all_model_tensors_unchanged=True,
        fresh_Q_and_banks0=True,training_NOT_started=True)))


if __name__=='__main__':main()
