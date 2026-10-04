#!/usr/bin/env python3
"""Initialize success-retaining SAC from a closed matching real run."""
import argparse
from datetime import datetime
import hashlib,json
from pathlib import Path

import h5py
import torch

from recover_pose_goal_actor import recover
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import (
    TrainSuccessBank,KEYS,retention_config,validate_success_outcome,match_measured_success_paths,
)
from kuavo_isaaclab_scene.rl.runners.storage import save_checkpoint


def prepare(checkpoint,best_checkpoint,output_dir):
    checkpoint,best_checkpoint,output_dir=map(Path,(checkpoint,best_checkpoint,output_dir))
    source=checkpoint.parent;manager=json.loads((source.parent/'status.json').read_text())
    if manager.get('phase')!='finished' or not manager.get('final_upload_verified') \
            or Path('/proc/'+str(manager['training_pid'])).exists():
        raise ValueError('Only a stopped writer with final verified backup can seed retention')
    manifest=json.loads((source/'manifest.json').read_text())
    if not manifest.get('training') or manifest.get('base_waypoint_probe',{}).get('enabled'):
        raise ValueError('Only the matching original TRAIN controller can seed real success')
    meta=json.loads((source/'metrics.json').read_text())
    selected=[o for o in meta['outcomes'] if o['split']=='train' and (o['result'] or {}).get('success')]
    if not selected:raise ValueError('Closed source has no actual TRAIN success')
    for outcome in selected:validate_success_outcome('train',outcome)
    actual=torch.load(source/'staged_goal_experience.pt',map_location='cpu',weights_only=True,mmap=True)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if actual['goal_contract']!=state['goal_contract']:raise ValueError('Source replay/checkpoint context differs')
    if 'train_success_retention' in state['goal_contract']:raise ValueError('Retention has already been enabled')
    outcomes={(o['wave'],o['environment']):o for o in selected};paths=[];evidence=[]
    with h5py.File(source/'executed_transitions.hdf5','r') as stream:
        for episode in stream['episodes'].values():
            key=(int(episode.attrs['wave']),int(episode.attrs['environment']))
            if key not in outcomes:continue
            outcome=outcomes[key];start=outcome['result']['staged_base']['manipulation_start']
            if not episode.attrs.get('success'):raise ValueError('Native episode success differs')
            transitions=episode['transitions']
            if not bool(transitions['success'][-1]) or bool(transitions['unsafe'][:].any()):
                raise ValueError('Native path has unsafe or missing measured success')
            paths.append(dict(critic_obs=torch.tensor(transitions['critic_obs'][start:]),
                next_critic_obs=torch.tensor(transitions['next_critic_obs'][start:]),
                reward=torch.tensor(transitions['reward'][start:],dtype=torch.float32),
                terminated=torch.tensor(transitions['terminated'][start:],dtype=torch.bool),first_held_step=0))
            evidence.append(outcome)
    if len(paths)!=len(selected):raise ValueError('Missing native successful TRAIN episodes')
    raw_width=paths[0]['critic_obs'].shape[1]
    horizon=state['frozen_warm_start']['bc_prior'].get('clock_horizon',410)
    ids=match_measured_success_paths(actual['executed_goal_transitions'],paths,raw_critic_dim=raw_width,clock_horizon=horizon)
    bank=TrainSuccessBank(state['actor_obs_dim'],state['critic_obs_dim'])
    for outcome,matched in zip(evidence,ids):
        bank.add_episode({k:actual['executed_goal_transitions'][k][matched] for k in KEYS},outcome,source_run=source.name,split='train')
    destination,audit=recover(checkpoint,best_checkpoint,output_dir,normalize_prior_loss_by_radius=True)
    updated=torch.load(destination,map_location='cpu',weights_only=True)
    experience=torch.load(output_dir/'staged_goal_experience.pt',map_location='cpu',weights_only=True)
    contract=updated['goal_contract']|dict(train_success_retention=retention_config())
    updated.update(goal_contract=contract,successful_train_transitions=bank.state(),
        success_schedule_actor_origin=updated['actor_updates'])
    experience.update(goal_contract=contract,successful_train_transitions=bank.state())
    audit.update(actual_success_retention=bank.report(),DEV_FINAL_imported=False,
        original_goal_Q_rows_matched_to_closed_native_success=True,physical_or_waypoint_changes=False,
        source_finite_checkpoint_from_closed_interrupted_run=manager.get('training_exit_code')!=0,
        source_final_backup_verified=True,source_writer_stopped=True,
        source_replay_tensors_unchanged=True,bank_initial_fraction=.2,bank_final_fraction=.05,
        source_successes=[dict(wave=o['wave'],environment=o['environment'],seed=o['layout']['seed'],region=o['layout']['target_region']) for o in evidence])
    updated['actor_recovery']=audit
    save_checkpoint(output_dir,updated,updated['critic_updates'],keep=None)
    torch.save(experience,output_dir/'staged_goal_experience.pt')
    new_manifest=json.loads((output_dir/'manifest.json').read_text())|dict(goal_contract=contract,actor_recovery=audit)
    (output_dir/'manifest.json').write_text(json.dumps(new_manifest,indent=2)+'\n')
    (output_dir/'status.json').write_text(json.dumps(dict(status='complete',initialized_not_new_training=True,success_bank=bank.report()))+'\n')
    (output_dir/'retention_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return destination,audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint','best-checkpoint','output-dir']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();destination,audit=prepare(args.checkpoint,args.best_checkpoint,args.output_dir)
    print(json.dumps(dict(checkpoint=str(destination),success_bank=audit['actual_success_retention'],no_evaluation_import=True)))


if __name__=='__main__':main()
