#!/usr/bin/env python3
"""Keep a finite staged SAC and only measured replay inside its decoder domain."""
import argparse
from datetime import datetime
import hashlib,json,shutil
from pathlib import Path
import torch
from kuavo_isaaclab_scene.rl.multi_box.demo_replay import _rotation_matrix
from kuavo_isaaclab_scene.rl.multi_box.geometry.projected_base import (
    outside_projected_base_domain,projected_base_safety_contract,BASE_PLANE_SAFETY_MIN_ABS_DETERMINANT,
)


def valid_replay_rows(rows,chunk=4096):
    n=len(rows['reward']);keep=torch.ones(n,dtype=torch.bool)
    for start in range(0,n,chunk):
        end=min(start+chunk,n)
        for key in ('actor_obs','next_actor_obs'):
            raw=rows[key][start:end]
            rotation=_rotation_matrix(raw[:,71:77])
            determinant=rotation[:,0,0]*rotation[:,1,1]-rotation[:,0,1]*rotation[:,1,0]
            keep[start:end]&=~outside_projected_base_domain(determinant,BASE_PLANE_SAFETY_MIN_ABS_DETERMINANT)
        for value in rows.values():
            keep[start:end]&=torch.isfinite(value[start:end]).reshape(end-start,-1).all(-1)
    return keep


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    source=args.checkpoint.resolve();replay=source.parent/'staged_goal_experience.pt'
    assert source.is_file() and replay.is_file() and not source.is_symlink()
    status=json.loads((source.parent/'status.json').read_text());assert status['status'] in ('failed','complete','interrupted')
    managed=source.parent.parent/'status.json'
    if managed.exists():assert not Path('/proc',str(json.loads(managed.read_text())['training_pid'])).exists()
    state=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
    experience=torch.load(replay,map_location='cpu',weights_only=True,mmap=True)
    assert state['goal_contract']==experience['goal_contract']
    assert 'projected_base_v2' in state['goal_contract']['action_coordinates']
    rows=experience['executed_goal_transitions'];n=len(rows['reward'])
    assert all(len(v)==n for v in rows.values()) and n>0
    assert rows['actor_obs'].shape[1]==state['actor_obs_dim']
    assert rows['critic_obs'].shape[1]==state['critic_obs_dim']
    def finite(value):
        if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        if isinstance(value,(tuple,list)):return all(finite(v) for v in value)
        return True
    assert finite(state),'Checkpoint model or optimizer is nonfinite'
    keep=valid_replay_rows(rows);assert keep.any()
    bank=experience.get('successful_train_transitions')
    bank_rows=0
    if bank:
        for episodes in bank['episodes'].values():
            for episode in episodes:
                assert bool(valid_replay_rows(episode['rows']).all()),'A successful TRAIN episode crosses the decoder domain; explicit episode migration required'
                bank_rows+=len(episode['rows']['reward'])
    args.output_dir.mkdir(parents=True,exist_ok=False)
    checkpoint=args.output_dir/source.name;shutil.copyfile(source,checkpoint)
    filtered={k:v[keep].contiguous() for k,v in rows.items()}
    # Clone only selected rows: torch.save must not retain discarded storage.
    experience=experience|{'executed_goal_transitions':filtered}
    torch.save(experience,args.output_dir/'staged_goal_experience.pt')
    report=dict(created_at=datetime.now().astimezone().isoformat(),source_checkpoint=str(source),
                source_experience=str(replay),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                actor_updates=state['actor_updates'],critic_updates=state['critic_updates'],
                original_rows=n,retained_rows=int(keep.sum()),excluded_rows=int((~keep).sum()),
                excluded_original_row_indices=torch.where(~keep)[0].tolist(),successful_TRAIN_bank_rows_kept=bank_rows,
                model_Q_optimizer_and_schedule_preserved=True,checkpoint_bytes_unchanged=True,
                retained_measured_labels_unchanged=True,unsupported_old_rows_excluded_not_relabelled=True,
                replay_contains_only_original_TRAIN=True,DEV_or_FINAL_imported=False,
                controller_coordinate_safety=projected_base_safety_contract())
    (args.output_dir/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
