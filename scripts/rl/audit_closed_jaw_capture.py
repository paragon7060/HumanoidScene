"""CPU audit of actual safe post-action TRAIN geometry; no rollout or update."""
from pathlib import Path
from datetime import datetime
import argparse, hashlib, json, math
from collections import defaultdict
import torch
from kuavo_isaaclab_scene.robots.end_effector import closed_offsets
from kuavo_isaaclab_scene.robots.gripper_config import load_gripper_settings
from kuavo_isaaclab_scene.robots.robot_model import resolve_robot_model
from kuavo_isaaclab_scene.rl.multi_box.geometry.grasp import nominal_flap_geometry,opposing_flap_reach_assignment,GRASP_ASSIGNMENT_SCALE_M
from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import GoalGripperProjector
from kuavo_isaaclab_scene.rl.multi_box.observations.flap_supplement import supplemental_perception_contract


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experience',type=Path,required=True,
        help='Immutable saved actual-flap TRAIN experience; never read an active writer file')
    parser.add_argument('--verified-receipt',type=Path,required=True,
        help='Existing input-directory Drive size/MD5 receipt, containing input_dir and files')
    parser.add_argument('--output-json',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[2]
    source=args.experience.resolve()
    receipt=json.loads(args.verified_receipt.read_text())
    assert receipt['all_files_size_MD5_verified'] and Path(receipt['input_dir']).resolve()==source.parent
    entry=next(x for x in receipt['files'] if x['file']==source.name)
    assert entry['size_MD5_verified'] and source.stat().st_size==entry['bytes']
    initial_signature=(source.stat().st_size,source.stat().st_mtime_ns)
    state=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
    assert state['goal_contract']['actor_dim']==518 and state['goal_contract']['critic_dim']==577
    assert state['goal_contract']['supplemental_perception']==supplemental_perception_contract()
    assert state['goal_contract']['physical_contract']['robot_model']=='s63'
    episodes={episode['identity']:episode for values in state['measured_train_credit_bank']['episodes'].values() for episode in values}
    for values in state['successful_train_transitions']['episodes'].values():
        for episode in values:
            if episode['identity'] in episodes:
                assert all(torch.equal(v,episodes[episode['identity']]['rows'][k]) for k,v in episode['rows'].items())
            episodes[episode['identity']]=episode
    assert episodes
    definition=json.loads((root/'configs/grasp_reference_points.json').read_text())
    model=resolve_robot_model('s63');gripper=load_gripper_settings('leju-twofinger')
    axes=closed_offsets(model.urdf_path,definition['offsets'],gripper,closing_axes=True)
    closing_axis=torch.tensor([axes['left'],axes['right']],dtype=torch.float32)
    gate=GoalGripperProjector()
    categories=('no_filtered_flap_contact','some_contact_without_unique_pinch','qualified_unique_pinch')
    groups=defaultdict(lambda:defaultdict(list));paths=[];all_assignment_disagreement=0

    def consecutive(mask):
        current=torch.zeros(2,dtype=torch.long);values=[]
        for row in mask:
            current=torch.where(row,current+1,0);values.append(current.clone())
        return torch.stack(values)

    def stats(values):
        values=torch.cat(values) if isinstance(values,list) and values else values
        if isinstance(values,list) or not len(values):return dict(count=0)
        return dict(count=len(values),minimum=float(values.min()),p10=float(values.quantile(.1)),
            median=float(values.median()),p90=float(values.quantile(.9)),maximum=float(values.max()))

    for identity,episode in sorted(episodes.items()):
        outcome=episode['outcome'];rows=episode['rows'];n=len(rows['reward'])
        assert outcome['split']=='train' and outcome['initial_layout_valid'] and outcome['complete']
        pre=rows['critic_obs'];post=rows['next_critic_obs'];raw=rows['next_actor_obs'];priv=post[:,464:530]
        safe=~(pre[:,523:530]>.5).any(-1)&~(post[:,523:530]>.5).any(-1)
        closed=rows['action'][:,19:21]>0;near=gate.near(rows['actor_obs'])
        assert not (closed&~near).any()
        closure=post[:,46:48]
        sustained=consecutive(closed&near)>=4
        valid=raw[:,510:512].sum(-1)>.5
        relations=raw[:,474:510].reshape(n,2,2,9)
        first=relations[...,3:6];second=relations[...,6:9]
        rotation=torch.stack((first,second,torch.cross(first,second,dim=-1)),-1)
        eye=torch.eye(3).expand(n,2,2,3,3)
        assert torch.allclose((rotation.transpose(-1,-2)@rotation)[valid],eye[valid],atol=1e-4,rtol=0)
        # The relation is panel center/orientation in TCP frame. Invert its
        # translation to measure the rigid CLOSED-calibrated TCP in panel frame.
        local=-(rotation.transpose(-1,-2)@relations[...,:3,None]).squeeze(-1)
        tokens=raw[:,86:350].reshape(n,12,22)
        target=raw[:,400:412].argmax(-1);target_token=tokens[torch.arange(n),target]
        size=target_token[:,5:8];types=target_token[:,3:5].argmax(-1)
        assert (target_token[:,0]>.5).all() and (size>0).all()
        _,halves,_=nominal_flap_geometry(size,types)
        excess=(local.abs()-halves[:,None]).clamp_min(0)
        normal_outside=excess[...,0]
        tangent_outside=excess[...,1:].norm(dim=-1)
        surface_distance=excess.norm(dim=-1)
        _,assignment=opposing_flap_reach_assignment(surface_distance,GRASP_ASSIGNMENT_SCALE_M)
        saved=raw[:,510:512].argmax(-1)
        disagreement=valid&(assignment[:,0]!=saved)
        assert not disagreement.any(),'Reconstructed perceived assignment differs'
        all_assignment_disagreement+=int(disagreement.sum())
        assigned=torch.stack((saved,1-saved),-1)
        indices=(torch.arange(n)[:,None],torch.arange(2)[None],assigned)
        nominal_alignment=(first*closing_axis[None,:,None]).sum(-1).abs().clamp(0,1)
        metric_values=dict(rigid_TCP_surface_distance_m=surface_distance[indices],
            rigid_TCP_normal_outside_m=normal_outside[indices],rigid_TCP_tangent_outside_m=tangent_outside[indices],
            rigid_TCP_panel_normal_delta_m=local[...,0][indices],
            rigid_TCP_panel_width_delta_m=local[...,1][indices],rigid_TCP_panel_height_delta_m=local[...,2][indices],
            nominal_closed_axis_alignment_cos=nominal_alignment[indices],measured_closure=closure)
        force=priv[:,15:23].reshape(n,2,2,2)*50
        in_region=priv[:,23:31].reshape(n,2,2,2)>.5;opposed=priv[:,31:35].reshape(n,2,2)>.5
        qualified=(force>=5).all(-1)&in_region.all(-1)&opposed
        unique=qualified.sum(-1)==1
        assert torch.equal(unique,priv[:,35:37]>.5)
        contact=(force>0).flatten(2).any(-1)
        category=torch.where(unique,2,torch.where(contact,1,0))
        mask=closed&near&sustained&(closure>=.95)&safe[:,None]&valid[:,None]
        region=outcome['layout']['target_region'];success=bool(outcome['result']['success'])
        perpath={}
        for hand,name in enumerate(('left','right')):
            entries={}
            for code,label in enumerate(categories):
                ids=mask[:,hand]&(category[:,hand]==code)
                key=(region,success,name,label)
                for metric,value in metric_values.items():groups[key][metric].append(value[ids,hand].clone())
                entries[label]=dict(rows=int(ids.sum()),
                    surface_distance_over2cm=int((metric_values['rigid_TCP_surface_distance_m'][ids,hand]>.02).sum()),
                    normal_outside_over2cm=int((metric_values['rigid_TCP_normal_outside_m'][ids,hand]>.02).sum()),
                    tangent_outside_over2cm=int((metric_values['rigid_TCP_tangent_outside_m'][ids,hand]>.02).sum()),
                    alignment_below_cos20deg=int((metric_values['nominal_closed_axis_alignment_cos'][ids,hand]<math.cos(math.radians(20))).sum()))
            perpath[name]=entries
        paths.append(dict(identity=identity,region=region,success=success,by_hand=perpath))
    summary=[]
    for (region,success,hand,label),values in sorted(groups.items()):
        summarized={name:stats(v) for name,v in values.items()}
        selected=[p['by_hand'][hand][label] for p in paths if p['region']==region and p['success']==success]
        summary.append(dict(region=region,episode_success=success,hand=hand,contact_category=label,
            episodes_with_rows=sum(v['rows']>0 for v in selected),rows=sum(v['rows'] for v in selected),
            flags={name:sum(v[name] for v in selected) for name in ('surface_distance_over2cm',
                'normal_outside_over2cm','tangent_outside_over2cm','alignment_below_cos20deg')},metrics=summarized))
    assert initial_signature==(source.stat().st_size,source.stat().st_mtime_ns)
    report=dict(recorded_at=datetime.now().astimezone().isoformat(),source=str(source.relative_to(root)) if source.is_relative_to(root) else str(source),
        verified_source_bytes=entry['bytes'],verified_source_MD5=entry['MD5'],episodes=len(episodes),
        sampled_dataset='biased_retained_actual_TRAIN_not_whole_failure_rate',
        scope='post_action_safe_measured_closure_ge0p95_after4_consecutive_near_close_commands',
        geometry='actual_perceived_panel_pose_known_shape_and_rigid_closed_calibrated_TCP',
        nominal_closed_axes_in_TCP_frame=axes,
        perceived_assignment_reconstructed_and_matches_saved=True,
        original_reference_points_SHA256=hashlib.sha256((root/'configs/grasp_reference_points.json').read_bytes()).hexdigest(),
        rigid_TCP_and_nominal_closed_axis_are_not_actual_dynamic_pad_points=True,
        geometry_comparison_does_not_prove_reachable_or_unreachable_physical_motion=True,
        no_reward_relabeling_or_success_threshold_or_safety_changes=True,
        CPU_only=True,optimizer_updates=0,physical_rollouts=0,live_jobs_replay_and_HDF_untouched=True,
        DEV_or_FINAL_imported=False,independent_FINAL_unused=True,goal_not_complete=True,
        groups=summary,paths=paths)
    output=args.output_json.resolve()
    output.write_text(json.dumps(report,indent=2)+'\n')
    no_contact=[g for g in summary if not g['episode_success']
                and g['contact_category']=='no_filtered_flap_contact']
    print(json.dumps(dict(output=str(output),retained_actual_TRAIN_paths=len(episodes),
        failed_near_closed_no_contact_samples=sum(g['rows'] for g in no_contact),
        failed_no_contact_rigid_TCP_over2cm=sum(g['flags']['surface_distance_over2cm'] for g in no_contact),
        qualified_pinch_samples_rigid_TCP_over2cm=sum(g['flags']['surface_distance_over2cm']
            for g in summary if g['contact_category']=='qualified_unique_pinch'),
        physical_rollouts=0,optimizer_updates=0),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
