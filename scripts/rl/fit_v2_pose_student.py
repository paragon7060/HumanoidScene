#!/usr/bin/env python3
"""Supervised pose-goal warm start from actual TRAIN success commands only.

This does not perform SAC/Q training and excludes heldout trials. The resulting
student must succeed in physical closed-loop evaluation before RL continuation.
"""
import argparse,hashlib,json
from pathlib import Path
import h5py
import torch
from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_student import PoseGoalCoordinates,PoseStudent
from kuavo_isaaclab_scene.rl.algorithms.asymmetric_sac import AsymmetricSAC
from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-suite',type=Path)
    p.add_argument('--native-dataset',type=Path,
                   help='Actual current GPU success, not legacy VR rewards. Excludes --training-suite.')
    p.add_argument('--initial-box-relative',action='store_true')
    p.add_argument('--clock-only-fit',action='store_true',
                   help='Freeze observation input weights at zero during fit, avoiding a learned feedback loop through current motion.')
    p.add_argument('--time-harmonics',type=int,default=0,
                   help='Optional sine/cosine time encoding; no recorded trajectory values are runtime inputs.')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=10000)
    args=p.parse_args()
    if bool(args.training_suite)==bool(args.native_dataset):p.error('Select training suite or one actual native dataset')
    if not 0<=args.time_harmonics<=64:p.error('Time harmonics must be within0..64')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(41)
    coordinates=PoseGoalCoordinates();inputs=[];labels=[];sources=[];roundtrip=0.
    rows=json.loads((args.training_suite/'results.json').read_text()) if args.training_suite else []
    datasets=([Path(row['run_dir'])/'executed_transitions.hdf5' for row in rows
               if row['split']=='train' and row['outcomes']==dict(success=1,unsafe=0,invalid_reset=0,time_out=0)]
              if args.training_suite else [args.native_dataset])
    for dataset in datasets:
        with h5py.File(dataset) as f:
            meta=json.loads(f.attrs['manifest_json'])
            allowed={'layout_reference_residual_sac'} if args.training_suite else {'current_v2_environment_executed_vr_reference'}
            if meta.get('collection_source') not in allowed or not meta.get('current_reward_verified_against_breakdown') \
                    or meta.get('old_demo_rewards_used') is not False or not str(meta.get('sim_device','')).startswith('cuda:'):
                raise ValueError('Only actual current GPU training success commands may provide labels')
            for episode in f['episodes'].values():
                if not episode.attrs.get('success',False):raise ValueError('Non-success source episode')
                data=episode['transitions']
                raw=torch.from_numpy(data['actor_obs'][:]);physical=torch.from_numpy(data['action'][:])
                if raw.shape!=(len(raw),464) or physical.shape!=(len(raw),24) or not torch.isfinite(raw).all():
                    raise ValueError('Bad physical label source')
                if not bool(data['terminated'][-1]) or not bool(data['success'][-1]) or data['unsafe'][:].any():
                    raise ValueError('Invalid physical success terminal')
                goal=coordinates.encode_physical(raw,physical)
                error=float((coordinates.decode(raw,goal)-physical).abs().max())
                roundtrip=max(roundtrip,error)
                if error>1e-4:raise ValueError(f'Physical action inverse differs:{error}')
                if args.initial_box_relative:
                    goal[:,19:21]-=coordinates.box_anchor(raw[:1])
                inputs.extend(coordinates.observations(raw[i:i+1],i,args.time_harmonics) for i in range(len(raw)))
                labels.append(goal)
        sources.append(dict(path=str(dataset.resolve()),sha256=hashlib.sha256(dataset.read_bytes()).hexdigest()))
    if not labels:raise ValueError('No physical training success labels')
    x=torch.cat(inputs);goal=torch.cat(labels)
    low,high=goal.min(0).values,goal.max(0).values
    margin=goal.new_tensor([.04]*17+[.01,.01]+[.02,.02,.03]+[0.,0.])
    center=(high+low)/2;scale=(high-low)/2+margin
    center[-2:]=0.;scale[-2:]=1.
    y=((goal-center)/scale).clamp(-.995,.995)
    config=SACConfig(hidden=256,initial_policy_std=.01,max_policy_std=.02,
        freeze_actor_normalizer=True,entropy_backup=False,initial_alpha=1e-5,min_alpha=1e-7,max_alpha=1e-3,
        critic_layer_norm=True,actor_q_normalize=True)
    agent=AsymmetricSAC(x.shape[1],531,24,config,'cpu')
    if args.clock_only_fit:
        with torch.no_grad():agent.actor.network[0].weight[:,:438].zero_()
    agent.actor_normalizer.update(x)
    # All-box tokens include empty slots and are needed at unseen layouts.
    agent.actor_normalizer.var[174:438].clamp_(min=.25)
    optimizer=torch.optim.Adam(agent.actor.parameters(),lr=3e-4)
    weights=goal.new_ones(24);weights[22:24]=3.
    for step in range(args.steps):
        ids=torch.randint(len(x),(512,))
        predicted,_=agent.actor(agent.actor_normalizer(x[ids]),deterministic=True)
        loss=((predicted-y[ids]).square()*weights).mean()
        optimizer.zero_grad();loss.backward()
        if args.clock_only_fit:agent.actor.network[0].weight.grad[:,:438]=0.
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),1.);optimizer.step()
        if (step+1)%1000==0:print(json.dumps(dict(fit_step=step+1,loss=float(loss))),flush=True)
    with torch.no_grad():
        prediction=agent.act(x,deterministic=True)
        errors=((center+scale*prediction)-goal).abs()
    state=agent.checkpoint()|dict(artifact_type=PoseStudent.artifact_type,format_version=1,
        action_coordinates=coordinates.name,goal_center=center,goal_scale=scale,
        actor_updates=0,critic_updates=0,actor_fit_steps=args.steps,reference_runtime_dependency=False,
        episode_clock_input=True,physical_reference_used_for_labels=True,source_train_episodes=len(sources),
        initial_box_relative_goals=args.initial_box_relative,clock_only_initial_fit=args.clock_only_fit,
        time_harmonics=args.time_harmonics,
        physical_config='unchanged_v2_grasp',sources=sources)
    torch.save(state,args.output_dir/'student.pt')
    report=dict(artifact_type=PoseStudent.artifact_type,rows=len(x),sources=sources,
        actor_fit_steps=args.steps,sac_actor_updates=0,sac_critic_updates=0,
        inverse_physical_command_max_error=roundtrip,
        mean_absolute_goal_error=errors.mean(0).tolist(),max_absolute_goal_error=errors.max(0).values.tolist(),
        runtime_reference_path_required=False,physical_success_verified=False)
    report.update(initial_box_relative_goals=args.initial_box_relative,clock_only_initial_fit=args.clock_only_fit,time_harmonics=args.time_harmonics)
    (args.output_dir/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output_dir/'status.json').write_text(json.dumps(dict(status='complete',physical_success_verified=False))+'\n')
    print(json.dumps(report|{'sources':len(sources)}),flush=True)


if __name__=='__main__':main()
