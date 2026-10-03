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
from kuavo_isaaclab_scene.rl.multi_box.experiments.executed_replay import read_executed_successes,PHYSICAL_KEYS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-suite',type=Path)
    p.add_argument('--native-dataset',type=Path,action='append',
                   help='Repeat for complete actual current GPU successes. Excludes --training-suite and legacy rewards.')
    p.add_argument('--initial-box-relative',action='store_true')
    p.add_argument('--clock-only-fit',action='store_true',
                   help='Freeze observation input weights at zero during fit, avoiding a learned feedback loop through current motion.')
    p.add_argument('--time-harmonics',type=int,default=0,
                   help='Optional sine/cosine time encoding; no recorded trajectory values are runtime inputs.')
    p.add_argument('--clock-horizon',type=int,default=410,
                   help='Explicit control-step horizon; longer upper-shelf demonstrations must not saturate at410.')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=10000)
    args=p.parse_args()
    if bool(args.training_suite)==bool(args.native_dataset):p.error('Select training suite or actual native datasets')
    if not 0<=args.time_harmonics<=64:p.error('Time harmonics must be within0..64')
    if not 1<=args.clock_horizon<=900:p.error('Clock horizon must be within1..900')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(41)
    coordinates=PoseGoalCoordinates(exact_projected_base=True);inputs=[];labels=[];sources=[];roundtrip=0.
    rows=json.loads((args.training_suite/'results.json').read_text()) if args.training_suite else []
    datasets=([Path(row['run_dir'])/'executed_transitions.hdf5' for row in rows
               if row['split']=='train' and row['outcomes']==dict(success=1,unsafe=0,invalid_reset=0,time_out=0)]
              if args.training_suite else args.native_dataset)
    physical_contract=None
    for dataset in datasets:
        with h5py.File(dataset) as f:
            meta=json.loads(f.attrs['manifest_json'])
            recorded=meta.get('training_contract',{})
            if physical_contract is None:physical_contract=recorded
            if any(recorded.get(key)!=physical_contract.get(key) for key in PHYSICAL_KEYS):
                raise ValueError('Pose labels may not mix physical/reward contracts')
            # Check actual seed state, timestamps, continuity and terminal
            # bilateral pinch before extracting any supervised goal labels.
            _,audit=read_executed_successes(dataset,physical_contract)
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
                inputs.extend(coordinates.observations(raw[i:i+1],i,args.time_harmonics,args.clock_horizon) for i in range(len(raw)))
                labels.append(goal)
        sources.append(dict(path=str(dataset.resolve()),sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
                            successful_episodes=audit['successful_episodes']))
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
        episode_clock_input=True,physical_reference_used_for_labels=True,
        source_train_episodes=sum(source['successful_episodes'] for source in sources),
        initial_box_relative_goals=args.initial_box_relative,clock_only_initial_fit=args.clock_only_fit,
        time_harmonics=args.time_harmonics,
        clock_horizon=args.clock_horizon,
        physical_config=physical_contract['action_contract'],sources=sources)
    state['physical_contract']={key:physical_contract[key] for key in (*PHYSICAL_KEYS,'flap_pose_source')}
    torch.save(state,args.output_dir/'student.pt')
    report=dict(artifact_type=PoseStudent.artifact_type,rows=len(x),sources=sources,
        actor_fit_steps=args.steps,sac_actor_updates=0,sac_critic_updates=0,
        inverse_physical_command_max_error=roundtrip,
        mean_absolute_goal_error=errors.mean(0).tolist(),max_absolute_goal_error=errors.max(0).values.tolist(),
        runtime_reference_path_required=False,physical_success_verified=False)
    report.update(initial_box_relative_goals=args.initial_box_relative,clock_only_initial_fit=args.clock_only_fit,
                  time_harmonics=args.time_harmonics,clock_horizon=args.clock_horizon)
    report['physical_contract']=state['physical_contract']
    (args.output_dir/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output_dir/'status.json').write_text(json.dumps(dict(status='complete',physical_success_verified=False))+'\n')
    print(json.dumps(report|{'sources':len(sources)}),flush=True)


if __name__=='__main__':main()
