#!/usr/bin/env python3
"""Measure the real goal-to-servo Jacobian on protected completed TRAIN paths.

This is a read-only model diagnostic, not a new physical evaluation or success
claim. It requires a protected checkpoint with a matching completed DEV proof.
No running replay/HDF, GPU, optimizer or Isaac runtime is used.
"""
import argparse
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from export_eval_q_videos import restored_agent
from prepare_actual_success_actor_tail import identical
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import validate_success_outcome


def load_owned(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid():
        raise ValueError('An owned regular protected file is required')
    return path.read_bytes()


def fractions(raw, agent, labels, generator):
    normalized = agent.actor_normalizer(agent.actor_features(raw))
    with torch.no_grad():
        mean, log_std, _ = agent.parameters_at(normalized)
        _, radius = agent.anchor_and_scale(raw)
        greedy = agent.act(raw, True)
        delta = agent.goal_servo_critic_encoder.unclipped_body(raw, greedy)
        command = agent.critic_action_features(raw, greedy)[:, :19]
    latent = mean.detach().clone().requires_grad_(True)
    body = agent.body_from_latent(normalized, latent.tanh(), raw)
    projected = agent.action_projector(raw, torch.cat((body, greedy[:, 19:]), -1))
    encoded = agent.critic_action_features(raw, projected)[:, :19]
    derivative = torch.autograd.grad(encoded.sum(), latent)[0]
    encoder = agent.goal_servo_critic_encoder
    steps = raw.new_tensor(encoder.coordinates.joints.scales + [.1/30, .1/30])
    analytic = encoder.scale[:19].to(raw)/steps*radius*(1-mean.tanh().square())
    analytic = analytic * (delta.abs() <= 1)
    if not torch.allclose(derivative, analytic, atol=2e-5, rtol=2e-5):
        raise ValueError('Production autograd and independently computed Jacobian differ')
    active = radius > 1e-8
    blocked = derivative.abs() < 1e-8
    denom = int(active.sum())
    if not denom:
        raise ValueError('No available body coordinates in these TRAIN states')
    with torch.no_grad():
        noisy_deltas = []
        for _ in range(16):
            noise = torch.randn(mean.shape, generator=generator, device='cpu').to(raw)
            noisy_body = agent.body_from_latent(normalized, (mean+log_std.exp()*noise).tanh(), raw)
            noisy_goals = agent.action_projector(raw, torch.cat((noisy_body, greedy[:,19:]), -1))
            noisy_command = agent.critic_action_features(raw, noisy_goals)[:, :19]
            noisy_deltas.append((noisy_command-command).abs())
        noise_delta = torch.stack(noisy_deltas)
        measured = agent.critic_action_features(raw, labels)[:, :19]
    row_active = active.sum(-1)
    row_blocked = (blocked & active).sum(-1)
    active_noise = noise_delta[:,active]
    return dict(
        states=len(raw), active_coordinate_samples=denom,
        servo_clipped_active_fraction=float(((delta.abs()>1)&active).sum())/denom,
        zero_servo_gradient_active_fraction=float((blocked&active).sum())/denom,
        tanh_mean_abs_over3_active_fraction=float(((mean.abs()>3)&active).sum())/denom,
        all_available_body_gradients_zero_states=int(((row_active>0)&(row_blocked==row_active)).sum()),
        active_gradient_abs_median=float(derivative.abs()[active].median()),
        Gaussian_std_min=float(log_std.exp().min()), Gaussian_std_max=float(log_std.exp().max()),
        independent_Gaussian_draws_per_state=16,
        active_command_noise_abs_median=float(active_noise.median()),
        active_command_noise_abs_p95=float(torch.quantile(active_noise.flatten(),.95)),
        active_command_noise_changes_over0p05_fraction=float((active_noise>.05).float().mean()),
        same_past_TRAIN_label_command_MAE=float((command-measured).abs().mean()),
        joint_jaw_greedy_label_match_fraction=float((greedy[:,19:]==labels[:,19:]).all(-1).float().mean()),
        recorded_both_closed_states=int((labels[:,19:]==1).all(-1).sum()),
        greedy_both_closed_on_recorded_closed_states=int(((labels[:,19:]==1).all(-1)&(greedy[:,19:]==1).all(-1)).sum()),
        all_jacobians_finite=bool(torch.isfinite(derivative).all()),
        discrete_jaws_and_20percent_bias_behavior_NOT_diagnosed=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-pointer',type=Path,required=True)
    p.add_argument('--completed-evaluation',type=Path,required=True)
    p.add_argument('--policy-pointer',type=Path,
        help='Optional different protected same-controller policy; diagnostic only, never a new evaluation claim')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    pointer_blob=load_owned(a.checkpoint_pointer)
    evaluation_blob=load_owned(a.completed_evaluation)
    pointer=json.loads(pointer_blob); evaluation=json.loads(evaluation_blob)
    checkpoint=Path(pointer['protected_checkpoint']); blob=load_owned(checkpoint)
    digest=hashlib.sha256(blob).hexdigest()
    if (digest!=pointer['checkpoint_SHA256'] or digest!=evaluation['matching_checkpoint_SHA256']
            or not evaluation['full_original_DEV128_per_region32']):
        raise ValueError('A completed matching model evaluation is required')
    state=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True)
    if (state['actor_updates'],state['critic_updates'])!=(pointer['actor_updates'],pointer['critic_updates']):
        raise ValueError('Protected model counters differ')
    policy_state=state
    policy_blob=None; policy_pointer_blob=None
    if a.policy_pointer is not None:
        policy_pointer_blob=load_owned(a.policy_pointer);candidate=json.loads(policy_pointer_blob)
        policy_blob=load_owned(candidate['protected_checkpoint'])
        if hashlib.sha256(policy_blob).hexdigest()!=candidate['checkpoint_SHA256']:
            raise ValueError('Protected diagnostic policy SHA256 differs')
        policy_state=torch.load(io.BytesIO(policy_blob),map_location='cpu',weights_only=True)
        if (policy_state['actor_updates'],policy_state['critic_updates'])!=(candidate['actor_updates'],candidate['critic_updates']):
            raise ValueError('Diagnostic policy counters differ')
        keys=('actor_dim','goal_center','goal_scale','action_coordinates','body_correction_radius',
              'supplemental_perception','validated_jaw_prior_confidence','jaw_prior_residual_gain',
              'critic_action_encoding','source_warm_start')
        if (policy_state['artifact_type']!=state['artifact_type']
                or not identical(policy_state['body_anchor_state'],state['body_anchor_state'])
                or not identical(policy_state['frozen_actor_prior'],state['frozen_actor_prior'])
                or any(policy_state['goal_contract'].get(k)!=state['goal_contract'].get(k) for k in keys)):
            raise ValueError('Diagnostic policy actor/servo/anchor coordinates differ')
    agent,_=restored_agent(policy_state)
    if not hasattr(agent,'goal_servo_critic_encoder'):
        raise ValueError('An actual production servo critic is required')
    groups=[]; generator=torch.Generator().manual_seed(237071801)
    episodes=state['successful_train_transitions']['episodes']
    for region,paths in episodes.items():
        if not paths:
            continue
        for phase in ('first16','last64'):
            observations=[]; actions=[]
            for e in paths:
                validate_success_outcome('train', e['outcome'])
                expected_identity=(f"{Path(pointer['source_run']).name}/wave{e['outcome']['wave']}"
                    f"/env{e['outcome']['environment']}/seed{e['outcome']['layout']['seed']}")
                if (e['identity']!=expected_identity
                        or e['outcome']['layout']['target_region']!=region):
                    raise ValueError('Only original completed successful matching TRAIN paths are allowed')
                rows=e['rows']; idx=slice(0,16) if phase=='first16' else slice(-64,None)
                observations.append(rows['actor_obs'][idx]);actions.append(rows['action'][idx])
            raw=torch.cat(observations); labels=torch.cat(actions)
            groups.append(dict(region=region,phase=phase,paths=len(paths),**fractions(raw,agent,labels,generator)))
    if not groups:
        raise ValueError('The protected model has no completed self-generated TRAIN states')
    assert digest==hashlib.sha256(load_owned(checkpoint)).hexdigest()
    assert pointer_blob==load_owned(a.checkpoint_pointer) and evaluation_blob==load_owned(a.completed_evaluation)
    if policy_blob is not None:
        assert policy_pointer_blob==load_owned(a.policy_pointer)
        assert policy_blob==load_owned(candidate['protected_checkpoint'])
    proof=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint_SHA256=digest,actor_updates=state['actor_updates'],critic_updates=state['critic_updates'],
        policy_checkpoint_SHA256=hashlib.sha256(policy_blob).hexdigest() if policy_blob is not None else digest,
        diagnostic_policy_actor_updates=policy_state['actor_updates'],diagnostic_policy_critic_updates=policy_state['critic_updates'],
        different_policy_on_same_past_TRAIN_states=a.policy_pointer is not None,
        source_completed_original_DEV_successes=evaluation['summary']['supported_successes'],
        groups=groups,production_goal_to_servo_autograd_matches_independent_analytic_Jacobian=True,
        same_past_completed_successful_TRAIN_only=True,NOT_new_physical_evaluation_or_learning_gain=True,
        policy_replay_rewards_optimizers_and_other_processes_unchanged=True,
        current_Gaussian_noise_diagnostic_NOT_full_collection_behavior=True,
        no_running_HDF_or_replay_read=True, no_GPU_or_Isaac=True, goal_not_complete=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(a.output),matching_DEV=evaluation['summary']['supported_successes'],groups=groups)),flush=True)


if __name__=='__main__':
    main()
