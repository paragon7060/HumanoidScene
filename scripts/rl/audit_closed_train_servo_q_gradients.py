"""Measure exact jaw-averaged Q gradients through the production goal servo.

Only a checksum-matched closed TRAIN replay and final model are queried.
No optimizer, normalizer, physics, current replay or alternative action is run.
These local gradients are not evidence of rollout success or a causal result.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path

import torch

from audit_closed_dev_critics import identity, require_closed_run
from export_eval_q_videos import restored_agent, sha256
from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_train_memory import structure_sha256
from kuavo_isaaclab_scene.rl.multi_box.experiments.kinematic_exploration import target_token
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import REGIONS


def query(agent, raw, critic, noise, groups):
    normalized = agent.actor_normalizer(agent.actor_features(raw))
    mean, log_std, logits = agent.continuous_parameters(normalized, raw)
    latent = (mean + log_std.exp()*noise).detach().requires_grad_(True)
    body = agent.body_from_latent(normalized, latent.tanh(), raw)
    goals, probability, _, _ = agent.enumerate_jaws(raw, body, logits)
    n = len(raw)
    repeated = raw[:, None].expand(-1, 4, -1).reshape(n*4, -1)
    flat = goals.reshape(n*4, 21)
    commands = agent.critic_action_features(repeated, flat).reshape(n, 4, 21)
    encoded = commands.detach().requires_grad_(True)
    normalized_critic = agent.critic_normalizer(critic)
    inputs = torch.cat((normalized_critic[:, None].expand(-1, 4, -1), encoded), -1)
    inputs = inputs.reshape(n*4, -1)
    q = torch.minimum(agent.q1(inputs), agent.q2(inputs)).reshape(n, 4)
    scale = q.detach().abs().mean().clamp_min(1).reciprocal() if agent.config.actor_q_normalize else 1.
    objective = -(probability.detach()*scale*q).sum()
    command_gradient = torch.autograd.grad(objective, encoded)[0]
    # Chain the exact command gradients through the actual decoder. No
    # straight-through approximation or counterfactual physics is introduced.
    q_gradient = torch.autograd.grad((commands*command_gradient.detach()).sum(),
                                    latent, retain_graph=True)[0]
    jacobian = torch.autograd.grad(commands[:, :, 1:15].sum(), latent)[0][:, 1:15]/4
    delta = agent.goal_servo_critic_encoder.unclipped_body(repeated, flat).reshape(n, 4, 19)
    encoder = agent.goal_servo_critic_encoder
    _, radius = agent.anchor_and_scale(raw)
    steps = raw.new_tensor(encoder.coordinates.joints.scales)
    expected = encoder.scale[1:15].to(raw)/steps[1:15]*radius[:, 1:15]*(1-latent[:, 1:15].tanh().square())
    expected *= delta[:, 0, 1:15].abs() <= 1
    if not torch.allclose(jacobian, expected, atol=2e-5, rtol=2e-5):
        raise ValueError('Actual arm servo derivative differs from the analytic decoder')
    if not torch.isfinite(q_gradient).all() or not torch.isfinite(command_gradient).all():
        raise ValueError('Nonfinite exact Q gradient')
    arms = q_gradient[:, 1:15]
    command_arms = command_gradient[:, :, 1:15].sum(1)
    def summarize(mask):
        selected_arms, selected_commands = arms[mask], command_arms[mask]
        return dict(states=int(mask.sum()),
            clipped_arm_coordinate_fraction=float((delta[mask, 0, 1:15].abs()>1).float().mean()),
            zero_arm_servo_derivative_fraction=float((jacobian[mask].abs()<1e-8).float().mean()),
            zero_arm_Q_gradient_fraction=float((selected_arms.abs()<1e-8).float().mean()),
            states_with_all_arm_Q_gradients_zero=int((selected_arms.abs()<1e-8).all(-1).sum()),
            nonzero_command_Q_but_zero_goal_Q_fraction=float(((selected_commands.abs()>=1e-8)&(selected_arms.abs()<1e-8)).float().mean()),
            arm_command_Q_gradient_abs_median=float(selected_commands.abs().median()),
            arm_goal_latent_Q_gradient_abs_median=float(selected_arms.abs().median()),
            jaw_averaged_Q_mean=float((probability.detach()[mask]*q.detach()[mask]).sum(-1).mean()),
            Gaussian_std_mean=float(log_std.exp()[mask].mean()))
    return dict(**summarize(torch.ones(n,dtype=torch.bool)),
        strata={name:summarize(mask) for name,mask in groups.items() if mask.any()},
        strata_share_original_query_and_Q_scaling=True,
        actual_arm_autograd_matches_analytic_servo=True,
        categorical_expectation_matches_production_actor_Q=True)


def measured_strata(raw, critic):
    """Separate measured held age and assigned midpoint proximity.

    Midpoint proximity is not actual contact-region distance or pinch success.
    The original held clock remains at530 when task remaining is added at571.
    """
    held_steps = critic[:,530]*900
    if not torch.allclose(held_steps,held_steps.round(),atol=1e-4,rtol=0) \
            or (held_steps<0).any() or (held_steps>900).any():
        raise ValueError('Expected original measured integer held clock /900')
    supplement=raw[:,474:512]
    if not torch.equal(supplement,critic[:,533:571]) \
            or not (supplement[:,36:].sum(-1)==1).all():
        raise ValueError('Measured actor/critic perceived midpoint supplement differs')
    assignment=supplement[:,36:].argmax(-1)
    flaps=torch.stack((assignment,1-assignment),-1)
    relation=supplement[:,:36].reshape(-1,2,2,9)
    selected=relation[torch.arange(len(raw))[:,None],torch.arange(2)[None],flaps]
    maximum_distance=selected[...,:3].norm(dim=-1).amax(-1)
    return dict(held_0_149_steps=held_steps<150,
        held_150_449_steps=(held_steps>=150)&(held_steps<450),
        held_450_or_more_steps=held_steps>=450,
        both_assigned_midpoints_within_10cm=maximum_distance<=.10,
        weaker_hand_assigned_midpoint_beyond_10cm=maximum_distance>.10,
        both_assigned_midpoints_within_5cm=maximum_distance<=.05,
        measured_both_hands_pinching=(critic[:,499:501]>.5).all(-1))


def audit(run, checkpoint, dev_proof, output, per_scope=64, seed=27011):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or output.exists() or not 1<=per_scope<=256:
        raise ValueError('Empty CUDA mask, new output and1..256 states per scope required')
    torch.set_num_threads(1)
    require_closed_run(run)
    proof = json.loads(dev_proof.read_text())
    checksum = sha256(checkpoint)
    if proof.get('run') != run.name or proof.get('final_model_SHA256') != checksum \
            or proof.get('fixed_requested_DEV_denominator') != 128 \
            or proof.get('learned', {}).get('attempts') != 128 \
            or len(proof.get('six_scopes', [])) != 6 \
            or proof.get('final_checkpoint_and_logs_Drive_verified') is not True:
        raise ValueError('Exact closed model and whole DEV128 proof required')
    payload = run/'staged_goal_experience.pt'
    checkpoint_before, payload_before = identity(checkpoint), identity(payload)
    payload_checksum = sha256(payload)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    experience = torch.load(payload, map_location='cpu', weights_only=True, mmap=True)
    if state['goal_contract'] != experience['goal_contract']:
        raise ValueError('Exact matching TRAIN replay required')
    replay = experience['executed_goal_transitions']
    agent, _ = restored_agent(state)
    agent.eval().requires_grad_(False)
    model_before = structure_sha256(agent.state_dict())
    records = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        token, valid = target_token(replay['critic_obs'][:, :464])
        if not valid.all() or not torch.equal(token[:, 8:12].argmax(-1),
                                             replay['actor_obs'][:, 94:98].argmax(-1)):
            raise ValueError('Perceived target and actor routing differ')
        for region_id, region in enumerate(REGIONS):
            for size_id, size in enumerate(('small', 'medium')):
                available = torch.where((token[:,8:12].argmax(-1)==region_id)&
                                        (token[:,3:5].argmax(-1)==size_id))[0]
                if not len(available):
                    continue
                indices = available[torch.randperm(len(available))[:per_scope]]
                raw, critic = [replay[k][indices].clone() for k in ('actor_obs', 'critic_obs')]
                if not (raw[:, -6]==1).all() or not torch.isfinite(raw).all():
                    raise ValueError('Measured finite held TRAIN states required')
                groups=measured_strata(raw,critic)
                for sampler in ('greedy', 'Gaussian'):
                    noise = torch.zeros(len(raw), 19) if sampler=='greedy' else torch.randn(len(raw), 19)
                    records.append(dict(region=region,box_type=size,sampler=sampler,
                        replay_rows_available=len(available),**query(agent,raw,critic,noise,groups)))
    if model_before != structure_sha256(agent.state_dict()) \
            or identity(checkpoint) != checkpoint_before or identity(payload) != payload_before \
            or sha256(checkpoint) != checksum or sha256(payload) != payload_checksum:
        raise ValueError('Protected model or closed TRAIN input changed')
    result = dict(recorded_at=datetime.now().astimezone().isoformat(),
        role='closed_TRAIN_exact_Q_to_arm_servo_gradient_READ_ONLY_diagnostic',
        run_directory_name=run.name,checkpoint_SHA256=checksum,replay_SHA256=payload_checksum,
        actor_updates=state['actor_updates'],critic_updates=state['critic_updates'],
        replay_rows=len(replay['reward']),per_scope=per_scope,seed=seed,records=records,
        source_model_and_replay_unchanged=True,no_DEV_FINAL_states_or_labels_used=True,
        no_optimizer_or_physics_update=True,no_GPU_or_Isaac=True,
        entropy_retention_and_regularizer_gradients_not_included=True,
        one_state_Gaussian_probe_not_full_AR1_or_guidance_collection=True,
        strata_are_same_queried_states_not_additional_rollouts=True,
        midpoint_distance_is_not_contact_region_distance_or_success=True,
        no_physical_success_or_causal_learning_improvement_claim=True,goal_not_complete=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:r[k] for k in ('region','box_type','sampler','states',
        'clipped_arm_coordinate_fraction','zero_arm_Q_gradient_fraction')} for r in records]))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('run','checkpoint','dev-proof','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--per-scope',type=int,default=64);p.add_argument('--seed',type=int,default=27011)
    a=p.parse_args();audit(a.run.resolve(),a.checkpoint.resolve(),a.dev_proof.resolve(),a.output,a.per_scope,a.seed)
