#!/usr/bin/env python3
"""Bounded SAC evaluation: hold, mean policy, sampled policy, position-only IK.

Uses the normal alternatives bootstrap and checkpoint contract. Replaces only
the evaluation callback in this process; never trains or changes saved weights.
Pass normal flap-pick arguments, --method play, and --diagnostic-steps N.
"""

import argparse
import json
import sys
from types import ModuleType


def evaluate(env, args, directory, state, steps):
    import torch
    from kuavo_isaaclab_scene.rl.algorithms.sac import SAC, SACConfig
    from kuavo_isaaclab_scene.teleop.urdf_arm_ik import UrdfArm, quat_matrix, skew
    from kuavo_isaaclab_scene.robots.robot_model import resolve_robot_model
    import numpy as np

    assert state['algorithm'] == 'sac' and env.num_envs % 4 == 0
    agent = SAC(state['obs_dim'], state['action_dim'], SACConfig(**state['config']), env.device)
    agent.restore(state, training=False)
    torch.manual_seed(42)
    obs = env.reset()[0]['policy']
    t = env.command_manager.get_term('workcell')
    groups = ('hold', 'deterministic', 'stochastic', 'position_ik')
    width = env.num_envs // 4
    arm = env.action_manager.get_term('upper_body')
    joints = list(arm._joint_ids)[7:14]
    urdf = UrdfArm(resolve_robot_model().urdf_path, 'right')
    parent_id = t.robot.find_bodies(urdf.parent)[0][0]
    tip_id = t.robot.find_bodies(urdf.tip)[0][0]
    fk_errors = []
    total_reward = torch.zeros(env.num_envs, device=env.device)
    completed = torch.zeros_like(total_reward)
    unsafe = torch.zeros_like(total_reward)
    successes = torch.zeros_like(total_reward)
    ready_steps = torch.zeros_like(total_reward)
    close_steps = torch.zeros_like(total_reward)
    grasp_steps = torch.zeros_like(total_reward)
    min_distance = torch.full_like(total_reward, float('inf'))
    action_sum = torch.zeros(env.num_envs, 16, device=env.device)
    discounted_reward = torch.zeros_like(total_reward)
    discounted_soft_reward = torch.zeros_like(total_reward)
    discount = torch.ones_like(total_reward)
    first_ready = None
    rows = []
    with torch.no_grad():
        for step in range(steps):
            t.refresh()
            was_ready = t.settling.ready.clone()
            action = torch.zeros(env.num_envs, 16, device=env.device)
            action[width:2*width] = agent.act(obs[width:2*width], deterministic=True)
            sampled, sampled_logp = agent.actor(agent.normalizer(obs[2*width:3*width]))
            action[2*width:3*width] = sampled
            # Position-only DLS reference: no grasp/pose teleport, no orientation
            # controller, same incremental action limits and collision checking.
            # Independently validate the URDF FK against live USD before using
            # its Jacobian; a failed position reference is not reachability evidence.
            parts = []
            for index in range(3*width, env.num_envs):
                q = t.robot.data.joint_pos[index, joints].cpu().numpy()
                p, _, jac, _ = urdf.fk(q)
                rotation = quat_matrix(t.robot.data.body_link_quat_w[index,parent_id].cpu().numpy())
                origin = t.robot.data.body_link_pos_w[index,parent_id].cpu().numpy()
                tip = t.robot.data.body_link_pos_w[index,tip_id].cpu().numpy()
                err = float(np.linalg.norm(origin + rotation @ p - tip))
                fk_errors.append(err)
                if err > .01:
                    raise ValueError(f'URDF/USD FK mismatch: {err} m')
                offset = rotation.T @ (t.tools[index,1].cpu().numpy() - tip)
                parts.append(rotation @ (jac[:3] - skew(offset) @ jac[3:]))
            j = torch.as_tensor(np.stack(parts), device=env.device, dtype=obs.dtype)
            error = (t.grips - t.tools)[3*width:, 1]
            error = error * (.005 / error.norm(dim=-1, keepdim=True).clamp_min(.005))
            delta = j.transpose(1, 2) @ torch.linalg.solve(
                j @ j.transpose(1, 2) + .01**2 * torch.eye(3, device=env.device), error[..., None])
            # Correct current joint error to avoid integrating tracking lag.
            desired = t.robot.data.joint_pos[3*width:, joints] + delta.squeeze(-1)
            action[3*width:, 7:14] = ((desired - arm.processed_actions[3*width:, 7:14]) / .02).clamp(-1, 1)
            if first_ready is None and was_ready.all():
                n = agent.normalizer(obs)
                mean, log_std = agent.actor.network(n).chunk(2, -1)
                entropy_sample, logp = agent.actor(n)
                q = torch.minimum(agent.q1(torch.cat((n, entropy_sample), -1)),
                                  agent.q2(torch.cat((n, entropy_sample), -1)))
                first_ready = {'step': step, 'mean_action': mean.tanh().mean(0).tolist(),
                    'latent_std': log_std.clamp(-5,2).exp().mean(0).tolist(),
                    'mean_logp': logp.mean().item(), 'mean_q_sampled': q.mean().item(),
                    'alpha': agent.log_alpha.exp().item()}
                first_ready['q_by_action'] = {}
                for name, candidate in [('hold',torch.zeros_like(action)),
                                         ('deterministic',mean.tanh()), ('sampled',sampled)]:
                    nn = n[2*width:3*width] if name == 'sampled' else n
                    qv = torch.minimum(agent.q1(torch.cat((nn,candidate),-1)),
                                       agent.q2(torch.cat((nn,candidate),-1)))
                    first_ready['q_by_action'][name] = qv.mean().item()
            ready_steps += was_ready
            close_steps += t.ready_to_close[:, 1] & was_ready
            grasp_steps += t.hand_grasp_flags[:, 1] & was_ready
            min_distance = torch.minimum(min_distance, torch.where(was_ready,
                t.hand_target_distance[:, 1], torch.full_like(min_distance, float('inf'))))
            action_sum += action * was_ready[:, None]
            next_obs, reward, terminated, truncated, _ = env.step(action)
            obs = next_obs['policy']
            total_reward += reward
            discounted_reward += discount * reward * was_ready
            soft = reward.clone()
            soft[2*width:3*width] -= agent.log_alpha.exp() * sampled_logp
            discounted_soft_reward += discount * soft * was_ready
            discount *= torch.where(was_ready, agent.config.gamma, 1.)
            completed += terminated | truncated
            unsafe += env.termination_manager.get_term('unsafe')
            successes += env.termination_manager.get_term('success')
            if step % 16 == 15 or step == steps-1:
                row = {'step': step+1, 'groups': {}}
                for i, name in enumerate(groups):
                    sl = slice(i*width, (i+1)*width)
                    row['groups'][name] = dict(distance=t.hand_target_distance[sl,1].mean().item(),
                        alignment=t.grasp_alignment[sl,1].mean().item(), gap=t.jaw_gap[sl,1].mean().item(),
                        reward_total=total_reward[sl].mean().item(), unsafe=unsafe[sl].sum().item(),
                        ready_to_close_steps=close_steps[sl].sum().item())
                rows.append(row)
                with (directory/'diagnostic_steps.jsonl').open('a') as f:
                    f.write(json.dumps(row, allow_nan=False)+'\n')
                print('[DIAG] '+json.dumps(row), flush=True)
    report = {'steps': steps, 'envs_per_group': width, 'checkpoint_iteration': state['iteration'],
              'first_ready': first_ready, 'max_urdf_fk_error_m': max(fk_errors), 'groups': {}}
    for i,name in enumerate(groups):
        sl = slice(i*width,(i+1)*width)
        report['groups'][name] = dict(mean_reward_total=total_reward[sl].mean().item(),
            completed=completed[sl].sum().item(), unsafe=unsafe[sl].sum().item(),
            successes=successes[sl].sum().item(), min_distance_mean=min_distance[sl].mean().item(),
            final_distance=t.hand_target_distance[sl,1].mean().item(),
            ready_steps=ready_steps[sl].sum().item(), close_steps=close_steps[sl].sum().item(),
            grasp_steps=grasp_steps[sl].sum().item(),
            discounted_reward=discounted_reward[sl].mean().item(),
            discounted_soft_reward=discounted_soft_reward[sl].mean().item(),
            mean_action=(action_sum[sl]/ready_steps[sl,None].clamp_min(1)).mean(0).tolist())
    (directory/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('[DIAG_RESULT] '+json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--diagnostic-steps', type=int, default=192)
    own, remaining = parser.parse_known_args()
    if not 1 <= own.diagnostic_steps <= 450:
        parser.error('diagnostic steps must be 1..450')
    sys.argv = [sys.argv[0], *remaining]
    callback = ModuleType('kuavo_isaaclab_scene.rl.runners.evaluate_alternative')
    callback.evaluate = lambda env,args,directory,state: evaluate(env,args,directory,state,own.diagnostic_steps)
    sys.modules[callback.__name__] = callback
    from kuavo_isaaclab_scene.rl.runners.alternatives import main
    main()
