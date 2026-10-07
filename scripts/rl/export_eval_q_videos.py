"""Annotate closed, held-base DEV videos with the matching frozen SAC critics.

No Isaac runtime, physics replay, GPU, optimizer update or replay import is used.
The reconstructed deterministic goals must decode to the recorded commands.
Q(s_t, a_t) is shown on the frame measured after executing that action; approach
frames have no Q because this critic was trained only on the held-grasp phase.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import h5py
import numpy as np
import torch

from browser_video import encode_browser_video
from kuavo_isaaclab_scene.rl.algorithms.common import ObservationNormalizer
from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig, SquashedActor
from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import (
    AbsoluteGoalJawProjector, BoundedCorrectionHybridSAC, ActualFlapResidualSACPilot,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_reanchored_sac import (
    ReanchoredActualFlapSACPilot, actual_actor_snapshot, source_actual_body_goal,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_servo_critic_sac import ServoCriticReanchoredSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.gentle_servo_critic_sac import GentleServoCriticSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.servo_critic import BodyServoCriticEncoder
from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_student import PoseStudent, pose_clock
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import (
    StagedGoalProjector, held_goal_coordinates, staged_context,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def restored_agent(state):
    from kuavo_isaaclab_scene.rl.multi_box.observations.task_timing import resolve_critic_episode_clock
    if resolve_critic_episode_clock(state) != state['goal_contract'].get('critic_episode_clock'):
        raise ValueError('Saved critic episode clock and feature contract differ')
    from kuavo_isaaclab_scene.rl.multi_box.experiments.servo_success_retention import (
        ServoRetentionGentleSACPilot,ServoRetainedCorrectionSAC,servo_success_retention_contract,
    )
    from kuavo_isaaclab_scene.rl.multi_box.experiments.conservative_servo_retention import (
        ConservativeServoRetentionSACPilot, validate_conservative_actor_state,
    )
    conservative = state.get('artifact_type') == ConservativeServoRetentionSACPilot.artifact_type
    if conservative:
        validate_conservative_actor_state(state)
    contract = state['goal_contract']
    critic_dim = 578 if state.get('critic_episode_clock') is not None else 577
    if (contract['actor_dim'], contract['critic_dim'], state['action_dim']) != (518, critic_dim, 21):
        raise ValueError('This exporter requires the actual-flap bounded held-base SAC contract')
    retained = conservative or state.get('artifact_type') == ServoRetentionGentleSACPilot.artifact_type
    if retained and contract.get('success_body_retention')!=servo_success_retention_contract():
        raise ValueError('Saved successful-servo retention contract differs')
    gentle = retained or state.get('artifact_type') == GentleServoCriticSACPilot.artifact_type
    if gentle:
        from kuavo_isaaclab_scene.rl.multi_box.experiments.body_policy_spread import validate_quarter_policy_state
        validate_quarter_policy_state(state)
    servo_critic = gentle or state.get('artifact_type') == ServoCriticReanchoredSACPilot.artifact_type
    reanchored = servo_critic or state.get('artifact_type') == ReanchoredActualFlapSACPilot.artifact_type
    if not reanchored and state.get('artifact_type') != ActualFlapResidualSACPilot.artifact_type:
        raise ValueError('Unknown actual-flap controller for Q video restoration')
    snapshot = state['body_anchor_state']
    nominal_snapshot = snapshot['nominal_body_anchor'] if reanchored else snapshot
    prior = PoseStudent(state['frozen_warm_start']['bc_prior'], 'cpu')
    nominal_width = contract['actor_dim'] - 38
    prefix = nominal_width - 6
    if prefix != prior.agent.actor_obs_dim + 2:
        raise ValueError('Frozen prior feature width differs')
    radius = nominal_snapshot['source_goal_contract']['fixed_prior_radius']
    projector = StagedGoalProjector(prior, prior.agent.actor_obs_dim, True, radius)
    anchor = torch.nn.ModuleDict(dict(
        actor=SquashedActor(nominal_width, 21, state['config']['hidden']),
        actor_normalizer=ObservationNormalizer(nominal_width),
    ))
    anchor.load_state_dict(nominal_snapshot['model'])
    anchor.requires_grad_(False)
    actual_anchor = actual_actor_snapshot(snapshot, 'cpu') if reanchored else None

    @torch.no_grad()
    def body_anchor(raw):
        nominal = torch.cat((raw[:, :prefix], raw[:, -6:]), -1).clone()
        nominal[:, -1] = radius
        command = anchor['actor'](anchor['actor_normalizer'](nominal), deterministic=True)[0]
        nominal_goal = projector(nominal, command)[:, :19]
        if actual_anchor is not None:
            return source_actual_body_goal(raw, nominal_goal, actual_anchor,
                snapshot['source_goal_contract']['fixed_prior_radius'])
        return nominal_goal

    agent_class=ServoRetainedCorrectionSAC if retained else BoundedCorrectionHybridSAC
    agent = agent_class(518, critic_dim, 21, SACConfig(**state['config']), 'cpu',
        action_projector=AbsoluteGoalJawProjector(),
        validated_jaw_prior_confidence=contract['validated_jaw_prior_confidence'],
        jaw_prior_residual_gain=contract['jaw_prior_residual_gain'])
    agent.correction_radius = contract['body_correction_radius']
    agent.executed_body_anchor = body_anchor
    if servo_critic:
        agent.goal_servo_critic_encoder=BodyServoCriticEncoder(prior.coordinates,
            contract['goal_center'],contract['goal_scale'])
        if contract.get('critic_action_encoding')!=agent.goal_servo_critic_encoder.contract:
            raise ValueError('Saved servo critic encoding contract differs')
    reference = deepcopy(anchor)
    reference.load_state_dict(state['frozen_actor_prior'])
    reference.requires_grad_(False)
    agent.validated_jaw_prior = lambda normal: reference['actor'].network(
        torch.cat((normal[:, :prefix], normal[:, -6:]), -1)).chunk(2, -1)[0][:, 19:21]
    agent.restore(state, training=False)
    agent.requires_grad_(False)
    if not all(torch.equal(value, agent.state_dict()[key]) for key, value in state['model'].items()):
        raise ValueError('Restored model does not exactly match the evaluation checkpoint')
    return agent, prior


@torch.no_grad()
def episode_values(episode, outcome, state, agent, prior):
    rows = episode['transitions']
    raw = torch.from_numpy(rows['actor_obs'][:])
    critic = torch.from_numpy(rows['critic_obs'][:])
    supplemental = torch.from_numpy(rows['actor_supplemental'][:])
    physical = torch.from_numpy(rows['action'][:])
    reward = rows['reward'][:]
    if len(raw) != outcome['steps'] or not np.isfinite(reward).all():
        raise ValueError('Episode rows do not match the closed video outcome')
    base = outcome['staged_base']
    start = base['manipulation_start']
    if base['phase'] != 'held_grasp' or not isinstance(start, int) or start >= len(raw):
        raise ValueError('Selected episode never entered the critic training phase')
    stage = SimpleNamespace(phase='held_grasp', manipulation_start=start,
        target_xy=raw.new_tensor([base['base_target_xy_rack_m']]),
        target_yaw=base['base_target_yaw_rack_rad'])
    initial = torch.from_numpy(episode['initial_state/observations/policy'][:])[None]
    anchor = prior.coordinates.box_anchor(initial)
    raw = raw[start:]
    clock_index = torch.arange(len(raw))
    config = prior.state
    horizon = config.get('clock_horizon', 410)
    features = prior.coordinates.observations(raw, clock_index,
        config.get('time_harmonics', 0), horizon,
        condition_on_shelf=config.get('shelf_conditioned_clock_fit', False),
        clock_limit=config.get('actor_clock_limit'))
    context = staged_context(raw, stage, state['goal_contract']['fixed_prior_radius'])
    anchor = anchor.expand(len(raw), -1)
    ao = torch.cat((features, anchor, supplemental[start:], context), -1)
    co = torch.cat((critic[start:], pose_clock(raw, clock_index, horizon), anchor,
        supplemental[start:], context), -1)
    clock = state.get('critic_episode_clock')
    if clock is not None:
        from kuavo_isaaclab_scene.rl.multi_box.observations.task_timing import (
            critic_episode_clock_config, VARIANT, add_critic_task_time)
        if clock != critic_episode_clock_config(VARIANT) or clock != state['goal_contract'].get('critic_episode_clock'):
            raise ValueError('Saved measured task-clock contract differs')
        if 'critic_episode_remaining' not in rows:
            raise ValueError('Q export requires actual recorded pre-autoreset task time')
        co = add_critic_task_time(co, torch.from_numpy(rows['critic_episode_remaining'][start:]))
    if ao.shape[1] != 518 or co.shape[1] != state['goal_contract']['critic_dim'] or not torch.isfinite(co).all():
        raise ValueError('Invalid reconstructed critic observations')
    goals = agent.act(ao, deterministic=True)
    center = raw.new_tensor(state['goal_contract']['goal_center'])
    scale = raw.new_tensor(state['goal_contract']['goal_scale'])
    reconstructed = held_goal_coordinates(prior.coordinates, raw, center + scale * goals, stage)
    error = (reconstructed - physical[start:]).abs()
    # Independent CPU GEMM batch sizes can differ slightly from the collection.
    if not torch.allclose(reconstructed, physical[start:], atol=2e-4, rtol=1e-5):
        raise ValueError(f'Policy goals do not match the executed commands: max error {float(error.max())}')
    qc = agent.critic_normalizer(co)
    qa = torch.cat((qc, agent.critic_action_features(ao,goals)), -1)
    q1, q2 = agent.q1(qa).flatten(), agent.q2(qa).flatten()
    if not torch.isfinite(q1).all() or not torch.isfinite(q2).all():
        raise ValueError('Non-finite Q estimate')
    arrays = {name: np.full(len(reward), np.nan) for name in ('q1', 'q2', 'q_min')}
    for name, value in [('q1', q1), ('q2', q2), ('q_min', torch.minimum(q1, q2))]:
        arrays[name][start:] = value.numpy()
    returns = np.zeros(len(reward))
    running = 0.
    for index in range(len(reward) - 1, -1, -1):
        running = float(reward[index]) * state['config']['reward_scale'] + state['config']['gamma'] * running
        returns[index] = running
    arrays.update(reward=reward, observed_return=returns)
    return arrays, start, float(error.max())


def dashboard(frame, index, arrays, start, title):
    canvas = np.full((920, frame.shape[1], 3), (242, 244, 246), np.uint8)
    canvas[:720] = frame
    cv2.rectangle(canvas, (0, 0), (960, 66), (23, 28, 36), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, title, (15, 24), font, .53, (245, 245, 245), 1, cv2.LINE_AA)
    if index < start:
        status = f't={(index+1)/30:.2f}s | approach controller | Q: N/A (held-grasp critic only)'
    else:
        status = (f't={(index+1)/30:.2f}s | Qmin={arrays["q_min"][index]:+.3f} '
                  f'Q1={arrays["q1"][index]:+.3f} Q2={arrays["q2"][index]:+.3f} '
                  f'r={arrays["reward"][index]:+.3f}')
    cv2.putText(canvas, status, (15, 50), font, .5, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Pre-action Q estimate (blue) | observed discounted return (orange, retrospective)',
        (18, 744), font, .48, (35, 40, 50), 1, cv2.LINE_AA)
    left, right, top, bottom = 85, 937, 765, 875
    values = np.r_[arrays['q_min'][start:], arrays['observed_return'][start:]]
    lo, hi = float(np.min(values)), float(np.max(values))
    span = max(hi - lo, .1); lo -= span * .12; hi += span * .12
    cv2.rectangle(canvas, (left, top), (right, bottom), (170, 178, 185), 1)
    for value in (lo, (lo+hi)/2, hi):
        y = int(bottom - (value-lo)/(hi-lo)*(bottom-top))
        cv2.putText(canvas, f'{value:+.2f}', (7, y+4), font, .4, (55, 65, 75), 1, cv2.LINE_AA)
    n = len(arrays['reward'])
    def points(series):
        xs = left + np.arange(start, index+1)/(n-1)*(right-left)
        ys = bottom - (series[start:index+1]-lo)/(hi-lo)*(bottom-top)
        return np.c_[xs, ys].round().astype(np.int32)
    for name, color in [('observed_return', (35, 125, 225)), ('q_min', (220, 110, 35))]:
        if index >= start:
            cv2.polylines(canvas, [points(arrays[name])], False, color, 2, cv2.LINE_AA)
    x = int(left + index/(n-1)*(right-left))
    cv2.line(canvas, (x, top), (x, bottom), (90, 100, 110), 1)
    cv2.putText(canvas, f'0s                    measured episode time                    {n/30:.2f}s',
        (left, 897), font, .45, (55, 65, 75), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Q is predicted cumulative reward, not a success probability. Original evaluation outcome unchanged.',
        (18, 915), font, .42, (55, 65, 75), 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--env-indices', type=int, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run = args.run_dir.resolve()
    status = json.loads((run/'status.json').read_text())
    verification = json.loads((run/'verification.json').read_text())
    if status.get('status') != 'complete' or not verification.get('writers_stopped_at'):
        raise ValueError('Only fully closed evaluation runs may be read')
    launch = json.loads((run.parent/'launch.json').read_text())
    command = launch['command']
    if Path(command[command.index('--checkpoint')+1]).resolve() != args.checkpoint.resolve():
        raise ValueError('Checkpoint differs from the policy used in these videos')
    checkpoint_sha = sha256(args.checkpoint)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    agent, prior = restored_agent(state)
    input_hdf = run/'executed_transitions.hdf5'
    hdf_sha = sha256(input_hdf)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    proofs = []
    with h5py.File(input_hdf, 'r') as dataset:
        episodes = {(int(e.attrs['wave']), int(e.attrs['environment'])): e for e in dataset['episodes'].values()}
        for env_index in args.env_indices:
            source = run/f'eval_wave_0000_env_{env_index:03d}_h264.mp4'
            record = json.loads(source.with_suffix('.json').read_text())
            if not record['source_video_writer_closed'] or record['role'] != 'frozen_DEV_policy_measured_body_poses_before_automatic_reset':
                raise ValueError('Only measured frozen DEV videos are accepted')
            if (record['actor_updates_at_start'], record['critic_updates_at_start']) != (state['actor_updates'], state['critic_updates']):
                raise ValueError('Checkpoint counters differ from the video policy')
            episode = episodes[(0, env_index)]
            if json.loads(episode.attrs['layout_json']) != record['layout']:
                raise ValueError('Video and episode initial layouts differ')
            arrays, start, error = episode_values(episode, record['actual_outcome'], state, agent, prior)
            final = len(arrays['reward']) - 1
            steps = list(range(0, final+1, record['capture_every_control_steps']))
            if steps[-1] != final: steps.append(final)
            if len(steps) != record['frames']:
                raise ValueError('Video capture steps differ from the closed trajectory')
            source_sha = sha256(source)
            capture = cv2.VideoCapture(str(source))
            fps = capture.get(cv2.CAP_PROP_FPS)
            raw_output = output/f'env_{env_index:03d}_q_pending.mp4'
            writer = cv2.VideoWriter(str(raw_output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (960, 920))
            if not capture.isOpened() or not writer.isOpened(): raise RuntimeError('Video codec could not open')
            outcome = record['actual_outcome']
            label = 'SUCCESS' if outcome['success'] else 'UNSAFE' if outcome['unsafe'] else 'TIMEOUT'
            title = f'DEV env{env_index} {record["region"]} | {label} | frozen SAC actor{state["actor_updates"]}'
            try:
                for step in steps:
                    ok, frame = capture.read()
                    if not ok: raise ValueError('Source video ended before its recorded frames')
                    preview = dashboard(frame, step, arrays, start, title)
                    writer.write(preview)
                if capture.read()[0]: raise ValueError('Source video has unexpected extra frames')
            finally:
                capture.release(); writer.release()
            destination = output/f'eval_env_{env_index:03d}_Q_h264.mp4'
            encoding = encode_browser_video(raw_output, destination)
            raw_output.unlink()
            cv2.imwrite(str(destination.with_suffix('.png')), preview)
            trace = [dict(control_step=i+1, q1=None if i<start else float(arrays['q1'][i]),
                q2=None if i<start else float(arrays['q2'][i]),
                q_min=None if i<start else float(arrays['q_min'][i]),
                reward=float(arrays['reward'][i]), observed_discounted_return=float(arrays['observed_return'][i]))
                for i in range(len(arrays['reward']))]
            destination.with_suffix('.trace.json').write_text(json.dumps(trace, indent=2)+'\n')
            q = arrays['q_min'][start:]
            observed = arrays['observed_return'][start:]
            proof = dict(environment=env_index, region=record['region'], layout=record['layout'],
                actual_outcome=outcome, actor_updates=state['actor_updates'], critic_updates=state['critic_updates'],
                source_checkpoint_SHA256=checkpoint_sha, source_HDF_SHA256=hdf_sha,
                original_video_SHA256=source_sha, output_video_SHA256=sha256(destination),
                output_video=destination.name, frames=len(steps), fps=fps,
                frame_control_steps=[i+1 for i in steps], manipulation_start_control_step=start+1,
                max_reconstructed_executed_command_error=error,
                Q_definition='minimum of matching frozen online Q1,Q2 at pre-action observation and executed deterministic goal',
                Q_is_success_probability=False, approach_Q_unavailable=True,
                observed_return_is_retrospective=True, entropy_backup=state['config']['entropy_backup'],
                discount=state['config']['gamma'], reward_scale=state['config']['reward_scale'],
                source_critic_physics=state['goal_contract']['physical_contract']['physics_dynamics'],
                evaluation_physics_device=record['simulation_device'],
                backend_transfer_diagnostic=json.loads((run/'manifest.json').read_text()).get(
                    'frozen_physics_backend_evaluation') is not None,
                casewise_return_difference_is_not_a_critic_learning_error=True,
                Q_min_first=float(q[0]), Q_min_last=float(q[-1]),
                Q_min_range=[float(q.min()),float(q.max())], observed_return_first=float(observed[0]),
                actual_Q_mean_absolute_return_error=float(np.abs(q-observed).mean()),
                no_physics_replay=True, no_GPU_used=True, optimizer_updates=0, replay_rows_imported=0,
                restored_model_tensors_exact=True,
                original_video_unchanged=sha256(source)==source_sha, browser_encoding=encoding)
            destination.with_suffix('.json').write_text(json.dumps(proof, indent=2)+'\n')
            proofs.append(proof)
    if sha256(input_hdf) != hdf_sha or sha256(args.checkpoint) != checkpoint_sha:
        raise ValueError('Closed source changed during export')
    result = dict(role='same_evaluation_policy_Q_annotation_NOT_new_evaluation', source_run=run.name,
        original_full_DEV_score_unchanged=True, sources_unchanged=True, videos=proofs)
    (output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output_dir=str(output), videos=len(proofs), sources_unchanged=True)))


if __name__ == '__main__':
    main()
