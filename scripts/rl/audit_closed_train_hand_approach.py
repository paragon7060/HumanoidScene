"""Locate bilateral approach loss in closed, original TRAIN trajectories.

Only scalar distances, angles and measured jaw summaries are exported.
Threshold crossings are observation evidence, not confirmed helper activation
or physical pinch. No replay import, model update or physics run is performed.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from audit_closed_dev_critics import identity, require_closed_run
from export_eval_q_videos import sha256
from kuavo_isaaclab_scene.rl.multi_box.demo_replay import _rotation_matrix
from kuavo_isaaclab_scene.robots.end_effector import closed_closing_axes
from summarize_batched_staged_run import read_snapshot, supported_success


def median(records, field):
    values = [r[field] for r in records if r[field] is not None]
    return float(np.median(values)) if values else None


def audit(run, output, wave):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only diagnostic requires an empty CUDA mask')
    torch.set_num_threads(1)
    require_closed_run(run)
    manifest = read_snapshot(run/'manifest.json')
    rows = [r for r in read_snapshot(run/'metrics.json')['outcomes'] if r['wave'] == wave]
    if (len(rows) != 128 or {r['environment'] for r in rows} != set(range(128))
            or manifest['layout_waves'][wave]['split'] != 'train'
            or any(r['split'] != 'train' or not r['complete'] for r in rows)):
        raise ValueError('A complete original TRAIN128 wave is required')
    expected = Counter({'shelf_2_left/small': 16, 'shelf_2_right/small': 16,
        'shelf_2_left/medium': 16, 'shelf_2_right/medium': 16,
        'shelf_3_left/small': 32, 'shelf_3_right/small': 32})
    if Counter(r['layout']['target_region']+'/'+r['layout']['target_box_type'] for r in rows) != expected:
        raise ValueError('All six original groups must be retained')
    path = run/'executed_transitions.hdf5'
    before = identity(path)
    checksum = sha256(path)
    axes = torch.tensor([closed_closing_axes()[s] for s in ('left', 'right')])
    records = []
    with h5py.File(path, 'r') as hdf:
        episodes = {(int(e.attrs['wave']), int(e.attrs['environment'])): e
                    for e in hdf['episodes'].values()}
        for row in sorted(rows, key=lambda r: r['environment']):
            index = row['environment']
            episode = episodes[wave, index]
            result = row['result']
            if (json.loads(episode.attrs['layout_json']) != row['layout']
                    or row['layout'] != manifest['layout_waves'][wave]['layouts'][index]['layout']
                    or bool(episode.attrs['success']) != supported_success(row)):
                raise ValueError('Actual trajectory and original outcome differ')
            mode = row.get('collection_policy_mode')
            if mode != episode.attrs.get('collection_policy_mode'):
                raise ValueError('Recorded TRAIN selection mode differs')
            start = result['staged_base'].get('manipulation_start')
            record = dict(environment=index, seed=row['layout']['seed'],
                region=row['layout']['target_region'], box_type=row['layout']['target_box_type'],
                mode=mode, success=supported_success(row), unsafe=result.get('unsafe', False),
                time_out=result.get('time_out', False), initial_layout_valid=row['initial_layout_valid'],
                rack_peak_body=result.get('rack_peak_body'), manipulation_start=start,
                terminal_surface_distances_m=result.get('flap_distances'))
            if start is None or not row['initial_layout_valid']:
                record['not_analyzed_reason'] = 'no_valid_held_phase'
                records.append(record)
                continue
            transitions = episode['transitions']
            supplemental = torch.from_numpy(transitions['actor_supplemental'][start:])
            if supplemental.ndim != 2 or supplemental.shape[1] != 38 or not len(supplemental):
                raise ValueError('Expected recorded midpoint perception throughout held phase')
            valid = torch.isfinite(supplemental).all(-1) & (supplemental[:, 36:].sum(-1) > .5)
            if not bool(valid.all()):
                raise ValueError('Do not silently analyze invalid perception rows')
            assignment = supplemental[:, 36:].argmax(-1)
            panel = supplemental[:, :36].reshape(-1, 2, 2, 9)
            selected = panel[torch.arange(len(panel))[:, None], torch.arange(2)[None],
                             torch.stack((assignment, 1-assignment), -1)]
            distance = selected[..., :3].norm(dim=-1)
            normal = _rotation_matrix(selected[..., 3:])[..., 0]
            dot = (normal*axes[None]).sum(-1).abs().clamp(0, 1)
            angle = torch.acos(dot)
            first = distance[0]
            worst = distance.amax(-1)
            together = worst <= .22
            aligned = (angle <= .25).all(-1)
            both_near = worst <= .10
            exact_rows = together.nonzero().flatten()
            first_near = int(exact_rows[0]) if len(exact_rows) else None
            # The original contact attempt has a finite360-tick lifetime.
            # This is a window aligned to the first necessary distance gate,
            # never a claim that the helper actually started on that row.
            window_end = min(len(distance), (first_near or 0)+360)
            raw_jaws = torch.from_numpy(transitions['actor_obs'][start:, 46:48])
            record.update(held_rows=len(distance),
                midpoint_first_left_right_m=first.tolist(),
                midpoint_last_left_right_m=distance[-1].tolist(),
                both_hands_best_midpoint_distance_m=float(worst.min()),
                both_hands_last_midpoint_distance_m=float(worst[-1]),
                both_hands_last_minus_first_distance_m=float(worst[-1]-worst[0]),
                first_both_midpoint_within22cm_held_row=first_near,
                both_within22cm_rows=int(together.sum()), both_within10cm_rows=int(both_near.sum()),
                both_axis_aligned_rows=int(aligned.sum()),
                both_close_and_aligned_rows=int((both_near&aligned).sum()),
                midpoint_at_distance_gate_plus360_left_right_m=distance[window_end-1].tolist(),
                measured_closure_best_lower_hand_fraction=float(raw_jaws.amin(-1).max()),
                measured_closure_last_left_right_fraction=raw_jaws[-1].tolist())
            records.append(record)
    if identity(path) != before:
        raise ValueError('Closed trajectory changed during inspection')
    grouped = defaultdict(list)
    for record in records:
        grouped[record['region']+'/'+record['box_type']+'/'+str(record['mode'])].append(record)
    summaries = {}
    for key, group in grouped.items():
        analyzed = [r for r in group if 'held_rows' in r]
        summaries[key] = dict(requested=len(group), analyzed=len(analyzed),
            success=sum(r['success'] for r in group), unsafe=sum(r['unsafe'] for r in group),
            time_out=sum(r['time_out'] for r in group),
            ever_both_within22cm=sum(r['both_within22cm_rows'] > 0 for r in analyzed),
            ever_both_within10cm=sum(r['both_within10cm_rows'] > 0 for r in analyzed),
            ever_both_close_and_aligned=sum(r['both_close_and_aligned_rows'] > 0 for r in analyzed),
            **{field+'_median': median(analyzed, field) for field in (
                'both_hands_best_midpoint_distance_m', 'both_hands_last_midpoint_distance_m',
                'both_hands_last_minus_first_distance_m', 'measured_closure_best_lower_hand_fraction')})
    proof = dict(recorded_at=datetime.now().astimezone().isoformat(),
        role='closed_original_TRAIN_bilateral_approach_diagnostic_NOT_new_evaluation',
        run_directory_name=run.name, wave=wave, source_HDF_SHA256=checksum,
        requested=128, all6_scope=True, records=records, by_group_and_mode=summaries,
        proximity_or_measured_closure_NOT_confirmed_pinch_or_success=True,
        distance_gate_crossing_NOT_proof_of_helper_activation=True,
        nearest_surface_distance_and_midpoint_distance_distinguished=True,
        no_GPU_physics_optimizer_replay_import_or_DEV_FINAL_rows=True,
        raw_observations_actions_NOT_exported=True, closed_sources_unchanged=True,
        existing_success_denominator_unchanged=True, goal_not_complete=True)
    output.mkdir(exist_ok=False, parents=True)
    (output/'verification.json').write_text(json.dumps(proof, indent=2)+'\n')
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--wave', type=int, default=1)
    parser.add_argument('--plot', action='store_true',
                        help='Save a scalar-only comparison of best and final pre-action midpoint distances')
    args = parser.parse_args()
    proof = audit(args.run_dir.resolve(), args.output_dir.resolve(), args.wave)
    if args.plot:
        plot(proof, args.output_dir.resolve()/'hand_approach.png')
    print(json.dumps(dict(wave=proof['wave'], groups=proof['by_group_and_mode'])))


def plot(proof, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = [('shelf_2_left/small', 'Middle left / small'),
              ('shelf_2_right/small', 'Middle right / small'),
              ('shelf_3_left/small', 'Upper left / small'),
              ('shelf_3_right/small', 'Upper right / small'),
              ('shelf_2_left/medium', 'Middle left / medium'),
              ('shelf_2_right/medium', 'Middle right / medium')]
    modes = [('greedy_current_policy', 'Greedy collection', '#1769aa'),
             ('perceived_contact_exploration', 'Selected TRAIN exploration', '#b36b00')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, (mode, title, color) in zip(axes, modes):
        for y, (key, _) in enumerate(groups):
            row = proof['by_group_and_mode'][key+'/'+mode]
            best = row['both_hands_best_midpoint_distance_m_median']
            last = row['both_hands_last_midpoint_distance_m_median']
            ax.plot([best, last], [y, y], color=color, linewidth=3)
            ax.scatter([best], [y], facecolors='white', edgecolors=color, s=65, zorder=3)
            ax.scatter([last], [y], color=color, s=65, zorder=3)
            ax.text(1.57, y, f"{row['success']}/{row['unsafe']}/{row['time_out']}",
                    ha='right', va='center', fontsize=10)
        ax.axvline(.10, color='#777777', linestyle=':', linewidth=1)
        ax.set_title(title)
        ax.set_xlabel('Worse-hand midpoint distance (m), episode median')
        ax.set_xlim(0, 1.60)
        ax.set_xticks([0, .1, .3, .6, .9, 1.2])
        ax.grid(axis='x', alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_yticks(range(len(groups)), [label for _, label in groups])
    axes[0].invert_yaxis()
    fig.suptitle(f'Closed original TRAIN{proof["wave"]}: where bilateral approach is lost', fontsize=15)
    fig.text(.5, .02, 'Open marker: best pre-action distance. Filled marker: last pre-action distance.\n'
             'Row counts: success / unsafe / timeout. Midpoint/current-assignment proximity is not pinch evidence. '
             'Early collisions shorten trajectories. '
             'Original 128 requests retained.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
