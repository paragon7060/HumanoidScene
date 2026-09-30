#!/usr/bin/env python3
"""Plot measured v2 SAC progress, preserving controller-specific success attribution."""
import argparse
import json
import os
from pathlib import Path
import tempfile


def read_rows(path):
    lines = path.read_text().splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            # A live logger may still be writing the final row.
    if not rows:
        raise ValueError('No completed metrics rows yet')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.run_dir / 'metrics.jsonl')
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'humanoid_rl_matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'SAC from reset': '#a93226', 'Online IK': '#239b56', 'IK warmup': '#2874a6'}
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    for label, prefix in [('SAC', 'sac_'), ('Online IK', 'online_ik_'), ('IK warmup', '')]:
        points = []
        for row in rows:
            if not prefix and row.get('rollout_policy') != 'ik_warmup':
                continue
            pair = [row.get(f'distance/{prefix}{hand}_flap_mean_m') for hand in ('left', 'right')]
            if all(value is not None for value in pair):
                points.append((row['iteration'], sum(pair) / 2))
        if points:
            color = colors.get(label, colors['SAC from reset'])
            axes[0].plot(*zip(*points), label=label, color=color)
    axes[0].set_ylabel('Mean hand-to-flap distance (m)')
    axes[0].legend(fontsize=8)
    success_keys = {
        'IK warmup': 'successful_warmup_episodes',
        'Online IK': 'successful_online_ik_episodes',
        'SAC from reset': 'successful_sac_from_reset_episodes',
    }
    summary = {'run': args.run_dir.name, 'iteration': rows[-1]['iteration'],
               'valid_transitions': rows[-1].get('valid_transitions'),
               'rollout_policy': rows[-1].get('rollout_policy')}
    def cumulative(key):
        total, values = 0, []
        for row in rows:
            total += row.get(key, 0)
            values.append(total)
        summary[key] = total
        return values
    iterations = [row['iteration'] for row in rows]
    for label, key in success_keys.items():
        axes[1].step(iterations, cumulative(key), where='post', label=label, color=colors[label])
    axes[1].set_ylabel('Held successes (cumulative)')
    axes[1].legend(fontsize=8)
    for key, label, color in [
        ('termination/unsafe', 'Unsafe', '#b03a2e'),
        ('termination/time_out', 'Timeout', '#7d3c98'),
        ('numerical_failure_episodes', 'Numerical failure (excluded)', '#d68910'),
    ]:
        axes[2].step(iterations, cumulative(key), where='post', label=label, color=color)
    axes[2].set(ylabel='Episode outcomes (cumulative)', xlabel='Iteration')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle(args.run_dir.name + '\nMeasured rollout metrics; expert success is not SAC success', fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
