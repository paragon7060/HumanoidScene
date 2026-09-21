#!/usr/bin/env python3
"""Convert normalized S63 arm policy actions to a portable deployment trajectory."""

import argparse
import json
from pathlib import Path

import numpy as np

from kuavo_real_control.contract import load_config
from kuavo_real_control.prepare import load_rl_action_rows, rl_actions_to_trajectory
from kuavo_real_control.trajectory import save_trajectory


ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help=".npy, .npz, or JSONL normalized actions")
    parser.add_argument("output", type=Path)
    parser.add_argument("--action-key", default="action", help="NPZ array or JSONL field")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "s63.json")
    args = parser.parse_args(argv)
    if not np.isfinite(args.hz) or not 10.0 <= args.hz <= 100.0:
        parser.error("--hz must be finite and in [10, 100]")
    try:
        config = load_config(args.config)
        actions = load_rl_action_rows(args.input, args.action_key)
        timestamps = np.arange(len(actions), dtype=np.float64) / args.hz
        trajectory = rl_actions_to_trajectory(
            actions,
            timestamps,
            source=str(args.input.resolve()),
            delta_scale_rad=config.rl_delta_scale_rad,
        )
        save_trajectory(args.output, trajectory)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, "ERROR: {}\n".format(exc))
    print(json.dumps({
        "output": str(args.output.resolve()),
        "kind": trajectory.kind,
        "samples": len(trajectory.timestamps_s),
        "duration_s": trajectory.duration_s,
        "delta_scale_rad": config.rl_delta_scale_rad,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
