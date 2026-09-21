#!/usr/bin/env python3
"""Create a zero-delta trajectory that resolves to the live measured arm pose."""

import argparse
import json
from pathlib import Path

import numpy as np

from kuavo_real_control.contract import load_config
from kuavo_real_control.prepare import rl_actions_to_trajectory
from kuavo_real_control.trajectory import save_trajectory


ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "s63.json")
    args = parser.parse_args(argv)
    if not np.isfinite(args.seconds) or args.seconds <= 0.0:
        parser.error("--seconds must be finite and positive")
    if not np.isfinite(args.hz) or not 10.0 <= args.hz <= 100.0:
        parser.error("--hz must be finite and in [10, 100]")
    try:
        config = load_config(args.config)
        count = max(1, int(round(args.seconds * args.hz)) + 1)
        timestamps = np.arange(count, dtype=np.float64) / args.hz
        trajectory = rl_actions_to_trajectory(
            np.zeros((count, 14), dtype=np.float64),
            timestamps,
            source="generated_zero_delta_hold",
            delta_scale_rad=config.rl_delta_scale_rad,
        )
        trajectory.metadata["source_type"] = "generated_zero_delta_hold"
        save_trajectory(args.output, trajectory)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, "ERROR: {}\n".format(exc))
    print(json.dumps({
        "output": str(args.output.resolve()),
        "samples": len(trajectory.timestamps_s),
        "duration_s": trajectory.duration_s,
        "motion": "zero normalized arm delta; gripper absent",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
