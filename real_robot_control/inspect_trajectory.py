#!/usr/bin/env python3
"""Validate and summarize a portable real-robot trajectory without ROS."""

import argparse
import json
from pathlib import Path

import numpy as np

from kuavo_real_control.trajectory import load_trajectory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    args = parser.parse_args(argv)
    try:
        trajectory = load_trajectory(args.trajectory)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, "ERROR: {}\n".format(exc))
    summary = {
        "path": str(args.trajectory.resolve()),
        "kind": trajectory.kind,
        "samples": len(trajectory.timestamps_s),
        "duration_s": trajectory.duration_s,
        "arm_min": np.min(trajectory.arm_values, axis=0).tolist(),
        "arm_max": np.max(trajectory.arm_values, axis=0).tolist(),
        "has_gripper": trajectory.gripper_close is not None,
        "metadata": trajectory.metadata,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
