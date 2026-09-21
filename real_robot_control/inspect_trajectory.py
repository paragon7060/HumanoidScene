#!/usr/bin/env python3
"""Validate and summarize a portable real-robot trajectory without ROS."""

import argparse
import hashlib
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
        "sha256": hashlib.sha256(args.trajectory.read_bytes()).hexdigest(),
        "kind": trajectory.kind,
        "samples": len(trajectory.timestamps_s),
        "duration_s": trajectory.duration_s,
        "first_arm_position_deg": np.rad2deg(trajectory.arm_values[0]).tolist(),
        "last_arm_position_deg": np.rad2deg(trajectory.arm_values[-1]).tolist(),
        "arm_min": np.min(trajectory.arm_values, axis=0).tolist(),
        "arm_max": np.max(trajectory.arm_values, axis=0).tolist(),
        "has_gripper": trajectory.gripper_close is not None,
        "metadata": trajectory.metadata,
    }
    if trajectory.kind == "joint_position_rad" and len(trajectory.timestamps_s) > 1:
        dt = np.diff(trajectory.timestamps_s)
        velocity = np.diff(trajectory.arm_values, axis=0) / dt[:, None]
        summary["max_velocity_rad_s"] = float(np.max(np.abs(velocity)))
        if len(velocity) > 1:
            velocity_dt = (dt[1:] + dt[:-1]) / 2.0
            acceleration = np.diff(velocity, axis=0) / velocity_dt[:, None]
            summary["max_acceleration_rad_s2"] = float(np.max(np.abs(acceleration)))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
