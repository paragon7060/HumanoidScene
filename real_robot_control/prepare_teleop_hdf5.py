#!/usr/bin/env python3
"""Convert one Quest HDF5 episode to a portable S63 arm trajectory."""

import argparse
import json
from pathlib import Path

from kuavo_real_control.prepare import teleop_hdf5_to_trajectory
from kuavo_real_control.trajectory import save_trajectory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episode", help="Episode group such as demo_00000; defaults to the first")
    args = parser.parse_args(argv)
    try:
        trajectory = teleop_hdf5_to_trajectory(args.input, args.episode)
        save_trajectory(args.output, trajectory)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, "ERROR: {}\n".format(exc))
    print(json.dumps({
        "output": str(args.output.resolve()),
        "kind": trajectory.kind,
        "samples": len(trajectory.timestamps_s),
        "duration_s": trajectory.duration_s,
        "episode": trajectory.metadata["source_episode"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
