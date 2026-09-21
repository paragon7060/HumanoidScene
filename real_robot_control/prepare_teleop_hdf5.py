#!/usr/bin/env python3
"""Convert one Quest HDF5 episode to a portable S63 arm trajectory."""

import argparse
import json
from pathlib import Path

from kuavo_real_control.contract import load_config
from kuavo_real_control.prepare import (
    TELEOP_COMMAND_SOURCES,
    retime_joint_position_trajectory,
    teleop_hdf5_to_trajectory,
)
from kuavo_real_control.trajectory import save_trajectory


ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episode", help="Episode group such as demo_00000; defaults to the first")
    parser.add_argument(
        "--source", choices=TELEOP_COMMAND_SOURCES, default="auto",
        help="Arm command source. auto prefers the post-IK self-collision-safe target.",
    )
    parser.add_argument(
        "--allow-unsuccessful", action="store_true",
        help="Offline review only: allow an episode not marked successful.",
    )
    parser.add_argument(
        "--keep-source-timing", action="store_true",
        help="Do not create a deployment-ready retimed trajectory; useful only for offline comparison.",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "s63.json")
    args = parser.parse_args(argv)
    try:
        trajectory = teleop_hdf5_to_trajectory(
            args.input,
            args.episode,
            command_source=args.source,
            require_success=not args.allow_unsuccessful,
        )
        if not args.keep_source_timing:
            trajectory = retime_joint_position_trajectory(trajectory, load_config(args.config))
        save_trajectory(args.output, trajectory)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.exit(1, "ERROR: {}\n".format(exc))
    print(json.dumps({
        "output": str(args.output.resolve()),
        "kind": trajectory.kind,
        "samples": len(trajectory.timestamps_s),
        "duration_s": trajectory.duration_s,
        "episode": trajectory.metadata["source_episode"],
        "source": trajectory.metadata["source_command_source"],
        "deployment_ready": trajectory.metadata["deployment_ready"],
        "retiming": trajectory.metadata.get("retiming"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
