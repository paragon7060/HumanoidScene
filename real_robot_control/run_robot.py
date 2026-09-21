#!/usr/bin/env python3
"""Dry-run or execute one validated S63 arm trajectory through the existing WBC."""

import argparse
from datetime import datetime
from pathlib import Path

from kuavo_real_control.contract import load_config
from kuavo_real_control.ros1_runtime import Ros1ArmRuntime
from kuavo_real_control.trajectory import load_trajectory


ROOT = Path(__file__).resolve().parent
LIVE_CONFIRMATION = "S63_CLEAR_AND_ESTOP_READY"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "s63.json")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--dry-run-speed", type=float, default=1.0)
    parser.add_argument("--enable-motion", action="store_true", help="Actually publish arm targets and change WBC mode")
    parser.add_argument("--enable-gripper", action="store_true", help="Also send binary Leju claw commands")
    parser.add_argument(
        "--approach-seconds", type=float, default=3.0,
        help="Minimum-jerk current-to-first-pose duration before the PLAY gate.",
    )
    parser.add_argument("--confirm", default="", help="Required exact live-motion acknowledgement")
    args = parser.parse_args(argv)
    if args.enable_motion and args.confirm != LIVE_CONFIRMATION:
        parser.error("live motion requires --confirm {}".format(LIVE_CONFIRMATION))
    if args.enable_gripper and not args.enable_motion:
        parser.error("--enable-gripper also requires --enable-motion")
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    log_path = args.log or ROOT / "logs" / ("run_{}.jsonl".format(stamp))
    try:
        config = load_config(args.config)
        trajectory = load_trajectory(args.trajectory)
        with Ros1ArmRuntime(
            config,
            trajectory,
            enable_motion=args.enable_motion,
            enable_gripper=args.enable_gripper,
            log_path=log_path,
            dry_run_speed=args.dry_run_speed,
            approach_seconds=args.approach_seconds,
        ) as runtime:
            runtime.run()
    except Exception as exc:
        parser.exit(1, "ERROR: {}\n".format(exc))
    print("Completed {}. Log: {}".format("LIVE run" if args.enable_motion else "dry-run", log_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
