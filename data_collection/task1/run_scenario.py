"""Run the canonical Task1 physical executor from a validated scenario manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from data_collection.task1.scenario import (
    PROJECT_DIR,
    build_physical_executor_argv,
    load_scenario_config,
    scenario_digest,
    validate_plan_files,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--scenario-config", type=Path, required=True)
    result.add_argument("--approach-plan", type=Path, required=True)
    result.add_argument("--retreat-plan", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--video-out", type=Path, required=True)
    result.add_argument("--headless", action="store_true")
    result.add_argument("--device", default="cuda:0")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if not args.headless:
        raise ValueError("Task1 scenario physical validation requires --headless")
    scenario = load_scenario_config(args.scenario_config)
    validate_plan_files(scenario, args.approach_plan, args.retreat_plan)
    executor_argv = build_physical_executor_argv(
        scenario,
        approach_plan=args.approach_plan,
        retreat_plan=args.retreat_plan,
        output=args.output,
        video_out=args.video_out,
    )
    command = [
        sys.executable,
        str(PROJECT_DIR / "scripts/task1_cumotion_grasp_pull_smoke.py"),
        *executor_argv,
        "--headless",
        "--device", args.device,
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        return completed.returncode

    report_path = args.output.expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["scenario"] = {
        "id": scenario["scenario_id"],
        "config_path": str(Path(scenario["_scenario_path"])),
        "config_sha256": scenario_digest(scenario),
        "planning_dof": scenario["planning"]["cspace_dof"],
        "verification_state": scenario["verification"]["state"],
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if report.get("passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
