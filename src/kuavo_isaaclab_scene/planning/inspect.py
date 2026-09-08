"""CPU-only pregrasp preflight. It never plans, drives a robot or starts Isaac."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

from .config import load_yaml, output_directory, validate_collection, validate_randomization
from .geometry import pose_matrix
from .robot_model import UrdfModel
from ..robots.gripper_config import load_gripper_settings
from ..robots.robot_model import resolve_robot_model, validate_robot_gripper


def file_identity(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def repository_identity(path: Path) -> dict:
    def git(*args):
        return subprocess.run(["git", "-C", str(path), *args], check=True,
                              capture_output=True, text=True, timeout=30).stdout.strip()
    return {"path": str(path), "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"), "status": git("status", "--porcelain")}


def inspect_robot(config: dict, gripper_config: Path, upstream_urdf: Path | None) -> dict:
    if not config.get("robot_model") or not config.get("hand_model"):
        raise ValueError("set robot_model/hand_model explicitly after inspecting the launcher")
    model = resolve_robot_model(config["robot_model"], config["hand_model"])
    validate_robot_gripper(model, config["hand_model"])
    gripper = load_gripper_settings(config["hand_model"], gripper_config)
    if not gripper.enabled or not gripper.integrated:
        raise ValueError("initial preflight supports integrated hands only; external hand adapter required")
    urdf = UrdfModel(model.urdf_path)
    if config.get("base_link") != urdf.root_link:
        raise ValueError("configured base_link differs from the URDF root")
    upstream, upstream_error = None, None
    if upstream_urdf is not None:
        try:
            upstream = UrdfModel(upstream_urdf)
        except ValueError as exc:
            # The upstream checkout is reference-only. Do not silently repair
            # it or claim equivalence when its full tree cannot be validated.
            upstream_error = str(exc)
    sides, all_arm_names = {}, []
    for side in ("left", "right"):
        link = config[f"{side}_tcp_link"]
        chain = urdf.chain(link)
        arm = [j for j in chain if re.fullmatch(f"zarm_{side[0]}[0-9]+_joint", j.name)]
        if not arm or any(j.kind != "revolute" for j in arm):
            raise ValueError(f"no revolute arm chain for {side}/{link}")
        if any(j.lower is None or j.upper is None or j.velocity is None for j in arm):
            raise ValueError(f"active arm is missing URDF position/velocity limits: {side}")
        names = [j.name for j in arm]
        all_arm_names.extend(names)
        tcp_offset = config.get(f"{side}_tcp_offset_pose")
        pose_matrix(tcp_offset)
        sides[side] = {
            "tcp_link": link, "tcp_offset_pose": tcp_offset,
            "root_to_tcp_joint_chain": [asdict(j) for j in chain],
            "arm_joint_names": names,
            "required_non_arm_ancestor_positions": [j.name for j in chain
                if j.kind != "fixed" and j.name not in names],
            "hand_drive_joint_names": list(gripper.joint_names_for(side)),
            "hand_open_drive_positions": gripper.command_for(side, gripper.open_command),
            "hand_close_drive_positions": gripper.command_for(side, gripper.close_command),
            "upstream_kinematic_chain_equal": (
                [asdict(j) for j in chain] == [asdict(j) for j in upstream.chain(link)]
                if upstream else None),
        }
        for name in gripper.joint_names_for(side):
            if name not in urdf.joints:
                raise ValueError(f"configured hand drive missing from URDF: {name}")
    specified = config.get("arm_joint_names")
    if specified is not None and specified != all_arm_names:
        raise ValueError("configured arm joint order differs from discovered left/right chains")
    return {
        "robot_model": model.name, "hand_model": gripper.name,
        "urdf": file_identity(Path(model.urdf_path)),
        "usd_path": model.usd_path, "usd_exists": Path(model.usd_path).is_file(),
        "upstream_urdf": file_identity(upstream_urdf) if upstream_urdf is not None else None,
        "upstream_validation_error": upstream_error,
        "base_link": urdf.root_link, "arm_joint_names": all_arm_names,
        "arm_command_dimension": len(all_arm_names),
        "hand_drive_dimension": sum(len(s["hand_drive_joint_names"]) for s in sides.values()),
        "hand_command_dimension": None,  # Policy channels are not drive joint count.
        "simulator_joint_indices": None, "sides": sides,
        "urdf_links_without_collision": sorted(set(urdf.links) - set(urdf.collision_links)),
        "movable_joints_without_velocity_limits": [j.name for j in urdf.joints.values()
                                                   if j.kind != "fixed" and j.velocity is None],
        "collision_note": "URDF collision presence is not evidence of USD collider or planner coverage",
        "tcp_physical_validation": "NOT_RUN",
    }


def build_report(config_dir: Path, repos: dict[str, Path], upstream_urdf: Path | None) -> dict:
    configs = {name: load_yaml(config_dir / f"{name}.yaml")
               for name in ("robot", "task1", "collection")}
    validate_randomization(configs["task1"])
    try:
        validate_collection(configs["collection"])
        collection_error = None
    except ValueError as exc:
        collection_error = str(exc)
    versions = {}
    for distribution in ("numpy", "PyYAML", "torch", "pin", "isaacsim", "isaaclab", "nvidia-curobo"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    # Discovery only. Importing a module/GPU context is intentionally avoided.
    curobo_present = importlib.util.find_spec("curobo") is not None
    blockers = ["live simulator joint mapping and multi-pose FK comparison not run",
                "USD physics-collider world adapter and collision coverage not implemented",
                "cuRobo robot collision model/backend and joint-drive runner not implemented",
                "pregrasp annotations and physical TCP calibration not validated"]
    if not curobo_present:
        blockers.insert(0, "cuRobo is not installed in this interpreter")
    if any(c.get("draft") is not False for c in configs.values()):
        blockers.append("configuration is still draft; this report is not an execution permit")
    robot = inspect_robot(configs["robot"], repos["HumanoidScene"] / "configs/grippers.json", upstream_urdf)
    return {
        "report_schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU source/config inspection only; no Isaac launch, no GPU allocation",
        "repositories": {name: repository_identity(path) for name, path in repos.items()},
        "configuration_files": {name: file_identity(config_dir / f"{name}.yaml") for name in configs},
        "distribution_versions": versions, "curobo_module_present": curobo_present,
        "robot": robot,
        "collection_config_error": collection_error,
        "stage1_blockers": blockers,
        "status": {"source_inspection_passed": True, "fk_validation_passed": "NOT_RUN",
                   "planning_success": "NOT_RUN", "execution_success": "NOT_RUN",
                   "pregrasp_verified": "NOT_RUN"},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--humanoid-repo", type=Path, required=True)
    parser.add_argument("--rwh-repo", type=Path, required=True)
    parser.add_argument("--ros-repo", type=Path, required=True)
    parser.add_argument("--upstream-urdf", type=Path)
    parser.add_argument("--run-name", help="One new directory under collection.output_root")
    args = parser.parse_args(argv)
    try:
        collection = load_yaml(args.config_dir / "collection.yaml")
        output = output_directory(collection, args.run_name)
        report = build_report(args.config_dir, {
            "HumanoidScene": args.humanoid_repo, "RwH-Kuavo_V2": args.rwh_repo,
            "kuavo-ros-opensource": args.ros_repo,
        }, args.upstream_urdf)
        # No audit resume: each report is immutable and never replaces an old run.
        output.mkdir(parents=True, exist_ok=False)
        for name in ("robot", "task1", "collection"):
            shutil.copyfile(args.config_dir / f"{name}.yaml", output / f"{name}.yaml")
        with (output / "preflight.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"preflight failed: {exc}\n")
    print(json.dumps({"report": str(output / "preflight.json"),
                      "source_inspection_passed": True, "pregrasp_verified": "NOT_RUN"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
