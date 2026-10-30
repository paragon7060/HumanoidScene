"""Build middleware-neutral physical-replay trajectories from Task1 plans."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil


def _output_frames(physics_steps: int, physics_dt_s: float, output_hz: int) -> int:
    value = physics_steps * physics_dt_s * output_hz
    rounded = round(value)
    if not math.isclose(value, rounded, abs_tol=1e-9):
        raise ValueError(
            f"phase duration does not align to {output_hz} Hz: {physics_steps=} {physics_dt_s=}"
        )
    if rounded <= 0:
        raise ValueError("every replay phase must contain at least one output frame")
    return rounded


def build_trajectory(
    approach: dict,
    retreat: dict,
    scenario: dict,
    report: dict,
    *,
    output_hz: int = 30,
) -> dict:
    """Reconstruct the command schedule used by the Task1 physics replay."""
    if output_hz <= 0:
        raise ValueError("output_hz must be positive")
    physics_dt_s = float(report["physics_dt_s"])
    execution = scenario["execution"]
    approach_q = approach["waypoint_q_rad"]
    retreat_q = retreat["waypoint_q_rad"]

    frames: list[dict] = []

    def append(phase: str, arm_q_rad: list[float], close_fraction: float) -> None:
        index = len(frames)
        frames.append(
            {
                "index": index,
                "t_s": index / output_hz,
                "phase": phase,
                "arm_q_rad": list(arm_q_rad),
                "active_gripper_close_fraction": close_fraction,
            }
        )

    for _ in range(_output_frames(execution["settle_steps"], physics_dt_s, output_hz)):
        append("settle", approach_q[0], 0.0)

    approach_hold = _output_frames(
        execution["approach_steps_per_waypoint"], physics_dt_s, output_hz
    )
    for waypoint in approach_q:
        for _ in range(approach_hold):
            append("approach", waypoint, 0.0)

    close_frames = _output_frames(execution["close_steps"], physics_dt_s, output_hz)
    for close_index in range(close_frames):
        append("close", approach_q[-1], (close_index + 1) / close_frames)

    for _ in range(
        _output_frames(execution["closed_hold_steps"], physics_dt_s, output_hz)
    ):
        append("closed_hold", approach_q[-1], 1.0)

    retreat_hold = _output_frames(
        execution["retreat_steps_per_waypoint"], physics_dt_s, output_hz
    )
    for waypoint in retreat_q:
        for _ in range(retreat_hold):
            append("retreat", waypoint, 1.0)

    for _ in range(
        _output_frames(execution["final_hold_steps"], physics_dt_s, output_hz)
    ):
        append("final_hold", retreat_q[-1], 1.0)

    phase_counts = Counter(frame["phase"] for frame in frames)
    return {
        "schema_version": 1,
        "output_hz": output_hz,
        "joint_names": list(approach["joint_names"]),
        "active_gripper": execution["active_gripper"],
        "phase_counts": dict(phase_counts),
        "frames": frames,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_bundle(
    *,
    approach_path: Path,
    retreat_path: Path,
    scenario_path: Path,
    report_path: Path,
    rack_box_poses_path: Path,
    output_dir: Path,
    output_hz: int = 30,
) -> dict:
    """Write an auditable neutral trajectory without binding actuator topics."""
    inputs = {
        "approach_plan": Path(approach_path),
        "retreat_plan": Path(retreat_path),
        "scenario": Path(scenario_path),
        "report": Path(report_path),
        "rack_box_poses": Path(rack_box_poses_path),
    }
    data = {name: json.loads(path.read_text()) for name, path in inputs.items()}
    actual_hashes = {name: _sha256(path) for name, path in inputs.items()}
    expected_hashes = data["report"].get("input_sha256", {})
    for name in ("approach_plan", "retreat_plan", "scenario"):
        expected = expected_hashes.get(name)
        if expected != actual_hashes[name]:
            raise ValueError(
                f"{name} SHA-256 mismatch: expected {expected!r}, got {actual_hashes[name]}"
            )
    trajectory = build_trajectory(
        data["approach_plan"],
        data["retreat_plan"],
        data["scenario"],
        data["report"],
        output_hz=output_hz,
    )

    output_dir = Path(output_dir)
    source_dir = output_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=False)
    for name, path in inputs.items():
        shutil.copyfile(path, source_dir / f"{name}.json")

    jsonl_path = output_dir / "trajectory.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(frame, separators=(",", ":")) + "\n" for frame in trajectory["frames"])
    )

    joint_names = trajectory["joint_names"]
    with (output_dir / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["index", "t_s", "phase", "active_gripper_close_fraction", *joint_names]
        )
        for frame in trajectory["frames"]:
            writer.writerow(
                [
                    frame["index"],
                    frame["t_s"],
                    frame["phase"],
                    frame["active_gripper_close_fraction"],
                    *frame["arm_q_rad"],
                ]
            )

    manifest = {
        "schema_version": 1,
        "robot_model": data["scenario"]["robot"]["model"],
        "gripper_preset": data["scenario"]["robot"]["gripper_preset"],
        "active_gripper": trajectory["active_gripper"],
        "torso_height_m": data["scenario"]["planning"]["torso_height_m"],
        "source_sha256": actual_hashes,
        "trajectory": {
            "output_hz": output_hz,
            "frame_count": len(trajectory["frames"]),
            "duration_s": len(trajectory["frames"]) / output_hz,
            "joint_names": joint_names,
            "phase_counts": trajectory["phase_counts"],
        },
        "verification_state": data["report"].get("scenario", {}).get(
            "verification_state", "unknown"
        ),
        "hardware_command_ready": False,
        "hardware_command_blocker": (
            "Map the neutral arm and gripper fields to the verified real-robot topics, "
            "message types, limits, and initial-state gate before publishing commands."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest
