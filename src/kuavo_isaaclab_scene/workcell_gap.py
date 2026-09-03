"""Inspect/set the rack-to-conveyor exterior gap without launching Isaac Sim.

Calibration: Rack.usd default-prim bounds in metres; NVIDIA Isaac 5.1
ConveyorBelt_A08_PR_NVD_01.usd default-prim bounds converted from centimetres
using the scene's fixed 0.01 scale. These include protruding exterior frames,
not just the invisible conveyor collision deck. Recalibrate for new assets.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import product
import json
import math
from pathlib import Path
import time

from .workcell_layout import AnchorPose, LAYOUT_PATH, load_layout, quat_rotate


RACK_BOUNDS_M = (
    (-1.0255, -0.855476901207163, 0.0),
    (0.0255, 0.025476901207163206, 2.165),
)
CONVEYOR_BOUNDS_M = (
    (-0.07405029296875, -2.7187200927734375, 0.0),
    (1.07690185546875, 0.0000396728515625, 1.1663306427001953),
)


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def _projection(pose: AnchorPose, bounds, axis) -> tuple[float, float]:
    values = []
    for corner in product(*zip(*bounds)):
        local = tuple(v * scale for v, scale in zip(corner, pose.scale))
        rotated = quat_rotate(pose.rot, local)
        world = tuple(a + b for a, b in zip(pose.pos, rotated))
        values.append(_dot(world, axis))
    return min(values), max(values)


def measure_gap(poses: dict[str, AnchorPose]) -> tuple[float, tuple[float, ...]]:
    """Return exterior projected gap and axis pointing from conveyor to rack.

    Only upright, parallel fixtures with overlapping frontage are accepted.
    This prevents treating an arbitrary diagonal/end-to-end arrangement as
    the same robot aisle. The conservative exterior bounds are not a collision
    clearance or an assurance that a robot of any size can safely enter.
    """
    for name in ("rack", "conveyor"):
        pose = poses[name]
        if not all(math.isfinite(v) for v in (*pose.pos, *pose.rot, *pose.scale)):
            raise ValueError(f"{name}: pose must contain only finite values")
        if any(v <= 0 for v in pose.scale):
            raise ValueError(f"{name}: scale must be positive")
        if _dot(quat_rotate(pose.rot, (0, 0, 1)), (0, 0, 1)) < 1 - 1e-6:
            raise ValueError("Gap adjustment requires upright rack and conveyor")
    if any(abs(v - 1) > 1e-8 for v in poses["conveyor"].scale):
        raise ValueError("Conveyor layout scale must be 1; the scene uses a fixed asset scale")

    # Rack native +Y points out of the shelf; -Y points into the rack.
    axis = quat_rotate(poses["rack"].rot, (0, -1, 0))
    width_axis = quat_rotate(poses["conveyor"].rot, (1, 0, 0))
    if abs(_dot(axis, width_axis)) < 1 - 1e-6:
        raise ValueError("Rack front and conveyor side must be parallel; edit rotations first")
    tangent = quat_rotate(poses["rack"].rot, (1, 0, 0))
    rack_span = _projection(poses["rack"], RACK_BOUNDS_M, tangent)
    belt_span = _projection(poses["conveyor"], CONVEYOR_BOUNDS_M, tangent)
    if min(rack_span[1], belt_span[1]) <= max(rack_span[0], belt_span[0]):
        raise ValueError("Rack and conveyor frontage does not overlap; no shared aisle")
    rack = _projection(poses["rack"], RACK_BOUNDS_M, axis)
    conveyor = _projection(poses["conveyor"], CONVEYOR_BOUNDS_M, axis)
    if sum(conveyor) >= sum(rack):
        raise ValueError("Conveyor must be on the open/front side of the rack")
    return rack[0] - conveyor[1], axis


def adjusted_position(poses: dict[str, AnchorPose], gap_m: float) -> tuple[float, ...]:
    if not math.isfinite(gap_m) or gap_m <= 0:
        raise ValueError("Requested gap must be a finite positive number in metres")
    before, axis = measure_gap(poses)
    # Snap tiny axis components to zero so rounded 90-degree captured
    # quaternions do not introduce meaningless changes to Y/Z coordinates.
    axis = tuple(0.0 if abs(v) < 1e-7 else v for v in axis)
    norm = math.sqrt(_dot(axis, axis))
    return tuple(
        round(value - direction / norm * (gap_m - before), 9)
        for value, direction in zip(poses["conveyor"].pos, axis)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, default=LAYOUT_PATH, help="Layout JSON to inspect/update")
    parser.add_argument("--gap", type=float, help="Desired exterior gap in metres; omit to inspect")
    parser.add_argument("--dry-run", action="store_true", help="Preview without modifying any file")
    parser.add_argument("--output", type=Path, help="Write a separate layout; input remains unchanged")
    args = parser.parse_args(argv)
    if args.output is not None and args.gap is None:
        parser.error("--output requires --gap")
    try:
        source = args.layout.expanduser().resolve()
        poses = load_layout(source)
        before, _ = measure_gap(poses)
        print(f"[LAYOUT] {source}")
        print(f"[GAP] Exterior rack/conveyor gap: {before:.6f} m")
        if args.gap is None:
            return 0
        target = adjusted_position(poses, args.gap)
        print(f"[MOVE] Conveyor: {poses['conveyor'].pos} -> {target}")
        print(f"[GAP] Requested: {args.gap:.6f} m; rack and robot unchanged")
        if args.dry_run:
            print("[DRY RUN] No files changed")
            return 0
        original_text = source.read_text(encoding="utf-8")
        raw = json.loads(original_text)
        updated = deepcopy(raw)
        if isinstance(updated["conveyor"], list):
            updated["conveyor"] = list(target)
        else:
            updated["conveyor"]["pos"] = list(target)
        destination = (args.output or source).expanduser().resolve()
        if destination != source and destination.exists():
            raise ValueError(f"Output already exists: {destination}; choose a new path")
        text = json.dumps(updated, indent=2, ensure_ascii=False) + "\n"
        if destination == source:
            if updated == raw:
                print("[LAYOUT] Already at requested gap; no write needed")
                return 0
            backup = source.with_name(source.name + f".bak.{time.time_ns()}")
            # Keep every previous version, including repeated adjustments.
            with backup.open("x", encoding="utf-8") as stream:
                stream.write(original_text)
            print(f"[BACKUP] {backup}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w" if destination == source else "x", encoding="utf-8") as stream:
            stream.write(text)
        print(f"[LAYOUT] Wrote {destination}; restart the scene/eval to apply")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
