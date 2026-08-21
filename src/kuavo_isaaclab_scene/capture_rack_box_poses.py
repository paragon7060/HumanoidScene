#!/usr/bin/env python3
"""Capture manually edited rack-box poses from an Isaac Sim stage.

The saved position and orientation are relative to the workcell ``Rack``
anchor Xform. Consequently, a captured arrangement follows later rack
translation, rotation, and scale changes.  Box scale remains independent and
is captured per instance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

from .paths import CONFIG_DIR


RACK_PRIM_PATH = "/World/envs/env_0/Workcell/Racks/Rack"
BOX_INSTANCE_NAMES = tuple(
    f"{box_type}_{index}"
    for box_type in ("SmallBox", "MediumBox", "LargeBox", "XLargeBox")
    for index in range(2)
)
BOX_PRIM_PATHS = {
    name: f"/World/envs/env_0/Workcell/StagingBoxes/{name}"
    for name in BOX_INSTANCE_NAMES
}

# Broad bounds in the authored Rack.usd coordinate system. They identify
# boxes placed on one of the three shelves while excluding floor staging.
RACK_CAPTURE_BOUNDS = (
    (-1.20, 0.20),
    (-1.05, 0.15),
    (0.25, 2.05),
)
RACK_SHELF_SURFACE_Z = (0.499122, 1.104122, 1.714122)


parser = argparse.ArgumentParser(
    description=(
        "Capture box position, rotation, and scale relative to the Rack anchor "
        "from a saved Isaac Sim stage."
    )
)
# Keep this optional during AppLauncher's preliminary parse, then validate it
# after the complete Isaac Lab v2.3 parse so both ``--help`` and missing-stage
# errors behave normally.
parser.add_argument(
    "stage",
    type=Path,
    nargs="?",
    help="USD/USDA stage saved after manual editing.",
)
parser.add_argument(
    "--output",
    type=Path,
    default=CONFIG_DIR / "rack_box_poses.json",
    help="Destination JSON (default: packaged configs/rack_box_poses.json).",
)
parser.add_argument(
    "--boxes",
    nargs="+",
    choices=BOX_INSTANCE_NAMES,
    default=None,
    metavar="NAME",
    help=(
        "Capture exactly these instances, including boxes outside the automatic "
        "rack bounds. If omitted, only boxes detected inside the rack are saved."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.stage is None:
    parser.error("the following arguments are required: stage")
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from pxr import Gf, Usd, UsdGeom


def _rounded_vector(vector: object, digits: int) -> list[float]:
    return [round(float(vector[index]), digits) for index in range(3)]


def _inside_rack(local_position: object) -> bool:
    return all(
        lower <= float(local_position[axis]) <= upper
        for axis, (lower, upper) in enumerate(RACK_CAPTURE_BOUNDS)
    )


def _nearest_shelf(local_z: float) -> int:
    return min(
        range(1, len(RACK_SHELF_SURFACE_Z) + 1),
        key=lambda shelf: abs(local_z - RACK_SHELF_SURFACE_Z[shelf - 1]),
    )


def main() -> None:
    stage_path = args_cli.stage.expanduser().resolve()
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Could not open stage: {stage_path}")

    rack_prim = stage.GetPrimAtPath(RACK_PRIM_PATH)
    if not rack_prim.IsValid():
        raise RuntimeError(f"Required rack prim is missing: {RACK_PRIM_PATH}")

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    rack_world_matrix = cache.GetLocalToWorldTransform(rack_prim)
    rack_world_transform = Gf.Transform(rack_world_matrix)
    rack_world_quaternion = rack_world_transform.GetRotation().GetQuat()
    rack_world_inverse = rack_world_matrix.GetInverse()

    requested = set(args_cli.boxes) if args_cli.boxes else None
    captured: dict[str, dict[str, object]] = {}
    for instance_name, prim_path in BOX_PRIM_PATHS.items():
        if requested is not None and instance_name not in requested:
            continue
        box_prim = stage.GetPrimAtPath(prim_path)
        if not box_prim.IsValid():
            if requested is not None:
                raise RuntimeError(f"Requested box prim is missing: {prim_path}")
            continue

        box_world_matrix = cache.GetLocalToWorldTransform(box_prim)
        box_world_transform = Gf.Transform(box_world_matrix)
        box_world_position = box_world_matrix.ExtractTranslation()
        local_position = rack_world_inverse.Transform(box_world_position)
        if requested is None and not _inside_rack(local_position):
            continue

        box_world_quaternion = box_world_transform.GetRotation().GetQuat()
        local_quaternion = rack_world_quaternion.GetInverse() * box_world_quaternion
        local_imaginary = local_quaternion.GetImaginary()
        captured[instance_name] = {
            "local_pos": _rounded_vector(local_position, 8),
            "local_rot": [
                round(float(local_quaternion.GetReal()), 10),
                round(float(local_imaginary[0]), 10),
                round(float(local_imaginary[1]), 10),
                round(float(local_imaginary[2]), 10),
            ],
            "scale": _rounded_vector(box_world_transform.GetScale(), 10),
            "shelf": _nearest_shelf(float(local_position[2])),
        }

    if not captured:
        bounds = ", ".join(f"{low}..{high}" for low, high in RACK_CAPTURE_BOUNDS)
        raise RuntimeError(
            "No boxes were detected inside the Rack.usd-local bounds "
            f"({bounds}). Move box root prims onto the rack, or pass --boxes NAME ... "
            "to capture explicit instances."
        )

    output_path = args_cli.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "reference": {
            "prim_path": RACK_PRIM_PATH,
            "source_stage": str(stage_path),
            "coordinate_frame": "Rack-anchor-local",
            "rotation_order": "quaternion-wxyz",
        },
        "boxes": captured,
    }
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")

    print(f"[RACK BOXES] Captured {len(captured)} box(es): {', '.join(captured)}")
    print(f"[RACK BOXES] Source: {stage_path}")
    print(f"[RACK BOXES] Wrote: {output_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
