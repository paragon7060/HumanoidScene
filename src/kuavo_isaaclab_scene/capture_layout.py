#!/usr/bin/env python3
"""Capture edited Isaac Sim prim positions, rotations, and scale.

This also captures each anchor's effective world scale. Only the rack
anchor's scale is functionally consumed by the scene configs (it drives
``RACK_SCALE`` in both Isaac Lab variants), so resizing the rack root
prim in Isaac Sim with the Scale tool and re-capturing is the supported
workflow for adjusting real-world rack dimensions without hand-editing
Python.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

from .paths import CONFIG_DIR


parser = argparse.ArgumentParser(description="Capture Kuavo workcell anchors from a saved USD stage.")
# Optional during AppLauncher's internal preliminary parse, then required
# immediately after the real parse. This preserves working ``--help`` output
# with Isaac Lab 0.45 while retaining the same runtime contract.
parser.add_argument(
    "stage",
    type=Path,
    nargs="?",
    help="Stage saved from the Isaac Sim layout-editing session.",
)
parser.add_argument(
    "--output",
    type=Path,
    default=CONFIG_DIR / "workcell_layout.json",
    help="Destination layout JSON.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.stage is None:
    parser.error("the following arguments are required: stage")
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from pxr import Gf, Usd, UsdGeom


ANCHOR_PATHS = {
    "robot": "/World/envs/env_0/Kuavo",
    "rack": "/World/envs/env_0/Workcell/Racks/Rack",
    "conveyor": "/World/envs/env_0/Workcell/ConveyorSystem/Visual",
    "fence": "/World/envs/env_0/Workcell/SafetySystem/Fence/Panel",
    "button_station": "/World/envs/env_0/Workcell/SafetySystem/ButtonStation",
}
# Anchors whose captured scale is actually consumed downstream. Every other
# anchor is captured with scale [1, 1, 1] regardless of any accidental Scale
# tool edits, since only the rack's physical size is meant to be adjustable
# this way.
SCALE_AWARE_ANCHORS = {"rack"}


def main() -> None:
    stage_path = args_cli.stage.expanduser().resolve()
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Could not open stage: {stage_path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    layout: dict[str, dict[str, list[float]]] = {}
    for name, prim_path in ANCHOR_PATHS.items():
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Required layout prim is missing: {prim_path}")
        matrix = cache.GetLocalToWorldTransform(prim)
        translation = matrix.ExtractTranslation()
        transform = Gf.Transform(matrix)
        quaternion = transform.GetRotation().GetQuat()
        imaginary = quaternion.GetImaginary()
        if name in SCALE_AWARE_ANCHORS:
            world_scale = transform.GetScale()
            scale_values = [round(float(world_scale[index]), 10) for index in range(3)]
        else:
            scale_values = [1.0, 1.0, 1.0]
        layout[name] = {
            "pos": [round(float(translation[index]), 6) for index in range(3)],
            "rot": [
                round(float(quaternion.GetReal()), 8),
                round(float(imaginary[0]), 8),
                round(float(imaginary[1]), 8),
                round(float(imaginary[2]), 8),
            ],
            "scale": scale_values,
        }

    output_path = args_cli.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(layout, stream, indent=2)
        stream.write("\n")
    print(f"[LAYOUT] Captured {len(layout)} anchors from {stage_path}")
    print(f"[LAYOUT] Wrote: {output_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
