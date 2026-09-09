#!/usr/bin/env python3
"""Generate checked-in S200062 hand spheres with NVIDIA cuMotion."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from kuavo_isaaclab_scene.core.paths import ASSET_DIR, PACKAGE_CONFIG_DIR
from kuavo_isaaclab_scene.planning.gripper_collision import (
    GRIPPER_COLLISION_FRAMES,
    SUPPORTED_MAX_OVERSHOOT_M,
)


def read_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read triangles without adding a second mesh dependency to the package."""
    raw = path.read_bytes()
    triangle_count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + 50 * triangle_count:
        raise ValueError(f"expected binary STL: {path}")
    vertices = np.empty((triangle_count * 3, 3), dtype=np.float64)
    triangles = np.arange(triangle_count * 3, dtype=np.int32).reshape(-1, 3)
    for index in range(triangle_count):
        record = struct.unpack_from("<12fH", raw, 84 + 50 * index)
        vertices[index * 3 : index * 3 + 3] = np.asarray(record[3:12]).reshape(3, 3)
    return vertices, triangles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_CONFIG_DIR / "task1_s200062_gripper_collision_spheres.json",
    )
    args = parser.parse_args()
    import cumotion

    mesh_root = ASSET_DIR / "kuavo_s200062" / "meshes"
    meshes = {
        frame: read_binary_stl(mesh_root / f"{frame}.STL")
        for frame in GRIPPER_COLLISION_FRAMES
    }
    payload = {
        "schema_version": 1,
        "generator": "NVIDIA cuMotion 1.1.0 generate_collision_spheres",
        "mesh_bounds": {
            frame: {
                "min": vertices.min(axis=0).tolist(),
                "max": vertices.max(axis=0).tolist(),
            }
            for frame, (vertices, _) in meshes.items()
        },
        "robot_model": "s200062",
        "presets": {},
    }
    for max_overshoot_m in SUPPORTED_MAX_OVERSHOOT_M:
        frames = {}
        for frame_name, (vertices, triangles) in meshes.items():
            spheres = cumotion.generate_collision_spheres(
                vertices, triangles, max_overshoot_m
            )
            frames[frame_name] = [
                {
                    "center": [float(value) for value in sphere.center],
                    "radius": float(sphere.radius),
                }
                for sphere in spheres
            ]
        payload["presets"][f"{max_overshoot_m:.3f}"] = {
            "frames": frames,
            "max_overshoot_m": max_overshoot_m,
            "sphere_count": sum(map(len, frames.values())),
        }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
