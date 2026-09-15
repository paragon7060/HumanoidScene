#!/usr/bin/env python3
"""Compose S63 with the packaged claw and a wrist visual without a second camera."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kuavo_isaaclab_scene.core.paths import ASSET_DIR
from kuavo_isaaclab_scene.robots.claw_assets import append_claw_branch, load_claw_asset


def wrist_without_camera(source: Path, output: Path, side: str) -> None:
    """Remove the three disconnected camera/bracket CAD parts, retaining the wrist.

    This is specific to the packaged S63 CAD revision. Do not silently apply
    its spatial selection to a new upstream mesh. Keep the original STL intact.
    """
    import numpy as np
    import trimesh

    expected = {
        "l": "1737a47cf34b44031e00d5b764e7a1d740e01f3f927054e000974898b0f40080",
        "r": "4c58f97b7d755da83f4c9977d21c9b8ed3d6fe9efed85607365c41a459ef5bbe",
    }
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected[side]:
        raise ValueError(f"Inspect changed S63 wrist CAD before removing its camera: {source}")
    mesh = trimesh.load(source, force="mesh", process=False)
    connected = mesh.copy()
    connected.merge_vertices()
    # Use original face indices: mesh.split() repairs holes and adds triangles.
    groups = trimesh.graph.connected_components(connected.face_adjacency,
                                                nodes=np.arange(len(mesh.faces)))
    camera_parts = [group for group in groups if mesh.triangles[group, :, 0].max() > 0.08]
    if sorted(map(len, camera_parts)) != [134, 1500, 1742]:
        raise ValueError(f"Unexpected S63 camera components: {source}")
    keep = np.ones(len(mesh.faces), dtype=bool)
    keep[np.concatenate(camera_parts)] = False
    wrist = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[keep], process=False)
    wrist.remove_unreferenced_vertices()
    output.parent.mkdir(parents=True, exist_ok=True)
    wrist.export(output)


def build(source: Path, output: Path) -> None:
    if source.resolve() == output.resolve():
        raise ValueError("S63 source must not be overwritten")
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("name", "kuavo_s63_twofinger")
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        prefix = "package://kuavo_assets/models/biped_s63/meshes/"
        path = (source.parent.parent / "meshes" / filename[len(prefix):]
                if filename.startswith(prefix) else source.parent / filename)
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set("filename", os.path.relpath(path.resolve(), output.resolve().parent))
    for side in ("l", "r"):
        visual = root.find(f"./link[@name='zarm_{side}7_link']/visual/geometry/mesh")
        wrist_source = (source.parent.parent / "meshes" / f"{side}_hand_pitch.STL").resolve()
        incoming_wrist = (output.parent / visual.get("filename")).resolve()
        if incoming_wrist not in (wrist_source, wrist_source.with_name(f"{side}_hand_pitch_noHand.STL")):
            raise ValueError(f"Expected packaged S63 hand-pitch visual for {side}")
        wrist_output = output.parent.parent / "meshes" / f"{side}_hand_pitch_wrist_only.STL"
        wrist_without_camera(wrist_source, wrist_output, side)
        visual.set("filename", os.path.relpath(wrist_output.resolve(), output.resolve().parent))
    for side in ("left", "right"):
        asset = load_claw_asset(side)
        mount = asset.metadata["sides"][side]
        append_claw_branch(root, side=side, parent_link=mount["source_mount_body"],
                           output_urdf=output, xyz=tuple(mount["source_mount_xyz_m"]),
                           rpy=tuple(mount["source_mount_rpy_rad"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    print(f"Generated {output}; duplicate S63 wrist cameras removed, Leju D405 mounts retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ASSET_DIR / "kuavo_s63/urdf/kuavo_s63.urdf")
    parser.add_argument("--output", type=Path, default=ASSET_DIR / "kuavo_s63_twofinger/urdf/kuavo_s63_twofinger.urdf")
    args = parser.parse_args()
    build(args.source, args.output)
