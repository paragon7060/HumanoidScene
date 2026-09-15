#!/usr/bin/env python3
"""Compose official S63 + packaged claw without changing any host mesh/frame."""
import argparse
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kuavo_isaaclab_scene.core.paths import ASSET_DIR
from kuavo_isaaclab_scene.robots.claw_assets import append_claw_branch, load_claw_asset


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
    for side in ("left", "right"):
        asset = load_claw_asset(side)
        mount = asset.metadata["sides"][side]
        append_claw_branch(root, side=side, parent_link=mount["source_mount_body"],
                           output_urdf=output, xyz=tuple(mount["source_mount_xyz_m"]),
                           rpy=tuple(mount["source_mount_rpy_rad"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    print(f"Generated {output}; official S63 visuals and EEF frames unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ASSET_DIR / "kuavo_s63/urdf/biped_s63.urdf")
    parser.add_argument("--output", type=Path, default=ASSET_DIR / "kuavo_s63_twofinger/urdf/kuavo_s63_twofinger.urdf")
    args = parser.parse_args()
    build(args.source, args.output)
