#!/usr/bin/env python3
"""Generate self-contained left/right claw URDFs, meshes and metadata.

This is an asset extraction/build step, not a simulation or a host-model edit.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kuavo_isaaclab_scene.core.paths import ASSET_DIR


def extract(donor_dir: Path, output: Path) -> None:
    if output.resolve() == donor_dir.resolve() or donor_dir.resolve() in output.resolve().parents:
        raise ValueError("Extraction must not overwrite the donor model")
    source = donor_dir / "urdf/biped_s200062.urdf"
    root = ET.parse(source).getroot()
    estimates = json.loads((donor_dir / "teleop_inertials.json").read_text())
    selected_estimates, hashes, sides = {}, {}, {}
    output.mkdir(parents=True, exist_ok=True)
    for side, prefix in (("left", "l"), ("right", "r")):
        base = f"{prefix}_twofinger_base"
        links, joint_names = {base}, set()
        changed = True
        while changed:
            changed = False
            for joint in root.findall("joint"):
                if joint.find("parent").get("link") in links:
                    child = joint.find("child").get("link")
                    joint_names.add(joint.get("name"))
                    if child not in links:
                        links.add(child)
                        changed = True
        if len(links) != 14:
            raise ValueError(f"Unexpected donor topology for {side}: {sorted(links)}")
        hand = ET.Element("robot", name=f"leju_claw_{side}")
        for node in root:
            if not ((node.tag == "link" and node.get("name") in links)
                    or (node.tag == "joint" and node.get("name") in joint_names)):
                continue
            node = deepcopy(node)
            if node.tag == "link":
                name = node.get("name")
                values = estimates["links"][name]
                selected_estimates[name] = values
                if node.find("inertial") is not None:
                    raise ValueError(f"Donor now has official inertia for {name}; review estimates")
                inertia = ET.SubElement(node, "inertial")
                ET.SubElement(inertia, "origin", xyz=" ".join(map(str, values["com_m"])), rpy="0 0 0")
                ET.SubElement(inertia, "mass", value=str(values["mass_kg"]))
                ixx, iyy, izz = values["diagonal_inertia_kg_m2"]
                ET.SubElement(inertia, "inertia", ixx=str(ixx), iyy=str(iyy), izz=str(izz),
                              ixy="0", ixz="0", iyz="0")
                # Same simplified runtime contact geometry: housing and separate
                # jaws, never convex hulls spanning the gripping gap or thin bars.
                if name == base or name in {f"{prefix}_{jaw}_finger" for jaw in "fb"}:
                    collision = deepcopy(node.find("visual"))
                    collision.tag = "collision"
                    material = collision.find("material")
                    if material is not None:
                        collision.remove(material)
                    node.append(collision)
            for mesh in node.findall(".//mesh"):
                filename = mesh.get("filename")
                original = (source.parent / filename).resolve()
                destination = output / "meshes" / original.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, destination)
                hashes[original.name] = hashlib.sha256(original.read_bytes()).hexdigest()
                mesh.set("filename", f"../meshes/{original.name}")
            hand.append(node)
        mount = root.find(f"./joint[@name='{prefix}_twofinger_base_joint']")
        sides[side] = {
            "root_link": base,
            "source_mount_body": mount.find("parent").get("link"),
            "source_mount_xyz_m": list(map(float, mount.find("origin").get("xyz").split())),
            "source_mount_rpy_rad": list(map(float, mount.find("origin").get("rpy").split())),
            "camera_body": f"{prefix}_d405_camera",
            "link_names": sorted(links),
            "total_mass_kg": sum(estimates["links"][name]["mass_kg"] for name in links),
        }
        ET.indent(hand, space="  ")
        (output / "urdf").mkdir(exist_ok=True)
        ET.ElementTree(hand).write(output / "urdf" / f"leju_claw_{side}.urdf",
                                   encoding="utf-8", xml_declaration=True)
    def write_json(name, data):
        (output / name).write_text(json.dumps(data, indent=2) + "\n")
    write_json("inertial_estimates.json", {"description": estimates["description"], "links": selected_estimates})
    write_json("config.json", {
        "schema_version": 1, "name": "leju_claw_two_finger",
        "source_model": "biped_s200062", "source_urdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "mesh_sha256": hashes, "sides": sides,
        "action": {"open": 1.0, "closed": -1.0, "driver_joints": ["{side}_f_bar_1_joint", "{side}_b_bar_1_joint"]},
        "actuator": {"effort_limit_sim": 100.0, "stiffness": 4000.0, "damping": 400.0, "friction": 0.02},
        "contact": {"finger_static_friction": 20.0, "finger_dynamic_friction": 16.0,
                    "housing_static_friction": 1.0, "housing_dynamic_friction": 0.8,
                    "friction_combine_mode": "average", "contact_offset_m": 0.002, "rest_offset_m": 0.0},
    })
    print(f"Extracted two 14-link claw packages into {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor-dir", type=Path, default=ASSET_DIR / "kuavo_s200062")
    parser.add_argument("--output", type=Path, default=ASSET_DIR / "leju_claw_two_finger")
    args = parser.parse_args()
    extract(args.donor_dir, args.output)
