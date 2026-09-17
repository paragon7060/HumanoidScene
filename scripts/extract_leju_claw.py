#!/usr/bin/env python3
"""Generate self-contained left/right claw URDFs, meshes and metadata.

This is an asset extraction/build step, not a simulation or a host-model edit.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kuavo_isaaclab_scene.core.paths import ASSET_DIR


def extract(donor_dir: Path, output: Path, *, mass_kg: float = 1.0) -> None:
    if not math.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("Claw package mass must be positive and finite")
    if output.resolve() == donor_dir.resolve() or donor_dir.resolve() in output.resolve().parents:
        raise ValueError("Extraction must not overwrite the donor model")
    source = donor_dir / "urdf/biped_s200062.urdf"
    previous_config_path = output / "config.json"
    previous_config = (json.loads(previous_config_path.read_text())
                       if previous_config_path.is_file() else {})
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
        source_mass = sum(estimates["links"][name]["mass_kg"] for name in links)
        mass_scale = mass_kg / source_mass
        hand = ET.Element("robot", name=f"leju_claw_{side}")
        for node in root:
            if not ((node.tag == "link" and node.get("name") in links)
                    or (node.tag == "joint" and node.get("name") in joint_names)):
                continue
            node = deepcopy(node)
            if node.tag == "link":
                name = node.get("name")
                values = deepcopy(estimates["links"][name])
                values["mass_kg"] *= mass_scale
                values["diagonal_inertia_kg_m2"] = [
                    value * mass_scale for value in values["diagonal_inertia_kg_m2"]]
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
            "total_mass_kg": mass_kg,
        }
        ET.indent(hand, space="  ")
        (output / "urdf").mkdir(exist_ok=True)
        ET.ElementTree(hand).write(output / "urdf" / f"leju_claw_{side}.urdf",
                                   encoding="utf-8", xml_declaration=True)
    def write_json(name, data):
        (output / name).write_text(json.dumps(data, indent=2) + "\n")
    write_json("inertial_estimates.json", {
        "description": f"Simulation estimates, not manufacturer calibration. Each claw package including "
                       f"D405 is {mass_kg:g} kg. Donor link mass ratios and CoMs are retained; "
                       "mass and inertia are scaled together. Host arm/torso inertials are unchanged.",
        "links": selected_estimates})
    action = previous_config.get("action", {
        "open": 1.0,
        "closed": -1.0,
        "driver_joints": ["{side}_f_bar_1_joint", "{side}_b_bar_1_joint"],
    })
    actuator = previous_config.get("actuator", {
        "effort_limit_sim": 100.0,
        "stiffness": 4000.0,
        "damping": 400.0,
        "friction": 0.02,
    })
    contact = previous_config.get("contact", {
        "finger_static_friction": 20.0,
        "finger_dynamic_friction": 16.0,
        "housing_static_friction": 1.0,
        "housing_dynamic_friction": 0.8,
        "friction_combine_mode": "average",
        "contact_offset_m": 0.002,
        "rest_offset_m": 0.0,
        "distal_pad": {
            "description": "Thin compliant fingertip pad covering the distal 20 mm grasping plane.",
            "size_m": [0.002, 0.018, 0.020],
            "center_m": {
                "f": [-0.031361, 0.0, -0.059024],
                "b": [0.031361, 0.0, -0.059024],
            },
            "compliance": {
                "enabled": True,
                "youngs_modulus_pa": 280000.0,
                "stiffness_n_per_m": 50400.0,
                "damping_n_s_per_m": 250.0,
            },
        },
    })
    runtime_defaults = previous_config.get("runtime_defaults", {
        "enabled": True,
        "integrated": True,
        "usd_path": "integrated://robot",
        "attachment_mount_body": "robot",
        "joint_names": ["{side}_f_bar_1_joint", "{side}_b_bar_1_joint"],
        "default_joint_pos": {"{side}_f_bar_1_joint": -0.25, "{side}_b_bar_1_joint": 0.25},
        "open_command": {"{side}_f_bar_1_joint": -0.25, "{side}_b_bar_1_joint": 0.25},
        "close_command": {"{side}_f_bar_1_joint": 0.0, "{side}_b_bar_1_joint": 0.0},
        "pinch_close_threshold_m": 0.055,
    })
    basic_sides = {side: {"enabled": True, "robot_mount_body": values["root_link"],
                           "robot_mount_pos": [0.0, 0.0, 0.0],
                           "robot_mount_rot": [1.0, 0.0, 0.0, 0.0]}
                   for side, values in sides.items()}
    runtime_presets = previous_config.get("runtime_presets", {
        name: {"sides": deepcopy(basic_sides)}
        for name in ("leju-twofinger", "s200062_integrated", "s56_twofinger")
    })
    write_json("config.json", {
        "schema_version": 4, "name": "leju_claw_two_finger",
        "source_model": "biped_s200062", "source_urdf_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "mesh_sha256": hashes, "sides": sides,
        "action": action,
        "actuator": actuator,
        "contact": contact,
        # Host selection stays in grippers.json; all physical settings and
        # host-specific two-finger calibration live with this package.
        "runtime_defaults": runtime_defaults,
        "runtime_presets": runtime_presets,
        "force_control": previous_config.get("force_control", {
            "close_force_n": 50.0,
            "max_velocity_rad_s": 0.5,
            "force_filter_time_constant_s": 0.02,
            "force_ramp_time_s": 0.3,
            "proportional_gain": 0.25,
            "integral_gain_per_s": 4.0,
            "integral_limit_multiplier": 4.0,
            "max_force_multiplier": 5.0,
        }),
    })
    print(f"Extracted two 14-link claw packages into {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor-dir", type=Path, default=ASSET_DIR / "kuavo_s200062")
    parser.add_argument("--output", type=Path, default=ASSET_DIR / "leju_claw_two_finger")
    parser.add_argument("--mass-kg", type=float, default=1.0,
                        help="Total mass per claw including camera; default 1 kg.")
    args = parser.parse_args()
    extract(args.donor_dir, args.output, mass_kg=args.mass_kg)
