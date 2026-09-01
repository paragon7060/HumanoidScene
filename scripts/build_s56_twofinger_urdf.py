#!/usr/bin/env python3
"""Build bare-wrist and S200062 two-finger/D405 S56 URDF variants.

The S56 and S200062 share the same arm wrist frame convention.  This script
removes the integrated QiangNao hand branches from the S56 source, replaces
the misleading ``*_hand_pitch_nohand.STL`` meshes (which actually contain a
complete five-finger hand) with the S200062 bare wrist links, and optionally
copies the complete S200062 two-finger and D405 branches onto the corresponding
``zarm_*7_link`` frames. Keeping these as generated variants avoids editing
either upstream-derived source model in place.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


PROJECT_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_DIR / "src" / "kuavo_isaaclab_scene" / "assets"
DEFAULT_S56 = ASSET_DIR / "kuavo_s56" / "urdf" / "kuavo_s56.urdf"
DEFAULT_DONOR = ASSET_DIR / "kuavo_s200062" / "urdf" / "biped_s200062.urdf"
DEFAULT_BARE_OUTPUT = ASSET_DIR / "kuavo_s56" / "urdf" / "kuavo_s56_bare.urdf"
DEFAULT_OUTPUT = ASSET_DIR / "kuavo_s56" / "urdf" / "kuavo_s56_twofinger.urdf"


def _branch_names(root: ET.Element, branch_roots: set[str]) -> tuple[set[str], set[str]]:
    """Return every link and joint below the named branch roots."""
    links = set(branch_roots)
    joints: set[str] = set()
    pending = list(root.findall("joint"))
    changed = True
    while changed:
        changed = False
        for joint in pending[:]:
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None or parent.get("link") not in links:
                continue
            joints.add(joint.get("name", ""))
            links.add(child.get("link", ""))
            pending.remove(joint)
            changed = True
    return links, joints


def _elements_by_name(root: ET.Element, tag: str) -> dict[str, ET.Element]:
    return {element.get("name", ""): element for element in root.findall(tag)}


def _rewrite_donor_mesh_paths(element: ET.Element) -> None:
    for mesh in element.findall(".//mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith("../meshes/"):
            mesh.set(
                "filename",
                f"../../kuavo_s200062/meshes/{filename.removeprefix('../meshes/')}",
            )


def _prepare_bare_s56(base: ET.Element, donor: ET.Element) -> None:
    """Remove every dexterous-hand geometry and install donor bare wrists."""

    qiangnao_links, qiangnao_joints = _branch_names(base, {"l_palm", "r_palm"})
    qiangnao_joints.update({"l_palm_fixed_joint", "r_palm_fixed_joint"})
    for element in list(base):
        if element.tag == "link" and element.get("name") in qiangnao_links:
            base.remove(element)
        elif element.tag == "joint" and element.get("name") in qiangnao_joints:
            base.remove(element)

    donor_link_elements = _elements_by_name(donor, "link")
    for side in "lr":
        name = f"zarm_{side}7_link"
        current = base.find(f"./link[@name='{name}']")
        replacement = donor_link_elements.get(name)
        if current is None or replacement is None:
            raise RuntimeError(f"Cannot construct bare S56 wrist {name!r}")
        replacement = deepcopy(replacement)
        _rewrite_donor_mesh_paths(replacement)
        index = list(base).index(current)
        base.remove(current)
        base.insert(index, replacement)


def _write_tree(tree: ET.ElementTree, output_path: Path, robot_name: str) -> None:
    tree.getroot().set("name", robot_name)
    ET.indent(tree, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def build_s56_bare_urdf(s56_path: Path, donor_path: Path, output_path: Path) -> None:
    tree = ET.parse(s56_path)
    root = tree.getroot()
    donor = ET.parse(donor_path).getroot()
    _prepare_bare_s56(root, donor)
    _write_tree(tree, output_path, "kuavo_s56_bare")


def build_s56_twofinger_urdf(s56_path: Path, donor_path: Path, output_path: Path) -> None:
    base_tree = ET.parse(s56_path)
    base = base_tree.getroot()
    donor = ET.parse(donor_path).getroot()
    _prepare_bare_s56(base, donor)

    donor_links, donor_joints = _branch_names(
        donor, {"l_twofinger_base", "r_twofinger_base"}
    )
    donor_joints.update({"l_twofinger_base_joint", "r_twofinger_base_joint"})
    # The tool reference links are siblings of the hand branches.  S56 already
    # contains the primary reference; copy the two auxiliary references used
    # by the S200062 inertial/contact corrections as well.
    donor_links.update({"zarm_l7_end_effector_1", "zarm_l7_end_effector_2",
                        "zarm_r7_end_effector_1", "zarm_r7_end_effector_2"})
    donor_joints.update({"zarm_l7_end_effector_1_joint", "zarm_l7_end_effector_2_joint",
                         "zarm_r7_end_effector_1_joint", "zarm_r7_end_effector_2_joint"})

    donor_link_elements = _elements_by_name(donor, "link")
    donor_joint_elements = _elements_by_name(donor, "joint")
    for name in sorted(donor_links):
        if name not in donor_link_elements:
            raise RuntimeError(f"S200062 donor is missing link {name!r}")
        element = deepcopy(donor_link_elements[name])
        _rewrite_donor_mesh_paths(element)
        base.append(element)
    for name in sorted(donor_joints):
        if name not in donor_joint_elements:
            raise RuntimeError(f"S200062 donor is missing joint {name!r}")
        base.append(deepcopy(donor_joint_elements[name]))

    _write_tree(base_tree, output_path, "kuavo_s56_twofinger")

    check = ET.parse(output_path).getroot()
    link_names = {link.get("name") for link in check.findall("link")}
    joint_names = {joint.get("name") for joint in check.findall("joint")}
    required_links = {"l_twofinger_base", "r_twofinger_base", "l_d405_camera", "r_d405_camera"}
    required_joints = {
        f"{side}_{jaw}_bar_{index}_joint"
        for side in "lr" for jaw in "fb" for index in (1, 3)
    }
    dexterous_tokens = ("palm", "thumb", "index", "middle", "ring", "little")
    forbidden_names = {
        name
        for name in (*link_names, *joint_names)
        if name and any(token in name.lower() for token in dexterous_tokens)
    }
    if not required_links <= link_names or not required_joints <= joint_names:
        raise RuntimeError("Generated S56 two-finger URDF is incomplete")
    if forbidden_names:
        raise RuntimeError(
            "Generated S56 two-finger URDF still contains QiangNao names: "
            + ", ".join(sorted(forbidden_names))
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s56", type=Path, default=DEFAULT_S56)
    parser.add_argument("--donor", type=Path, default=DEFAULT_DONOR)
    parser.add_argument("--bare-output", type=Path, default=DEFAULT_BARE_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_s56_bare_urdf(
        args.s56.expanduser().resolve(),
        args.donor.expanduser().resolve(),
        args.bare_output.expanduser().resolve(),
    )
    build_s56_twofinger_urdf(
        args.s56.expanduser().resolve(),
        args.donor.expanduser().resolve(),
        args.output.expanduser().resolve(),
    )
    print(f"Generated {args.bare_output} and {args.output}")


if __name__ == "__main__":
    main()
