#!/usr/bin/env python3
"""Build the minimal 14-DoF Kuavo5W arm model used by plantIK.

The simulator keeps the complete Kuavo5W URDF.  plantIK's distributed binary
is an arm solver, so it is more reliable to derive an arm-only model from the
same source URDF than to duplicate a second hand-written kinematic model.
Visual and collision elements are omitted; inertial data and all joint origins
are retained for Drake's kinematics and CoM term.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


ARM_LINKS = {
    "base_link",
    "waist_yaw_link",
    *(f"zarm_{side}{index}_link" for side in ("l", "r") for index in range(1, 8)),
    "zarm_l7_end_effector",
    "zarm_r7_end_effector",
}
ARM_JOINTS = {
    *(f"zarm_{side}{index}_joint" for side in ("l", "r") for index in range(1, 8)),
    "zarm_l7_end_effector_joint",
    "zarm_r7_end_effector_joint",
}


def _without_geometry(link: ET.Element) -> ET.Element:
    result = deepcopy(link)
    for tag in ("visual", "collision"):
        for element in result.findall(tag):
            result.remove(element)
    return result


def make_model(source: Path, output: Path) -> None:
    source_root = ET.parse(source).getroot()
    output_root = ET.Element("robot", {"name": "kuavo5w_arm"})

    for link in source_root.findall("link"):
        if link.get("name") in ARM_LINKS:
            output_root.append(_without_geometry(link))

    # The complete robot has a movable waist-yaw joint.  The arm solver keeps
    # the base posture fixed, so retain its exact origin as a fixed adapter.
    waist = source_root.find("joint[@name='waist_yaw_joint']")
    if waist is None:
        raise ValueError("source URDF has no waist_yaw_joint")
    waist_fixed = ET.Element("joint", {"name": "waist_yaw_fixed", "type": "fixed"})
    for child in waist:
        if child.tag in {"origin", "parent", "child"}:
            waist_fixed.append(deepcopy(child))
    output_root.append(waist_fixed)

    for joint in source_root.findall("joint"):
        if joint.get("name") in ARM_JOINTS:
            output_root.append(deepcopy(joint))

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(output_root, space="  ")
    ET.ElementTree(output_root).write(output, encoding="utf-8", xml_declaration=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_model(args.source, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
