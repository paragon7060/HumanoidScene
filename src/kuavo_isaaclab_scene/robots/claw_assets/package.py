"""Self-contained claw paths and optional URDF composition, without Kit imports."""

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from ...core.paths import ASSET_DIR

CLAW_ASSET_DIR = ASSET_DIR / "leju_claw_two_finger"


@dataclass(frozen=True)
class ClawAsset:
    side: str
    prefix: str
    root_link: str
    urdf_path: Path
    usd_path: Path
    metadata: dict

    @property
    def motor_names(self) -> tuple[str, str]:
        return tuple(f"{self.prefix}_{jaw}_bar_1_joint" for jaw in "fb")

    def motor_positions(self, signed_action: float = 1.0) -> dict[str, float]:
        """+1=open, -1=closed; return targets for only the two driven hinges."""
        if not math.isfinite(signed_action):
            raise ValueError("Claw action must be finite")
        fraction = (1.0 + max(-1.0, min(1.0, signed_action))) / 2.0
        return dict(zip(self.motor_names, (-0.25 * fraction, 0.25 * fraction)))

    def initial_positions(self, signed_action: float = 1.0) -> dict[str, float]:
        from ..twofinger_linkage import initial_passive_positions
        commands = self.motor_positions(signed_action)
        return {**commands, **initial_passive_positions(commands)}


def load_claw_asset(side: str, asset_dir: Path = CLAW_ASSET_DIR) -> ClawAsset:
    if side not in ("left", "right"):
        raise ValueError("Claw side must be 'left' or 'right'")
    metadata = json.loads((asset_dir / "config.json").read_text())
    return ClawAsset(side, side[0], metadata["sides"][side]["root_link"],
                     asset_dir / "urdf" / f"leju_claw_{side}.urdf",
                     asset_dir / "usd" / side / f"leju_claw_{side}.usd", metadata)


def append_claw_branch(robot: ET.Element, *, side: str, parent_link: str,
                       output_urdf: Path, xyz: tuple[float, float, float],
                       rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
                       asset_dir: Path = CLAW_ASSET_DIR) -> str:
    """Append an independent package to a robot tree; never remove host visuals.

    Host mesh paths are the caller's responsibility. Output is not written here.
    Coordinates are metres/radians relative to parent_link, not world or EEF.
    Fixed links must NOT be merged when importing the resulting URDF into Isaac.
    """
    if (len(xyz) != 3 or len(rpy) != 3
            or not all(math.isfinite(value) for value in (*xyz, *rpy))):
        raise ValueError("Mount xyz/rpy must contain three finite values each")
    asset = load_claw_asset(side, asset_dir)
    branch = ET.parse(asset.urdf_path).getroot()
    names = {node.get("name") for node in robot if node.tag in ("link", "joint")}
    if robot.find(f"./link[@name='{parent_link}']") is None:
        raise ValueError(f"Missing host mount link: {parent_link}")
    attachment = f"{asset.prefix}_twofinger_base_joint"
    incoming = {node.get("name") for node in branch}
    if names & (incoming | {attachment}):
        raise ValueError("Claw branch already exists or has conflicting names")
    elements = [deepcopy(node) for node in branch]
    for node in elements:
        for mesh in node.findall(".//mesh"):
            source = (asset.urdf_path.parent / mesh.get("filename")).resolve()
            mesh.set("filename", os.path.relpath(source, output_urdf.resolve().parent))
    joint = ET.Element("joint", name=attachment, type="fixed")
    ET.SubElement(joint, "parent", link=parent_link)
    ET.SubElement(joint, "child", link=asset.root_link)
    ET.SubElement(joint, "origin", xyz=" ".join(map(str, xyz)), rpy=" ".join(map(str, rpy)))
    robot.extend([*elements, joint])
    return attachment
