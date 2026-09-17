"""Static extraction/composition checks; no simulator or image rendering."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from kuavo_isaaclab_scene.core.paths import ASSET_DIR
from kuavo_isaaclab_scene.robots.claw_assets import (
    CLAW_ASSET_DIR,
    append_claw_branch,
    load_claw_asset,
    load_claw_config,
)
from kuavo_isaaclab_scene.robots.gripper_config import load_gripper_settings


TWO_FINGER_PRESETS = ("leju-twofinger", "s200062_integrated", "s56_twofinger")


def test_twofinger_registry_entries_are_package_aliases():
    expected = "${KUAVO_PACKAGE_ASSET_DIR}/leju_claw_two_finger/config.json"
    for registry in (
        Path(__file__).resolve().parents[1] / "configs/grippers.json",
        Path(__file__).resolve().parents[1] / "src/kuavo_isaaclab_scene/configs/grippers.json",
    ):
        presets = json.loads(registry.read_text())["presets"]
        for name in TWO_FINGER_PRESETS:
            assert presets[name] == {"package_config": expected, "package_preset": name}


def test_twofinger_presets_share_package_physics_and_keep_host_calibration():
    package = load_claw_config()
    assert package["schema_version"] == 3
    assert set(package["runtime_presets"]) == set(TWO_FINGER_PRESETS)
    assert package["force_control"]["close_force_n"] == 50.0
    for name in TWO_FINGER_PRESETS:
        settings = load_gripper_settings(name)
        assert settings.package_config_path == CLAW_ASSET_DIR / "config.json"
        assert settings.actuator.stiffness == package["actuator"]["stiffness"]
        assert settings.actuator.damping == package["actuator"]["damping"]
        assert settings.finger_contact.static_friction == package["contact"]["finger_static_friction"]
        assert settings.finger_contact.dynamic_friction == package["contact"]["finger_dynamic_friction"]


@pytest.mark.parametrize("side", ["left", "right"])
def test_independent_claw_is_self_contained(side):
    asset = load_claw_asset(side)
    root = ET.parse(asset.urdf_path).getroot()
    assert len(root.findall("link")) == 14
    assert len(root.findall("joint")) == 13
    children = {joint.find("child").get("link") for joint in root.findall("joint")}
    assert {link.get("name") for link in root.findall("link")} - children == {asset.root_link}
    assert all(link.find("inertial") is not None for link in root.findall("link"))
    assert sum(float(link.find("inertial/mass").get("value")) for link in root.findall("link")) == pytest.approx(1.)
    assert asset.metadata["sides"][side]["total_mass_kg"] == 1.
    assert len(root.findall(".//collision")) == 3
    assert not any("zarm" in link.get("name") for link in root.findall("link"))
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        path = (asset.urdf_path.parent / filename).resolve()
        assert filename.startswith("../meshes/") and path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == asset.metadata["mesh_sha256"][path.name]
    assert asset.usd_path.is_file()
    for path in asset.usd_path.parent.rglob("*.usd"):
        payload = path.read_bytes()
        assert b"kuavo_s200062" not in payload
        assert b"/home/" not in payload


@pytest.mark.parametrize("side", ["left", "right"])
def test_claw_mass_and_inertia_match_host_and_preserve_donor_distribution(side):
    asset = load_claw_asset(side)
    donor = json.loads((ASSET_DIR / "kuavo_s200062/teleop_inertials.json").read_text())["links"]
    estimates = json.loads((ASSET_DIR / "leju_claw_two_finger/inertial_estimates.json").read_text())["links"]
    host = ET.parse(ASSET_DIR / "kuavo_s63_twofinger/urdf/kuavo_s63_twofinger.urdf").getroot()
    names = asset.metadata["sides"][side]["link_names"]
    scale = 1. / sum(donor[name]["mass_kg"] for name in names)
    for name in names:
        values = estimates[name]
        assert values["mass_kg"] == pytest.approx(donor[name]["mass_kg"] * scale)
        assert values["diagonal_inertia_kg_m2"] == pytest.approx(
            [value * scale for value in donor[name]["diagonal_inertia_kg_m2"]])
        assert values["com_m"] == donor[name]["com_m"]
        assert float(host.find(f"./link[@name='{name}']/inertial/mass").get("value")) == pytest.approx(values["mass_kg"])


def test_claw_action_and_passive_reset_contract():
    claw = load_claw_asset("right")
    assert tuple(claw.motor_positions().values()) == (-.25, .25)
    assert tuple(claw.motor_positions(-1).values()) == (0, 0)
    assert tuple(claw.motor_positions(0).values()) == (-.125, .125)
    assert len(claw.initial_positions()) == 6
    assert claw.initial_positions()["r_f_bar_4_joint"] == pytest.approx(-.417458492389)
    with pytest.raises(ValueError):
        claw.motor_positions(float("nan"))
    with pytest.raises(ValueError):
        load_claw_asset("l")


def test_claw_has_flat_distal_contact_pads():
    config = load_claw_asset("right").metadata
    assert config["schema_version"] == 3
    pad = config["contact"]["distal_pad"]
    assert pad["size_m"] == [0.002, 0.018, 0.020]
    assert pad["center_m"]["f"] == [-0.031361, 0.0, -0.059024]
    assert pad["center_m"]["b"] == [0.031361, 0.0, -0.059024]
    for jaw in "fb":
        center, size = pad["center_m"][jaw], pad["size_m"]
        assert center[2] - size[2] / 2 == pytest.approx(-0.069024)
        assert center[2] + size[2] / 2 == pytest.approx(-0.049024)


def test_s63_composition_is_explicit_and_preserves_host(tmp_path):
    root = ET.parse(ASSET_DIR / "kuavo_s63/urdf/kuavo_s63.urdf").getroot()
    originals = [ET.tostring(node) for node in root]
    joint_name = append_claw_branch(root, side="left", parent_link="zarm_l7_link",
                                   output_urdf=tmp_path / "s63_claw.urdf", xyz=(0, -.0005, -.041))
    assert [ET.tostring(node) for node in list(root)[:len(originals)]] == originals
    joint = root.find(f"./joint[@name='{joint_name}']")
    assert joint.find("child").get("link") == "l_twofinger_base"
    assert joint.find("origin").get("xyz") == "0 -0.0005 -0.041"
    with pytest.raises(ValueError, match="conflicting"):
        append_claw_branch(root, side="left", parent_link="zarm_l7_link",
                           output_urdf=tmp_path / "s63_claw.urdf", xyz=(0, 0, 0))
    with pytest.raises(ValueError, match="Missing host"):
        append_claw_branch(root, side="right", parent_link="missing",
                           output_urdf=tmp_path / "s63_claw.urdf", xyz=(0, 0, 0))
