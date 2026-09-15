"""S63 selection and host-preserving claw composition; no simulation startup."""
import xml.etree.ElementTree as ET

import pytest

from kuavo_isaaclab_scene.core.paths import ASSET_DIR
from kuavo_isaaclab_scene.robots.gripper_config import resolve_gripper_settings
from kuavo_isaaclab_scene.robots.robot_model import resolve_robot_model, validate_robot_gripper


def test_s63_gripper_cli_environment_selects_integrated_variant(monkeypatch):
    monkeypatch.setenv("KUAVO_ROBOT_MODEL", "s63")
    monkeypatch.setenv("KUAVO_GRIPPER", "leju-twofinger")
    model = resolve_robot_model()
    hand = resolve_gripper_settings()
    assert model.integrated_gripper_preset == hand.name == "leju-twofinger"
    assert hand.integrated and hand.asset_name_for("left") == "robot"
    assert model.usd_path.endswith("kuavo_s63_twofinger/usd/kuavo_s63_twofinger_fixed.usd")
    assert model.wrist_camera_bodies == {"left": "l_d405_camera", "right": "r_d405_camera"}
    assert hand.joint_names_for("left") == ("l_f_bar_1_joint", "l_b_bar_1_joint")
    validate_robot_gripper(model, hand.name)
    plain = resolve_robot_model("s63", "none")
    assert not plain.has_integrated_grippers
    assert plain.usd_path.endswith("kuavo_s63/usd/kuavo_s63_fixed.usd")


def test_explicit_s63_selection_and_other_models_reject_preset():
    assert resolve_robot_model("s63", "leju-twofinger").has_integrated_grippers
    with pytest.raises(ValueError):
        validate_robot_gripper(resolve_robot_model("s200062"), "leju-twofinger")


def test_s63_variant_preserves_official_host_geometry_frames_and_inertia():
    original_path = ASSET_DIR / "kuavo_s63/urdf/biped_s63.urdf"
    variant_path = ASSET_DIR / "kuavo_s63_twofinger/urdf/kuavo_s63_twofinger.urdf"
    original, variant = ET.parse(original_path).getroot(), ET.parse(variant_path).getroot()
    # Normalize only mesh package/relative path representations. All original
    # numeric data, collision geometry and visual material definitions stay exact.
    for node in original:
        if node.tag not in ("link", "joint"):
            continue
        replacement = variant.find(f"./{node.tag}[@name='{node.get('name')}']")
        assert replacement is not None
        for mesh in node.findall(".//mesh") + replacement.findall(".//mesh"):
            mesh.set("filename", mesh.get("filename").split("/")[-1])
        def structure(element):
            return element.tag, element.attrib, [structure(child) for child in element]
        assert structure(node) == structure(replacement), node.get("name")
    assert len(variant.findall("link")) == len(original.findall("link")) + 28
    assert len(variant.findall("joint")) == len(original.findall("joint")) + 28
    for side in ("l", "r"):
        mount = variant.find(f"./joint[@name='{side}_twofinger_base_joint']")
        assert mount.find("parent").get("link") == f"zarm_{side}7_link"
    for mesh in ET.parse(variant_path).getroot().findall(".//mesh"):
        assert (variant_path.parent / mesh.get("filename")).resolve().is_file()
