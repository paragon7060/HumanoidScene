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
from kuavo_isaaclab_scene.robots.claw_assets.usd import (
    author_claw_distal_pads, author_claw_jaw_contact, pad_compliance, resolve_finger_contact)
from kuavo_isaaclab_scene.robots.gripper_config import FingerContactSettings
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
    assert package["schema_version"] == 4
    assert set(package["runtime_presets"]) == set(TWO_FINGER_PRESETS)
    assert package["force_control"]["close_force_n"] == 50.0
    assert package["action"]["binary_open"] == 0.0
    assert package["action"]["binary_closed"] == 1.0
    assert package["action"]["signed_target_open"] == 1.0
    assert package["action"]["signed_target_closed"] == -1.0
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
    assert config["schema_version"] == 4
    pad = config["contact"]["distal_pad"]
    assert pad["size_m"] == [0.002, 0.018, 0.020]
    assert pad["center_m"]["f"] == [-0.031361, 0.0, -0.059024]
    assert pad["center_m"]["b"] == [0.031361, 0.0, -0.059024]
    for jaw in "fb":
        center, size = pad["center_m"][jaw], pad["size_m"]
        assert center[2] - size[2] / 2 == pytest.approx(-0.069024)
        assert center[2] + size[2] / 2 == pytest.approx(-0.049024)


def test_distal_pads_are_soft_and_stay_within_their_travel():
    contact = load_claw_config()["contact"]
    pad = contact["distal_pad"]
    thickness, width, length = pad["size_m"]
    stiffness, damping = pad_compliance(contact)
    # The spring is the pad's own compression stiffness, so a softer or thicker
    # pad cannot be configured without restating its modulus.
    modulus = pad["compliance"]["youngs_modulus_pa"]
    assert stiffness == pytest.approx(modulus * width * length / thickness)
    assert damping > 0
    # PhysX applies the spring per contact point, and a flat pad on a flat flap
    # generates a four-point patch. At the package close force that patch has to
    # deform visibly yet stay ahead of the finger hull waiting 0.6 mm behind it.
    squeeze_m = load_claw_config()["force_control"]["close_force_n"] / (4 * stiffness)
    assert 0.0001 < squeeze_m < 0.0006


@pytest.mark.parametrize("field, value", [
    ("stiffness_n_per_m", 0.0), ("damping_n_s_per_m", -1.0), ("youngs_modulus_pa", float("nan")),
])
def test_soft_pad_rejects_unusable_compliance(field, value):
    contact = json.loads(json.dumps(load_claw_config()["contact"]))
    contact["distal_pad"]["compliance"][field] = value
    with pytest.raises(ValueError):
        pad_compliance(contact)


def test_soft_pad_stiffness_must_match_its_geometry():
    contact = json.loads(json.dumps(load_claw_config()["contact"]))
    contact["distal_pad"]["size_m"][0] *= 2
    with pytest.raises(ValueError, match="E\\*A/t"):
        pad_compliance(contact)


def _stage_with_jaw_and_wrist():
    """Minimal stand-in for a donor USD: a jaw mesh and a wrist primitive collider."""
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    for name in ("l_f_finger", "zarm_l7_link"):
        UsdGeom.Xform.Define(stage, f"/Robot/{name}")
        UsdGeom.Mesh.Define(stage, f"/Robot/{name}/visuals/mesh")
    cylinder = UsdGeom.Cylinder.Define(stage, "/Robot/zarm_l7_link/cylinder").GetPrim()
    UsdPhysics.CollisionAPI.Apply(cylinder).CreateCollisionEnabledAttr(True)
    return stage, root


def test_jaw_contact_replaces_primitive_wrist_colliders_with_the_mesh_hull():
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    UsdShade = pytest.importorskip("pxr.UsdShade")
    stage, root = _stage_with_jaw_and_wrist()
    counts = author_claw_jaw_contact(
        root, load_claw_config()["contact"], links={"l_f_finger", "zarm_l7_link"},
        finger_links={"l_f_finger"}, mesh_scope="visuals", replace_colliders={"zarm_l7_link"})
    assert counts == {"l_f_finger": 1, "zarm_l7_link": 1}
    cylinder = stage.GetPrimAtPath("/Robot/zarm_l7_link/cylinder")
    assert UsdPhysics.CollisionAPI(cylinder).GetCollisionEnabledAttr().Get() is False
    bindings = {}
    for name in ("l_f_finger", "zarm_l7_link"):
        mesh = stage.GetPrimAtPath(f"/Robot/{name}/visuals/mesh")
        assert UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get() is True
        assert UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == "convexHull"
        bindings[name] = UsdShade.MaterialBindingAPI(mesh).GetDirectBindingRel(
            "physics").GetTargets()[0].name
    assert bindings == {"l_f_finger": "FingerContactMaterial",
                        "zarm_l7_link": "HandContactMaterial"}


def test_jaw_contact_reports_links_the_host_usd_does_not_provide():
    _, root = _stage_with_jaw_and_wrist()
    with pytest.raises(RuntimeError, match="collisions"):
        author_claw_jaw_contact(root, load_claw_config()["contact"],
                                links={"l_f_finger"}, finger_links={"l_f_finger"},
                                mesh_scope="collisions")


def test_rigid_pad_mode_is_selectable_without_editing_the_package():
    contact = resolve_finger_contact(load_claw_config(), FingerContactSettings(soft_pad=False))
    assert pad_compliance(contact) is None
    assert pad_compliance(resolve_finger_contact(load_claw_config())) == (50400.0, 250.0)
    # Selecting the rigid pad must not silently alter the pad's geometry.
    assert contact["distal_pad"]["size_m"] == load_claw_config()["contact"]["distal_pad"]["size_m"]


def _stage_with_one_jaw_pair():
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    for jaw in "fb":
        UsdGeom.Xform.Define(stage, f"/Robot/l_{jaw}_finger")
        UsdGeom.Mesh.Define(stage, f"/Robot/l_{jaw}_finger/visuals/mesh")
    return stage, root


@pytest.mark.parametrize("soft_pad, material, visibility", [
    (True, "SoftPadContactMaterial", "inherited"),
    (False, "FingerContactMaterial", "invisible"),
])
def test_both_pad_models_author_the_same_slab(soft_pad, material, visibility):
    UsdShade = pytest.importorskip("pxr.UsdShade")
    stage, root = _stage_with_one_jaw_pair()
    contact = resolve_finger_contact(load_claw_config(), FingerContactSettings(soft_pad=soft_pad))
    jaws = {"l_f_finger", "l_b_finger"}
    author_claw_jaw_contact(root, contact, links=jaws, finger_links=jaws, mesh_scope="visuals")
    count, compliance = author_claw_distal_pads(root, contact, sides="l")
    assert count == 2
    assert (compliance is None) is not soft_pad
    pad = stage.GetPrimAtPath("/Robot/l_f_finger/distal_contact_pad")
    assert pad.GetAttribute("visibility").Get() == visibility
    assert UsdShade.MaterialBindingAPI(pad).GetDirectBindingRel(
        "physics").GetTargets()[0].name == material
    spring = pad.GetCustomData().get("kuavo", {}).get("distalPadStiffnessNPerM")
    assert (spring is None) is not soft_pad


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
