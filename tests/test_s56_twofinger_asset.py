from pathlib import Path
import xml.etree.ElementTree as ET

from kuavo_isaaclab_scene.core.paths import ASSET_DIR


def test_generated_s56_twofinger_asset_replaces_qiangnao_without_overlap() -> None:
    path = ASSET_DIR / "kuavo_s56/urdf/kuavo_s56_twofinger.urdf"
    root = ET.parse(path).getroot()
    links = {link.get("name") for link in root.findall("link")}
    joints = {joint.get("name") for joint in root.findall("joint")}

    dexterous_tokens = ("palm", "thumb", "index", "middle", "ring", "little")
    assert not {
        name
        for name in (*links, *joints)
        if name and any(token in name.lower() for token in dexterous_tokens)
    }
    assert {name for name in links if name.endswith("_finger")} == {
        "l_f_finger",
        "l_b_finger",
        "r_f_finger",
        "r_b_finger",
    }
    assert {"l_twofinger_base", "r_twofinger_base", "l_d405_camera", "r_d405_camera"} <= links
    assert {
        f"{side}_{jaw}_bar_{index}_joint"
        for side in "lr" for jaw in "fb" for index in (1, 3)
    } <= joints
    for side in "lr":
        wrist = root.find(f"./link[@name='zarm_{side}7_link']")
        meshes = [mesh.get("filename", "") for mesh in wrist.findall(".//visual/geometry/mesh")]
        assert meshes == [f"../../kuavo_s200062/meshes/{side}_hand_pitch.STL"]
        assert not any("nohand" in mesh.lower() for mesh in meshes)


def test_generated_s56_bare_asset_has_no_hand_or_gripper_geometry() -> None:
    path = ASSET_DIR / "kuavo_s56/urdf/kuavo_s56_bare.urdf"
    root = ET.parse(path).getroot()
    forbidden = ("palm", "thumb", "index", "middle", "ring", "little", "finger", "bar")
    names = [element.get("name", "") for tag in ("link", "joint") for element in root.findall(tag)]
    assert not [name for name in names if any(token in name.lower() for token in forbidden)]
    for side in "lr":
        wrist = root.find(f"./link[@name='zarm_{side}7_link']")
        meshes = [mesh.get("filename", "") for mesh in wrist.findall(".//visual/geometry/mesh")]
        assert meshes == [f"../../kuavo_s200062/meshes/{side}_hand_pitch.STL"]


def test_transplanted_mesh_references_resolve_to_s200062_assets() -> None:
    path = ASSET_DIR / "kuavo_s56/urdf/kuavo_s56_twofinger.urdf"
    root = ET.parse(path).getroot()
    transplanted_links = {
        link.get("name"): link
        for link in root.findall("link")
        if "twofinger" in link.get("name", "")
        or "d405" in link.get("name", "")
        or "_bar_" in link.get("name", "")
        or "_finger" in link.get("name", "")
    }
    assert transplanted_links
    for link in transplanted_links.values():
        for mesh in link.findall(".//mesh"):
            mesh_path = (path.parent / mesh.get("filename")).resolve()
            assert mesh_path.is_file(), mesh_path
            assert "kuavo_s200062/meshes" in str(mesh_path)
