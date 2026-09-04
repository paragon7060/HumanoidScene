"""Checks that required deployment resources are present in the package."""

from pathlib import Path
import xml.etree.ElementTree as ET

from kuavo_isaaclab_scene.core.paths import ASSET_DIR, CONFIG_DIR, PACKAGE_CONFIG_DIR


def test_required_assets_are_packaged() -> None:
    required = (
        ASSET_DIR / "Rack.usd",
        ASSET_DIR / "SmallBox.usd",
        ASSET_DIR / "MediumBox.usd",
        ASSET_DIR / "LargeBox.usd",
        ASSET_DIR / "XLargeBox.usd",
        ASSET_DIR / "SmallBox_physical.usda",
        ASSET_DIR / "MediumBox_physical.usda",
        ASSET_DIR / "LargeBox_physical.usda",
        ASSET_DIR / "XLargeBox_physical.usda",
        ASSET_DIR / "SmallBox_atlas.usda",
        ASSET_DIR / "MediumBox_atlas.usda",
        ASSET_DIR / "LargeBox_atlas.usda",
        ASSET_DIR / "XLargeBox_atlas.usda",
        ASSET_DIR / "textures" / "boxes" / "bottom.png",
        ASSET_DIR / "textures" / "boxes" / "small.png",
        ASSET_DIR / "textures" / "boxes" / "medium.png",
        ASSET_DIR / "textures" / "boxes" / "large.png",
        ASSET_DIR / "textures" / "boxes" / "xlarge.png",
        ASSET_DIR / "textures" / "boxes" / "generated" / "small_atlas.png",
        ASSET_DIR / "textures" / "boxes" / "generated" / "medium_atlas.png",
        ASSET_DIR / "textures" / "boxes" / "generated" / "large_atlas.png",
        ASSET_DIR / "textures" / "boxes" / "generated" / "xlarge_atlas.png",
        ASSET_DIR / "button_station.usda",
        ASSET_DIR / "kuavo_s56" / "usd" / "kuavo_s56_fixed.usd",
        ASSET_DIR / "kuavo_s56_bare" / "usd" / "kuavo_s56_bare_fixed.usd",
        ASSET_DIR / "kuavo_s56_twofinger" / "usd" / "kuavo_s56_twofinger_fixed.usd",
        ASSET_DIR / "kuavo_s63" / "usd" / "kuavo_s63_fixed.usd",
        ASSET_DIR / "kuavo_s200062" / "usd" / "kuavo_s200062_fixed.usd",
        ASSET_DIR / "robotiq_2f85" / "usd" / "robotiq_2f85.usd",
    )
    assert all(path.is_file() for path in required)
    assert not (ASSET_DIR / "s200049_gripper").exists()
    assert not (ASSET_DIR / "kuavo_s62").exists()


def test_packaged_robot_and_gripper_assets_have_no_remote_references() -> None:
    paths = [
        *(ASSET_DIR / "kuavo_s56").rglob("*"),
        *(ASSET_DIR / "kuavo_s56_bare").rglob("*"),
        *(ASSET_DIR / "kuavo_s56_twofinger").rglob("*"),
        *(ASSET_DIR / "kuavo_s63").rglob("*"),
        *(ASSET_DIR / "robotiq_2f85").rglob("*"),
    ]
    forbidden = (b"http://", b"https://", b"omniverse://")
    for path in paths:
        if path.is_file():
            payload = path.read_bytes().lower()
            assert not any(token in payload for token in forbidden), path


def test_s63_visual_colors_match_the_requested_s62_material_profile() -> None:
    """The official S63 already carries the requested S62 white/gray profile."""
    urdf = ASSET_DIR / "kuavo_s63" / "urdf" / "kuavo_s63.urdf"
    root = ET.parse(urdf).getroot()
    colors = [color.attrib["rgba"] for color in root.findall(".//visual/material/color")]
    assert colors.count("1 1 1 1") == 28
    assert colors.count(
        "0.611764705882353 0.658823529411765 0.670588235294118 1"
    ) == 1
    radar = root.find("./link[@name='radar']/visual/material/color")
    assert radar is not None
    assert radar.attrib["rgba"] == colors[-1]


def test_s200062_runtime_urdf_is_complete_and_local() -> None:
    asset = ASSET_DIR / "kuavo_s200062"
    runtime_urdf = asset / "urdf" / "biped_s200062.urdf"
    source_urdf = asset / "urdf" / "biped_s200062.source.urdf"
    root = ET.parse(runtime_urdf).getroot()

    assert source_urdf.is_file()
    assert root.attrib["name"] == "biped_s200062"
    assert root.find("./link[@name='l_twofinger_base']") is not None
    assert root.find("./link[@name='r_twofinger_base']") is not None
    assert root.find("./link[@name='l_d405_camera']") is not None
    assert root.find("./link[@name='r_d405_camera']") is not None

    camera_base_joint = root.find("./joint[@name='camera_base_joint']")
    camera_joint = root.find("./joint[@name='camera']")
    assert camera_base_joint.find("child").attrib["link"] == "head_camera_base"
    assert camera_joint.find("parent").attrib["link"] == "head_camera_base"

    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        assert not filename.startswith(("package://", "http://", "https://", "omniverse://"))
        assert (runtime_urdf.parent / filename).resolve().is_file(), filename


def test_s63_is_handless_with_only_its_referenced_meshes_packaged() -> None:
    asset = ASSET_DIR / "kuavo_s63"
    root = ET.parse(asset / "urdf" / "kuavo_s63.urdf").getroot()
    assert root.attrib["name"] == "biped_s63"
    finger_tokens = ("finger", "thumb", "index", "middle", "ring", "little")
    assert not any(
        any(token in link.attrib["name"].lower() for token in finger_tokens)
        for link in root.findall("./link")
    )
    referenced = {
        Path(mesh.attrib["filename"]).name
        for mesh in root.findall(".//visual/geometry/mesh")
    }
    packaged = {path.name for path in (asset / "meshes").glob("*.STL")}
    assert packaged == referenced
    assert {"l_hand_pitch.STL", "r_hand_pitch.STL"} <= referenced
    assert not any("nohand" in name.lower() for name in referenced)
    expected_mounts = {
        "l": (
            "0.135599 -0.017281 -0.115070",
            "2.779374508 -0.751879414 0.253261563",
        ),
        "r": (
            "0.135599 0.017281 -0.115070",
            "-2.779374508 -0.751879414 -0.253261563",
        ),
    }
    for side, (xyz, rpy) in expected_mounts.items():
        tip = f"zarm_{side}7_end_effector"
        assert root.find(f"./link[@name='{tip}']/visual") is None
        joint = root.find(f"./joint[@name='{tip}_joint']")
        assert joint is not None
        assert joint.find("parent").attrib["link"] == f"zarm_{side}7_link"
        assert joint.find("origin").attrib == {"xyz": xyz, "rpy": rpy}


def test_s56_runtime_urdf_contains_local_qiangnao_hands() -> None:
    asset = ASSET_DIR / "kuavo_s56"
    root = ET.parse(asset / "urdf" / "kuavo_s56.urdf").getroot()
    assert root.attrib["name"] == "biped_s56"
    revolute = {
        joint.attrib["name"] for joint in root.findall("./joint[@type='revolute']")
    }
    assert {f"leg_{side}{index}_joint" for side in ("l", "r") for index in range(1, 7)} <= revolute
    assert {f"zarm_{side}{index}_joint" for side in ("l", "r") for index in range(1, 8)} <= revolute
    assert {
        f"{side}_{finger}{joint}"
        for side in ("l", "r")
        for finger in ("index", "middle", "ring", "little")
        for joint in ("MCP", "PIP")
    } <= revolute
    assert {f"{side}_thumb{joint}" for side in ("l", "r") for joint in ("CMC", "MCP")} <= revolute
    assert not any(name.startswith("wheel_") for name in revolute)

    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        assert not filename.startswith(("package://", "http://", "https://", "omniverse://"))
        assert (asset / "urdf" / filename).resolve().is_file(), filename
    referenced = {
        Path(mesh.attrib["filename"]).name
        for mesh in root.findall(".//visual/geometry/mesh")
    }
    assert {"l_hand_pitch_nohand.STL", "r_hand_pitch_nohand.STL"} <= referenced
    assert {"l_palm.STL", "r_palm.STL", "l_thumb_dist.STL", "r_little_dist.STL"} <= referenced

    for side in ("l", "r"):
        joint = root.find(f"./joint[@name='zarm_{side}7_end_effector_joint']")
        assert joint is not None
        assert joint.find("origin").attrib == {
            "xyz": "0 0.0 -0.17",
            "rpy": "0 0 0",
        }
        palm_joint = root.find(f"./joint[@name='{side}_palm_fixed_joint']")
        assert palm_joint is not None
        assert palm_joint.find("parent").attrib["link"] == f"zarm_{side}7_link"


def test_s63_chassis_wheels_and_wrist_inertials_are_preserved() -> None:
    """Guard the physical S63 model against accidental S62 substitution."""
    root = ET.parse(
        ASSET_DIR / "kuavo_s63" / "urdf" / "kuavo_s63.urdf"
    ).getroot()

    base = root.find("./link[@name='base_link']/inertial")
    assert base is not None
    assert base.find("origin").attrib["xyz"] == "0.00196618 -0.00243108 0.17326354"
    assert base.find("mass").attrib["value"] == "129.40001"
    assert base.find("inertia").attrib == {
        "ixx": "6.70252462",
        "ixy": "0.0",
        "ixz": "0.0",
        "iyy": "6.79536961",
        "iyz": "0.0",
        "izz": "9.12227793",
    }

    wheel_origins = {
        "wheel_left_front_joint": ("0.23865 0.23865 0.13035", "0 0 0.7854"),
        "wheel_right_front_joint": ("0.23865 -0.23865 0.13035", "0 0 -0.7854"),
        "wheel_left_behind_joint": ("-0.23865 0.23865 0.13035", "0 0 2.3562"),
        "wheel_right_behind_joint": ("-0.23865 -0.23865 0.13035", "0 0 -2.3562"),
    }
    for joint_name, (xyz, rpy) in wheel_origins.items():
        origin = root.find(f"./joint[@name='{joint_name}']/origin")
        assert origin is not None
        assert origin.attrib == {"xyz": xyz, "rpy": rpy}

    for wheel_name in (
        "wheel_left_front",
        "wheel_right_front",
        "wheel_left_behind",
        "wheel_right_behind",
    ):
        inertial = root.find(f"./link[@name='{wheel_name}']/inertial")
        assert inertial is not None
        assert inertial.find("mass").attrib["value"] == "3.85"
        assert inertial.find("inertia").attrib["ixx"] == "0.03331258"

    wrist_masses = {
        "zarm_l5_link": "0.67878399",
        "zarm_l6_link": "0.51230538",
        "zarm_l7_link": "0.7374",
        "zarm_r5_link": "0.67878399",
        "zarm_r6_link": "0.51230538",
        "zarm_r7_link": "0.7374",
    }
    for link_name, mass in wrist_masses.items():
        inertial = root.find(f"./link[@name='{link_name}']/inertial")
        assert inertial is not None
        assert inertial.find("mass").attrib["value"] == mass


def test_robotiq_2f85_tree_port_keeps_challenge_topology_and_materials() -> None:
    urdf = ASSET_DIR / "robotiq_2f85" / "urdf" / "robotiq_2f85.urdf"
    root = ET.parse(urdf).getroot()
    revolute = {
        joint.attrib["name"]
        for joint in root.findall("./joint[@type='revolute']")
    }
    expected = {
        f"{side}_{name}_joint"
        for side in ("left", "right")
        for name in ("driver", "coupler", "spring_link", "follower")
    }
    assert revolute == expected
    assert root.find("./link[@name='base_mount']") is not None
    assert root.find("./link[@name='left_pad']") is not None
    assert root.find("./link[@name='right_pad']") is not None
    materials = {item.attrib["name"] for item in root.findall("./material")}
    assert materials == {"leju_black", "leju_gray", "leju_silicone"}


def test_default_runtime_configs_are_packaged() -> None:
    assert (CONFIG_DIR / "workcell_layout.json").is_file()
    assert (CONFIG_DIR / "rack_box_poses.json").is_file()
    assert (CONFIG_DIR / "grippers.json").is_file()
    assert (PACKAGE_CONFIG_DIR / "workcell_layout.json").is_file()
    assert (PACKAGE_CONFIG_DIR / "rack_box_poses.json").is_file()
    assert (PACKAGE_CONFIG_DIR / "grippers.json").is_file()


def test_deployment_and_wheel_fallback_configs_are_synchronized() -> None:
    repository_configs = Path(__file__).resolve().parents[1] / "configs"
    for name in ("workcell_layout.json", "rack_box_poses.json", "grippers.json"):
        assert (repository_configs / name).read_bytes() == (
            PACKAGE_CONFIG_DIR / name
        ).read_bytes()
