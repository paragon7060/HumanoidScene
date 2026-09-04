import math
from dataclasses import replace
import importlib.util
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from kuavo_isaaclab_scene.core.paths import ASSET_DIR
from kuavo_isaaclab_scene.robots.twofinger_linkage import (
    FINGER_PIN, FOLLOWER_PIN, initial_passive_positions, passive_joint_angles, pin_for,
    validate_motor_commands,
)


def rotate(q, v):
    return (math.cos(q)*v[0]+math.sin(q)*v[1], -math.sin(q)*v[0]+math.cos(q)*v[1])


@pytest.mark.parametrize("jaw", ["f", "b"])
def test_physical_pin_closure_across_entire_command_range(jaw):
    sign = 1 if jaw == "f" else -1
    for step in range(101):
        q = sign * (-.25 * step / 100)
        q3, q4 = passive_joint_angles(q, jaw)
        assert abs(q3) < .698 and abs(q4) < .698
        crank = rotate(q, (sign*.027533, -.0085573))
        coupler = rotate(q+q3, (sign*(-.0081591+FINGER_PIN[0]), -.047301+FINGER_PIN[2]))
        follower = rotate(q4, (sign*FOLLOWER_PIN[0], FOLLOWER_PIN[2]))
        a = (sign*.0125+crank[0]+coupler[0], -.063137+crank[1]+coupler[1])
        b = (sign*.02+follower[0], -.09+follower[1])
        assert math.dist(a, b) < 1e-12


def test_fourbar_reset_is_not_four_identical_joint_targets():
    positions = initial_passive_positions({"l_f_bar_1_joint": -.25, "l_b_bar_1_joint": .25})
    assert positions["l_f_bar_4_joint"] == pytest.approx(-.41745849)
    assert positions["l_f_bar_3_joint"] == pytest.approx(-.0213475)
    assert positions["l_b_bar_4_joint"] == -positions["l_f_bar_4_joint"]
    assert passive_joint_angles(0) == pytest.approx((0,0), abs=1e-12)
    assert pin_for("b", FINGER_PIN) == (.0125, 0, -.021)


@pytest.mark.parametrize("q", [float("nan"), float("inf"), -.4, .1])
def test_invalid_driver_pose_fails_loudly(q):
    with pytest.raises(ValueError):
        passive_joint_angles(q)


@pytest.mark.parametrize("asset", ["kuavo_s200062/urdf/biped_s200062.urdf", "kuavo_s56/urdf/kuavo_s56_twofinger.urdf"])
def test_urdf_central_links_are_passive_revolute_not_fixed(asset):
    root = ET.parse(ASSET_DIR / asset).getroot()
    for side in "lr":
        for jaw in "fb":
            joint = root.find(f"./joint[@name='{side}_{jaw}_bar_4_joint']")
            assert joint.get("type") == "revolute"
            assert joint.find("axis").get("xyz") == "0 1 0"
            assert joint.find("limit") is not None


def test_legacy_passive_joint_commands_are_rejected():
    from kuavo_isaaclab_scene.robots.gripper_config import load_gripper_settings
    settings = load_gripper_settings("s56_twofinger")
    validate_motor_commands(settings)
    with pytest.raises(ValueError, match="only bar_1"):
        validate_motor_commands(replace(settings, joint_names=(*settings.joint_names, "{side}_f_bar_3_joint")))


def test_pin_constants_match_the_actual_donor_joint_frames():
    root = ET.parse(ASSET_DIR / "kuavo_s200062/urdf/biped_s200062.urdf").getroot()
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    for side in "lr":
        for jaw in "fb":
            tip = list(pin_for(jaw, FINGER_PIN))
            link = f"{side}_{jaw}_finger"
            while link != f"{side}_twofinger_base":
                j = by_child[link]
                origin = j.find("origin")
                assert origin.get("rpy", "0 0 0") == "0 0 0"
                xyz = list(map(float, origin.get("xyz").split()))
                tip = [a+b for a,b in zip(tip, xyz)]
                link = j.find("parent").get("link")
            origin4 = list(map(float, by_child[f"{side}_{jaw}_bar_4"].find("origin").get("xyz").split()))
            assert tuple(a-b for a,b in zip(tip, origin4)) == pytest.approx(pin_for(jaw, FOLLOWER_PIN), abs=1e-12)


def test_packaged_usd_has_closed_hinges_and_no_passive_position_servos():
    if importlib.util.find_spec("isaacsim") is None and importlib.util.find_spec("pxr") is None:
        pytest.skip("Offline USD inspection needs pxr or Isaac Sim libraries (no simulator startup)")
    script = Path(__file__).resolve().parents[1] / "scripts/finalize_twofinger_usd.py"
    result = subprocess.run([sys.executable, str(script), "--check"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("Validated four closed loops") == 2
