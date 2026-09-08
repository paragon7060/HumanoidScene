"""CPU contracts only. Passing these tests does NOT prove Isaac pregrasp success."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from kuavo_isaaclab_scene.core.paths import ASSET_DIR
from kuavo_isaaclab_scene.planning import config
from kuavo_isaaclab_scene.planning.geometry import (
    inverse_transform, origin_matrix, pose_matrix, pregrasp_transform,
)
from kuavo_isaaclab_scene.planning.inspect import inspect_robot
from kuavo_isaaclab_scene.planning.robot_model import UrdfModel, joint_indices


@pytest.fixture
def urdf(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text('''<robot name="test"><link name="base"/><link name="arm"/>
    <link name="tip"/>
    <joint name="shoulder" type="revolute"><parent link="base"/><child link="arm"/>
      <origin xyz="1 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
      <limit lower="-2" upper="2" velocity="1" effort="1"/></joint>
    <joint name="tcp" type="fixed"><parent link="arm"/><child link="tip"/>
      <origin xyz="0.17 0 0" rpy="0 0 0"/></joint></robot>''')
    return path


def test_pose_composition_preserves_nonzero_center_offset():
    world = origin_matrix([3, 4, 5], [0, 0, np.pi / 2])
    grasp = [0.2, 0.1, 0.05, 1, 0, 0, 0]
    result = pregrasp_transform(world, grasp, [1, 0, 0], 0.1)
    np.testing.assert_allclose(result[:3, 3], [2.9, 4.1, 5.05], atol=1e-12)
    np.testing.assert_allclose(inverse_transform(world) @ world, np.eye(4), atol=1e-12)


def test_flap_target_updates_with_reference_motion():
    grasp = [0.1, 0.2, 0, 1, 0, 0, 0]
    first = pregrasp_transform(np.eye(4), grasp, [0, 1, 0], .1)
    moved = origin_matrix([1, 2, 3], [.2, -.1, .3])
    second = pregrasp_transform(moved, grasp, [0, 1, 0], .1)
    np.testing.assert_allclose(second, moved @ first)


@pytest.mark.parametrize("pose", [
    [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 2, 0, 0, 0], [0]*6,
    [float("nan"), 0, 0, 1, 0, 0, 0],
])
def test_bad_pose_rejected(pose):
    with pytest.raises(ValueError):
        pose_matrix(pose)


def test_quaternion_sign_invariance_and_wxyz():
    q = np.sqrt(.5)
    positive = pose_matrix([0, 0, 0, q, 0, 0, q])
    negative = pose_matrix([0, 0, 0, -q, 0, 0, -q])
    np.testing.assert_allclose(positive, negative)
    np.testing.assert_allclose(positive[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)


@pytest.mark.parametrize("axis,distance", [([0, 0, 0], .1), ([2, 0, 0], .1),
                                                ([1, 0, 0], 0), ([1, 0, 0], float("nan"))])
def test_bad_pregrasp_rejected(axis, distance):
    with pytest.raises(ValueError):
        pregrasp_transform(np.eye(4), [0, 0, 0, 1, 0, 0, 0], axis, distance)


def test_scale_is_not_silently_treated_as_rigid_pose():
    with pytest.raises(ValueError, match="rigid"):
        inverse_transform(np.diag([2, 1, 1, 1]))


def test_joint_mapping_uses_names_not_urdf_indices():
    assert joint_indices(["left", "right"], ["head", "right", "left"]) == (2, 1)


@pytest.mark.parametrize("requested,available", [([], ["a"]), (["a", "a"], ["a"]),
                                                         (["a"], ["a", "a"]), (["a"], ["b"])])
def test_invalid_joint_mapping_rejected(requested, available):
    with pytest.raises(ValueError):
        joint_indices(requested, available)


def test_urdf_fk_includes_fixed_tcp_offset(urdf):
    model = UrdfModel(urdf)
    result = model.fk("tip", {"shoulder": np.pi / 2})
    np.testing.assert_allclose(result[:3, 3], [1, .17, 0], atol=1e-12)
    assert [j.name for j in model.chain("tip")] == ["shoulder", "tcp"]


@pytest.mark.parametrize("positions", [{}, {"shoulder": 3}, {"shoulder": float("nan")}])
def test_fk_does_not_guess_or_clamp(urdf, positions):
    with pytest.raises(ValueError):
        UrdfModel(urdf).fk("tip", positions)


def test_unknown_link_rejected(urdf):
    with pytest.raises(ValueError, match="unknown link"):
        UrdfModel(urdf).chain("flap")


def test_duplicate_joint_child_rejected(urdf):
    original = urdf.read_text()
    urdf.write_text(original.replace('name="tcp"', 'name="shoulder"'))
    with pytest.raises(ValueError, match="duplicate"):
        UrdfModel(urdf)


def test_mimic_requires_explicit_linkage_adapter(urdf):
    urdf.write_text(urdf.read_text().replace('<limit lower=', '<mimic joint="other"/><limit lower='))
    with pytest.raises(ValueError, match="linkage adapter"):
        UrdfModel(urdf).fk("tip", {"shoulder": 0})


def test_duplicate_yaml_keys_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("dataset_fps: 10\ndataset_fps: 30\n")
    with pytest.raises(ValueError, match="duplicate"):
        config.load_yaml(path)


@pytest.fixture
def collection(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_ROOT", tmp_path)
    return {
        "draft": False, "episodes": 5, "max_attempts": 20, "dataset_fps": 10,
        "output_root": str(tmp_path), "run_name": "test_run", "resume": False,
        "target_format": "lerobot_v3", "write_strategy": "direct", "save_hdf5": True,
        "save_raw_control": True, "preserve_privileged_gt": True, "record_failure_metadata": True,
        "extra_modalities": {"depth": False, "segmentation": False},
        "cameras": {name: {"enabled": True, "width": 320, "height": 240}
                    for name in ("head", "left_wrist", "right_wrist")},
    }


def test_complete_collection_config(collection):
    config.validate_collection(collection)


@pytest.mark.parametrize("key,value", [("draft", True), ("episodes", None), ("episodes", True),
    ("max_attempts", 1), ("dataset_fps", float("nan")), ("output_root", "/tmp"),
    ("run_name", "../escape"), ("run_name", "/absolute"), ("write_strategy", None),
    ("save_hdf5", None), ("resume", "false")])
def test_incomplete_or_unsafe_collection_rejected(collection, key, value):
    collection[key] = value
    with pytest.raises(ValueError):
        config.validate_collection(collection)


def test_camera_dimensions_are_required_only_if_enabled(collection):
    collection["cameras"]["head"]["width"] = None
    with pytest.raises(ValueError, match="width"):
        config.validate_collection(collection)
    collection["cameras"]["head"]["enabled"] = False
    config.validate_collection(collection)


def test_run_symlink_rejected(collection, tmp_path):
    (tmp_path / "test_run").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        config.output_directory(collection)


def test_disabled_randomization_allows_unset_ranges():
    config.validate_randomization({"randomization": {"enabled": False}})


def test_randomization_ranges_are_explicit():
    value = {"enabled": True, "box_types": ["small"], "rack_slot_ids": ["actual_slot_id"],
             "box_position_offset_m": {"frame": "rack", "x": [-.01, .01], "y": [0, 0], "z": [0, 0]},
             "box_yaw_offset_rad": [-.1, .1], "lighting": {"enabled": False}, "friction": {"enabled": False}}
    config.validate_randomization({"randomization": value})
    invalid = deepcopy(value)
    invalid["box_position_offset_m"]["frame"] = None
    with pytest.raises(ValueError, match="frame"):
        config.validate_randomization({"randomization": invalid})
    invalid = deepcopy(value)
    invalid["box_yaw_offset_rad"] = [1, -1]
    with pytest.raises(ValueError, match="range"):
        config.validate_randomization({"randomization": invalid})


def test_packaged_robot_contract_and_no_gpu_import(urdf):
    from kuavo_isaaclab_scene.robots.gripper_config import DEFAULT_GRIPPER_CONFIG
    settings = {"robot_model": "s200062", "hand_model": "s200062_integrated",
                "base_link": "base_link"}
    for side in ("left", "right"):
        settings[f"{side}_tcp_link"] = f"zarm_{side[0]}7_end_effector"
        settings[f"{side}_tcp_offset_pose"] = [0, 0, 0, 1, 0, 0, 0]
    before = set(sys.modules)
    report = inspect_robot(settings, DEFAULT_GRIPPER_CONFIG, None)
    assert report["arm_command_dimension"] == 14
    assert report["hand_drive_dimension"] == 4
    assert report["hand_command_dimension"] is None
    assert report["simulator_joint_indices"] is None
    assert report["tcp_physical_validation"] == "NOT_RUN"
    json.dumps(report, allow_nan=False)
    imported = set(sys.modules) - before
    assert not any(name.split(".")[0] in {"torch", "isaaclab", "curobo"} for name in imported)
    urdf.write_text(urdf.read_text().replace('child link="tip"', 'child link="missing"'))
    report = inspect_robot(settings, DEFAULT_GRIPPER_CONFIG, urdf)
    assert "missing parent/child" in report["upstream_validation_error"]
    assert report["sides"]["left"]["upstream_kinematic_chain_equal"] is None
    assert report["upstream_urdf"]["path"] == str(urdf)


def test_audit_output_is_new_and_never_overwrites(collection, tmp_path, monkeypatch):
    from kuavo_isaaclab_scene.planning import inspect
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    for name in ("robot", "task1", "collection"):
        (config_dir / f"{name}.yaml").write_text(json.dumps(collection))
    monkeypatch.setattr(inspect, "build_report", lambda *args: {"inspection_test": True})
    args = ["--config-dir", str(config_dir), "--humanoid-repo", str(tmp_path),
            "--rwh-repo", str(tmp_path), "--ros-repo", str(tmp_path)]
    assert inspect.main(args) == 0
    report = tmp_path / "test_run/preflight.json"
    previous = report.read_bytes()
    with pytest.raises(SystemExit) as exc:
        inspect.main(args)
    assert exc.value.code == 2
    assert report.read_bytes() == previous


def test_s200062_tree_fk_against_pinocchio_multiple_poses():
    pin = pytest.importorskip("pinocchio")
    path = ASSET_DIR / "kuavo_s200062/urdf/biped_s200062.urdf"
    model = UrdfModel(path)
    reference = pin.buildModelFromUrdf(str(path))
    data = reference.createData()
    for fraction in (.2, .5, .8):
        for side in ("l", "r"):
            link = f"zarm_{side}7_end_effector"
            chain = [j for j in model.chain(link) if j.kind != "fixed"]
            positions = {j.name: j.lower + fraction * (j.upper - j.lower) for j in chain}
            q = pin.neutral(reference)
            for name, value in positions.items():
                joint = reference.joints[reference.getJointId(name)]
                assert joint.nq == 1
                q[joint.idx_q] = value
            pin.forwardKinematics(reference, data, q)
            pin.updateFramePlacements(reference, data)
            expected = data.oMf[reference.getFrameId(link)].homogeneous
            np.testing.assert_allclose(model.fk(link, positions), expected, atol=1e-10, rtol=0)
