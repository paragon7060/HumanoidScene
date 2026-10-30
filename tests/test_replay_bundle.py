import hashlib
import json

import pytest


JOINT_NAMES = [
    *(f"zarm_l{i}_joint" for i in range(1, 8)),
    *(f"zarm_r{i}_joint" for i in range(1, 8)),
]


def _inputs():
    approach = {
        "status": "SUCCESS",
        "joint_names": JOINT_NAMES,
        "waypoint_q_rad": [[0.0] * 14, [0.1] * 14],
    }
    retreat = {
        "status": "SUCCESS",
        "joint_names": JOINT_NAMES,
        "waypoint_q_rad": [[0.1] * 14, [0.2] * 14],
    }
    scenario = {
        "robot": {"model": "s200062", "gripper_preset": "s200062_integrated"},
        "planning": {
            "cspace_dof": 14,
            "cspace_joint_names": JOINT_NAMES,
            "torso_height_m": 0.25,
        },
        "execution": {
            "active_gripper": "left",
            "settle_steps": 4,
            "approach_steps_per_waypoint": 4,
            "close_steps": 8,
            "closed_hold_steps": 4,
            "retreat_steps_per_waypoint": 8,
            "final_hold_steps": 4,
        },
    }
    report = {"physics_dt_s": 1 / 120, "scenario": {"verification_state": "experimental_unverified"}}
    return approach, retreat, scenario, report


def test_build_trajectory_preserves_phase_timing_and_joint_targets():
    """Catches lost waypoint holds or an incorrect 120 Hz to 30 Hz conversion."""
    try:
        from kuavo_isaaclab_scene.replay_bundle import build_trajectory
    except ImportError as exc:
        pytest.fail(f"replay bundle feature is missing: {exc}")

    trajectory = build_trajectory(*_inputs(), output_hz=30)

    assert trajectory["phase_counts"] == {
        "settle": 1,
        "approach": 2,
        "close": 2,
        "closed_hold": 1,
        "retreat": 4,
        "final_hold": 1,
    }
    assert len(trajectory["frames"]) == 11
    assert trajectory["frames"][0] == {
        "index": 0,
        "t_s": 0.0,
        "phase": "settle",
        "arm_q_rad": [0.0] * 14,
        "active_gripper_close_fraction": 0.0,
    }
    assert [row["arm_q_rad"][0] for row in trajectory["frames"][1:3]] == [0.0, 0.1]
    assert [row["active_gripper_close_fraction"] for row in trajectory["frames"][3:5]] == [0.5, 1.0]
    assert [row["arm_q_rad"][0] for row in trajectory["frames"][6:10]] == [0.1, 0.1, 0.2, 0.2]
    assert trajectory["frames"][-1]["t_s"] == pytest.approx(10 / 30)


def test_export_bundle_writes_auditable_neutral_replay_files(tmp_path):
    """Catches a replay export that drops provenance or emits no inspectable trajectory."""
    try:
        from kuavo_isaaclab_scene.replay_bundle import export_bundle
    except ImportError as exc:
        pytest.fail(f"bundle export feature is missing: {exc}")

    approach, retreat, scenario, report = _inputs()
    rack_poses = {"version": 1, "boxes": {"SmallBox_0": {"local_pos": [0, 0, 1]}}}
    source_values = {
        "approach_plan": approach,
        "retreat_plan": retreat,
        "scenario": scenario,
        "rack_box_poses": rack_poses,
    }
    paths = {}
    expected_hashes = {}
    for key, value in source_values.items():
        path = tmp_path / f"{key}.json"
        payload = json.dumps(value, sort_keys=True).encode()
        path.write_bytes(payload)
        paths[key] = path
        expected_hashes[key] = hashlib.sha256(payload).hexdigest()
    report["input_sha256"] = {
        "approach_plan": expected_hashes["approach_plan"],
        "retreat_plan": expected_hashes["retreat_plan"],
        "scenario": expected_hashes["scenario"],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    output_dir = tmp_path / "bundle"

    manifest = export_bundle(
        approach_path=paths["approach_plan"],
        retreat_path=paths["retreat_plan"],
        scenario_path=paths["scenario"],
        report_path=report_path,
        rack_box_poses_path=paths["rack_box_poses"],
        output_dir=output_dir,
        output_hz=30,
    )

    assert manifest["source_sha256"]["approach_plan"] == expected_hashes["approach_plan"]
    assert manifest["trajectory"]["frame_count"] == 11
    assert manifest["hardware_command_ready"] is False
    assert json.loads((output_dir / "manifest.json").read_text()) == manifest
    assert len((output_dir / "trajectory.jsonl").read_text().splitlines()) == 11
    assert (output_dir / "trajectory.csv").read_text().splitlines()[0].startswith(
        "index,t_s,phase,active_gripper_close_fraction,zarm_l1_joint"
    )
    assert (output_dir / "source" / "approach_plan.json").read_bytes() == paths[
        "approach_plan"
    ].read_bytes()


def test_export_bundle_rejects_a_source_hash_mismatch_before_writing(tmp_path):
    """Catches replaying a plan different from the one that produced the report."""
    from kuavo_isaaclab_scene.replay_bundle import export_bundle

    approach, retreat, scenario, report = _inputs()
    values = {
        "approach_plan": approach,
        "retreat_plan": retreat,
        "scenario": scenario,
        "rack_box_poses": {"version": 1},
    }
    paths = {}
    for name, value in values.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        paths[name] = path
    report["input_sha256"] = {
        "approach_plan": "0" * 64,
        "retreat_plan": hashlib.sha256(paths["retreat_plan"].read_bytes()).hexdigest(),
        "scenario": hashlib.sha256(paths["scenario"].read_bytes()).hexdigest(),
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    output_dir = tmp_path / "bundle"

    with pytest.raises(ValueError, match="approach_plan SHA-256 mismatch"):
        export_bundle(
            approach_path=paths["approach_plan"],
            retreat_path=paths["retreat_plan"],
            scenario_path=paths["scenario"],
            report_path=report_path,
            rack_box_poses_path=paths["rack_box_poses"],
            output_dir=output_dir,
        )

    assert not output_dir.exists()
