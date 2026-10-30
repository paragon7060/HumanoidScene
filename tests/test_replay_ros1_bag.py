import json
import math
import os
from pathlib import Path
import subprocess

import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_ros1_bag_export_records_synchronized_neutral_replay_topics(tmp_path):
    """Catches a ROS1 bag that drops fields or cannot be deserialized."""
    rosbags_python = os.environ.get("ROSBAGS_PYTHON")
    if not rosbags_python:
        pytest.skip("set ROSBAGS_PYTHON to an interpreter with rosbags installed")
    trajectory = tmp_path / "trajectory.jsonl"
    rows = [
        {
            "index": 0,
            "t_s": 0.0,
            "phase": "settle",
            "arm_q_rad": [0.0, 0.1],
            "active_gripper_close_fraction": 0.0,
        },
        {
            "index": 1,
            "t_s": 1 / 30,
            "phase": "approach",
            "arm_q_rad": [0.2, 0.3],
            "active_gripper_close_fraction": 0.25,
        },
    ]
    trajectory.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"trajectory": {"joint_names": ["joint_a", "joint_b"]}})
    )
    output = tmp_path / "replay.bag"

    result = subprocess.run(
        [
            rosbags_python,
            str(PROJECT_DIR / "scripts" / "export_replay_ros1_bag.py"),
            "--trajectory",
            str(trajectory),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    verifier = """
import json, sys
from collections import Counter
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
store=get_typestore(Stores.ROS1_NOETIC)
counts=Counter(); arm=[]
with Reader(sys.argv[1]) as reader:
    for conn, timestamp, raw in reader.messages():
        counts[conn.topic] += 1
        if conn.topic == '/ssm_replay/arm_joint_targets':
            msg=store.deserialize_ros1(raw, conn.msgtype)
            arm.append({'name': msg.name, 'position': msg.position.tolist()})
print(json.dumps({'counts': counts, 'arm': arm}, sort_keys=True))
"""
    inspected = subprocess.run(
        [rosbags_python, "-c", verifier, str(output)],
        text=True,
        capture_output=True,
        check=True,
    )
    summary = json.loads(inspected.stdout)
    assert summary["counts"] == {
        "/ssm_replay/active_gripper_close_fraction": 2,
        "/ssm_replay/arm_joint_targets": 2,
        "/ssm_replay/phase": 2,
    }
    assert summary["arm"] == [
        {"name": ["joint_a", "joint_b"], "position": [0.0, 0.1]},
        {"name": ["joint_a", "joint_b"], "position": [0.2, 0.3]},
    ]


def test_kuavo_arm_degree_profile_writes_only_direct_arm_commands(tmp_path):
    """Catches a hardware bag with neutral topics, radians, or gripper commands."""
    rosbags_python = os.environ.get("ROSBAGS_PYTHON")
    if not rosbags_python:
        pytest.skip("set ROSBAGS_PYTHON to an interpreter with rosbags installed")
    trajectory = tmp_path / "trajectory.jsonl"
    rows = [
        {
            "index": 0,
            "t_s": 0.0,
            "phase": "settle",
            "arm_q_rad": [0.0, math.pi / 2, -math.pi],
            "active_gripper_close_fraction": 0.0,
        },
        {
            "index": 1,
            "t_s": 1 / 30,
            "phase": "approach",
            "arm_q_rad": [math.pi / 4, -math.pi / 6, math.pi],
            "active_gripper_close_fraction": 1.0,
        },
    ]
    trajectory.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"trajectory": {"joint_names": ["joint_a", "joint_b", "joint_c"]}}
        )
    )
    output = tmp_path / "kuavo_arm_deg.bag"

    result = subprocess.run(
        [
            rosbags_python,
            str(PROJECT_DIR / "scripts" / "export_replay_ros1_bag.py"),
            "--trajectory",
            str(trajectory),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--profile",
            "kuavo-arm-deg",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    verifier = """
import json, sys
from collections import Counter
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
store=get_typestore(Stores.ROS1_NOETIC)
counts=Counter(); arm=[]; timestamps=[]
with Reader(sys.argv[1]) as reader:
    for conn, timestamp, raw in reader.messages():
        counts[conn.topic] += 1
        timestamps.append(timestamp)
        if conn.topic == '/kuavo_arm_traj':
            msg=store.deserialize_ros1(raw, conn.msgtype)
            arm.append({'name': msg.name, 'position': msg.position.tolist()})
print(json.dumps({'counts': counts, 'arm': arm, 'timestamps': timestamps}, sort_keys=True))
"""
    inspected = subprocess.run(
        [rosbags_python, "-c", verifier, str(output)],
        text=True,
        capture_output=True,
        check=True,
    )
    summary = json.loads(inspected.stdout)
    assert summary["counts"] == {"/kuavo_arm_traj": 2}
    assert summary["arm"] == [
        {
            "name": ["joint_a", "joint_b", "joint_c"],
            "position": [0.0, 90.0, -180.0],
        },
        {
            "name": ["joint_a", "joint_b", "joint_c"],
            "position": [45.0, -29.999999999999996, 180.0],
        },
    ]
    assert summary["timestamps"] == [1_000_000_000, 1_033_333_333]


@pytest.mark.parametrize(
    ("active_gripper", "expected_positions"),
    [
        ("left", [[100.0, 100.0], [0.0, 100.0]]),
        ("right", [[100.0, 100.0], [100.0, 0.0]]),
    ],
)
def test_kuavo_source_claw_policy_preserves_the_active_side(
    tmp_path, active_gripper, expected_positions
):
    """Catches reversed percentages, mirrored motion, or a swapped active side."""
    rosbags_python = os.environ.get("ROSBAGS_PYTHON")
    if not rosbags_python:
        pytest.skip("set ROSBAGS_PYTHON to an interpreter with rosbags installed")
    trajectory = tmp_path / "trajectory.jsonl"
    rows = [
        {
            "index": 0,
            "t_s": 0.0,
            "phase": "approach",
            "arm_q_rad": [0.0, math.pi / 2],
            "active_gripper_close_fraction": 0.0,
        },
        {
            "index": 1,
            "t_s": 1 / 30,
            "phase": "close",
            "arm_q_rad": [math.pi / 4, -math.pi / 6],
            "active_gripper_close_fraction": 1.0,
        },
    ]
    trajectory.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "active_gripper": active_gripper,
                "trajectory": {"joint_names": ["joint_a", "joint_b"]},
            }
        )
    )
    output = tmp_path / f"kuavo_{active_gripper}.bag"

    result = subprocess.run(
        [
            rosbags_python,
            str(PROJECT_DIR / "scripts" / "export_replay_ros1_bag.py"),
            "--trajectory",
            str(trajectory),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--profile",
            "kuavo-arm-deg",
            "--claw-policy",
            "source",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    verifier = """
import json, sys
from collections import Counter, defaultdict
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
store=get_typestore(Stores.ROS1_NOETIC)
store.register(get_types_from_msg('''
string[] name
float64[] position
float64[] velocity
float64[] effort
''', 'kuavo_msgs/msg/endEffectorData'))
store.register(get_types_from_msg('''
std_msgs/Header header
kuavo_msgs/endEffectorData data
''', 'kuavo_msgs/msg/lejuClawCommand'))
counts=Counter(); timestamps=defaultdict(list); claws=[]
with Reader(sys.argv[1]) as reader:
    for conn, timestamp, raw in reader.messages():
        counts[conn.topic] += 1
        timestamps[conn.topic].append(timestamp)
        if conn.topic == '/leju_claw_command':
            msg=store.deserialize_ros1(raw, conn.msgtype)
            claws.append({
                'name': msg.data.name,
                'position': msg.data.position.tolist(),
                'velocity': msg.data.velocity.tolist(),
                'effort': msg.data.effort.tolist(),
            })
print(json.dumps({'counts': counts, 'timestamps': timestamps, 'claws': claws}, sort_keys=True))
"""
    inspected = subprocess.run(
        [rosbags_python, "-c", verifier, str(output)],
        text=True,
        capture_output=True,
        check=True,
    )
    summary = json.loads(inspected.stdout)
    assert summary["counts"] == {
        "/kuavo_arm_traj": 2,
        "/leju_claw_command": 2,
    }
    assert [claw["position"] for claw in summary["claws"]] == expected_positions
    assert all(
        claw == {
            "name": ["left_claw", "right_claw"],
            "position": expected_position,
            "velocity": [50.0, 50.0],
            "effort": [1.0, 1.0],
        }
        for claw, expected_position in zip(summary["claws"], expected_positions)
    )
    assert summary["timestamps"]["/kuavo_arm_traj"] == [
        1_000_000_000,
        1_033_333_333,
    ]
    assert summary["timestamps"]["/leju_claw_command"] == [
        1_000_000_000,
        1_033_333_333,
    ]
