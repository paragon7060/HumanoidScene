#!/usr/bin/env python3
"""Export an SSM replay trajectory to a selected ROS1 ``.bag`` contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_types_from_msg, get_typestore


ARM_TOPIC = "/ssm_replay/arm_joint_targets"
GRIPPER_TOPIC = "/ssm_replay/active_gripper_close_fraction"
PHASE_TOPIC = "/ssm_replay/phase"
KUAVO_ARM_TOPIC = "/kuavo_arm_traj"
KUAVO_CLAW_TOPIC = "/leju_claw_command"
END_EFFECTOR_DATA_TYPE = "kuavo_msgs/msg/endEffectorData"
LEJU_CLAW_COMMAND_TYPE = "kuavo_msgs/msg/lejuClawCommand"
END_EFFECTOR_DATA_MSG = """
string[] name
float64[] position
float64[] velocity
float64[] effort
"""
LEJU_CLAW_COMMAND_MSG = """
std_msgs/Header header
kuavo_msgs/endEffectorData data
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("neutral", "kuavo-arm-deg"),
        default="neutral",
        help="output contract; kuavo-arm-deg writes only direct degree arm commands",
    )
    parser.add_argument(
        "--claw-policy",
        choices=("none", "source"),
        default="none",
        help="source commands only the manifest active_gripper and keeps the other open",
    )
    return parser.parse_args()


def _load_rows(path: Path, joint_count: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("trajectory is empty")
    previous_t = -math.inf
    for expected_index, row in enumerate(rows):
        if row["index"] != expected_index:
            raise ValueError("trajectory indexes must be contiguous from zero")
        if len(row["arm_q_rad"]) != joint_count:
            raise ValueError("arm target dimension does not match manifest joint names")
        if not math.isfinite(row["t_s"]) or row["t_s"] <= previous_t:
            raise ValueError("trajectory timestamps must be finite and strictly increasing")
        previous_t = row["t_s"]
    return rows


def main() -> None:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing bag: {args.output}")
    manifest = json.loads(args.manifest.read_text())
    joint_names = manifest["trajectory"]["joint_names"]
    rows = _load_rows(args.trajectory, len(joint_names))
    kuavo_arm_degree_profile = args.profile == "kuavo-arm-deg"
    if args.claw_policy != "none" and not kuavo_arm_degree_profile:
        raise ValueError("claw policies require the kuavo-arm-deg profile")
    active_gripper = manifest.get("active_gripper")
    if args.claw_policy == "source" and active_gripper not in ("left", "right"):
        raise ValueError("source claw policy requires active_gripper left or right")

    store = get_typestore(Stores.ROS1_NOETIC)
    if args.claw_policy != "none":
        store.register(get_types_from_msg(END_EFFECTOR_DATA_MSG, END_EFFECTOR_DATA_TYPE))
        store.register(
            get_types_from_msg(LEJU_CLAW_COMMAND_MSG, LEJU_CLAW_COMMAND_TYPE)
        )
    JointState = store.types["sensor_msgs/msg/JointState"]
    Header = store.types["std_msgs/msg/Header"]
    Time = store.types["builtin_interfaces/msg/Time"]
    Float64 = store.types["std_msgs/msg/Float64"]
    String = store.types["std_msgs/msg/String"]
    empty = np.asarray([], dtype=np.float64)
    if args.claw_policy != "none":
        EndEffectorData = store.types[END_EFFECTOR_DATA_TYPE]
        LejuClawCommand = store.types[LEJU_CLAW_COMMAND_TYPE]

    with Writer(args.output) as writer:
        arm_connection = writer.add_connection(
            KUAVO_ARM_TOPIC if kuavo_arm_degree_profile else ARM_TOPIC,
            "sensor_msgs/msg/JointState",
            typestore=store,
        )
        if not kuavo_arm_degree_profile:
            gripper_connection = writer.add_connection(
                GRIPPER_TOPIC, "std_msgs/msg/Float64", typestore=store
            )
            phase_connection = writer.add_connection(
                PHASE_TOPIC, "std_msgs/msg/String", typestore=store
            )
        if args.claw_policy != "none":
            claw_connection = writer.add_connection(
                KUAVO_CLAW_TOPIC, LEJU_CLAW_COMMAND_TYPE, typestore=store
            )
        epoch_offset_ns = 1_000_000_000
        for row in rows:
            timestamp_ns = epoch_offset_ns + round(row["t_s"] * 1_000_000_000)
            arm_position = np.asarray(row["arm_q_rad"], dtype=np.float64)
            if kuavo_arm_degree_profile:
                arm_position = np.rad2deg(arm_position)
            arm = JointState(
                header=Header(
                    seq=row["index"],
                    stamp=Time(
                        sec=timestamp_ns // 1_000_000_000,
                        nanosec=timestamp_ns % 1_000_000_000,
                    ),
                    frame_id="",
                ),
                name=joint_names,
                position=arm_position,
                velocity=empty,
                effort=empty,
            )
            writer.write(
                arm_connection,
                timestamp_ns,
                store.serialize_ros1(arm, arm_connection.msgtype),
            )
            if kuavo_arm_degree_profile:
                if args.claw_policy != "none":
                    close_fraction = float(row["active_gripper_close_fraction"])
                    active_position = 100.0 * (1.0 - close_fraction)
                    claw_positions = (
                        [active_position, 100.0]
                        if active_gripper == "left"
                        else [100.0, active_position]
                    )
                    claw = LejuClawCommand(
                        header=arm.header,
                        data=EndEffectorData(
                            name=["left_claw", "right_claw"],
                            position=np.asarray(
                                claw_positions,
                                dtype=np.float64,
                            ),
                            velocity=np.asarray([50.0, 50.0], dtype=np.float64),
                            effort=np.asarray([1.0, 1.0], dtype=np.float64),
                        ),
                    )
                    writer.write(
                        claw_connection,
                        timestamp_ns,
                        store.serialize_ros1(claw, claw_connection.msgtype),
                    )
                continue
            gripper = Float64(
                data=float(row["active_gripper_close_fraction"])
            )
            writer.write(
                gripper_connection,
                timestamp_ns,
                store.serialize_ros1(gripper, gripper_connection.msgtype),
            )
            phase = String(data=row["phase"])
            writer.write(
                phase_connection,
                timestamp_ns,
                store.serialize_ros1(phase, phase_connection.msgtype),
            )


if __name__ == "__main__":
    main()
