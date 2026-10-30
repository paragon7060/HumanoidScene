"""Static contract checks for the Isaac-starting Task1 pose editor module."""

from __future__ import annotations

import ast
from pathlib import Path


POSE_EDITOR = Path(__file__).resolve().parents[1] / "data_collection/task1/pose_editor.py"


def test_pose_editor_declares_pair_snapshot_arguments():
    source = POSE_EDITOR.read_text(encoding="utf-8")
    tree = ast.parse(source)
    option_names = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }

    assert {"--paired-boxes", "--active-arm"} <= option_names


def test_pose_editor_pair_snapshot_serializes_required_contract_fields():
    source = POSE_EDITOR.read_text(encoding="utf-8")

    for field in (
        '"target_mode": "paired"',
        '"paired_box_keys"',
        '"paired_box_body_poses_b"',
        '"pair_grasp"',
        '"reference_arm_q_rad"',
    ):
        assert field in source
