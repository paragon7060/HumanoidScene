import ast
from pathlib import Path


SOURCE = Path("data_collection/task1/execute.py")


def test_executor_exposes_paired_box_cli_and_metrics_without_importing_isaac():
    text = SOURCE.read_text()
    ast.parse(text)

    for token in (
        '"--scenario-path"',
        '"--paired-boxes"',
        '"--clear-same-shelf-boxes"',
        '"--pair-separation-drift-max-m"',
        '"box_body_positions_b_m"',
        'paired_retention_metrics(',
        'paired_box_acceptance(',
    ):
        assert token in text
