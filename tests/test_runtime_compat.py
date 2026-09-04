from __future__ import annotations

from kuavo_isaaclab_scene.core.runtime_compat import validate_runtime


SUPPORTED = {
    "python": "3.11.13",
    "isaacsim": "5.1.0.0",
    "isaaclab": "0.54.2",
    "torch": "2.7.0+cu128",
    "torchvision": "0.22.0+cu128",
    "gymnasium": "1.2.1",
    "numpy": "1.26.4",
    "typing_extensions": "4.12.2",
    "psutil": "5.9.8",
    "wheel": "0.45.1",
    "onnx": "1.18.0",
}


def test_stable_stack_is_accepted() -> None:
    assert validate_runtime(SUPPORTED, machine="x86_64", system="Linux") == []


def test_old_isaac_stack_is_rejected() -> None:
    versions = {**SUPPORTED, "isaacsim": "5.0.0.0", "isaaclab": "0.45.7"}
    issues = validate_runtime(versions, machine="x86_64", system="Linux")
    assert {issue.component for issue in issues} == {"isaacsim", "isaaclab"}


def test_python_numpy_and_quest_architecture_are_checked() -> None:
    versions = {**SUPPORTED, "python": "3.12.1", "numpy": "2.0.0"}
    issues = validate_runtime(versions, machine="aarch64", system="Linux")
    assert {issue.component for issue in issues} == {
        "architecture",
        "python",
        "numpy",
    }
