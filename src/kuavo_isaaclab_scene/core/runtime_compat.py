"""Runtime compatibility checks for the repository's stable Isaac stack."""

from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import Mapping, Sequence

from packaging.version import InvalidVersion, Version


STABLE_STACK = {
    "python": "3.11",
    "isaacsim": "5.1.0",
    "isaaclab": "0.54.2",  # Isaac Lab tag v2.3.2
    "torch": "2.7.0",
    "torchvision": "0.22.0",
    "gymnasium": "1.2.1",
    "typing_extensions": "4.12.2",
    "psutil": "5.9.8",
    "wheel": "0.45.1",
    "onnx": "1.18.0",
}


@dataclass(frozen=True)
class RuntimeIssue:
    component: str
    installed: str
    expected: str

    def __str__(self) -> str:
        return f"{self.component}: found {self.installed}, expected {self.expected}"


def _installed_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not installed"


def collect_versions() -> dict[str, str]:
    """Collect versions without importing Kit or starting Isaac Sim."""
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "isaacsim": _installed_version("isaacsim"),
        "isaaclab": _installed_version("isaaclab"),
        "torch": _installed_version("torch"),
        "torchvision": _installed_version("torchvision"),
        "gymnasium": _installed_version("gymnasium"),
        "numpy": _installed_version("numpy"),
        "typing_extensions": _installed_version("typing_extensions"),
        "psutil": _installed_version("psutil"),
        "wheel": _installed_version("wheel"),
        "onnx": _installed_version("onnx"),
    }


def _version(value: str) -> Version | None:
    try:
        return Version(value)
    except InvalidVersion:
        return None


def validate_runtime(
    versions: Mapping[str, str] | None = None,
    *,
    machine: str | None = None,
    system: str | None = None,
) -> list[RuntimeIssue]:
    """Return all mismatches against the supported Linux x86_64 stable stack."""
    found = dict(versions or collect_versions())
    issues: list[RuntimeIssue] = []

    runtime_system = system or platform.system()
    runtime_machine = machine or platform.machine()
    if runtime_system != "Linux":
        issues.append(RuntimeIssue("platform", runtime_system, "Linux"))
    if runtime_machine not in {"x86_64", "AMD64"}:
        issues.append(RuntimeIssue("architecture", runtime_machine, "x86_64 (OpenXR/Quest target)"))

    python_version = _version(found.get("python", ""))
    if python_version is None or python_version.release[:2] != (3, 11):
        issues.append(RuntimeIssue("python", found.get("python", "unknown"), "3.11.x"))

    exact_release_prefixes = {
        "isaacsim": (5, 1, 0),
        "isaaclab": (0, 54, 2),
        "torch": (2, 7, 0),
        "torchvision": (0, 22, 0),
        "gymnasium": (1, 2, 1),
        "typing_extensions": (4, 12, 2),
        "psutil": (5, 9, 8),
        "wheel": (0, 45, 1),
        "onnx": (1, 18, 0),
    }
    for component, expected_release in exact_release_prefixes.items():
        value = found.get(component, "not installed")
        parsed = _version(value)
        if parsed is None or parsed.release[: len(expected_release)] != expected_release:
            issues.append(
                RuntimeIssue(component, value, STABLE_STACK[component])
            )

    numpy_value = found.get("numpy", "not installed")
    numpy_version = _version(numpy_value)
    if numpy_version is None or numpy_version.major >= 2:
        issues.append(RuntimeIssue("numpy", numpy_value, ">=1.24,<2"))
    return issues


def format_report(versions: Mapping[str, str], issues: Sequence[RuntimeIssue]) -> str:
    lines = [
        "Supported runtime: Isaac Sim 5.1.0 + Isaac Lab v2.3.2 (package 0.54.2)",
        *(f"  {name:11s} {value}" for name, value in versions.items()),
    ]
    if issues:
        lines.append("Compatibility: FAILED")
        lines.extend(f"  - {issue}" for issue in issues)
    else:
        lines.append("Compatibility: OK")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="Only print failures.")
    parser.add_argument(
        "--allow-unsupported",
        action="store_true",
        help="Report mismatches without returning a failing exit status.",
    )
    args = parser.parse_args(argv)

    versions = collect_versions()
    issues = validate_runtime(versions)
    if issues or not args.quiet:
        print(format_report(versions, issues), file=sys.stderr if issues else sys.stdout)
    return 0 if not issues or args.allow_unsupported else 1


if __name__ == "__main__":
    raise SystemExit(main())
