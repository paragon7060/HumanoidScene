"""Non-GUI preflight checks for the Kuavo Meta Quest/OpenXR path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
from typing import Sequence


XR_EXPERIENCES = (
    "isaaclab.python.xr.openxr.kit",
    "isaaclab.python.xr.openxr.headless.kit",
)
XR_EXTENSIONS = (
    "omni.kit.xr.core-",
    "omni.kit.xr.scene_view.core-",
    "omni.kit.xr.scene_view.utils-",
    "omni.kit.xr.system.openxr-",
    "isaacsim.xr.openxr-",
)


@dataclass(frozen=True)
class QuestRuntimeIssue:
    component: str
    detail: str

    def __str__(self) -> str:
        return f"{self.component}: {self.detail}"


def validate_openxr_manifest(path: Path) -> list[QuestRuntimeIssue]:
    """Validate the fields and referenced runtime library in an OpenXR manifest."""
    path = path.expanduser().resolve()
    if not path.is_file():
        return [QuestRuntimeIssue("XR_RUNTIME_JSON", f"manifest does not exist: {path}")]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return [QuestRuntimeIssue("XR_RUNTIME_JSON", f"cannot parse {path}: {error}")]
    if not isinstance(payload, dict):
        return [QuestRuntimeIssue("XR_RUNTIME_JSON", "manifest root must be a JSON object")]
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        return [QuestRuntimeIssue("XR_RUNTIME_JSON", "missing runtime object")]
    library_value = runtime.get("library_path")
    if not isinstance(library_value, str) or not library_value.strip():
        return [QuestRuntimeIssue("XR_RUNTIME_JSON", "missing runtime.library_path")]
    library_path = Path(os.path.expandvars(library_value)).expanduser()
    if not library_path.is_absolute():
        library_path = path.parent / library_path
    if "$" not in str(library_path) and not library_path.resolve().is_file():
        return [
            QuestRuntimeIssue(
                "OpenXR runtime library",
                f"runtime.library_path does not exist: {library_path.resolve()}",
            )
        ]
    return []


def _package_locations(package: str) -> list[Path]:
    spec = importlib.util.find_spec(package)
    if spec is None:
        return []
    locations = [Path(value).resolve() for value in spec.submodule_search_locations or ()]
    if spec.origin and spec.origin not in {"built-in", "frozen"}:
        locations.append(Path(spec.origin).resolve().parent)
    return list(dict.fromkeys(locations))


def _find_isaaclab_apps() -> Path | None:
    for package_dir in _package_locations("isaaclab"):
        for parent in (package_dir, *package_dir.parents):
            apps = parent / "apps"
            if all((apps / name).is_file() for name in XR_EXPERIENCES):
                return apps
    return None


def _find_isaacsim_extension(extension_prefix: str) -> Path | None:
    for package_dir in _package_locations("isaacsim"):
        for extension_root in (package_dir / "extscache", package_dir / "exts"):
            matches = sorted(extension_root.glob(f"{extension_prefix}*"))
            if matches:
                return matches[-1]
    return None


def collect_quest_issues(runtime_json: Path | None = None) -> tuple[list[QuestRuntimeIssue], list[str]]:
    """Return blocking issues and a concise component report without launching Kit."""
    issues: list[QuestRuntimeIssue] = []
    report: list[str] = []

    apps = _find_isaaclab_apps()
    if apps is None:
        issues.append(QuestRuntimeIssue("Isaac Lab OpenXR", "OpenXR experience files were not found"))
    else:
        report.append(f"Isaac Lab OpenXR experiences: {apps}")

    for prefix in XR_EXTENSIONS:
        extension = _find_isaacsim_extension(prefix)
        if extension is None:
            issues.append(QuestRuntimeIssue("Isaac Sim OpenXR", f"missing extension {prefix}*"))
        else:
            report.append(f"OpenXR extension: {extension.name}")

    if runtime_json is None:
        report.append("CloudXR runtime: not configured (browser/IWER preview remains available)")
    else:
        runtime_json = runtime_json.expanduser().resolve()
        issues.extend(validate_openxr_manifest(runtime_json))
        if not issues or not any(issue.component.startswith(("XR_RUNTIME_JSON", "OpenXR runtime")) for issue in issues):
            report.append(f"CloudXR runtime manifest: {runtime_json}")
    return issues, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xr-runtime-json", type=Path, default=None)
    parser.add_argument(
        "--require-runtime",
        action="store_true",
        help="Fail when neither --xr-runtime-json nor XR_RUNTIME_JSON is configured.",
    )
    args = parser.parse_args(argv)

    runtime_json = args.xr_runtime_json
    if runtime_json is None and os.environ.get("XR_RUNTIME_JSON"):
        runtime_json = Path(os.environ["XR_RUNTIME_JSON"])
    issues, report = collect_quest_issues(runtime_json)
    if args.require_runtime and runtime_json is None:
        issues.append(QuestRuntimeIssue("CloudXR runtime", "set XR_RUNTIME_JSON or pass --xr-runtime-json"))

    print("Quest target: Meta Quest 3/3S + OpenXR on Linux x86_64")
    for line in report:
        print(f"  {line}")
    if issues:
        print("Quest compatibility: FAILED")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    print("Quest compatibility: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
