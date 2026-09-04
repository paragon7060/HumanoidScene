import json
from pathlib import Path

from kuavo_isaaclab_scene.teleop.quest_runtime import validate_openxr_manifest


def test_openxr_manifest_accepts_existing_relative_library(tmp_path: Path) -> None:
    library = tmp_path / "libopenxr_cloudxr.so"
    library.touch()
    manifest = tmp_path / "openxr_cloudxr.json"
    manifest.write_text(
        json.dumps(
            {
                "file_format_version": "1.0.0",
                "runtime": {"name": "CloudXR", "library_path": library.name},
            }
        ),
        encoding="utf-8",
    )

    assert validate_openxr_manifest(manifest) == []


def test_openxr_manifest_rejects_missing_library(tmp_path: Path) -> None:
    manifest = tmp_path / "openxr_cloudxr.json"
    manifest.write_text(
        json.dumps({"runtime": {"library_path": "missing.so"}}),
        encoding="utf-8",
    )

    issues = validate_openxr_manifest(manifest)

    assert len(issues) == 1
    assert issues[0].component == "OpenXR runtime library"


def test_openxr_manifest_rejects_invalid_json(tmp_path: Path) -> None:
    manifest = tmp_path / "openxr_cloudxr.json"
    manifest.write_text("not-json", encoding="utf-8")

    issues = validate_openxr_manifest(manifest)

    assert len(issues) == 1
    assert issues[0].component == "XR_RUNTIME_JSON"
