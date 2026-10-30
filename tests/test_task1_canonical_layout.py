from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_task1_files_are_the_only_active_data_collection_pipeline():
    expected = {
        "__init__.py",
        "contract.py",
        "collision.py",
        "pose_editor.py",
        "endpoint.py",
        "approach.py",
        "retreat.py",
        "execute.py",
        "ui/editor.html",
    }
    actual = {
        path.relative_to(ROOT / "data_collection/task1").as_posix()
        for path in (ROOT / "data_collection/task1").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }

    assert expected <= actual
    assert not list((ROOT / "data_collection").glob("task1_*.py"))


def test_legacy_is_history_only_and_has_manifest():
    legacy = ROOT / "data_collection/legacy"

    assert (legacy / "README.md").is_file()
    assert (legacy / "MANIFEST.md").is_file()
    assert (legacy / "scripts/task1_pregrasp_smoke.py").is_file()
    assert (legacy / "tests/task1_wrist_schedule_test.py").is_file()
    assert not (legacy / "__init__.py").exists()


def test_superseded_pregrasp_pipeline_is_not_active():
    assert not (ROOT / "scripts/task1_pregrasp_smoke.py").exists()
    assert not (ROOT / "tests/test_task1_wrist_schedule.py").exists()


def test_physical_runner_wrapper_calls_the_canonical_entrypoint():
    source = (ROOT / "scripts/task1_cumotion_grasp_pull_smoke.py").read_text()

    assert "from data_collection.task1.execute import _main as main" in source
