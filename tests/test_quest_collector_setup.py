import importlib.util
import io
from pathlib import Path
import subprocess
import tarfile

import pytest


spec = importlib.util.spec_from_file_location(
    "collector_setup", Path(__file__).parents[1] / "scripts/setup_quest_collector.py"
)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def test_host_requires_explicit_non_loopback_ipv4():
    assert setup.ipv4("192.168.0.18") == "192.168.0.18"
    for value in ("0.0.0.0", "127.0.0.1", "224.0.0.1", "host", "::1"):
        with pytest.raises(setup.argparse.ArgumentTypeError):
            setup.ipv4(value)


def make_archive(path, name, *, symlink=None):
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        if symlink is not None:
            member.type = tarfile.SYMTYPE
            member.linkname = symlink
            archive.addfile(member)
        else:
            member.size = 3
            archive.addfile(member, io.BytesIO(b"sdk"))


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "include/../../escape"])
def test_archive_rejects_traversal(tmp_path, name):
    archive = tmp_path / "sdk.tar.gz"
    make_archive(archive, name)
    destination = tmp_path / "extract"
    destination.mkdir()
    with pytest.raises(ValueError):
        setup.safe_extract(archive, destination)
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("target", ["../../escape", "/tmp/escape", "../lib.so"])
def test_archive_rejects_external_symlinks(tmp_path, target):
    archive = tmp_path / "sdk.tar.gz"
    make_archive(archive, "library.so", symlink=target)
    with pytest.raises(ValueError):
        setup.safe_extract(archive, tmp_path / "extract")


def test_archive_allows_sdk_library_symlink(tmp_path):
    archive = tmp_path / "sdk.tar.gz"
    make_archive(archive, "library.so", symlink="library.so.1")
    destination = tmp_path / "extract"
    destination.mkdir()
    setup.safe_extract(archive, destination)
    assert (destination / "library.so").is_symlink()


def test_wrong_checksum_stops_install(tmp_path):
    archive = tmp_path / "sdk.tar.gz"
    archive.write_bytes(b"wrong SDK")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        setup.install_sdk(tmp_path, archive, False)
    assert not (tmp_path / "runtime").exists()


def test_env_is_private_quoted_and_idempotent(tmp_path):
    path = tmp_path / "session.env"
    values = {"CLOUDXR_HOST": "192.168.0.18", "TEST_PATH": "/path with spaces/'literal $HOME"}
    setup.write_env(path, values, False)
    first_mtime = path.stat().st_mtime_ns
    setup.write_env(path, values, False)
    assert path.stat().st_mtime_ns == first_mtime
    assert path.stat().st_mode & 0o777 == 0o600
    result = subprocess.run(["bash", "-c", 'source "$1"; printf "%s" "$TEST_PATH"', "bash", str(path)],
                            text=True, capture_output=True, check=True)
    assert result.stdout == values["TEST_PATH"]


def test_env_change_requires_opt_in_and_preserves_previous(tmp_path):
    path = tmp_path / "session.env"
    setup.write_env(path, {"CLOUDXR_HOST": "192.168.0.18"}, False)
    original = path.read_text()
    with pytest.raises(ValueError, match="--update-config"):
        setup.write_env(path, {"CLOUDXR_HOST": "192.168.0.19"}, False)
    assert path.read_text() == original
    setup.write_env(path, {"CLOUDXR_HOST": "192.168.0.19"}, True)
    backup, = tmp_path.glob("session.env.backup-*")
    assert backup.read_text() == original
    assert backup.stat().st_mode & 0o777 == 0o600


def test_browser_snapshot_does_not_modify_preview(tmp_path):
    source = tmp_path / "preview"
    source.mkdir()
    (source / "index.html").write_text("old")
    state = tmp_path / "collector"
    state.mkdir()
    snapshot = setup.snapshot_browser(state, source, False)
    (source / "index.html").write_text("new")
    setup.snapshot_browser(state, source, False)
    assert (snapshot / "simple/build/index.html").read_text() == "old"
    setup.snapshot_browser(state, source, True)
    assert (snapshot / "simple/build/index.html").read_text() == "new"
    backup, = state.glob("browser.backup-*")
    assert (backup / "simple/build/index.html").read_text() == "old"
    assert (source / "index.html").read_text() == "new"


def run_collector_wrapper(tmp_path, *extra_args):
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    launcher = scripts / "quest_collector.sh"
    launcher.write_text((Path(__file__).parents[1] / "scripts/quest_collector.sh").read_text())
    # Replace the actual collector with an argument echo, not an Isaac import.
    (scripts / "collect_quest_teleop.sh").write_text('printf "%s\\n" "$@"\n')
    for filename in ("manifest.json", "server.crt", "server.key"):
        (project / filename).touch()
    env = project / "session.env"
    setup.write_env(env, {
        "ISAACLAB_PYTHON": "/unused/python", "XR_RUNTIME_JSON": str(project / "manifest.json"),
        "CLOUDXR_RUNTIME_DIR": str(project), "CLOUDXR_HOST": "192.168.0.18",
        "CLOUDXR_CERTIFICATE": str(project / "server.crt"), "CLOUDXR_KEY": str(project / "server.key"),
        "CLOUDXR_JS_SAMPLES_DIR": str(project), "QUEST_COLLECTOR_WEB_PORT": "8443",
    }, False)
    result = subprocess.run(["bash", str(launcher), "--config", str(env), "collect", *extra_args],
                            text=True, capture_output=True, check=True)
    return result.stdout.splitlines()


def test_wrapper_defaults_without_launching_simulator(tmp_path):
    lines = run_collector_wrapper(
        tmp_path, "--dataset-format", "both", "--desktop-render"
    )
    assert "--no-auto-start" in lines
    assert "--no-desktop-render" in lines
    assert "--no-head-camera" in lines
    assert lines[-3:] == ["--dataset-format", "both", "--desktop-render"]


def test_wrapper_reward_debug_one_applies_collection_preset(tmp_path):
    lines = run_collector_wrapper(tmp_path, "--rl-reward-debug", "1")
    preset = [
        "--controller-mapping", "absolute",
        "--absolute-orientation", "downward",
        "--arm-response", "responsive",
        "--rl-task", "pick_place",
        "--rack-rollers",
        "--no-quest-camera-overlay",
        "--no-camera-preview",
        "--no-wrist-cameras",
        "--no-head-camera",
        "--no-rl-obstacle-collision",
        "--arm-orientation-weight", "0.2",
    ]
    preset_start = lines.index("--absolute-orientation") - 2
    assert lines[preset_start:preset_start + len(preset)] == preset
    assert lines[-2:] == ["--rl-reward-debug", "1"]


def test_wrapper_reward_debug_preset_allows_explicit_overrides(tmp_path):
    lines = run_collector_wrapper(
        tmp_path,
        "--rl-reward-debug=1",
        "--controller-mapping", "scaled",
        "--wrist-cameras",
        "--rl-obstacle-collision",
        "--arm-orientation-weight", "0.7",
    )
    assert lines[-6:] == [
        "--controller-mapping", "scaled",
        "--wrist-cameras",
        "--rl-obstacle-collision",
        "--arm-orientation-weight", "0.7",
    ]


@pytest.mark.parametrize("debug_args", [("--rl-reward-debug",), ("--rl-reward-debug", "0")])
def test_wrapper_arms_only_reward_debug_keeps_regular_defaults(tmp_path, debug_args):
    lines = run_collector_wrapper(tmp_path, *debug_args)
    assert "--rack-rollers" not in lines
    assert "--no-wrist-cameras" not in lines
    assert "--rl-task" not in lines
