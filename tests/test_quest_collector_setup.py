import ast
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


def test_host_is_local_detects_addresses_this_pc_does_not_own():
    assert setup.host_is_local("127.0.0.1") is True
    # 192.0.2.0/24 is TEST-NET-1 (RFC 5737) and is never assigned to a real PC.
    assert setup.host_is_local("192.0.2.1") is False


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


def run_wrapper(tmp_path, command, *extra_args, host="192.168.0.18", extra_env=None):
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
        "CLOUDXR_RUNTIME_DIR": str(project), "CLOUDXR_HOST": host,
        "CLOUDXR_CERTIFICATE": str(project / "server.crt"), "CLOUDXR_KEY": str(project / "server.key"),
        "CLOUDXR_JS_SAMPLES_DIR": str(project), "QUEST_COLLECTOR_WEB_PORT": "8443",
        **(extra_env or {}),
    }, False)
    return subprocess.run(["bash", str(launcher), "--config", str(env), command, *extra_args],
                          text=True, capture_output=True)


def run_collector_wrapper(tmp_path, *extra_args):
    result = run_wrapper(tmp_path, "collect", *extra_args)
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


def test_check_stops_when_configured_ip_left_this_pc(tmp_path):
    result = run_wrapper(tmp_path, "check", host="192.0.2.1",
                         extra_env={"LEROBOT_PYTHON": "/unused/lerobot"})
    assert result.returncode == 1
    assert "[ERROR] Configured CLOUDXR_HOST 192.0.2.1 is not assigned" in result.stderr
    # The suggested command carries over the custom arguments that --update-config needs.
    assert "--isaaclab-python /unused/python" in result.stderr
    assert "--lerobot-python /unused/lerobot" in result.stderr
    assert "--update-config" in result.stderr


def test_info_only_warns_about_stale_ip(tmp_path):
    result = run_wrapper(tmp_path, "info", host="192.0.2.1")
    assert result.returncode == 0
    assert "Quest page: https://192.0.2.1:8443" in result.stdout
    assert "[WARN] Configured CLOUDXR_HOST 192.0.2.1 is not assigned" in result.stderr


def test_check_passes_host_test_for_local_ip(tmp_path):
    # The empty test certificate fails later; only the host test is asserted here.
    result = run_wrapper(tmp_path, "check", host="127.0.0.1")
    assert "is not assigned" not in result.stderr


def test_collect_does_not_require_host_test(tmp_path):
    result = run_wrapper(tmp_path, "collect", host="192.0.2.1")
    assert result.returncode == 0, result.stderr
    assert "is not assigned" not in result.stderr


def test_wrapper_defaults_without_launching_simulator(tmp_path):
    lines = run_collector_wrapper(
        tmp_path, "--dataset-format", "both", "--desktop-render"
    )
    assert "--no-auto-start" in lines
    assert "--no-desktop-render" in lines
    assert "--no-head-camera" in lines
    initial = lines.index("--initial-state")
    assert lines[initial:initial + 2] == ["--initial-state", "s63_leju_vr_collect_01"]
    assert lines[-3:] == ["--dataset-format", "both", "--desktop-render"]


def test_wrapper_initial_state_can_be_overridden(tmp_path):
    lines = run_collector_wrapper(
        tmp_path, "--initial-state", "s63_leju_ready_01"
    )
    assert lines[-2:] == ["--initial-state", "s63_leju_ready_01"]


def test_wrapper_reward_debug_one_applies_collection_preset(tmp_path):
    lines = run_collector_wrapper(tmp_path, "--rl-reward-debug", "1")
    assert "--initial-state" not in lines
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
        "--arm-orientation-weight", "0.5",
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


def test_wrapper_reward_debug_two_inherits_whole_body_preset_with_rollers(tmp_path):
    lines = run_collector_wrapper(tmp_path, "--rl-reward-debug", "2")
    assert "--initial-state" not in lines
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
        "--arm-orientation-weight", "0.5",
    ]
    preset_start = lines.index("--absolute-orientation") - 2
    assert lines[preset_start:preset_start + len(preset)] == preset
    assert lines[-2:] == ["--rl-reward-debug", "2"]


@pytest.mark.parametrize("debug_args", [("--rl-reward-debug",), ("--rl-reward-debug", "0")])
def test_wrapper_arms_only_reward_debug_keeps_regular_defaults(tmp_path, debug_args):
    lines = run_collector_wrapper(tmp_path, *debug_args)
    assert "--rack-rollers" not in lines
    assert "--no-wrist-cameras" not in lines
    assert "--rl-task" not in lines


def test_direct_quest_teleop_holds_the_nonzero_reset_torso_pose():
    source = (
        Path(__file__).parents[1]
        / "src/kuavo_isaaclab_scene/teleop/collect_quest_teleop.py"
    ).read_text()
    assert "def reset_body_mapper_from_robot()" in source
    assert "robot.data.joint_pos[0, body_joint_ids]" in source
    # One definition-time synchronization and one synchronization after env.reset().
    assert source.count("reset_body_mapper_from_robot()") >= 3


def test_direct_quest_teleop_imports_body_joints_used_during_startup():
    source = (
        Path(__file__).parents[1]
        / "src/kuavo_isaaclab_scene/teleop/collect_quest_teleop.py"
    ).read_text()
    tree = ast.parse(source)
    body_import = next(
        node for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "teleop_body"
    )
    assert "BODY_JOINTS" in {alias.name for alias in body_import.names}


def test_direct_quest_teleop_applies_and_records_named_initial_state():
    source = (
        Path(__file__).parents[1]
        / "src/kuavo_isaaclab_scene/teleop/collect_quest_teleop.py"
    ).read_text()
    assert "initial_state_metadata = configure_initial_state(cfg, args_cli)" in source
    assert '"initial_state_name"' in source
    assert '"initial_state": initial_state_metadata or {}' in source
