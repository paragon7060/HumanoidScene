"""Cloud failures must never discard the only local recovery checkpoints."""

import importlib.util
import os
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "drive_backup", Path(__file__).resolve().parents[1] / "scripts/rl/drive_backup.py")
backup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backup)


class Remote:
    def __init__(self, fail=None, mutate=False):
        self.files = {}
        self.fail = fail
        self.mutate = mutate

    def upload(self, path, destination):
        self.files[destination] = path.read_bytes()
        if self.mutate and path.suffix == ".pt":
            path.write_bytes(b"changed during transfer")

    def verify(self, path, destination, digest):
        if path.name == self.fail:
            raise RuntimeError("remote checksum mismatch")
        import hashlib
        assert hashlib.md5(self.files[destination]).hexdigest() == digest


@pytest.fixture
def run(tmp_path):
    (tmp_path / "manifest.json").write_text('{"contract_hash":"test"}')
    for iteration in (0, 9, 10, 100):
        path = tmp_path / f"model_{iteration}.pt"
        path.write_bytes(f"checkpoint {iteration}".encode())
        os.utime(path, (1, 1))
    return tmp_path


def test_verified_backup_keeps_two_numerically_latest_and_remote_history(run):
    remote = Remote()
    assert backup.backup_once(run, "drive:run", remote) == ["model_0.pt", "model_9.pt"]
    assert {p.name for p in run.glob("*.pt")} == {"model_10.pt", "model_100.pt"}
    assert len(remote.files) == 5


@pytest.mark.parametrize("failed", ["manifest.json", "model_100.pt"])
def test_metadata_or_checkpoint_verification_failure_preserves_all_local_files(run, failed):
    with pytest.raises(RuntimeError):
        backup.backup_once(run, "drive:run", Remote(fail=failed))
    assert len(list(run.glob("*.pt"))) == 4


def test_changed_source_is_never_pruned(run):
    with pytest.raises(RuntimeError, match="Source changed"):
        backup.backup_once(run, "drive:run", Remote(mutate=True))
    assert len(list(run.glob("*.pt"))) == 4


def test_fresh_checkpoint_does_not_replace_two_verified_recovery_copies(run):
    os.utime(run / "model_100.pt", None)
    removed = backup.backup_once(run, "drive:run", Remote())
    assert removed == ["model_0.pt"]
    assert len(list(run.glob("*.pt"))) == 3


def test_live_logs_and_unrelated_files_are_not_uploaded(run):
    (run / "events.out.tfevents.test").write_text("active log")
    (run / ".env").write_text("secret")
    (run / "model_999.pt").symlink_to(run / ".env")
    remote = Remote()
    backup.backup_once(run, "drive:run", remote)
    assert not any("999" in name or ".env" in name or "tfevents" in name for name in remote.files)
    backup.backup_once(run, "drive:run", remote, finished=True)
    assert "drive:run/events.out.tfevents.test" in remote.files


def test_manifest_required_before_any_transfer(tmp_path):
    remote = Remote()
    with pytest.raises(ValueError, match="manifest"):
        backup.backup_once(tmp_path, "drive:run", remote)
    assert not remote.files


def test_dppo_logs_wait_for_finished_and_verified_retention_keeps_two(run):
    for index in range(4):
        path = run / f"checkpoint_{index:08d}.pt"
        path.write_bytes(bytes([index]))
    (run / "status.json").write_text('{"status":"complete"}')
    (run / "resources.jsonl").write_text('{"process_mib":30000}\n')
    remote = Remote()
    backup.backup_once(run, "drive:run", remote, min_age=0)
    assert "drive:run/status.json" not in remote.files
    assert len(list(run.glob("checkpoint_*.pt"))) == 2
    assert len([p for p in remote.files if "checkpoint_" in p]) == 4
    backup.backup_once(run, "drive:run", remote, min_age=0, finished=True)
    assert "drive:run/status.json" in remote.files and "drive:run/resources.jsonl" in remote.files


def test_real_rclone_local_backend_roundtrip_and_pruning(tmp_path):
    """Exercise actual copyto/lsjson hash formats without network or credentials."""
    binary = Path(__file__).resolve().parents[1] / ".external/rclone/rclone"
    if not binary.exists():
        pytest.skip("Optional project-local rclone is not installed")
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text("{}")
    for iteration in (0, 1, 2):
        (run / f"model_{iteration}.pt").write_bytes(bytes([iteration]) * 1024)
    wrapper = tmp_path / "rclone.sh"
    wrapper.write_text('#!/bin/sh\nexec "' + str(binary) + '" --config /dev/null "$@"\n')
    destination = tmp_path / "remote"
    assert backup.backup_once(run, str(destination), backup.Rclone(wrapper), min_age=0) == ["model_0.pt"]
    assert (destination / "model_0.pt").read_bytes() == bytes(1024)
    assert len(list(run.glob("*.pt"))) == 2
