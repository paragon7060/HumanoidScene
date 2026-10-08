"""Immutable model identity and continuously valid reader paths during moves."""
import hashlib
import importlib.util
from pathlib import Path
import threading

import pytest

PATH = Path(__file__).resolve().parents[1]/"scripts/rl/relocate_protected_dev_models.py"
SPEC = importlib.util.spec_from_file_location("relocate_protected_dev_models", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def example(tmp_path):
    source_root = tmp_path/"ram";source_root.mkdir()
    destination_root = tmp_path/"disk";destination_root.mkdir()
    role = "URDF_test_DEV4"
    folder = source_root/("CPU_"+role+"_frozen_20261009_070000");folder.mkdir()
    source = folder/"checkpoint_00004828.pt"
    value = bytes(range(256))*4096
    source.write_bytes(value);source.chmod(0o400)
    entry = dict(source_run="/owned/batch_sac_20261009_071056_f6e97e",role=role,source_commit="a"*40,wave=4)
    proof = entry|dict(protected_checkpoint=str(source),split="validation",critic_updates=4828,
        checkpoint_SHA256=hashlib.sha256(value).hexdigest(),actual_model_finite=True,
        exact_actual_runtime_goal_contract_verified=True)
    return source_root,destination_root,source,value,proof,entry


def test_atomic_folder_exchange_preserves_reader_path_and_exact_bytes(tmp_path):
    ram,disk,source,value,proof,entry = example(tmp_path)
    errors = [];stop = threading.Event();ready = threading.Event()
    def read():
        while not stop.is_set():
            try:
                assert source.read_bytes() == value
                ready.set()
            except BaseException as error:
                errors.append(error);stop.set()
    reader = threading.Thread(target=read);reader.start();assert ready.wait(5)
    try:
        result = MODULE.relocate(proof,entry,ram,disk,minimum_age=0)
    finally:
        stop.set();reader.join(5)
    assert not errors
    assert result["state"] == "relocated"
    assert source.parent.is_symlink() and not source.is_symlink()
    assert source.is_file() and source.read_bytes() == value
    assert Path(result["destination"]).read_bytes() == value
    assert source.stat().st_mode & 0o222 == 0
    assert not list(ram.glob("*.exchange-*"))
    again = MODULE.relocate(proof,entry,ram,disk,minimum_age=0)
    assert again["state"] == "already_relocated" and again["SHA256"] == result["SHA256"]


@pytest.mark.parametrize("change", ["checksum", "writable", "source", "extra_file"])
def test_reject_invalid_proof_or_mutable_source_without_removal(tmp_path, change):
    ram,disk,source,value,proof,entry = example(tmp_path)
    if change == "checksum":proof["checkpoint_SHA256"] = "0"*64
    if change == "writable":source.chmod(0o600)
    if change == "source":proof["source_run"] = "/another/run"
    if change == "extra_file":(source.parent/"unrelated.txt").write_text("keep")
    with pytest.raises(ValueError):MODULE.relocate(proof,entry,ram,disk,minimum_age=0)
    assert source.read_bytes() == value and not source.parent.is_symlink()
    assert list(disk.iterdir()) == []


def test_destination_conflict_is_retained_without_touching_source(tmp_path):
    ram,disk,source,value,proof,entry = example(tmp_path)
    target = disk/Path(entry["source_run"]).name/source.parent.name/source.name
    target.parent.mkdir(parents=True);target.write_bytes(b"other data");target.chmod(0o400)
    with pytest.raises(ValueError):MODULE.relocate(proof,entry,ram,disk,minimum_age=0)
    assert source.read_bytes() == value and target.read_bytes() == b"other data"
    assert not source.parent.is_symlink()


def test_too_recent_source_waits_without_copy_or_exchange(tmp_path):
    ram,disk,source,value,proof,entry = example(tmp_path)
    result = MODULE.relocate(proof,entry,ram,disk,minimum_age=3600)
    assert result["state"] == "waiting_immutable_age"
    assert source.read_bytes() == value and list(disk.iterdir()) == []
