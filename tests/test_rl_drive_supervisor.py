"""Exercise lifecycle/backup ordering with real CPU-only child processes."""

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from kuavo_isaaclab_scene.rl.runners.checkpoints import AtomicCheckpointMixin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/rl"))
spec = importlib.util.spec_from_file_location("train_with_drive", ROOT / "scripts/rl/train_with_drive.py")
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)
sys.path.pop(0)


def test_final_logs_uploaded_only_after_child_exit_and_retry(tmp_path):
    script = """
import pathlib, time
p = pathlib.Path(__import__('sys').argv[1]) / 'train_test'
p.mkdir()
for name in ('manifest.json', 'env.yaml', 'agent.yaml'):
    (p / name).write_text('{}')
print('training output', flush=True)
time.sleep(.3)
(p / 'writers_done').touch()
print('final output', flush=True)
"""
    calls = []
    def backup(source, finished):
        calls.append(finished)
        if finished:
            assert (source / "writers_done").exists()
            assert "final output" in (source / "console.log").read_text()
            assert json.loads((source / "verification.json").read_text())["training_exit_code"] == 0
            if calls.count(True) == 1:
                raise RuntimeError("temporary network failure")
        return []
    result = supervisor.supervise([sys.executable, "-c", script, str(tmp_path)], tmp_path,
                                  os.environ.copy(), backup, interval=.05, poll=.01, min_free_bytes=0)
    assert result == 0
    assert False in calls and calls[-2:] == [True, True]
    assert json.loads((tmp_path / "status.json").read_text())["final_upload_verified"]


def test_failed_initialization_still_uploads_closed_console(tmp_path):
    calls = []
    def backup(source, finished):
        assert finished
        assert "startup failed" in (source / "console.log").read_text()
        calls.append(source)
        return []
    result = supervisor.supervise([sys.executable, "-c", "print('startup failed'); raise SystemExit(3)"],
                                  tmp_path, os.environ.copy(), backup, poll=.01, min_free_bytes=0)
    assert result == 3 and len(calls) == 1


def test_dppo_prefix_and_masked_kit_failure_are_reported(tmp_path):
    script = """
import pathlib, sys, json
p = pathlib.Path(sys.argv[1]) / 'dppo_test'
p.mkdir()
for name in ('manifest.json', 'env.yaml', 'agent.yaml'):
    (p / name).write_text('{}')
(p / 'status.json').write_text(json.dumps({'status': 'failed', 'error': 'test failure'}))
print('Kit returned zero despite failed training')
"""
    def backup(source, finished):
        if finished:
            assert source.name == "dppo_test"
            assert json.loads((source / "verification.json").read_text())["training_exit_code"] == 1
        return []
    result = supervisor.supervise([sys.executable, "-c", script, str(tmp_path)], tmp_path,
        os.environ.copy(), backup, poll=.01, min_free_bytes=0, run_prefix="dppo_", require_run_status=True)
    assert result == 1


def test_low_disk_stops_owned_child_then_finalizes_logs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    ready = tmp_path / "train_test" / "ready"
    monkeypatch.setattr(supervisor.shutil, "disk_usage",
                        lambda path: SimpleNamespace(free=0 if ready.exists() else 100))
    script = """
import pathlib, signal, sys, time
p = pathlib.Path(sys.argv[1]) / 'train_test'
p.mkdir()
def stop(number, frame):
    print('child stopped cleanly', flush=True)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
(p / 'ready').touch()
while True:
    time.sleep(.01)
"""
    def backup(source, finished):
        assert finished
        assert "child stopped cleanly" in (source / "console.log").read_text()
        return []
    assert supervisor.supervise([sys.executable, "-c", script, str(tmp_path)], tmp_path,
                                os.environ.copy(), backup, poll=.01, min_free_bytes=1) == 0
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["stop_reason"] == "low_disk_space"
    assert status["final_upload_verified"]


@pytest.mark.parametrize("usage,free,other_gpu,reason", [
    (30720, 12000, False, None),
    (35000, 12000, False, "child_gpu_memory_limit"),
    (30720, 7000, False, "gpu_free_memory_reserve"),
    (1000, 12000, True, "child_created_context_on_unselected_gpu"),
])
def test_gpu_budget_tracks_only_its_child_and_moves_log(tmp_path, monkeypatch, usage, free, other_gpu, reason):
    module_spec = importlib.util.spec_from_file_location("gpu_budget", ROOT / "scripts/rl/gpu_budget.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    def query(command, **kwargs):
        if "--query-gpu=uuid,memory.free" in command:
            return f"GPU-zero, {free}"
        gpu = "GPU-one" if other_gpu else "GPU-zero"
        return f"GPU-zero, 999, 60000\n{gpu}, 123, {usage}\n"
    monkeypatch.setattr(module.subprocess, "check_output", query)
    monitor = module.GpuBudget(tmp_path, 0, 34816, 8192, 60, "sac_")
    assert monitor(123) == reason
    run = tmp_path / "sac_test"
    run.mkdir()
    (run / "manifest.json").write_text("{}")
    assert monitor(123) == reason
    assert not (tmp_path / "resources.jsonl").exists()
    rows = [json.loads(line) for line in (run / "resources.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and all(row["process_peak_mib"] == usage for row in rows)


@pytest.mark.parametrize("interrupt", [False, True])
def test_atomic_checkpoint_does_not_expose_partial_writes(tmp_path, interrupt):
    destination = tmp_path / "model_4.pt"
    destination.write_bytes(b"previous complete checkpoint")
    class Serializer:
        def save(self, path, infos=None):
            Path(path).write_bytes(b"partial")
            assert destination.read_bytes() == b"previous complete checkpoint"
            if interrupt:
                raise KeyboardInterrupt
            Path(path).write_bytes(b"complete")
    class Runner(AtomicCheckpointMixin, Serializer):
        pass
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            Runner().save(destination)
        assert destination.read_bytes() == b"previous complete checkpoint"
    else:
        Runner().save(destination)
        assert destination.read_bytes() == b"complete"
    assert list(tmp_path.iterdir()) == [destination]
