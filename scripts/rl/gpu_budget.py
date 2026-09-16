"""Read-only process GPU accounting shared by supervised experiments."""

from datetime import datetime
import json
import subprocess
import time


class GpuBudget:
    """Read-only monitoring; returns a stop reason for this supervisor's child only."""
    def __init__(self, parent, gpu, limit_mib, reserve_mib, max_seconds, run_prefix="dppo_"):
        self.parent, self.gpu = parent, gpu
        self.run_prefix = run_prefix
        self.limit, self.reserve, self.max_seconds = limit_mib, reserve_mib, max_seconds
        self.started = time.monotonic()
        self.peak = 0
        self.path = parent / "resources.jsonl"

    def __call__(self, pid):
        run = next(iter(self.parent.glob(self.run_prefix + "*/manifest.json")), None)
        if run is not None and self.path.parent == self.parent:
            destination = run.parent / self.path.name
            if self.path.exists():
                self.path.rename(destination)
            self.path = destination
        gpu_row = subprocess.check_output(
            ["nvidia-smi", f"--id={self.gpu}", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"],
            text=True, timeout=15).strip().split(",")
        uuid, free = gpu_row[0].strip(), int(gpu_row[1])
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
            text=True, timeout=15).splitlines()
        usage, wrong_gpu = 0, False
        for row in rows:
            fields = [x.strip() for x in row.split(",")]
            if len(fields) == 3 and fields[1] == str(pid):
                wrong_gpu |= fields[0] != uuid
                usage += int(fields[2]) if fields[2].isdigit() else 0
        self.peak = max(self.peak, usage)
        with self.path.open("a") as stream:
            stream.write(json.dumps(dict(time=datetime.now().astimezone().isoformat(), training_pid=pid,
                gpu=self.gpu, process_mib=usage, process_peak_mib=self.peak, gpu_free_mib=free)) + "\n")
        if wrong_gpu:
            return "child_created_context_on_unselected_gpu"
        if usage > self.limit:
            return "child_gpu_memory_limit"
        if free < self.reserve:
            return "gpu_free_memory_reserve"
        if time.monotonic() - self.started > self.max_seconds:
            return "maximum_test_duration"
        return None
