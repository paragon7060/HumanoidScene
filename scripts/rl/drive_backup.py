#!/usr/bin/env python3
"""Archive one RL run with rclone; prune only checksum-verified checkpoints.

No Isaac/PyTorch imports, GPU use, mounts, or remote deletions. Run separately
from training. Active logs are deliberately excluded; --finished includes them.
"""

import argparse
import fcntl
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = re.compile(r"(model|checkpoint)_(\d+)\.pt$")
METADATA = ("manifest.json", "env.yaml", "agent.yaml", "verification.json")


def signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def md5(path):
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Rclone:
    def __init__(self, wrapper):
        self.wrapper = str(wrapper)

    def call(self, *args):
        result = subprocess.run(
            ["bash", self.wrapper, *map(str, args), "--retries", "3",
             "--contimeout", "15s", "--timeout", "60s"],
            check=True, capture_output=True, text=True, timeout=600,
        )
        return result.stdout

    def upload(self, source, destination):
        self.call("copyto", source, destination, "--checksum", "--immutable")

    def verify(self, source, destination, digest):
        record = json.loads(self.call("lsjson", destination, "--stat", "--hash", "--hash-type", "md5"))
        hashes = {key.lower(): value.lower() for key, value in record.get("Hashes", {}).items()}
        if (record.get("Size") != source.stat().st_size
                or hashes.get("md5") != digest):
            raise RuntimeError(f"Remote size/MD5 verification failed: {source.name}")


def archive_file(path, destination, remote):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Refusing non-regular file: {path}")
    before = signature(path)
    digest = md5(path)
    remote_path = f"{destination}/{path.name}"
    remote.upload(path, remote_path)
    remote.verify(path, remote_path, digest)
    if signature(path) != before:
        raise RuntimeError(f"Source changed during upload; not pruning: {path.name}")
    print(f"[Drive] verified {path.name}", flush=True)
    return before


def backup_once(source, destination, remote, keep=2, min_age=120, finished=False):
    if keep < 1 or min_age < 0:
        raise ValueError("keep must be >= 1 and min_age >= 0")
    # A run's contract must be available in Drive alongside its checkpoints.
    manifest = source / "manifest.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("Select one run directory containing manifest.json")
    json.loads(manifest.read_text())
    for name in METADATA:
        path = source / name
        if path.exists():
            archive_file(path, destination, remote)
    checkpoints = sorted(
        (p for p in source.iterdir() if CHECKPOINT.fullmatch(p.name)
         and p.is_file() and not p.is_symlink()),
        key=lambda p: (CHECKPOINT.fullmatch(p.name)[1], int(CHECKPOINT.fullmatch(p.name)[2])),
    )
    verified = {}
    for path in checkpoints:
        if time.time() - path.stat().st_mtime >= min_age:
            verified[path] = archive_file(path, destination, remote)
    if finished:
        for path in sorted(source.iterdir()):
            if (path.name.startswith("events.out.tfevents.")
                    or path.name in ("metrics.jsonl", "metrics.json", "status.json", "resources.jsonl")
                    or path.suffix == ".log"):
                archive_file(path, destination, remote)
    # A failed copy or verification above exits before ANY local pruning.
    removed = []
    for family in ("model", "checkpoint"):
        members = [p for p in checkpoints if CHECKPOINT.fullmatch(p.name)[1] == family]
        protected = set([p for p in members if p in verified][-keep:])
        for path in members[:-keep]:
            if (path in verified and path not in protected and not path.is_symlink()
                    and signature(path) == verified[path]):
                path.unlink()
                removed.append(path.name)
                print(f"[Drive] pruned verified local checkpoint {path.name}", flush=True)
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--remote-root", default="gdrive:HumanoidScene-RL")
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--min-age", type=int, default=120)
    parser.add_argument("--watch", type=int, default=0, metavar="SECONDS")
    parser.add_argument("--finished", action="store_true", help="Include logs only after the writer has stopped")
    args = parser.parse_args()
    source = args.run_dir.expanduser().resolve()
    if not source.is_dir() or not (source / "manifest.json").is_file():
        parser.error("--run-dir must be one existing run directory containing manifest.json")
    if args.keep < 1 or args.min_age < 0 or (args.watch and args.watch < 30):
        parser.error("--keep >= 1, --min-age >= 0; --watch is 0 or >= 30 seconds")
    if args.finished and args.watch:
        parser.error("--finished requires the run to be stopped; do not combine with --watch")
    if not re.fullmatch(r"[\w-]+:[^\r\n]+", args.remote_root) or ".." in args.remote_root.split("/"):
        parser.error("--remote-root must name an rclone remote and dedicated folder")
    destination = f"{args.remote_root.rstrip('/')}/{source.name}"
    remote = Rclone(ROOT / "scripts/rl/gdrive.sh")
    with (source / ".drive-backup.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another backup process already owns this run")
        while True:
            try:
                backup_once(source, destination, remote, args.keep, args.min_age, args.finished)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                print(f"[Drive] backup failed; remaining local files retained: {error}", flush=True)
                if not args.watch:
                    raise SystemExit(1) from error
            if not args.watch:
                break
            time.sleep(args.watch)


if __name__ == "__main__":
    main()
