"""Move explicit immutable DEV model copies from RAM storage to local disk.

Only SHA256-proven, read-only model copies are eligible. Their original folder
paths remain valid through an atomic directory/symlink exchange. Training
checkpoints, replay, HDF, active logs, and other users' files are never inputs.
This is local relocation, not Drive backup or checkpoint retention/pruning.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def atomic_exchange(left, right):
    library = ctypes.CDLL(None, use_errno=True)
    function = library.renameat2
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                         ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    # AT_FDCWD=-100; Linux RENAME_EXCHANGE=2, verified in system headers.
    result = function(-100, os.fsencode(left), -100, os.fsencode(right), 2)
    if result:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def read_owned_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid():
        raise ValueError("Owned regular proof/config required")
    before = path.stat()
    value = json.loads(path.read_text())
    if identity(before) != identity(path.stat()):
        raise ValueError("Proof changed while reading")
    return value


def relocate(proof, entry, source_root, destination_root, *, minimum_age=120):
    if not re.fullmatch(r"batch_sac_\d{8}_\d{6}_[a-f0-9]{6}", Path(entry["source_run"]).name) \
            or not re.fullmatch(r"URDF_[A-Za-z0-9_]+_DEV\d+", entry["role"]):
        raise ValueError("Expected explicit batch run and DEV role")
    if proof.get("source_run") != entry["source_run"] or proof.get("role") != entry["role"] \
            or proof.get("source_commit") != entry["source_commit"] \
            or proof.get("wave") != entry["wave"] or proof.get("split") != "validation" \
            or not proof.get("actual_model_finite") or not proof.get("exact_actual_runtime_goal_contract_verified"):
        raise ValueError("Matching DEV source/role/model proof required")
    source = Path(proof["protected_checkpoint"])
    if source.parent.parent != source_root or not source.parent.name.startswith("CPU_"+entry["role"]+"_frozen_") \
            or source.name != f"checkpoint_{proof['critic_updates']:08d}.pt" \
            or not re.fullmatch(r"[a-f0-9]{64}", proof["checkpoint_SHA256"]):
        raise ValueError("Outside explicit protected-model namespace")
    if source.is_symlink() or not source.is_file():
        raise ValueError("Read-only regular model file required")
    before = source.stat()
    if before.st_uid != os.getuid() or before.st_mode & 0o222 or before.st_size <= 0:
        raise ValueError("Model owner or immutable permissions differ")
    target_directory = destination_root/Path(entry["source_run"]).name/source.parent.name
    target = target_directory/source.name
    if source.parent.is_symlink():
        if source.parent.resolve() != target_directory.resolve() or target.is_symlink() \
                or not target.is_file() or sha256(target) != proof["checkpoint_SHA256"]:
            raise ValueError("Existing relocation identity differs")
        return dict(state="already_relocated", source=str(source), destination=str(target),
                    bytes=before.st_size, SHA256=proof["checkpoint_SHA256"])
    if source.parent.stat().st_uid != os.getuid() or sorted(p.name for p in source.parent.iterdir()) != [source.name]:
        raise ValueError("Expected one owned model-only directory")
    if time.time()-before.st_mtime < minimum_age:
        return dict(state="waiting_immutable_age", source=str(source))
    if sha256(source) != proof["checkpoint_SHA256"] or identity(before) != identity(source.stat()):
        raise ValueError("Source checksum or file identity changed")
    if target_directory.parent.is_symlink():
        raise ValueError("Destination run namespace cannot be a symlink")
    target_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if target_directory.is_symlink() or target_directory.stat().st_uid != os.getuid():
        raise ValueError("Destination must be an owned real directory")
    if target.exists():
        if target.is_symlink() or target.stat().st_uid != os.getuid() \
                or target.stat().st_size != before.st_size or sha256(target) != proof["checkpoint_SHA256"]:
            raise ValueError("Destination conflict; preserve both inputs")
    else:
        temporary = target.with_name(target.name+".copy-"+uuid.uuid4().hex)
        try:
            with source.open("rb") as reader, temporary.open("xb") as writer:
                shutil.copyfileobj(reader, writer, 1024*1024)
                writer.flush(); os.fsync(writer.fileno())
            shutil.copystat(source, temporary)
            if temporary.stat().st_size != before.st_size or sha256(temporary) != proof["checkpoint_SHA256"] \
                    or identity(source.stat()) != identity(before):
                raise ValueError("Copy verification failed; original remains")
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
    if sha256(target) != proof["checkpoint_SHA256"] or identity(source.stat()) != identity(before):
        raise ValueError("Input/copy changed before exchange")
    # Both directory entries live on the source filesystem. Exchange preserves
    # the original public path at every instant, including for active readers.
    exchange = source.parent.with_name(source.parent.name+".exchange-"+uuid.uuid4().hex)
    os.symlink(target_directory, exchange, target_is_directory=True)
    try:
        atomic_exchange(source.parent, exchange)
    except BaseException:
        exchange.unlink()
        raise
    old_file = exchange/source.name
    if not source.is_file() or source.is_symlink() or source.stat().st_size != before.st_size \
            or sha256(source) != proof["checkpoint_SHA256"] or identity(old_file.stat()) != identity(before):
        # Preserve the old copy for recovery; do not unlink on failed verification.
        raise ValueError("Exchange verification failed; retained old directory")
    old_file.unlink(); exchange.rmdir()
    return dict(state="relocated", source=str(source), destination=str(target),
                bytes=before.st_size, SHA256=proof["checkpoint_SHA256"],
                original_path_continuously_available=True, original_model_bits_preserved=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--watch", type=int, default=0)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.watch and args.watch<30:
        parser.error("CPU-only worker and watch>=30 seconds required")
    config = read_owned_json(args.config)
    source_root = Path(config["source_root"])
    destination_root = Path(config["destination_root"])
    if source_root.is_symlink() or destination_root.is_symlink() or source_root == destination_root:
        raise ValueError("Distinct real local storage roots required")
    destination_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    for root in (source_root, destination_root):
        if root.stat().st_uid != os.getuid():
            raise ValueError("Storage roots must belong to this user")
    if source_root.stat().st_dev == destination_root.stat().st_dev:
        raise ValueError("Relocation requires different filesystems")
    proof_paths = [x["proof_path"] for x in config["entries"]]
    if len(proof_paths) != len(set(proof_paths)):
        raise ValueError("Duplicate proof entries")
    status_path = args.config.with_suffix(".status.json")
    with args.config.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger = read_owned_json(status_path).get("relocated", {}) if status_path.exists() else {}
        while True:
            waiting, errors = [], []
            for entry in config["entries"]:
                if entry["proof_path"] in ledger:
                    continue
                try:
                    path = Path(entry["proof_path"])
                    if not path.exists():
                        waiting.append(entry["role"]); continue
                    if shutil.disk_usage(destination_root).free < config.get("minimum_disk_free_bytes", 20*2**30):
                        waiting.append(entry["role"]); continue
                    result = relocate(read_owned_json(path), entry, source_root, destination_root)
                    if result["state"] == "waiting_immutable_age":
                        waiting.append(entry["role"]); continue
                    ledger[entry["proof_path"]] = result
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    errors.append(dict(role=entry["role"],error=str(error)))
            status = dict(recorded_at=datetime.now().astimezone().isoformat(), worker_pid=os.getpid(),
                phase="watching" if args.watch and (waiting or errors) else "done",
                relocated=ledger, relocated_files=len(ledger), relocated_bytes=sum(x["bytes"] for x in ledger.values()),
                pending_roles=waiting, errors=errors, GPU_mask_empty=True,
                local_relocation_NOT_Drive_backup=True, no_active_training_checkpoint_replay_HDF_or_logs_touched=True,
                no_process_signals_or_other_user_changes=True)
            temporary = status_path.with_suffix(".atomic.tmp")
            temporary.write_text(json.dumps(status,indent=2)+"\n");temporary.replace(status_path)
            print(json.dumps({k:status[k] for k in ("recorded_at","phase","relocated_files","relocated_bytes","errors")}),flush=True)
            if not args.watch or not waiting and not errors:
                return
            time.sleep(args.watch)


if __name__ == "__main__":
    main()
