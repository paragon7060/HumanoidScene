"""Delete only explicit unused raw data with matching immutable Drive copies."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts/rl"))
SPEC = importlib.util.spec_from_file_location("offload_closed_sac_payloads",ROOT/"scripts/rl/offload_closed_sac_payloads.py")
MODULE = importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)


class Remote:
    def __init__(self):self.files={};self.verify_count=0
    def upload(self,path,destination):self.files[destination]=path.read_bytes()
    def verify(self,path,destination,digest):
        self.verify_count+=1
        value=self.files[destination]
        if len(value)!=path.stat().st_size or hashlib.md5(value).hexdigest()!=digest:
            raise RuntimeError("checksum differs")


def example(tmp_path,monkeypatch):
    parent=tmp_path/"closed";parent.mkdir()
    run=parent/"batch_sac_20261009_070000_abcdef";run.mkdir()
    launch=dict(run=str(run),command=["python","trainer.py","--output-dir",str(run)],backup_scope="checkpoint_contract_logs_only")
    status=dict(run_dir=str(run),phase="finished",training_exit_code=0,final_upload_verified=True,
        training_pid=2**30,supervisor_pid=2**30+1)
    (parent/"launch.json").write_text(json.dumps(launch));(parent/"status.json").write_text(json.dumps(status))
    (run/"manifest.json").write_text("{}")
    for name in MODULE.PAYLOADS:(run/name).write_bytes(name.encode()*10)
    for name in ("checkpoint_00001000.pt","console.log","policy.mp4"):(run/name).write_bytes(b"keep")
    monkeypatch.setattr(MODULE,"process_inventory",lambda:([],set()))
    return parent,run,launch,status


def test_verified_raw_supplement_preserves_original_scope_metadata_and_other_files(tmp_path,monkeypatch):
    parent,run,launch,status=example(tmp_path,monkeypatch)
    metadata={p:p.read_bytes() for p in (parent/"launch.json",parent/"status.json")}
    originals={name:(run/name).read_bytes() for name in MODULE.PAYLOADS}
    remote=Remote();result=MODULE.offload(parent,"existing:HumanoidScene-RL",remote)
    assert len(result["removed"])==2 and remote.verify_count==4
    for name,value in originals.items():
        assert not (run/name).exists()
        assert remote.files[f"existing:HumanoidScene-RL/{run.name}/{name}"]==value
    for name in ("checkpoint_00001000.pt","console.log","policy.mp4"):assert (run/name).read_bytes()==b"keep"
    assert all(p.read_bytes()==value for p,value in metadata.items())
    journal=json.loads((run/"supplemental_raw_Drive_offload.json").read_text())
    assert journal["original_backup_scope"]=="checkpoint_contract_logs_only"
    assert all(x["local_removal_complete"] for x in journal["files"])
    assert "existing:" not in json.dumps(journal)


def test_remote_verification_failure_keeps_local_payloads(tmp_path,monkeypatch):
    parent,run,_,_=example(tmp_path,monkeypatch)
    class Bad(Remote):
        def verify(self,*args):raise RuntimeError("unverified")
    with pytest.raises(RuntimeError):MODULE.offload(parent,"existing:HumanoidScene-RL",Bad())
    assert all((run/name).exists() for name in MODULE.PAYLOADS)


def test_runtime_file_reference_is_retained_without_upload(tmp_path,monkeypatch):
    parent,run,_,_=example(tmp_path,monkeypatch)
    monkeypatch.setattr(MODULE,"process_inventory",lambda:([str(run).encode()],set()))
    remote=Remote();result=MODULE.offload(parent,"existing:HumanoidScene-RL",remote)
    assert len(result["retained"])==2 and not remote.files


def test_open_inode_and_protected_inputs_are_retained(tmp_path,monkeypatch):
    parent,run,_,_=example(tmp_path,monkeypatch)
    value=(run/MODULE.PAYLOADS[0]).stat()
    monkeypatch.setattr(MODULE,"process_inventory",lambda:([],{(value.st_dev,value.st_ino)}))
    result=MODULE.offload(parent,"existing:HumanoidScene-RL",Remote(),protected=[run/MODULE.PAYLOADS[1]])
    assert not result["removed"] and len(result["retained"])==2


def test_same_bytes_rewrite_during_upload_is_not_deleted(tmp_path,monkeypatch):
    parent,run,_,_=example(tmp_path,monkeypatch)
    class Changed(Remote):
        def upload(self,path,destination):
            super().upload(path,destination)
            info=path.stat();os.utime(path,ns=(info.st_atime_ns,info.st_mtime_ns+10000000))
    result=MODULE.offload(parent,"existing:HumanoidScene-RL",Changed())
    assert len(result["retained"])==2 and all((run/name).exists() for name in MODULE.PAYLOADS)


def test_existing_writer_or_unverified_final_backup_rejects_offload(tmp_path,monkeypatch):
    parent,run,_,status=example(tmp_path,monkeypatch)
    status["training_pid"]=os.getpid();(parent/"status.json").write_text(json.dumps(status))
    with pytest.raises(ValueError):MODULE.offload(parent,"existing:HumanoidScene-RL",Remote())
    status["training_pid"]=2**30;status["final_upload_verified"]=False
    (parent/"status.json").write_text(json.dumps(status))
    with pytest.raises(ValueError):MODULE.offload(parent,"existing:HumanoidScene-RL",Remote())
    assert all((run/name).exists() for name in MODULE.PAYLOADS)


def test_original_directory_alias_resolves_but_conflicting_binding_is_rejected(tmp_path,monkeypatch):
    parent,run,launch,status=example(tmp_path,monkeypatch)
    alias=tmp_path/"original_parent";alias.symlink_to(parent,target_is_directory=True)
    launch["run"]=str(alias/run.name);launch["command"][-1]=launch["run"]
    status["run_dir"]=launch["run"]
    (parent/"launch.json").write_text(json.dumps(launch));(parent/"status.json").write_text(json.dumps(status))
    assert MODULE.closed_source(parent)[0]==run
    status["run_dir"]=str(tmp_path/"wrong");(parent/"status.json").write_text(json.dumps(status))
    with pytest.raises(ValueError):MODULE.closed_source(parent)
