"""Diagnose torso tracking in an immutable copy of completed TRAIN waves.

Read actual measured joints, logical targets and safety/contact labels offline.
Never read an active replay, import rows into training, or infer motor torques.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import stat

import torch

from analyze_closed_upright_tracking import torso_tracking_summary
from compare_actor_train_memory import owned_stable_bytes


def verified_snapshot(proof_path):
    proof_path = Path(proof_path)
    proof = json.loads(owned_stable_bytes(proof_path))
    path = Path(proof["snapshot"])
    if path.parent != proof_path.parent or path.is_symlink() \
            or path.resolve() == (Path(proof["source_run"])/"staged_goal_experience.pt").resolve():
        raise ValueError("Only the separate completed replay copy is eligible")
    before = path.stat()
    if before.st_mode & 0o222 or not stat.S_ISREG(before.st_mode):
        raise ValueError("Readonly regular replay copy required")
    for key in ("source_stat_unchanged_during_copy", "completed_metadata_unchanged_during_copy",
                "source_NOT_removed_or_modified", "no_live_HDF_read", "no_live_GPU_replay_read"):
        if proof.get(key) is not True:
            raise ValueError("Completed-wave snapshot provenance is incomplete")
    waves = proof.get("completed_TRAIN_waves")
    if not waves or len(waves) != len(set(waves)) or any(type(w) is not int or w<0 for w in waves):
        raise ValueError("Distinct completed TRAIN waves required")
    blob = owned_stable_bytes(path)
    if len(blob) != proof["bytes"] or hashlib.sha256(blob).hexdigest() != proof["snapshot_SHA256"]:
        raise ValueError("Snapshot differs from its verified identity")
    saved = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    metrics = json.loads(owned_stable_bytes(proof_path.parent/"completed_waves_metrics.json"))
    if len(saved["executed_goal_transitions"]["action"]) != proof["completed_online_rows"] \
            or metrics["learner"]["critic_updates"] != proof["completed_critic_updates"]:
        raise ValueError("Rows or completed update counters differ")
    after = path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns) != (after.st_ino,after.st_size,after.st_mtime_ns):
        raise ValueError("Readonly source changed during analysis")
    return proof, saved, metrics


def map_completed_rows(rows, metrics, waves):
    """Recover exact environment IDs from the collector's chronological order."""
    n = len(rows["action"])
    groups, offset = [], 0
    for wave in waves:
        cases = sorted((x for x in metrics["outcomes"] if x["wave"]==wave),
                       key=lambda x:x["environment"])
        if not cases or len({x["environment"] for x in cases}) != len(cases) \
                or any(x["split"]!="train" or not x["complete"] or not x["initial_layout_valid"]
                       or x["result"].get("numerical_failure") for x in cases):
            raise ValueError("Only completed valid TRAIN trajectories can be mapped")
        starts = [x["result"]["staged_base"]["manipulation_start"] for x in cases]
        ends = [x["result"]["steps"] for x in cases]
        if any(type(a) is not int or type(b) is not int or not 0<=a<b for a,b in zip(starts,ends)):
            raise ValueError("Missing actual held-phase clocks")
        indices = [[] for x in cases]
        for step in range(max(ends)):
            active = [i for i,(a,b) in enumerate(zip(starts,ends)) if a<=step<b]
            if offset+len(active)>n:
                raise ValueError("Completed trajectories exceed saved rows")
            for i in active:
                clock = rows["critic_obs"][offset,530]
                if not torch.isclose(clock,clock.new_tensor((step-starts[i])/900),atol=1e-7,rtol=0):
                    raise ValueError("Saved held clock differs from environment row mapping")
                indices[i].append(offset);offset+=1
        for case,ix in zip(cases,indices):
            final = rows["next_critic_obs"][ix[-1],464:530]
            result = case["result"]
            if (final[35:37]>.5).tolist()!=result["pinching"] \
                    or bool(final[54]>.5)!=result["success"] \
                    or (final[41:43]>.5).tolist()!=result["stable_hands"]:
                raise ValueError("Mapped terminal contact differs from actual episode")
            groups.append((case,torch.tensor(ix,dtype=torch.long)))
    if offset!=n:
        raise ValueError("Some saved rows do not belong to declared completed TRAIN waves")
    return groups


def summarize(rows, contract, groups):
    result = torso_tracking_summary(rows, contract)
    episodes = []
    for case,indices in groups:
        subset = {k:v[indices] for k,v in rows.items()}
        summary = torso_tracking_summary(subset,contract)
        episodes.append(dict(wave=case["wave"],environment=case["environment"],
            region=case["layout"]["target_region"],box_type=case["layout"]["target_box_type"],
            collection_mode=case["collection_policy_mode"],success=case["result"]["success"],
            unsafe=case["result"]["unsafe"],time_out=case["result"]["time_out"],
            **summary))
    keys = sorted({x["region"]+"/"+x["box_type"] for x in episodes})
    region_modes = []
    for key in keys:
        for mode in sorted({x["collection_mode"] for x in episodes}):
            selected = [x for x in episodes if x["region"]+"/"+x["box_type"]==key
                        and x["collection_mode"]==mode]
            if not selected:
                continue
            region_modes.append(dict(region_size=key,collection_mode=mode,episodes=len(selected),
                successful_episodes=sum(x["success"] for x in selected),
                unsafe_episodes=sum(x["unsafe"] for x in selected),
                actual_held_rows=sum(x["actual_held_transition_rows"] for x in selected),
                episodes_with_pitch_error_above3deg=sum(x["rows_above_3deg_pitch_tracking_error"]>0 for x in selected),
                episodes_with_measured_XZ_outside_support=sum(x["pre_measured_outside_support_rows"]>0 for x in selected),
                maximum_pitch_error_deg=max(x["measured_minus_logical_target_pitch_absolute_error_deg"]["max"] for x in selected),
                maximum_X_tracking_error_mm=max(x["measured_minus_logical_target_XZ_absolute_error_mm"]["X"]["max"] for x in selected)))
    result.update(completed_episodes_mapped=len(episodes),by_region_and_collection_mode=region_modes,
                  episodes=episodes,source_data_NO_import_or_policy_updates=True,
                  contact_and_tracking_association_NOT_cause_or_torque_evidence=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--snapshot-proof",type=Path,required=True)
    parser.add_argument("--output-json",type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        parser.error("CPU-only diagnosis requires empty CUDA_VISIBLE_DEVICES")
    if args.output_json.exists():
        parser.error("Use a unique new diagnosis output")
    torch.set_num_threads(1)
    proof,saved,metrics=verified_snapshot(args.snapshot_proof)
    rows=saved["executed_goal_transitions"]
    groups=map_completed_rows(rows,metrics,proof["completed_TRAIN_waves"])
    if len(groups)!=128*len(proof["completed_TRAIN_waves"]):
        raise ValueError("Expected all128 requested TRAIN environments per completed wave")
    result=summarize(rows,saved["goal_contract"],groups)
    # The model/data copy is normally immutable. Recheck it after computation
    # as well so an operator edit cannot silently invalidate the report.
    snapshot=Path(proof["snapshot"])
    digest=hashlib.sha256()
    before=snapshot.stat()
    with snapshot.open("rb") as source:
        for block in iter(lambda:source.read(1024*1024),b""):
            digest.update(block)
    after=snapshot.stat()
    if snapshot.is_symlink() or before.st_mode&0o222 or after.st_mode&0o222 \
            or (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns) \
            or digest.hexdigest()!=proof["snapshot_SHA256"]:
        raise ValueError("Completed snapshot changed during diagnosis")
    result.update(source_run_directory_name=Path(proof["source_run"]).name,
        source_completed_snapshot_SHA256=proof["snapshot_SHA256"],
        verified_completed_TRAIN_waves=proof["completed_TRAIN_waves"],
        readonly_completed_snapshot_only_no_active_replay_or_HDF=True,
        source_snapshot_and_trajectory_mapping_verified=True,
        source_snapshot_checksum_reverified_after_diagnosis=True,
        motor_torque_gravity_feedforward_and_effort_saturation_NOT_recorded=True,
        independent_FINAL_unused=True,goal_not_complete=True)
    args.output_json.parent.mkdir(parents=True,exist_ok=True)
    with args.output_json.open("x") as output:
        output.write(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="episodes"},ensure_ascii=False))


if __name__=="__main__":
    main()
