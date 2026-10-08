import hashlib
import json
import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/rl"))
from analyze_completed_upright_tracking import map_completed_rows, summarize, verified_snapshot
from analyze_completed_upright_tracking import main
from analyze_closed_upright_tracking import torso_tracking_summary


def sample():
    rows={k:torch.zeros(4,d) for k,d in (("actor_obs",518),("critic_obs",578),
        ("next_critic_obs",578),("action",21))}
    for key in ("critic_obs","next_critic_obs"):
        rows[key][:,:3]=torch.tensor([.2,-.5,.3])
        rows[key][:,439]=1
        rows[key][1::2,2]+=.1
        rows[key][1::2,418]=-.1
        rows[key][:,530]=torch.tensor([0.,0.,1/900,1/900])
    rows["next_critic_obs"][2,499:501]=1
    rows["next_critic_obs"][2,505:507]=1
    rows["next_critic_obs"][2,518]=1
    center=torch.zeros(21);center[17:19]=torch.tensor([.03,.63])
    scale=torch.ones(21);scale[17:19]=torch.tensor([.02,.14])
    contract=dict(goal_center=center.tolist(),goal_scale=scale.tolist(),
                  URDF_upright_support=dict(upright_goal_columns=[17,18]))
    cases=[]
    for i in range(2):
        cases.append(dict(wave=1,split="train",environment=i,complete=True,initial_layout_valid=True,
            layout=dict(target_region="shelf_2_left",target_box_type="small"),
            collection_policy_mode="perceived_contact_exploration" if i else "greedy_current_policy",
            result=dict(staged_base=dict(manipulation_start=0),steps=2,
                pinching=[i==0,i==0],stable_hands=[i==0,i==0],success=i==0,
                unsafe=False,time_out=i==1)))
    metrics=dict(outcomes=cases,learner=dict(critic_updates=10))
    return rows,contract,metrics


def snapshot(tmp_path):
    rows,contract,metrics=sample()
    path=tmp_path/"completed_copy.pt"
    torch.save(dict(executed_goal_transitions=rows,goal_contract=contract),path)
    path.chmod(0o400)
    proof=dict(snapshot=str(path),source_run=str(tmp_path/"live_run"),
        bytes=path.stat().st_size,snapshot_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
        completed_TRAIN_waves=[1],completed_online_rows=4,completed_critic_updates=10,
        source_stat_unchanged_during_copy=True,completed_metadata_unchanged_during_copy=True,
        source_NOT_removed_or_modified=True,no_live_HDF_read=True,no_live_GPU_replay_read=True)
    proof_path=tmp_path/"snapshot_verified.json";proof_path.write_text(json.dumps(proof))
    (tmp_path/"completed_waves_metrics.json").write_text(json.dumps(metrics))
    return path,proof_path,proof


def test_distinguishes_measured_pitch_from_logical_goal_and_maps_interleaved_environments():
    rows,contract,metrics=sample()
    groups=map_completed_rows(rows,metrics,[1])
    assert [i.tolist() for _,i in groups]==[[0,2],[1,3]]
    result=summarize(rows,contract,groups)
    assert result["logical_pitch_target_deg"]["max"]<1e-5
    assert result["rows_above_5deg_pitch_tracking_error"]==2
    assert result["episodes"][0]["rows_above_5deg_pitch_tracking_error"]==0
    assert result["episodes"][1]["rows_above_5deg_pitch_tracking_error"]==2
    assert result["by_region_and_collection_mode"][0]["successful_episodes"]==1


def test_refuses_swapped_environment_clocks_and_terminal_contact():
    rows,_,metrics=sample();rows["critic_obs"][2,530]=0
    with pytest.raises(ValueError,match="clock"):
        map_completed_rows(rows,metrics,[1])
    rows,_,metrics=sample();rows["next_critic_obs"][2,499]=0
    with pytest.raises(ValueError,match="terminal contact"):
        map_completed_rows(rows,metrics,[1])


@pytest.mark.parametrize("change",["incomplete","invalid","validation"])
def test_rejects_incomplete_invalid_or_eval_episodes(change):
    rows,_,metrics=sample();case=metrics["outcomes"][0]
    if change=="incomplete":case["complete"]=False
    elif change=="invalid":case["initial_layout_valid"]=False
    else:case["split"]="validation"
    with pytest.raises(ValueError,match="completed valid TRAIN"):
        map_completed_rows(rows,metrics,[1])


def test_requires_exact_completed_row_coverage():
    rows,_,metrics=sample();rows={k:torch.cat([v,v[-1:]]) for k,v in rows.items()}
    with pytest.raises(ValueError,match="Some saved rows"):
        map_completed_rows(rows,metrics,[1])


def test_readonly_copy_identity_and_provenance_are_required(tmp_path):
    path,proof_path,proof=snapshot(tmp_path)
    loaded,_,_=verified_snapshot(proof_path);assert loaded["snapshot_SHA256"]==proof["snapshot_SHA256"]
    path.chmod(0o600)
    with pytest.raises(ValueError,match="Readonly"):
        verified_snapshot(proof_path)
    path.write_bytes(path.read_bytes()+b"changed");path.chmod(0o400)
    with pytest.raises(ValueError,match="verified identity"):
        verified_snapshot(proof_path)


def test_never_accepts_actual_live_replay_path_even_if_readonly(tmp_path):
    path,proof_path,proof=snapshot(tmp_path)
    renamed=tmp_path/"staged_goal_experience.pt";path.rename(renamed)
    proof.update(snapshot=str(renamed),source_run=str(tmp_path));proof_path.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match="separate completed replay"):
        verified_snapshot(proof_path)


def test_rejects_unproven_completed_snapshot_and_out_of_support_goals(tmp_path):
    _,proof_path,proof=snapshot(tmp_path);proof["completed_metadata_unchanged_during_copy"]=False
    proof_path.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match="provenance"):
        verified_snapshot(proof_path)
    rows,contract,_=sample();rows["action"][:,17]=2
    with pytest.raises(AssertionError,match="Executed goals"):
        torso_tracking_summary(rows,contract)


def test_complete128_cli_records_mapping_and_rechecks_copy_without_modifying_it(tmp_path,monkeypatch):
    path,proof_path,proof=snapshot(tmp_path)
    rows,contract,metrics=sample()
    ix=torch.tensor([0,1]*64)
    expanded={k:v[ix].clone() for k,v in rows.items()}
    expanded["next_critic_obs"][::2]=rows["next_critic_obs"][2]
    metrics["outcomes"]=[]
    for i in range(128):
        case=copy.deepcopy(sample()[2]["outcomes"][i%2])
        case["environment"]=i;case["result"]["steps"]=1
        metrics["outcomes"].append(case)
    path.chmod(0o600);torch.save(dict(executed_goal_transitions=expanded,goal_contract=contract),path)
    path.chmod(0o400);identity=hashlib.sha256(path.read_bytes()).hexdigest()
    proof.update(bytes=path.stat().st_size,snapshot_SHA256=identity,completed_online_rows=128)
    proof_path.write_text(json.dumps(proof));(tmp_path/"completed_waves_metrics.json").write_text(json.dumps(metrics))
    output=tmp_path/"diagnosis.json"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    monkeypatch.setattr(sys,"argv",["diagnosis","--snapshot-proof",str(proof_path),"--output-json",str(output)])
    main();result=json.loads(output.read_text())
    assert result["completed_episodes_mapped"]==128
    assert result["source_snapshot_checksum_reverified_after_diagnosis"]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==identity
