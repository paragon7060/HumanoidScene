"""Guard diagnostic isolation and compare the actual production close gate."""
import gzip
import json
from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.debug.grasp_observation_audit import (
    GraspObservationAudit, compare_close_gates, finite_json,
    validate_grasp_observation_audit, valid_measured_poses,
)


def waves(split='validation',count=2,background='original'):
    return [dict(split=split,layouts=[{} for _ in range(count)],background_placement=background)]


@pytest.mark.parametrize('change',[
    dict(training=True),dict(other_probe=True),dict(steps=30),dict(steps=901),
    dict(waves=waves('train')),dict(waves=waves('final')),dict(waves=waves(count=17)),
    dict(waves=waves(background='packed')),dict(waves=waves()*2),
])
def test_only_short_frozen_original_dev_audits_are_allowed(change):
    args=dict(waves=waves(),enabled=True,training=False,steps=900,other_probe=False)
    args.update(change)
    with pytest.raises(ValueError):validate_grasp_observation_audit(**args)


def test_disabled_audit_does_not_restrict_training():
    validate_grasp_observation_audit(waves('train',128),enabled=False,training=True,steps=900)


def test_actual_gate_separates_midpoint_motion_from_assignment_without_mutation():
    raw=torch.zeros(1,464)
    distances=torch.tensor([[[.11,.04],[.04,.11]]])
    nominal=torch.zeros(1,2,2,9);nominal[...,0]=distances
    raw[:,350:386]=nominal.flatten(1);raw[:,386]=1
    original=raw.clone()
    actual=nominal.clone();actual[0,0,0,0]=.15
    result=compare_close_gates(raw,actual,torch.tensor([[1,0]]))
    assert result['nominal'].tolist()==[[True,True]]
    assert result['actual_same_assignment'].tolist()==[[False,True]]
    assert result['actual_reassigned'].tolist()==[[True,True]]
    assert torch.equal(raw,original)
    raw[:,386:388]=0
    assert not any(g.any() for g in compare_close_gates(raw,actual,torch.tensor([[1,0]])).values())


def test_invalid_physical_pose_is_marked_per_environment_and_json_stays_strict():
    poses=torch.tensor([[0.,0.,0.,1.,0.,0.,0.]]).repeat(3,1)
    panels=poses[:,None].repeat(1,2,1)
    panels[1,0,3:]=0;panels[2,1,0]=float('nan')
    assert valid_measured_poses(poses,panels).tolist()==[True,False,False]
    assert json.loads(json.dumps(finite_json(dict(x=panels[2,1,0],y=float('inf'))),allow_nan=False))==dict(x=None,y=None)


def test_duplicate_termination_compute_does_not_duplicate_terminal_capture(tmp_path):
    audit=GraspObservationAudit.__new__(GraspObservationAudit)
    audit.path=tmp_path/'grasp_observation_audit.jsonl.gz'
    audit.stream=gzip.open(audit.path,'wt')
    audit.frames=audit.rows=audit.invalid_rows=0;audit.completed_step=None
    audit.pending=dict(step=7,ids=[0],held={0},action=[[1.,-1.]],goals={0:[1.,-1.]},before=[{'measurement_valid':True}])
    calls=[]
    audit.measure=lambda: calls.append('pre_reset') or [dict(measurement_valid=True,hold_time_s=.27)]
    safety=SimpleNamespace(**{k:torch.tensor([False]) for k in ('invalid_box_pose','invalid_flap_pose',
        'robot_rack_collision','self_collision','obstacle_collision','workspace_limit','box_drop',
        'box_lift_limit','box_speed_limit')})
    g=SimpleNamespace(success=SimpleNamespace(success=torch.tensor([True])))
    audit.finish(g,safety);audit.finish(g,safety);audit.close()
    with gzip.open(audit.path,'rt') as stream:rows=[json.loads(x) for x in stream]
    assert calls==['pre_reset'] and len(rows)==1 and rows[0]['success'] and rows[0]['step']==8
    summary=json.loads((tmp_path/'grasp_observation_audit_summary.json').read_text())
    assert summary['frames']==1 and not summary['Q_import_eligible']
