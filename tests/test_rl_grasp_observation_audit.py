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


def full_dev_distribution():
    regions=('shelf_2_left','shelf_2_right','shelf_3_left','shelf_3_right')
    return [dict(split='validation',layouts=[dict(layout=dict(seed=121000+i,
        split='holdout',target_region=regions[i%4]))]) for i in range(128)]


def test_full_distribution_is_frozen_balanced_and_keeps_legacy_layout_labels():
    rows=full_dev_distribution();original=json.dumps(rows,sort_keys=True)
    validate_grasp_observation_audit(rows,enabled=True,training=False,steps=900,full_distribution=True)
    assert json.dumps(rows,sort_keys=True)==original


@pytest.mark.parametrize('change',['TRAIN','FINAL','duplicate','unbalanced','selected_subset','packed'])
def test_full_distribution_cannot_import_training_final_or_easier_subset(change):
    rows=full_dev_distribution()
    if change=='TRAIN':rows[7]['split']='train'
    elif change=='FINAL':rows[7]['split']='holdout'
    elif change=='duplicate':rows[7]['layouts'][0]['layout']['seed']=121000
    elif change=='unbalanced':rows[7]['layouts'][0]['layout']['target_region']='shelf_2_left'
    elif change=='selected_subset':rows=rows[:16]
    elif change=='packed':rows[7]['background_placement']='packed'
    with pytest.raises(ValueError):
        validate_grasp_observation_audit(rows,enabled=True,training=False,steps=900,full_distribution=True)


@pytest.mark.parametrize('change',[dict(training=True),dict(enabled=False),dict(other_probe=True)])
def test_full_distribution_needs_explicit_audit_without_training_or_physics_probe(change):
    args=dict(waves=full_dev_distribution(),enabled=True,training=False,steps=900,full_distribution=True)
    args.update(change)
    with pytest.raises(ValueError):validate_grasp_observation_audit(**args)


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


def test_same_local_step_in_next_wave_is_captured_for_its_original_case(tmp_path):
    audit=GraspObservationAudit.__new__(GraspObservationAudit)
    audit.path=tmp_path/'grasp_observation_audit.jsonl.gz'
    audit.stream=gzip.open(audit.path,'wt')
    audit.frames=audit.rows=audit.invalid_rows=0
    audit.measure=lambda: [dict(measurement_valid=True)]
    safety=SimpleNamespace(**{k:torch.tensor([False]) for k in ('invalid_box_pose','invalid_flap_pose',
        'robot_rack_collision','self_collision','obstacle_collision','workspace_limit','box_drop',
        'box_lift_limit','box_speed_limit')})
    g=SimpleNamespace(success=SimpleNamespace(success=torch.tensor([False])))
    cases=full_dev_distribution()
    for i in range(2):
        audit.begin_wave(i,cases[i]['layouts'])
        assert audit.pending is None and audit.completed_step is None
        audit.pending=dict(step=0,ids=[0],held={0},action=[[1.,-1.]],goals={},before=[{'measurement_valid':True}])
        audit.finish(g,safety);audit.finish(g,safety)
    audit.close()
    with gzip.open(audit.path,'rt') as stream:rows=[json.loads(x) for x in stream]
    assert [r['step'] for r in rows]==[1,1]
    assert [r['wave'] for r in rows]==[0,1]
    assert [r['layout_seed'] for r in rows]==[121000,121001]
    assert [r['target_region'] for r in rows]==['shelf_2_left','shelf_2_right']
