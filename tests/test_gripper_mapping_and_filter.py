"""Runtime mapping/filter tests; no real-robot capture tools or data required."""
import json
import math

import pytest
import torch

from kuavo_isaaclab_scene.robots.gripper_action import DirectionalGripperMapping, GripperTargetFilter
from kuavo_isaaclab_scene.robots.gripper_config import load_gripper_settings
from kuavo_isaaclab_scene.robots.twofinger_linkage import DRIVER_OPEN_MIN, FINGER_PIN, FOLLOWER_PIN, passive_joint_angles


def test_target_filter_uses_physics_time_and_independent_direction_constants():
    cfg = dict(closing_time_constant_s=.12, opening_time_constant_s=.08)
    initial = torch.tensor([[-.36,.36],[-.36,.36]],dtype=torch.float64)
    close_direction = -initial[0]
    first = GripperTargetFilter(initial,close_direction,cfg)
    second = GripperTargetFilter(initial,close_direction,cfg)
    desired = torch.tensor([[0.,0.],[-.36,.36]],dtype=torch.float64)
    for _ in range(120):first.advance(desired,1/120)
    for _ in range(60):second.advance(desired,1/60)
    assert torch.allclose(first.current,second.current,atol=1e-12)
    assert torch.allclose(first.current[0],initial[0]*math.exp(-1/.12),atol=1e-12)
    assert torch.equal(first.current[1],initial[1])
    first.reset(torch.zeros(1,2),[0])
    first.advance(initial, .08)
    assert torch.allclose(first.current[0], initial[0]*(1-math.exp(-1)),atol=1e-12)
    assert torch.equal(first.current[1],initial[1])



@pytest.mark.parametrize("stages", [1, 2])
def test_target_filter_direction_reversal_is_monotonic_and_reset_clears_pending_motion(stages):
    filt = GripperTargetFilter(torch.tensor([[0.,0.],[0.,0.]]),torch.tensor([1.,-1.]),
                              dict(closing_time_constant_s=.12,opening_time_constant_s=.08,stages=stages))
    closed = torch.tensor([[1.,-1.],[1.,-1.]])
    before = filt.advance(closed,.01).clone()
    reversed_target = torch.zeros_like(closed)
    after = filt.advance(reversed_target,.01).clone()
    assert torch.all(after.abs()<before.abs())
    captured = torch.tensor([[.3,-.31]])
    filt.reset(captured,[0])
    desired = after.clone()
    desired[0]=captured[0]
    for _ in range(12):filt.advance(desired,1/120)
    assert torch.equal(filt.current[0],captured[0])



def test_two_stage_filter_is_timestep_independent_and_starts_gently():
    initial = torch.zeros(2,2,dtype=torch.float64)
    cfg = dict(closing_time_constant_s=.1, opening_time_constant_s=.08, stages=2)
    first = GripperTargetFilter(initial,torch.tensor([1.,-1.]),cfg)
    second = GripperTargetFilter(initial,torch.tensor([1.,-1.]),cfg)
    target = torch.tensor([[1.,-1.],[0.,0.]],dtype=torch.float64)
    early = first.advance(target,.01).clone()
    assert early[0,0] < 1-math.exp(-.01/.1)
    for _ in range(99):first.advance(target,.01)
    for _ in range(50):second.advance(target,.02)
    assert torch.allclose(first.current,second.current,atol=1e-12)
    assert first.current[0,0].item() == pytest.approx(1-math.exp(-10)*(1+10),abs=1e-12)
    assert torch.equal(first.current[1], initial[1])



@pytest.mark.parametrize("value", [0,3,2.,True,float('nan')])
def test_invalid_target_filter_stages_rejected(tmp_path,value):
    payload=json.loads(load_gripper_settings('leju-twofinger').config_path.read_text())
    side=payload['presets']['leju-twofinger'].setdefault('sides',{}).setdefault('left',{})
    side.setdefault('target_filter',{})['stages']=value
    path=tmp_path/'grippers.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):load_gripper_settings('leju-twofinger',path)



@pytest.mark.parametrize("value", [0,-1,float('nan'),float('inf'),True])
def test_invalid_target_filter_settings_rejected(tmp_path,value):
    payload=json.loads(load_gripper_settings("leju-twofinger").config_path.read_text())
    side=payload['presets']['leju-twofinger'].setdefault('sides',{}).setdefault('left',{})
    side['target_filter'] = dict(closing_time_constant_s=value,opening_time_constant_s=.1)
    path=tmp_path/'grippers.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):load_gripper_settings('leju-twofinger',path)



def test_partial_reversals_hold_until_envelope_catches_up_and_reset_is_local():
    mapping = load_gripper_settings("leju-twofinger").sides["left"].position_mapping
    mapper = DirectionalGripperMapping(mapping, 2, "cpu")
    mapper.process(torch.tensor([[-.5], [0.0]]))  # 75 and 50 closing
    before = mapper.fraction.clone()
    mapper.process(torch.tensor([[-.48], [0.02]]))  # Tiny opening reversals
    assert torch.equal(mapper.fraction, before)
    mapper.process(torch.tensor([[.5], [.5]]))
    assert torch.all(mapper.fraction < before)
    held = mapper.fraction.clone()
    mapper.process(torch.tensor([[.5], [.5]]))
    assert torch.equal(mapper.fraction, held)
    mapper.reset([0], torch.tensor([.63]))
    neutral = 1 - mapper.previous_percent / 50
    assert torch.allclose(mapper.process(neutral), torch.tensor([[.63], [held[1,0]]]), atol=1e-6)
    mapper.reset([0])
    assert mapper.fraction[0] == 0 and mapper.fraction[1] == held[1]



@pytest.mark.parametrize("jaw", ["f", "b"])
def test_expanded_linkage_range_closes_and_respects_follower_limits(jaw):
    sign = 1 if jaw == "f" else -1
    def rotate(q, point):
        return (math.cos(q)*point[0]+math.sin(q)*point[1], -math.sin(q)*point[0]+math.cos(q)*point[1])
    for step in range(101):
        q = sign * DRIVER_OPEN_MIN * step / 100
        q3, q4 = passive_joint_angles(q, jaw)
        assert abs(q3) < .698 and abs(q4) < .698
        crank = rotate(q, (sign*.027533, -.0085573))
        coupler = rotate(q+q3, (sign*(-.0081591+FINGER_PIN[0]), -.047301+FINGER_PIN[2]))
        follower = rotate(q4, (sign*FOLLOWER_PIN[0], FOLLOWER_PIN[2]))
        a = (sign*.0125+crank[0]+coupler[0], -.063137+crank[1]+coupler[1])
        b = (sign*.02+follower[0], -.09+follower[1])
        assert math.dist(a, b) < 1e-12



@pytest.mark.parametrize("field,value", [
    ("command_scale", 0), ("command_scale", math.nan),
    ("position_mapping", {"command_percent": [0, 50, 50, 100], "closing_fractions": [0,.3,.4,1], "opening_fractions": [0,.4,.5,1]}),
    ("position_mapping", {"command_percent": [0,100], "closing_fractions": [0,.9], "opening_fractions": [0,1]}),
    ("position_mapping", {"command_percent": [0,50,100], "closing_fractions": [0,.8,1], "opening_fractions": [0,.5,1]})])
def test_bad_calibration_rejected(tmp_path, field, value):
    source = load_gripper_settings("leju-twofinger").config_path
    config = json.loads(source.read_text())
    side=config["presets"]["leju-twofinger"].setdefault("sides",{}).setdefault("left",{})
    side[field] = value
    path = tmp_path / "grippers.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_gripper_settings("leju-twofinger", path)
