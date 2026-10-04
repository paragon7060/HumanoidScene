"""A candidate moves the actual base controller without changing resets/Q."""
from copy import deepcopy
import math
import pytest
import torch
from test_rl_staged_base_hold import Coordinates,scene
from kuavo_isaaclab_scene.rl.multi_box.experiments.batched_staged_goal import BatchedBaseStages
from kuavo_isaaclab_scene.rl.multi_box.experiments.waypoint_probe import validate_waypoint_probe,apply_waypoint_probe


def candidate(name,offset):
    return dict(layout={'seed':42},waypoint_probe=dict(name=name,offset_xy_yaw=offset))


def test_candidates_drive_approach_and_held_context_without_touching_initial_state():
    raw,templates=scene();raw=raw.repeat(2,1);original=raw.clone();old=deepcopy(templates)
    stages=BatchedBaseStages(Coordinates(),templates,raw)
    layouts=[candidate('baseline',[0.,0.,0.]),candidate('center',[.09,.05,.1])]
    validate_waypoint_probe([dict(layouts=layouts)],enabled=True)
    apply_waypoint_probe(stages,layouts)
    commands=stages.approach_commands(raw,torch.ones(2,dtype=torch.bool))
    assert torch.allclose(commands[:,:3],torch.tensor([[0.,.7,0.],[.09,.75,.1]]))
    assert torch.equal(raw,original) and templates==old
    assert torch.equal(commands[:,3:20],torch.zeros(2,17))
    assert commands[:,20:22].tolist()==[[-1.,-1.],[-1.,-1.]]
    for stage in stages.stages:stage.phase='held_grasp';stage.manipulation_start=10
    held=stages.held_context(torch.tensor([1,0]))
    assert torch.allclose(held.target_xy,stages.target_xy[[1,0]])
    assert torch.allclose(held.target_yaw,stages.target_yaw[[1,0]])
    assert stages.stages[1].report()['waypoint_probe']['name']=='center'
    assert stages.stages[0].template==old['shelves']['middle']


def test_candidate_metadata_cannot_silently_change_training_or_duplicate_cases():
    waves=[dict(layouts=[candidate('baseline',[0.,0.,0.])])]
    for enabled,training in [(False,False),(False,True),(True,True)]:
        with pytest.raises(ValueError):validate_waypoint_probe(waves,enabled=enabled,training=training)
    validate_waypoint_probe([dict(layouts=[dict(layout={'seed':42})])])
    with pytest.raises(ValueError,match='distinct'):
        validate_waypoint_probe([dict(layouts=waves[0]['layouts']*2)],enabled=True)
    for bad in [[.151,0.,0.],[0.,0.,math.pi/11],[math.nan,0.,0.],[True,0.,0.]]:
        with pytest.raises(ValueError):
            validate_waypoint_probe([dict(layouts=[candidate('bad',bad)])],enabled=True)
