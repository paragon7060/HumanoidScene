import copy
import torch
import pytest

from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_base_hold import StagedBaseHoldDiagnostic


class Coordinates:
    def box_anchor(self,raw):
        return raw[:,98:100]
    def current(self,raw):
        return raw[:,:17],raw[:,17:19],raw[:,20:22],raw[:,22],None
    def decode(self,raw,goal):
        action=raw.new_zeros(1,24)
        action[:,:2]=goal[:,19:21]-raw[:,20:22]
        action[:,2]=goal[:,21]-raw[:,22]
        return action


def scene():
    raw=torch.zeros(1,464)
    raw[:,86]=raw[:,388]=raw[:,400]=1
    raw[:,91:94]=torch.tensor([.266,.185,.13])
    raw[:,94]=1
    templates=dict(format=StagedBaseHoldDiagnostic.name,shelves={'middle':dict(
        source_split='train',measured_success=True,box_size_m=[.266,.185,.13],
        base_minus_initial_box_xy_rack_m=[0.,.7],base_yaw_rack_rad=0.)})
    return raw,templates


def test_dwell_requires_consecutive_stationary_observations_and_neutral_approach():
    raw,templates=scene();stage=StagedBaseHoldDiagnostic(Coordinates(),templates,raw)
    action=stage.action(raw)
    assert action[0,1]>.5
    assert torch.equal(action[0,3:20],torch.zeros(17))
    assert action[0,20:22].tolist()==[-1.,-1.]
    raw[:,21]=.7
    for step in range(14):stage.update(raw,torch.zeros(1,3),torch.zeros(1,3),step)
    assert stage.phase=='approach'
    stage.update(raw,torch.tensor([[.03,0.,0.]]),torch.zeros(1,3),14)
    assert stage.stable_steps==0
    for step in range(15,30):stage.update(raw,torch.zeros(1,3),torch.zeros(1,3),step)
    assert stage.phase=='held_grasp' and stage.manipulation_index(29)==0
    raw[:,21]=.72
    proposed=torch.ones(1,24)*.3
    executed=stage.action(raw,proposed)
    assert executed[0,1]<0
    assert torch.equal(executed[:,3:],proposed[:,3:])
    assert stage.target_xy[0,1]==pytest.approx(.7)


def test_waypoints_reject_heldout_success_and_unmeasured_box_size():
    raw,templates=scene()
    heldout=copy.deepcopy(templates);heldout['shelves']['middle']['source_split']='holdout'
    with pytest.raises(ValueError,match='TRAIN'):StagedBaseHoldDiagnostic(Coordinates(),heldout,raw)
    raw[:,91]=.4
    with pytest.raises(ValueError,match='box size'):StagedBaseHoldDiagnostic(Coordinates(),templates,raw)
