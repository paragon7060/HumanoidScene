import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.state.grasp_target import select_grasp_target,ExplicitGraspTargetSelector


def test_distractors_do_not_replace_an_explicit_active_target():
    active=torch.tensor([[True,False,True],[False,True,True]])
    assert select_grasp_target(active,torch.tensor([2,1])).tolist()==[2,1]
    with pytest.raises(ValueError,match='explicit target'):
        select_grasp_target(active)


def test_partial_normal_reset_can_return_to_default_single_target():
    active=torch.tensor([[False,True,False],[True,False,True]])
    assert select_grasp_target(active,torch.tensor([-1,2])).tolist()==[1,2]
    assert select_grasp_target(active[:1]).tolist()==[1]


@pytest.mark.parametrize('override', [torch.tensor([1]),torch.tensor([3]),torch.tensor([-1]),torch.tensor([2.])])
def test_invalid_or_inactive_target_is_not_silently_substituted(override):
    active=torch.tensor([[True,False,True]])
    with pytest.raises(ValueError):
        select_grasp_target(active,override)


def test_perception_and_privileged_target_share_explicit_command_after_partial_reset():
    from kuavo_isaaclab_scene.rl.multi_box.runtime import FirstSelectableBoxSelector
    selectable=torch.zeros(2,12,dtype=torch.bool);selectable[:,2]=True;selectable[:,9]=True
    overrides=torch.tensor([9,-1,2])
    selector=ExplicitGraspTargetSelector(lambda:overrides,FirstSelectableBoxSelector())
    selection=selector.select(None,selectable,torch.tensor([0,1]))
    assert selection.box_index.tolist()==[9,2]
    selector=ExplicitGraspTargetSelector(lambda:None,FirstSelectableBoxSelector())
    assert selector.select(None,selectable,torch.tensor([0,1])).box_index.tolist()==[2,2]
