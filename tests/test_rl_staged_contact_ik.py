import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_contact_ik import (
    coordinated_close, contact_handoff_phase)


def test_contact_teacher_waits_for_both_hands_and_preserves_measured_pinch():
    distance=torch.tensor([[.001,.02],[.01,.01],[.01,.01]])
    angle=torch.tensor([[.01,.01],[.01,.20],[.01,.01]])
    pinch=torch.tensor([[True,False],[False,False],[False,False]])
    assert coordinated_close(distance,angle,pinch,torch.zeros(3,dtype=torch.bool)).tolist()==[
        [True,False],[False,False],[True,True]]


def test_confirmed_lift_keeps_the_jaws_closed_even_after_the_hand_goals_move():
    distance=torch.ones(1,2)
    angle=torch.ones(1,2)
    assert coordinated_close(distance,angle,torch.zeros(1,2,dtype=torch.bool),torch.ones(1,dtype=torch.bool)).all()


def test_handoff_from_a_held_base_cannot_skip_the_safe_front_stage():
    assert contact_handoff_phase(torch.ones(1,2),'after-base-hold') == 0
    assert contact_handoff_phase(torch.tensor([[.02,.11]]),'near-contact') is None
    assert contact_handoff_phase(torch.tensor([[.02,.09]]),'near-contact') == 1
