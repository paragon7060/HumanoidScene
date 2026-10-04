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


def test_velocity_teacher_respects_executed_position_clipping_and_direction():
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_contact_ik import executed_velocity_feedforward
    current=torch.zeros(1,4)
    command=torch.tensor([[.01,-.01,-.01,.04]])
    proposed=torch.tensor([[2.,-2.,2.,.2]])
    torch.testing.assert_close(executed_velocity_feedforward(current,command,proposed,.1),
                               torch.tensor([[.1,-.1,0.,.2]]))


def test_teacher_identity_does_not_change_actual_actor_observation_or_live_geometry():
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_contact_ik import tracked_flap_observation
    from kuavo_isaaclab_scene.rl.multi_box.experiments.guided_exploration import ASSIGNMENT_START
    raw=torch.randn(1,464)
    raw[:,ASSIGNMENT_START:ASSIGNMENT_START+2]=torch.tensor([[0.,1.]])
    frozen_identity=torch.tensor([[1.,0.]])
    result=tracked_flap_observation(raw,frozen_identity)
    assert raw[0,ASSIGNMENT_START:ASSIGNMENT_START+2].tolist()==[0.,1.]
    assert result[0,ASSIGNMENT_START:ASSIGNMENT_START+2].tolist()==[1.,0.]
    torch.testing.assert_close(result[:,:ASSIGNMENT_START],raw[:,:ASSIGNMENT_START])
    torch.testing.assert_close(result[:,ASSIGNMENT_START+2:],raw[:,ASSIGNMENT_START+2:])
