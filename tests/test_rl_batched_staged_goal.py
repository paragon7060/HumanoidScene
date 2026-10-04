"""Prevent cross-environment clock/waypoint leakage and reset-seam replay."""
from types import SimpleNamespace
import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.batched_staged_goal import BatchedBaseStages
from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_student import pose_clock
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import staged_context,held_goal_coordinates
from test_rl_staged_base_hold import scene,Coordinates


def test_feedback_rate_contract_checks_every_vector_environment():
    from kuavo_isaaclab_scene.rl.multi_box.experiments.reference_residual import validate_goal_feedback_rates
    base=torch.tensor([[.15,.15,.5]]).repeat(4,1)
    upper=torch.tensor([[.01]+[.02]*14]).repeat(4,1)
    validate_goal_feedback_rates(base,upper,torch.full((4,2),.01),.1,1/30)
    upper[3,4]=.03
    with pytest.raises(ValueError,match='rates differ'):
        validate_goal_feedback_rates(base,upper,.01,.1,1/30)


def test_different_physical_settling_times_have_independent_anchors_and_clocks():
    raw,templates=scene();raw=raw.repeat(3,1)
    raw[:,98]=torch.tensor([-.2,.1,.35])
    stages=BatchedBaseStages(Coordinates(),templates,raw)
    raw[:,20]=raw[:,98];raw[:,21]=.7
    zero=torch.zeros(3,3);active=torch.ones(3,dtype=torch.bool)
    for step in range(15):
        velocity=zero.clone();velocity[1,0]=.03
        held=stages.update(raw,velocity,zero,active,step)
    assert held.tolist()==[0,2]
    with pytest.raises(ValueError,match='unconfirmed'):stages.held_context(torch.tensor([1]))
    # The measured initial box anchor stays fixed even if its box later moves.
    raw[0,98]=9
    active[2]=False
    for step in range(15,30):held=stages.update(raw,zero,zero,active,step)
    assert held.tolist()==[0,1]
    torch.testing.assert_close(stages.anchors[:2,0],torch.tensor([-.2,.1]))
    assert stages.clocks(held,29).tolist()==[15,0]
    context=staged_context(raw[held],stages.held_context(held),.06)
    torch.testing.assert_close(context[:,1],torch.tensor([-.2,.1]))
    assert context[:,0].eq(1).all()


def test_per_environment_clock_and_yaw_match_individual_decoding():
    raw=torch.zeros(2,464);times=torch.tensor([0,700])
    torch.testing.assert_close(pose_clock(raw,times,594,593),torch.tensor([[0.],[593/594]]))
    for bad in (torch.tensor([-1,0]),torch.tensor([0]),torch.tensor([float('nan'),0])):
        with pytest.raises(ValueError,match='elapsed clock'):pose_clock(raw,bad,594)
    class Decoder:
        def decode(self,raw,goal):return goal
    stage=SimpleNamespace(phase='held_grasp',manipulation_start=True,
        target_xy=torch.tensor([[-.2,.7],[.1,.6]]),target_yaw=torch.tensor([.2,-.3]))
    requested=torch.randn(2,21)
    batched=held_goal_coordinates(Decoder(),raw,requested,stage)
    for i in range(2):
        single=SimpleNamespace(target_xy=stage.target_xy[i:i+1],target_yaw=float(stage.target_yaw[i]))
        torch.testing.assert_close(batched[i:i+1],held_goal_coordinates(Decoder(),raw[i:i+1],requested[i:i+1],single))
    context=staged_context(raw,stage,.05)
    torch.testing.assert_close(context[:,3],stage.target_yaw.sin())
