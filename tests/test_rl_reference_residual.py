"""The contextual residual MDP must preserve actual-command bookkeeping."""
import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.reference_residual import ReferenceResidual, ReferenceGoalResidual, ResidualSACPilot, validate_goal_feedback_rates


def test_zero_residual_reproduces_measured_commands_and_grippers():
    commands = torch.linspace(-1, 1, 48).reshape(2, 24)
    controller = ReferenceResidual(commands)
    assert torch.equal(controller.physical_commands(torch.zeros(1, 22), 0), commands[:1])
    physical = controller.physical_commands(torch.ones(1, 22), 1)
    assert torch.equal(physical[:, 20:22], commands[1:2, 20:22])
    assert physical.abs().max() <= 1
    assert (physical-commands[1:2]).abs().max() <= .050001


def test_horizon_holds_goals_and_exposes_reference_progress():
    commands = torch.ones(2, 24)
    controller = ReferenceResidual(commands)
    context = controller.context(2, 1)
    assert context[0, -1] == 1
    assert not context[0, controller.columns].any()
    assert context[0, 20:22].eq(1).all()
    ao, co = controller.observations(torch.zeros(1, 464), torch.zeros(1, 530), 1)
    assert ao.shape == (1, 199) and co.shape == (1, 555)
    assert torch.equal(ao[:, -25:], co[:, -25:])
    assert ao[0, -1] == .5
    with pytest.raises(ValueError):
        controller.context(-1, 1)


@pytest.mark.parametrize('scale', [0, .3, float('nan')])
def test_reject_invalid_scale(scale):
    with pytest.raises(ValueError):
        ReferenceResidual(torch.zeros(2, 24), scale)


def test_q_coordinates_stay_residual_while_physical_output_saturates():
    controller = ReferenceResidual(torch.ones(2, 24))
    residual = torch.ones(1, 22)
    physical = controller.physical_commands(residual, 0)
    # Clipping is part of the contextual environment transform. It must not
    # overwrite replay's issued residual with a physical action in another MDP.
    assert residual.eq(1).all() and physical.eq(1).all()


def test_online_update_enables_gradient_inside_replay_no_grad(tmp_path):
    path = tmp_path/'measured.bin'; path.write_bytes(b'actual-source')
    actor = torch.zeros(2, 464); critic = torch.zeros(2, 530)
    measured = dict(actor_obs=actor, critic_obs=critic, next_actor_obs=actor,
        next_critic_obs=critic, action=torch.zeros(2, 24),
        reward=torch.zeros(2), terminated=torch.tensor([False, True]))
    pilot = ResidualSACPilot(measured, path, tmp_path, training=False, updates_per_step=1)
    pilot.training = True; pilot.online_rows = 63
    with torch.no_grad():
        _, previous = pilot.act(actor[:1], critic[:1], 0)
        pilot.observe(previous, actor[:1], critic[:1], torch.zeros(1), torch.tensor([False]), 0)
    assert pilot.actor_updates == 1 and pilot.critic_updates == 1
    assert pilot.latest['actor_updated']
    assert all(torch.isfinite(p).all() for p in pilot.agent.parameters())
    checkpoint = pilot.save(final=True)
    resumed = ResidualSACPilot(measured,path,tmp_path,checkpoint=checkpoint,training=True)
    assert resumed.actor_updates == 1
    assert resumed.replay.size == 3  # two actual seed rows and the issued online row
    assert len(resumed.online_history) == 1


def test_resume_rejects_changed_reference_or_residual_scale(tmp_path):
    path=tmp_path/'source';path.write_bytes(b'measured')
    actor=torch.zeros(1,464);critic=torch.zeros(1,530)
    measured=dict(actor_obs=actor,critic_obs=critic,next_actor_obs=actor,
        next_critic_obs=critic,action=torch.zeros(1,24),reward=torch.zeros(1),terminated=torch.ones(1,dtype=torch.bool))
    pilot=ResidualSACPilot(measured,path,tmp_path,training=False)
    checkpoint=pilot.save()
    with pytest.raises(ValueError,match='reference/scale'):
        ResidualSACPilot(measured,path,tmp_path,scale=.1,checkpoint=checkpoint,training=False)


def test_goal_residual_is_a_position_offset_not_an_accumulating_command():
    raw=torch.zeros(3,464);raw[:,439]=1
    raw[:,71:77]=torch.tensor([1.,0,0,0,1,0])
    controller=ReferenceGoalResidual(dict(action=torch.zeros(3,24),actor_obs=raw,next_actor_obs=raw))
    residual=torch.zeros(1,22);residual[:,4]=1
    first=controller.physical_commands(residual,0,raw[:1])
    assert first[0,4] == .05
    actual=raw[:1].clone();actual[:,420]=first[:,4]*.02
    second=controller.physical_commands(residual,1,actual)
    assert second[0,4].abs() < 1e-7
    assert controller.context(4,1)[0,-1]>1  # elapsed time does not alias after reference


def test_goal_base_feedback_restores_reference_pose():
    raw=torch.zeros(2,464);raw[:,439]=1
    raw[:,71:77]=torch.tensor([1.,0,0,0,1,0])
    controller=ReferenceGoalResidual(dict(action=torch.zeros(2,24),actor_obs=raw,next_actor_obs=raw))
    residual=torch.zeros(1,22);residual[:,0]=1
    actual=raw[:1].clone();actual[:,68]=-.05*.15/30
    physical=controller.physical_commands(residual,1,actual)
    assert physical[0,0].abs()<1e-7


def test_uniform_scalar_and_per_joint_isaac_scales_are_supported():
    validate_goal_feedback_rates(torch.tensor([.15,.15,.5]),
        torch.tensor([[.01]+[.02]*14]),.01,.1,1/30)
    with pytest.raises(ValueError,match='rates differ'):
        validate_goal_feedback_rates([.25,.25,.7],[.01]+[.02]*14,.01,.1,1/30)
