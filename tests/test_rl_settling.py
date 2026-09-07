"""Reset settling must not let a policy redefine the lift baseline mid-episode."""

from types import SimpleNamespace
import torch

from kuavo_isaaclab_scene.rl.tasks.specs import task_spec
from kuavo_isaaclab_scene.rl.mdp.settling import ResetSettling, gate_actions


def test_each_environment_requires_stability_and_partial_reset_is_isolated():
    spec = task_spec("pick", reset_settle_seconds=.5)
    state = ResetSettling(2, "cpu", spec)
    velocities = torch.zeros(2, 1, 6)
    velocities[1, 0, 2] = -.4
    update = torch.ones(2, dtype=torch.bool)
    for _ in range(4):
        assert not state.advance(velocities, update, .1).any()
    assert not state.advance(velocities, update, .1).any()
    assert state.advance(velocities, update, .1).tolist() == [True, False]
    # An already latched reference cannot be re-latched after the box moves.
    for _ in range(15):
        assert not state.advance(velocities, update, .1)[0]
    assert state.failed.tolist() == [False, True]
    elapsed = state.elapsed.clone()
    state.reset(torch.tensor([1]))
    assert state.ready.tolist() == [True, False]
    assert state.elapsed[0] == elapsed[0] and state.elapsed[1] == 0


def test_nonconsecutive_stability_does_not_complete_settling_and_actions_hold():
    spec = task_spec("pick", reset_settle_seconds=.1, reset_settle_hold_seconds=.2)
    state = ResetSettling(2, "cpu", spec)
    vel = torch.zeros(2, 1, 6)
    update = torch.ones(2, dtype=torch.bool)
    state.advance(vel, update, .1)
    vel[1, 0, 3] = 1
    state.advance(vel, update, .1)
    vel.zero_()
    state.advance(vel, update, .1)
    assert state.ready.tolist() == [True, False]
    env = SimpleNamespace(cfg=SimpleNamespace(task=spec),
        command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(settling=state)))
    actions = torch.ones(2, 16)
    masked = gate_actions(env, actions)
    assert masked[0].eq(1).all() and masked[1].eq(0).all()
    assert actions.eq(1).all()
