"""CPU checks for the multi-box deployable-actor/privileged-critic SAC."""

from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.algorithms.asymmetric_sac import (
    AsymmetricReplayBuffer,
    AsymmetricSAC,
)
from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
from kuavo_isaaclab_scene.rl.runners.train_asymmetric_sac import (
    _reset_settling_metrics,
    _reward_breakdown,
    _termination_snapshot,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.train_grasp_v2_sac import (
    apply_run_profile,
)


def _batch(count=32):
    actor = torch.randn(count, 4)
    critic = torch.randn(count, 7)
    return {
        "actor_obs": actor,
        "critic_obs": critic,
        "action": torch.rand(count, 2) * 2 - 1,
        "reward": torch.randn(count),
        "next_actor_obs": actor + 0.1,
        "next_critic_obs": critic - 0.1,
        "terminated": torch.zeros(count, dtype=torch.bool),
    }


def test_asymmetric_replay_keeps_actor_and_critic_views_separate():
    replay = AsymmetricReplayBuffer(5, 4, 7, 2)
    replay.add(**_batch(8))
    assert replay.size == 5
    assert replay.data["actor_obs"].shape == (5, 4)
    assert replay.data["critic_obs"].shape == (5, 7)
    sample = replay.sample(12, "cpu")
    assert sample["next_actor_obs"].shape == (12, 4)
    assert sample["next_critic_obs"].shape == (12, 7)


def test_asymmetric_sac_updates_and_restores_without_privileged_actor_input():
    torch.manual_seed(7)
    agent = AsymmetricSAC(4, 7, 2, SACConfig(hidden=32))
    batch = _batch()
    agent.update_normalizers(batch["actor_obs"], batch["critic_obs"])
    before = {name: value.clone() for name, value in agent.actor.state_dict().items()}
    report = agent.update(batch)
    assert all(torch.isfinite(torch.tensor(value)) for value in report.values())
    assert any(
        not torch.equal(value, before[name])
        for name, value in agent.actor.state_dict().items()
    )
    # Deployment action requires exactly the actor view; privileged features
    # are accepted only by the Q networks during training.
    assert agent.act(torch.zeros(3, 4)).shape == (3, 2)
    assert agent.q1[0].in_features == 7 + 2

    state = agent.checkpoint()
    restored = AsymmetricSAC(4, 7, 2, SACConfig(hidden=32))
    restored.restore(state)
    torch.testing.assert_close(
        restored.act(torch.zeros(3, 4), deterministic=True),
        agent.act(torch.zeros(3, 4), deterministic=True),
    )


def test_asymmetric_sac_rejects_checkpoint_dimension_mismatch():
    state = AsymmetricSAC(4, 7, 2, SACConfig(hidden=16)).checkpoint()
    target = AsymmetricSAC(4, 8, 2, SACConfig(hidden=16))
    try:
        target.restore(state)
    except ValueError as error:
        assert "dimensions differ" in str(error)
    else:
        raise AssertionError("Mismatched critic dimensions must be rejected")


def test_v2_reward_breakdown_is_reconstructed_before_sac_storage():
    env = SimpleNamespace(
        num_envs=2,
        _multi_box_grasp_reward_breakdown=SimpleNamespace(
            terms={
                "progress": torch.tensor([0.2, -0.1]),
                "success_event": torch.tensor([0.0, 3.0]),
            },
            total=torch.tensor([0.2, 2.9]),
        ),
    )
    terms, total = _reward_breakdown(env)
    assert set(terms) == {"progress", "success_event"}
    torch.testing.assert_close(total, torch.tensor([0.2, 2.9]))


def test_v2_reward_breakdown_rejects_inconsistent_total():
    env = SimpleNamespace(
        num_envs=1,
        _multi_box_grasp_reward_breakdown=SimpleNamespace(
            terms={"progress": torch.tensor([0.2])},
            total=torch.tensor([0.3]),
        ),
    )
    with pytest.raises(RuntimeError, match="do not sum"):
        _reward_breakdown(env)


def test_v2_terminal_snapshot_separates_failures_from_timeout():
    class Manager:
        active_terms = ["success", "unsafe", "time_out"]
        _term_cfgs = [
            SimpleNamespace(time_out=False),
            SimpleNamespace(time_out=False),
            SimpleNamespace(time_out=True),
        ]
        values = {
            "success": torch.tensor([True, False, False]),
            "unsafe": torch.tensor([False, True, False]),
            "time_out": torch.tensor([False, False, True]),
        }

        def get_term(self, name):
            return self.values[name]

    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        termination_manager=Manager(),
    )
    terms, terminated, truncated = _termination_snapshot(env)
    assert set(terms) == {"success", "unsafe", "time_out"}
    assert terminated.tolist() == [True, True, False]
    assert truncated.tolist() == [False, False, True]


def test_v2_reset_settling_metrics_expose_rejection_causes():
    settling = SimpleNamespace(
        ready=torch.tensor([True, False, False]),
        invalid=torch.tensor([False, True, False]),
        invalid_count=torch.tensor([0, 2, 0]),
        region_invalid_count=torch.tensor([0, 2, 0]),
        footprint_invalid_count=torch.tensor([0, 1, 0]),
        shelf_invalid_count=torch.tensor([0, 2, 0]),
        timeout_invalid_count=torch.tensor([0, 0, 0]),
        nonfinite_invalid_count=torch.tensor([0, 1, 0]),
    )
    metrics = _reset_settling_metrics(SimpleNamespace(
        _multi_box_reset_settling=settling))
    assert metrics == {
        "reset_ready_envs": 1,
        "reset_settling_envs": 1,
        "reset_invalid_total": 2,
        "reset_region_invalid_total": 2,
        "reset_footprint_invalid_total": 1,
        "reset_shelf_invalid_total": 2,
        "reset_timeout_invalid_total": 0,
        "reset_nonfinite_invalid_total": 1,
    }


def test_v2_sac_pilot_profile_is_bounded_but_performs_updates():
    args = SimpleNamespace(
        smoke_test=False, pilot=True, num_envs=4096, max_iterations=2000,
        rollout_steps=128, batch_size=1024, replay_capacity=250_000,
        learning_starts=100_000, warmup_vector_steps=450,
        updates_per_step=4, save_interval=50,
    )
    apply_run_profile(args)
    assert args.num_envs == 64
    assert args.max_iterations == 20
    assert args.rollout_steps == 32
    assert args.batch_size == 512
    assert args.replay_capacity == 50_000
    assert args.learning_starts == 4_096
    assert args.warmup_vector_steps == 64
    assert args.updates_per_step == 1
    assert args.save_interval == 5


def test_v2_sac_profiles_are_mutually_exclusive():
    args = SimpleNamespace(smoke_test=True, pilot=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        apply_run_profile(args)
