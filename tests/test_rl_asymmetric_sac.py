"""CPU checks for the multi-box deployable-actor/privileged-critic SAC."""

import json
from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.algorithms.asymmetric_sac import (
    AsymmetricReplayBuffer,
    AsymmetricSAC,
)
from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
from kuavo_isaaclab_scene.rl.runners.train_asymmetric_sac import (
    _SafetyDiagnostics,
    _reset_settling_metrics,
    _reward_breakdown,
    _sample_mixed_replay,
    _sample_warmup_action,
    _settle_initial_resets,
    _termination_snapshot,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.train_grasp_v2_sac import (
    _compatible_checkpoint,
    apply_run_profile,
)


def test_v2_safety_diagnostics_separates_unsafe_causes_and_force_bands():
    monitor = _SafetyDiagnostics("cpu")
    false = torch.zeros(3, dtype=torch.bool)
    monitor.record(SimpleNamespace(
        robot_rack_collision=torch.tensor([False, True, False]),
        obstacle_collision=torch.tensor([False, True, False]),
        workspace_limit=false, box_drop=torch.tensor([False, False, True]),
        box_lift_limit=false, box_speed_limit=false, self_collision=false,
        contact_eligible=torch.tensor([True, True, False]),
        rack_force_n=torch.tensor([0.2, 12.0, 100.0]),
        obstacle_force_n=torch.tensor([0.0, 6.0, 100.0]),
    ), torch.tensor([False, True, True]))

    result = monitor.report()
    assert result["unsafe_cause/robot_rack_collision"] == 1
    assert result["unsafe_cause/obstacle_collision"] == 1
    assert result["unsafe_cause/box_drop"] == 1
    assert result["unsafe_cause/overlap"] == 1
    assert result["unsafe_cause/unattributed"] == 0
    assert result["contact_force/eligible_samples"] == 2
    assert result["contact_force/rack_gt_0p1_n"] == 2
    assert result["contact_force/rack_gt_10p0_n"] == 1
    assert result["contact_force/obstacle_gt_5p0_n"] == 1
    assert result["contact_force/rack_max_n"] == 12.0


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


def test_demo_replay_remains_in_each_minibatch_after_online_replay_wraps():
    online = AsymmetricReplayBuffer(5, 4, 7, 2)
    demo = AsymmetricReplayBuffer(2, 4, 7, 2)
    online_batch = _batch(12)
    online_batch["reward"].fill_(0)
    online.add(**online_batch)
    demo_batch = _batch(2)
    demo_batch["reward"].fill_(5)
    demo.add(**demo_batch)
    sampled = _sample_mixed_replay(online, demo, 20, 0.2, "cpu")
    assert len(sampled["reward"]) == 20
    assert int((sampled["reward"] == 5).sum()) == 4


def test_warmup_limits_continuous_actions_but_explores_binary_grippers():
    widths = {"base": 3, "left_gripper": 1, "right_gripper": 1, "head": 2}
    env = SimpleNamespace(
        num_envs=100,
        device="cpu",
        action_manager=SimpleNamespace(
            total_action_dim=sum(widths.values()),
            active_terms=tuple(widths),
            get_term=lambda name: SimpleNamespace(action_dim=widths[name]),
        ),
    )
    action = _sample_warmup_action(env, 0.35)
    assert action.shape == (100, 7)
    assert bool((action[:, :3].abs() <= 0.35).all())
    assert bool((action[:, 5:].abs() <= 0.35).all())
    assert set(action[:, 3:5].unique().tolist()) == {-1.0, 1.0}


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


def test_asymmetric_sac_exploration_floor_and_diagnostics():
    agent = AsymmetricSAC(4, 7, 2, SACConfig(hidden=16, min_alpha=0.01))
    with torch.no_grad():
        agent.log_alpha.fill_(torch.tensor(0.001).log())
    report = agent.update(_batch())
    assert report["alpha"] >= 0.01 - 1e-7
    assert torch.isfinite(torch.tensor(report["policy_logp_mean"]))
    assert report["policy_action_std_mean"] > 0


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


def test_v2_initial_reset_settling_uses_zero_actions_until_every_env_is_ready():
    class Environment:
        step_dt = 0.1
        cfg = SimpleNamespace(multi_box=SimpleNamespace(
            reset_settle_timeout_seconds=1.0))
        action_manager = SimpleNamespace(action=torch.ones(2, 3))
        _multi_box_reset_settling = SimpleNamespace(
            ready=torch.tensor([False, False]),
            invalid=torch.tensor([False, False]),
            invalid_count=torch.zeros(2, dtype=torch.long),
            region_invalid_count=torch.zeros(2, dtype=torch.long),
            footprint_invalid_count=torch.zeros(2, dtype=torch.long),
            shelf_invalid_count=torch.zeros(2, dtype=torch.long),
            timeout_invalid_count=torch.zeros(2, dtype=torch.long),
            nonfinite_invalid_count=torch.zeros(2, dtype=torch.long),
        )

        def __init__(self):
            self.actions = []

        def step(self, action):
            self.actions.append(action.clone())
            if len(self.actions) == 1:
                self._multi_box_reset_settling.ready[0] = True
            if len(self.actions) == 2:
                self._multi_box_reset_settling.ready[1] = True
            return {"policy": torch.full((2, 1), len(self.actions))}, None, None, None, None

    env = Environment()
    observations, steps = _settle_initial_resets(
        env, {"policy": torch.zeros(2, 1)})
    assert steps == 2
    assert observations["policy"].tolist() == [[2], [2]]
    assert all(torch.equal(action, torch.zeros(2, 3)) for action in env.actions)


def test_v2_initial_reset_settling_can_start_with_ready_majority():
    class Environment:
        num_envs = 10
        step_dt = 0.1
        cfg = SimpleNamespace(multi_box=SimpleNamespace(
            reset_settle_timeout_seconds=0.1))
        action_manager = SimpleNamespace(action=torch.ones(10, 3))
        _multi_box_reset_settling = SimpleNamespace(
            ready=torch.tensor([True] * 9 + [False]),
            invalid=torch.zeros(10, dtype=torch.bool),
        )

        def step(self, action):
            assert torch.equal(action, torch.zeros(10, 3))
            return {"policy": torch.zeros(10, 1)}, None, None, None, None

    observations, steps = _settle_initial_resets(
        Environment(), {"policy": torch.ones(10, 1)})
    assert steps == 4
    assert torch.equal(observations["policy"], torch.zeros(10, 1))


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


def test_same_dimension_checkpoint_cannot_resume_with_changed_flap_goal(tmp_path):
    original = {name: None for name in (
        "task_family", "schema_version", "skill", "algorithm", "robot_model",
        "gripper", "actions", "observations", "observation_contract",
        "critic_mapping", "reward_profile", "exploration", "demonstrations",
        "self_collision",
    )}
    original["observation_contract"] = "nearest_surface_tcp_frame_v1"
    (tmp_path / "manifest.json").write_text(json.dumps(original))
    checkpoint = tmp_path / "checkpoint_00000001.pt"
    checkpoint.touch()
    updated = dict(original, observation_contract="neutral_flap_center_tcp_frame_v1")
    with pytest.raises(ValueError, match="observation_contract"):
        _compatible_checkpoint(checkpoint, updated)
