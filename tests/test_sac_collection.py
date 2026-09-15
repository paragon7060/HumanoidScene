"""Focused SAC regression: pre-action readiness, terminal replay and warmup."""

import json
from types import SimpleNamespace

import torch

from kuavo_isaaclab_scene.rl.runners import train_sac


def test_settling_is_excluded_but_enabled_terminal_transition_is_kept(tmp_path, monkeypatch):
    buffers = []
    original = train_sac.ReplayBuffer

    def capture_buffer(*args):
        buffer = original(*args)
        buffers.append(buffer)
        return buffer

    monkeypatch.setattr(train_sac, "ReplayBuffer", capture_buffer)

    class Env:
        device, num_envs, step_dt = "cpu", 2, .1
        action_manager = SimpleNamespace(total_action_dim=1)

        def __init__(self):
            self.i = 0
            self.command = SimpleNamespace(
                settling=SimpleNamespace(ready=torch.zeros(2, dtype=torch.bool)),
                metrics={"grasp_right": torch.zeros(2)})
            self.command_manager = SimpleNamespace(get_term=lambda _: self.command)
            self.reward_manager = SimpleNamespace(active_terms=["cost"], _step_reward=-torch.ones(2, 1))

        def reset(self):
            return {"policy": torch.zeros(2, 1)}, {}

        def step(self, action):
            self.i += 1
            self.command.settling.ready[:] = torch.tensor(
                [(True, False), (False, True), (True, True), (True, True)][self.i - 1])
            obs = torch.full((2, 1), float(self.i))
            final = obs + 100
            terminated = torch.tensor([self.i == 2, False])
            # Returned terminal metrics differ from the post-reset zero metrics.
            info = {"transition_next_obs": final,
                    "transition_task_metrics": {"grasp_right": terminated.float()}}
            return {"policy": obs}, torch.full((2,), -.1), terminated, torch.zeros(2, dtype=torch.bool), info

    args = SimpleNamespace(replay_capacity=16, replay_device="cpu", learning_starts=2,
        warmup_vector_steps=1, updates_per_step=2, batch_size=2, max_iterations=1,
        rollout_steps=4, save_interval=1, keep_checkpoints=2)
    train_sac.train(Env(), args, tmp_path)
    replay = buffers[0]
    assert replay.size == 4
    assert replay.data["obs"][:4, 0].tolist() == [1, 2, 3, 3]
    assert replay.data["next_obs"][:4, 0].tolist() == [102, 103, 104, 104]
    assert replay.data["terminated"][:4].tolist() == [True, False, False, False]
    row = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert row["valid_transitions"] == row["skipped_settling_transitions"] == 4
    assert row["warmup_transitions_this_iteration"] == 2
    assert row["optimizer_updates"] == 3
    assert row["workcell/grasp_right"] == .25
    assert abs(row["reward/cost"] - row["reward_per_valid_step"]) < 1e-6
    state = torch.load(next(tmp_path.glob("*.pt")), weights_only=True)
    assert state["model"]["normalizer.count"].item() == 4


def test_empty_replay_add_is_a_noop():
    replay = train_sac.ReplayBuffer(2, 1, 1)
    replay.add(obs=torch.empty(0, 1))
    assert replay.size == replay.cursor == 0


def test_nonfinite_observations_and_terminal_data_are_excluded(tmp_path, monkeypatch):
    buffers = []
    original = train_sac.ReplayBuffer
    def capture_buffer(*args):
        buffer = original(*args)
        buffers.append(buffer)
        return buffer
    monkeypatch.setattr(train_sac, "ReplayBuffer", capture_buffer)
    class Env:
        device, num_envs, step_dt = "cpu", 2, .1
        action_manager = SimpleNamespace(total_action_dim=1)
        command = SimpleNamespace(settling=SimpleNamespace(ready=torch.ones(2, dtype=torch.bool)), metrics={})
        command_manager = SimpleNamespace(get_term=lambda _: Env.command)
        reward_manager = SimpleNamespace(active_terms=[], _step_reward=torch.empty(2, 0))
        def reset(self):
            return {"policy": torch.tensor([[0.], [float("nan")]])}, {}
        def step(self, action):
            assert torch.isfinite(action).all()
            return ({"policy": torch.zeros(2, 1)}, torch.tensor([-.1, -.1]),
                    torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool),
                    {"transition_next_obs": torch.tensor([[1.], [float("nan")]])})
    args = SimpleNamespace(replay_capacity=4, replay_device="cpu", learning_starts=100,
        warmup_vector_steps=1, updates_per_step=1, batch_size=2, max_iterations=1,
        rollout_steps=1, save_interval=1, keep_checkpoints=2)
    train_sac.train(Env(), args, tmp_path)
    assert buffers[0].size == 1
    assert torch.isfinite(buffers[0].data["next_obs"][:1]).all()
    metrics = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert metrics["nonfinite_transitions"] == 1
    assert metrics["skipped_settling_transitions"] == 0
    state = torch.load(next(tmp_path.glob("*.pt")), weights_only=True)
    assert state["model"]["normalizer.count"].item() == 1
