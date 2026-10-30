"""CPU numerical/integration checks for the new learners, independent of Isaac."""

import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from kuavo_isaaclab_scene.rl.algorithms.common import ObservationNormalizer, generalized_advantage
from kuavo_isaaclab_scene.rl.algorithms.sac import SAC, SACConfig, ReplayBuffer, SquashedActor, soft_target
from kuavo_isaaclab_scene.rl.algorithms.diffusion import DiffusionPolicy, DiffusionConfig, load_diffusion
from kuavo_isaaclab_scene.rl.algorithms.dppo import DPPO, DPPOConfig, clipped_objective
from kuavo_isaaclab_scene.rl.data.episodes import EpisodeDataset, EpisodeWriter
from kuavo_isaaclab_scene.rl.envs.terminal_observation import TerminalObservationMixin
from kuavo_isaaclab_scene.rl.runners.storage import save_checkpoint, load_checkpoint, check_contract


@pytest.fixture(autouse=True)
def deterministic_cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(12)
    yield
    torch.set_num_threads(old)


def manifest():
    return {"contract_hash": "test", "actions": {"arm": 2}, "action_config": {"scale": .02},
            "observations": {"policy": [4]}, "task": {"name": "pick"}}


def test_normalizer_merges_batches_without_changing_statistics():
    x = torch.randn(100, 4) * 3 + 5
    normalizer = ObservationNormalizer(4)
    normalizer.update(x[:17])
    normalizer.update(x[17:])
    torch.testing.assert_close(normalizer.mean, x.mean(0))
    torch.testing.assert_close(normalizer.var, x.var(0, unbiased=False))
    torch.testing.assert_close(normalizer(x).mean(0), torch.zeros(4), atol=1e-6, rtol=0)


def test_replay_wrap_and_oversize_retain_recent_transitions():
    replay = ReplayBuffer(5, 1, 1)
    def add(values):
        data = torch.tensor(values, dtype=torch.float32)
        replay.add(obs=data[:, None], action=data[:, None], reward=data, next_obs=(data + 1)[:, None],
                   terminated=torch.zeros(len(data), dtype=torch.bool))
    add([0, 1, 2])
    add([3, 4, 5, 6])
    assert replay.size == 5 and sorted(replay.data["reward"].tolist()) == [2, 3, 4, 5, 6]
    add(list(range(10, 20)))
    assert sorted(replay.data["reward"].tolist()) == [15, 16, 17, 18, 19]
    sample = replay.sample(100, "cpu")
    torch.testing.assert_close(sample["next_obs"] - sample["obs"], torch.ones(100, 1))


def test_sac_squash_log_probability_matches_transformed_distribution():
    actor = SquashedActor(4, 2, 16)
    obs = torch.randn(32, 4)
    action, logp = actor(obs)
    mean, log_std = actor.network(obs).chunk(2, -1)
    distribution = torch.distributions.TransformedDistribution(
        torch.distributions.Normal(mean, log_std.clamp(-5, 2).exp()),
        [torch.distributions.TanhTransform(cache_size=1)])
    torch.testing.assert_close(logp, distribution.log_prob(action).sum(-1), atol=2e-4, rtol=2e-4)
    with torch.no_grad():
        actor.network[-1].bias[:2].fill_(100)
    saturated, logp = actor(obs)
    assert (saturated.abs() <= 1).all() and torch.isfinite(logp).all()


def test_sac_terminal_mask_and_timeout_bootstrap():
    target = soft_target(torch.tensor([1., 1.]), torch.tensor([True, False]),
                         torch.tensor([10., 10.]), torch.zeros(2), .2, .9)
    torch.testing.assert_close(target, torch.tensor([1., 10.]))


def test_gae_uses_terminal_observation_but_stops_trace_at_timeout():
    advantage, returns = generalized_advantage(
        torch.tensor([[1.], [100.]]), torch.tensor([[2.], [3.]]), torch.tensor([[10.], [50.]]),
        torch.tensor([[False], [True]]), torch.tensor([[True], [True]]), gamma=.9, lam=.95)
    torch.testing.assert_close(advantage, torch.tensor([[8.], [97.]]))
    torch.testing.assert_close(returns, torch.tensor([[10.], [100.]]))


def test_terminal_mixin_preserves_pre_reset_obs_for_only_reset_envs():
    class FakeBase:
        def __init__(self):
            self.state = torch.zeros(3, 4)
            self.observation_manager = SimpleNamespace(compute=lambda **kw: {"policy": self.state})
        def _reset_idx(self, ids):
            self.state[ids] = -5
        def step(self, action):
            self.state[:] = torch.tensor([[1.], [2.], [3.]])
            self._reset_idx(torch.tensor([1]))
            return {"policy": self.state}, torch.zeros(3), torch.zeros(3, dtype=torch.bool), torch.tensor([False, True, False]), {}
    class Env(TerminalObservationMixin, FakeBase):
        pass
    env = Env()
    env._reset_idx(torch.tensor([0]))  # Explicit initial reset must not capture.
    obs, _, _, _, extras = env.step(None)
    torch.testing.assert_close(obs["policy"][1], torch.full((4,), -5.))
    torch.testing.assert_close(extras["transition_next_obs"], torch.tensor([[1.]*4, [2.]*4, [3.]*4]))


@pytest.mark.parametrize("steps", [1, 4, 20])
def test_diffusion_stored_chain_likelihoods_match_recomputation(steps):
    policy = DiffusionPolicy(4, 2, DiffusionConfig(3, steps, 32, .1))
    obs = torch.randn(8, 4)
    action, chain, old = policy.sample(obs)
    assert action.shape == (8, 3, 2) and (action.abs() <= 1).all()
    assert chain.shape == (8, steps + 1, 3, 2)
    for i in range(steps):
        new = policy.transition_log_prob(obs, chain[:, i], chain[:, i + 1], torch.full((8,), steps - i - 1))
        torch.testing.assert_close(new, old[:, i])
    # The latent distribution remains Gaussian; clipping is only applied to executed actions.
    assert chain.abs().max() > 1 and torch.isfinite(old).all()


def test_ppo_clipping_limits_improvements_but_not_harmful_updates():
    new, old, adv = torch.tensor([2., .5]).log(), torch.zeros(2), torch.tensor([1., -1.])
    loss, kl = clipped_objective(new, old, adv, torch.full((2,), .2))
    torch.testing.assert_close(loss, torch.tensor(-.2))  # mean(-1.2, +0.8)
    assert kl > 0


def test_diffusion_bc_loss_learns_fixed_denoising_problem():
    policy = DiffusionPolicy(4, 2, DiffusionConfig(2, 4, 32, .1))
    obs, actions = torch.randn(32, 4), torch.randn(32, 2, 2).tanh()
    t, noise = torch.randint(4, (32,)), torch.randn_like(actions)
    optimizer = torch.optim.Adam(policy.network.parameters(), lr=.01)
    initial = policy.loss(obs, actions, t, noise).item()
    for _ in range(80):
        loss = policy.loss(obs, actions, t, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert policy.loss(obs, actions, t, noise).item() < initial * .1


def test_dppo_updates_network_and_keeps_bc_normalizer_frozen():
    policy = DiffusionPolicy(4, 2, DiffusionConfig(2, 4, 32, .1))
    obs = torch.randn(64, 4)
    policy.normalizer.update(obs)
    _, chain, old = policy.sample(obs)
    agent = DPPO(policy, DPPOConfig(epochs=2, minibatch_size=32))
    before = {key: value.clone() for key, value in policy.state_dict().items()}
    report = agent.update(obs, chain, old, torch.linspace(-1, 1, 64), torch.randn(64))
    assert report["actor_updates"] > 0
    assert any(not torch.equal(v, before[k]) for k, v in policy.state_dict().items() if k.startswith("network"))
    assert all(torch.equal(v, before[k]) for k, v in policy.state_dict().items() if k.startswith("normalizer"))
    restored_policy = load_diffusion(agent.checkpoint())
    restored = DPPO(restored_policy, agent.config)
    restored.restore(agent.checkpoint())
    torch.testing.assert_close(restored.value(obs), agent.value(obs))


def test_sac_update_and_checkpoint_round_trip(tmp_path):
    agent = SAC(4, 2, SACConfig(hidden=32))
    obs = torch.randn(32, 4)
    batch = dict(obs=obs, action=torch.rand(32, 2) * 2 - 1, next_obs=obs + .1,
                 reward=torch.randn(32), terminated=torch.zeros(32, dtype=torch.bool))
    original = [p.clone() for p in agent.actor.parameters()]
    report = agent.update(batch)
    assert all(np.isfinite(v) for v in report.values())
    assert any(not torch.equal(a, b) for a, b in zip(original, agent.actor.parameters()))
    for i in range(4):
        path = save_checkpoint(tmp_path, agent.checkpoint(), i, keep=2)
    assert len(list(tmp_path.glob("*.pt"))) == 2 and not list(tmp_path.glob("*.tmp"))
    state = load_checkpoint(path)
    restored = SAC(4, 2, SACConfig(**state["config"]))
    restored.restore(state)
    torch.testing.assert_close(agent.act(obs, True), restored.act(obs, True))
    assert "replay" not in state


def test_dataset_chunks_do_not_cross_episode_boundaries(tmp_path):
    path = tmp_path / "episodes.hdf5"
    writer = EpisodeWriter(path, manifest())
    writer.add(np.zeros((5, 4)), np.zeros((5, 2)), True)
    writer.add(np.ones((5, 4)), np.ones((5, 2)), True)
    writer.add(np.ones((5, 4))*2, np.ones((5, 2))*.5, False)
    writer.close()
    data = EpisodeDataset(path, horizon=3)
    try:
        assert len(data) == 6
        assert data[2][1].sum() == 0 and data[3][1].sum() == 6
        normalizer = ObservationNormalizer(4)
        data.fit_normalizer(normalizer)
        torch.testing.assert_close(normalizer.mean, torch.full((4,), .5))
        torch.testing.assert_close(normalizer.var, torch.full((4,), .25))
    finally:
        data.close()


def test_empty_success_data_and_wrong_contract_are_rejected(tmp_path):
    path = tmp_path / "episodes.hdf5"
    writer = EpisodeWriter(path, manifest())
    writer.add(np.zeros((5, 4)), np.zeros((5, 2)), False)
    writer.close()
    with pytest.raises(ValueError, match="No eligible"):
        EpisodeDataset(path, 2)
    changed = dict(manifest(), actions={"absolute_joints": 36})
    with pytest.raises(ValueError, match="actions"):
        check_contract(manifest(), changed)
    with pytest.raises(ValueError, match="native"):
        load_diffusion({"algorithm": "lerobot"})


class ToyEnv:
    """Small vector MDP for executing actual training loops on CPU."""
    device, num_envs = "cpu", 4
    action_manager = SimpleNamespace(total_action_dim=2)
    def __init__(self):
        self.steps = 0
    def reset(self):
        self.obs = torch.zeros(4, 4)
        return {"policy": self.obs}, {}
    def step(self, actions):
        assert torch.isfinite(actions).all() and actions.abs().max() <= 1
        self.steps += 1
        self.obs = torch.cat((actions, actions.square()), -1)
        trunc = torch.full((4,), self.steps % 3 == 0, dtype=torch.bool)
        final = self.obs.clone()
        self.obs[trunc] = 0
        return {"policy": self.obs}, -actions.square().sum(-1), torch.zeros(4, dtype=torch.bool), trunc, {"transition_next_obs": final}


def test_actual_sac_and_dppo_loops_complete_and_resume(tmp_path):
    from kuavo_isaaclab_scene.rl.runners.train_sac import train as train_sac
    from kuavo_isaaclab_scene.rl.runners.train_dppo import train as train_dppo
    args = SimpleNamespace(replay_capacity=64, replay_device="cpu", updates_per_step=1, batch_size=8,
                           max_iterations=2, rollout_steps=4, learning_starts=4, save_interval=1000,
                           keep_checkpoints=2, epochs=1, critic_warmup=0)
    sac_path, dp_path = tmp_path / "sac", tmp_path / "dppo"
    sac_path.mkdir()
    dp_path.mkdir()
    train_sac(ToyEnv(), args, sac_path)
    sac = load_checkpoint(next(sac_path.glob("*.pt")))
    train_sac(ToyEnv(), args, sac_path, sac)
    policy = DiffusionPolicy(4, 2, DiffusionConfig(2, 4, 32, .1))
    train_dppo(ToyEnv(), args, dp_path, policy.checkpoint())
    dp = load_checkpoint(next(dp_path.glob("*.pt")))
    train_dppo(ToyEnv(), args, dp_path, dp)
    assert load_checkpoint(sorted(dp_path.glob("*.pt"))[-1])["iteration"] == 4
    assert len((sac_path / "metrics.jsonl").read_text().splitlines()) == 4


def test_offline_pretrain_entrypoint_produces_dppo_loadable_checkpoint(tmp_path, monkeypatch):
    from kuavo_isaaclab_scene.rl.runners.pretrain_diffusion import main
    dataset_path = tmp_path / "demo.hdf5"
    writer = EpisodeWriter(dataset_path, manifest())
    writer.add(np.random.default_rng(4).normal(size=(12, 4)), np.zeros((12, 2)), True)
    writer.close()
    output = tmp_path / "pretrained"
    monkeypatch.setattr("sys.argv", ["pretrain", "--dataset", str(dataset_path), "--output", str(output),
                                    "--steps", "4", "--batch-size", "8", "--horizon", "2", "--hidden", "16",
                                    "--denoising-steps", "4", "--save-interval", "1"])
    main()
    state = load_checkpoint(sorted(output.glob("*.pt"))[-1])
    policy = load_diffusion(state)
    assert policy.normalizer.count.item() == 12
    assert len(list(output.glob("*.pt"))) == 2
    check_contract(json.loads((output / "manifest.json").read_text()), manifest())
    assert torch.isfinite(policy.sample(torch.zeros(2, 4))[0]).all()


def test_runner_requires_one_masked_gpu_and_maps_renderer_physically(monkeypatch):
    from kuavo_isaaclab_scene.rl.runners.alternatives import validate_args
    args = SimpleNamespace(rollout_steps=2, batch_size=8, replay_capacity=64, updates_per_step=1,
                           epochs=1, keep_checkpoints=2, collect_max_steps=32, learning_starts=4,
                           critic_warmup=0, method="sac", checkpoint=None, num_envs=2,
                           device="cuda:0", kit_args="", save_interval=None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(ValueError, match="ONE"):
        validate_args(args)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    validate_args(args)
    assert args.device == "cuda:0" and "activeGpu=2" in args.kit_args
    assert "multiGpu/enabled=false" in args.kit_args
