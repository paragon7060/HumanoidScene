"""Real RSL-RL likelihood, normalization, update and checkpoint checks on CPU."""
import io
import pytest
import torch
pytest.importorskip('rsl_rl')
from tensordict import TensorDict
from torch.distributions import Normal, TransformedDistribution, TanhTransform
from kuavo_isaaclab_scene.rl.agents.flap_ppo import FlapActorCritic, FlapPPO, LimitedNormalization


def policy():
    obs = TensorDict({'policy': torch.randn(8, 12)}, batch_size=[8])
    p = FlapActorCritic(obs, {'policy':['policy'], 'critic':['policy']}, 3,
        actor_obs_normalization=True, critic_obs_normalization=True,
        actor_hidden_dims=[16], critic_hidden_dims=[16])
    return p, obs


def test_rare_flags_bounded_and_noise_cannot_grow_without_limit():
    norm = LimitedNormalization(2)
    norm.update(torch.zeros(10000, 2))
    assert norm(torch.ones(1,2)).abs().max() <= 5
    p, obs = policy()
    for raw in (-100., 100.):
        with torch.no_grad(): p.std_logit.fill_(raw)
        a = p.act(obs)
        assert a.abs().max() < 1
        assert p.action_std.min() >= .05-1e-6 and p.action_std.max() <= .35+1e-6
        assert p.act_inference(obs).abs().max() < 1


def test_squashed_likelihood_matches_pytorch_and_gradients_are_finite():
    p, obs = policy()
    actions = p.act(obs)
    ref = TransformedDistribution(Normal(p.action_mean, p.action_std), [TanhTransform()])
    torch.testing.assert_close(p.get_actions_log_prob(actions), ref.log_prob(actions).sum(-1), atol=2e-5, rtol=2e-5)
    old_log = p.get_actions_log_prob(actions).detach()
    p.act(obs)
    torch.testing.assert_close((p.get_actions_log_prob(actions)-old_log).exp(), torch.ones(8))
    loss = -p.get_actions_log_prob(actions.detach()).mean()-.001*p.entropy.mean()+p.evaluate(obs).square().mean()
    loss.backward()
    assert all(torch.isfinite(x.grad).all() for x in p.parameters() if x.grad is not None)


def test_real_ppo_update_and_checkpoint_roundtrip():
    p, obs = policy()
    alg = FlapPPO(p, num_learning_epochs=2, num_mini_batches=2, device='cpu')
    alg.init_storage('rl', 8, 4, obs, [3])
    for _ in range(4):
        with torch.inference_mode():
            a = alg.act(obs)
            alg.process_env_step(obs, -a.square().sum(-1), torch.zeros(8,dtype=torch.bool), {})
    with torch.inference_mode(): alg.compute_returns(obs)
    stats=alg.update()
    assert all(torch.isfinite(torch.tensor(x)) for x in stats.values())
    assert 'policy_kl_sample' in stats
    buf=io.BytesIO();torch.save(p.state_dict(),buf);buf.seek(0)
    restored,_=policy();restored.load_state_dict(torch.load(buf,weights_only=True))
    torch.testing.assert_close(p.act_inference(obs),restored.act_inference(obs))
