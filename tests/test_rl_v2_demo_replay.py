"""The checked-in two Quest successes must be usable by the current SAC."""

from pathlib import Path

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.demo_replay import load_v2_grasp_demonstrations


DATASET = Path(__file__).resolve().parents[1] / "examples/demos/v2_grasp_quest_success.hdf5"


def test_two_successful_quest_episodes_convert_to_current_replay_contract():
    batch, metadata = load_v2_grasp_demonstrations(
        DATASET, self_collision_enabled=False)
    assert metadata["episodes"] == 2
    assert metadata["transitions"] == 910
    assert batch["actor_obs"].shape == (910, 441)
    assert batch["critic_obs"].shape == (910, 507)
    assert batch["next_actor_obs"].shape == (910, 441)
    assert batch["next_critic_obs"].shape == (910, 507)
    assert batch["action"].shape == (910, 25)
    assert int(batch["terminated"].sum()) == 2
    torch.testing.assert_close(batch["critic_obs"][:, :441], batch["actor_obs"])
    torch.testing.assert_close(batch["next_critic_obs"][:, :441], batch["next_actor_obs"])
    assert bool(torch.isfinite(batch["actor_obs"]).all())
    assert bool(torch.isfinite(batch["next_actor_obs"]).all())
    assert bool((batch["actor_obs"][:, 350:386].abs().sum(-1) > 0).all())


def test_demo_rejects_self_collision_contract_mismatch():
    with pytest.raises(ValueError, match="self-collision"):
        load_v2_grasp_demonstrations(DATASET, self_collision_enabled=True)
