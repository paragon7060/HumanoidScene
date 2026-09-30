"""The checked-in two Quest successes must be usable by the current SAC."""

from pathlib import Path
import json

import h5py

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.demo_replay import load_v2_grasp_demonstrations


DATASET = Path(__file__).resolve().parents[1] / "examples/demos/v2_grasp_quest_success.hdf5"


def test_two_successful_quest_episodes_convert_to_current_replay_contract():
    batch, metadata = load_v2_grasp_demonstrations(
        DATASET, self_collision_enabled=False)
    assert metadata["episodes"] == 2
    assert metadata["transitions"] == 910
    assert batch["actor_obs"].shape == (910, 440)
    assert batch["critic_obs"].shape == (910, 506)
    assert batch["next_actor_obs"].shape == (910, 440)
    assert batch["next_critic_obs"].shape == (910, 506)
    assert batch["action"].shape == (910, 24)
    assert int(batch["terminated"].sum()) == 2
    torch.testing.assert_close(batch["critic_obs"][:, :440], batch["actor_obs"])
    torch.testing.assert_close(batch["next_critic_obs"][:, :440], batch["next_actor_obs"])
    assert metadata["action_terms"][2] == ["height", 2]
    assert metadata["action_conversion"] == "s63_upright_xz_jacobian_binary_gripper_v2"
    assert set(batch["action"][:, 20:22].unique().tolist()) == {-1.0, 1.0}
    assert 0 <= metadata["torso_action_saturated_fraction"] < 1
    assert metadata["mean_discarded_waist_pitch_command_rad"] >= 0
    assert bool(torch.isfinite(batch["actor_obs"]).all())
    assert bool(torch.isfinite(batch["next_actor_obs"]).all())
    assert bool((batch["actor_obs"][:, 350:386].abs().sum(-1) > 0).all())


def test_demo_rejects_self_collision_contract_mismatch():
    with pytest.raises(ValueError, match="self-collision"):
        load_v2_grasp_demonstrations(DATASET, self_collision_enabled=True)


def test_native_upright_recordings_are_accepted_without_legacy_pose_conversion(tmp_path):
    batch, _ = load_v2_grasp_demonstrations(DATASET, self_collision_enabled=False)
    with h5py.File(DATASET) as legacy:
        manifest = json.loads(legacy.attrs["manifest_json"])
    manifest.update(action_dim=24, actor_obs_dim=440, critic_obs_dim=506)
    manifest["action_terms"][2] = ["height", 2]
    path = tmp_path / "native.hdf5"
    with h5py.File(path, "w") as output:
        output.attrs.update(format="kuavo_v2_grasp_sac_transitions", format_version=1,
                            manifest_json=json.dumps(manifest))
        episode = output.create_group("episodes/episode_000000")
        episode.attrs["success"] = True
        transitions = episode.create_group("transitions")
        for key, value in batch.items():
            transitions.create_dataset(key, data=value[:389].numpy())
        success = torch.zeros(389, dtype=torch.bool)
        success[-1] = True
        transitions.create_dataset("success", data=success.numpy())
    restored, metadata = load_v2_grasp_demonstrations(path, self_collision_enabled=False)
    torch.testing.assert_close(restored["actor_obs"], batch["actor_obs"][:389])
    torch.testing.assert_close(restored["action"], batch["action"][:389])
    assert metadata["action_conversion"] == "native_upright_binary_gripper_v3"
