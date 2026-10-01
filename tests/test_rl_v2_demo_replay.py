"""The checked-in two Quest successes must be usable by the current SAC."""

from pathlib import Path
import json

import h5py

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.demo_replay import load_v2_grasp_demonstrations
from kuavo_isaaclab_scene.rl.multi_box.experiments.kinematic_exploration import successful_demo_grasp_offsets


DATASET = Path(__file__).resolve().parents[1] / "examples/demos/v2_grasp_quest_success.hdf5"


def test_two_successful_quest_episodes_convert_to_current_replay_contract():
    batch, metadata = load_v2_grasp_demonstrations(
        DATASET, self_collision_enabled=False)
    assert metadata["episodes"] == 2
    assert metadata["transitions"] == 910
    assert batch["actor_obs"].shape == (910, 464)
    assert batch["critic_obs"].shape == (910, 530)
    assert batch["next_actor_obs"].shape == (910, 464)
    assert batch["next_critic_obs"].shape == (910, 530)
    assert batch["action"].shape == (910, 24)
    assert int(batch["terminated"].sum()) == 2
    torch.testing.assert_close(batch["critic_obs"][:, :464], batch["actor_obs"])
    torch.testing.assert_close(batch["next_critic_obs"][:, :464], batch["next_actor_obs"])
    assert (batch["actor_obs"][:, 416:440] == 0).all()
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


def test_nominal_demos_do_not_silently_become_articulated_observations():
    with pytest.raises(ValueError, match="flap pose source"):
        load_v2_grasp_demonstrations(
            DATASET, self_collision_enabled=False, flap_pose_source="articulated")
    _, metadata = load_v2_grasp_demonstrations(
        DATASET, self_collision_enabled=False, flap_pose_source="articulated",
        allow_nominal_flap_prior=True)
    assert metadata["source_flap_pose_source"] == "nominal"
    assert metadata["requested_flap_pose_source"] == "articulated"
    assert metadata["nominal_flap_prior"] is True


def test_executed_success_import_rejects_equal_dimensions_with_changed_geometry(tmp_path):
    from kuavo_isaaclab_scene.rl.multi_box.experiments.train_grasp_v2_sac import _compatible_checkpoint
    manifest = {"observation_contract": "neutral_flap_center_controller_state_actual_base_twist_v2"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    requested = {"observation_contract": "perceived_articulated_flap_center_controller_state_actual_base_twist_v3"}
    with pytest.raises(ValueError, match="observation_contract"):
        _compatible_checkpoint(tmp_path / "checkpoint.pt", requested, data_only=True)


def test_retargeted_grasp_requires_physical_pinch_not_only_a_close_command():
    batch, _ = load_v2_grasp_demonstrations(DATASET, self_collision_enabled=False)
    offset = successful_demo_grasp_offsets(batch, 0.335)
    assert offset.shape == (2, 3) and torch.isfinite(offset).all()
    assert ((offset.norm(dim=-1) > 0.03) & (offset.norm(dim=-1) < 0.15)).all()
    missing = {**batch, "critic_obs": batch["critic_obs"].clone()}
    missing["critic_obs"][:, 464 + 35:464 + 37] = 0
    with pytest.raises(ValueError, match="physical pinch"):
        successful_demo_grasp_offsets(missing, 0.335)


def test_native_bent_panel_goals_are_calibrated_in_panel_frame():
    import math
    from kuavo_isaaclab_scene.rl.multi_box.demo_replay import _rotation_matrix
    from kuavo_isaaclab_scene.rl.multi_box.experiments.kinematic_exploration import target_token
    batch, _ = load_v2_grasp_demonstrations(DATASET, self_collision_enabled=False)
    original = successful_demo_grasp_offsets(batch, .335)
    changed = batch["actor_obs"].clone()
    box, _ = target_token(changed)
    box_rotation = _rotation_matrix(box[:, 15:21])
    tcp_rotation = _rotation_matrix(changed[:, 50:68].reshape(-1, 2, 9)[..., 3:])
    c = math.sqrt(.5)
    tilt = torch.tensor([[c, 0., c], [0., 1., 0.], [-c, 0., c]])
    panel_rotation = box_rotation @ tilt
    local = tcp_rotation.transpose(-1, -2) @ panel_rotation[:, None]
    relative6d = torch.cat((local[..., :, 0], local[..., :, 1]), -1)
    relations = changed[:, 350:386].reshape(-1, 2, 2, 9)
    relations[..., 3:] = relative6d[:, :, None]
    offsets = successful_demo_grasp_offsets({**batch, "actor_obs": changed}, .335)
    torch.testing.assert_close(offsets, (tilt.T @ original[..., None]).squeeze(-1), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("source_actor_dim", [440, 464])
@pytest.mark.parametrize("source_flap", ["nominal", "articulated"])
def test_native_upright_recordings_are_accepted_without_legacy_pose_conversion(tmp_path, source_actor_dim, source_flap):
    batch, _ = load_v2_grasp_demonstrations(DATASET, self_collision_enabled=False)
    with h5py.File(DATASET) as legacy:
        manifest = json.loads(legacy.attrs["manifest_json"])
    manifest.update(action_dim=24, actor_obs_dim=source_actor_dim, critic_obs_dim=source_actor_dim + 66)
    manifest["action_terms"][2] = ["height", 2]
    manifest["multi_box"]["flap_pose_source"] = source_flap
    path = tmp_path / "native.hdf5"
    with h5py.File(path, "w") as output:
        output.attrs.update(format="kuavo_v2_grasp_sac_transitions", format_version=1,
                            manifest_json=json.dumps(manifest))
        episode = output.create_group("episodes/episode_000000")
        episode.attrs["success"] = True
        transitions = episode.create_group("transitions")
        for key, value in batch.items():
            if source_actor_dim == 440 and key in ("actor_obs", "next_actor_obs", "critic_obs", "next_critic_obs"):
                value = torch.cat((value[:, :416], value[:, 440:]), -1)
            transitions.create_dataset(key, data=value[:389].numpy())
        success = torch.zeros(389, dtype=torch.bool)
        success[-1] = True
        transitions.create_dataset("success", data=success.numpy())
    restored, metadata = load_v2_grasp_demonstrations(
        path, self_collision_enabled=False, flap_pose_source=source_flap)
    torch.testing.assert_close(restored["actor_obs"], batch["actor_obs"][:389])
    torch.testing.assert_close(restored["action"], batch["action"][:389])
    assert metadata["action_conversion"] == "native_upright_binary_gripper_v3"
    assert metadata["source_flap_pose_source"] == source_flap
    if source_flap == "articulated":
        with pytest.raises(ValueError, match="flap pose source"):
            load_v2_grasp_demonstrations(path, self_collision_enabled=False)
