"""Protect current measured success replay from stale VR rewards/reset seams."""

import json

import h5py
import numpy as np
import pytest

from kuavo_isaaclab_scene.rl.multi_box.experiments.executed_replay import (
    PHYSICAL_KEYS, read_executed_successes,
)


def _record(tmp_path):
    contract = {key: f"reviewed-{key}" for key in PHYSICAL_KEYS}
    contract.update(actions={"base": 3, "upper_body": 15, "height": 2,
                             "left_gripper": 1, "right_gripper": 1, "head": 2},
                    observations={"policy": [464], "critic": [66]},
                    reward_profile={"weights": {"success_event": 1}}, flap_pose_source="nominal")
    actor = np.zeros((3, 464), dtype=np.float32)
    actor[:, 0] = np.arange(3)
    actor[:, 439] = 1
    next_actor = np.concatenate((actor[1:], actor[-1:])).copy()
    next_actor[-1, 0] = 3
    critic = np.concatenate((actor, np.zeros((3, 66), dtype=np.float32)), -1)
    next_critic = np.concatenate((next_actor, np.zeros((3, 66), dtype=np.float32)), -1)
    next_critic[-1, 464+35:464+37] = 1
    next_critic[-1, 464+54] = 1
    meta = {"collection_source": "current_v2_environment_executed_vr_reference",
            "current_reward_verified_against_breakdown": True, "sim_device": "cuda:0",
            "old_demo_rewards_used": False, "training_contract": contract,
            "actor_obs_dim": 464, "critic_obs_dim": 530, "action_dim": 24,
            "control_dt": 1/30, "episode_seconds": 30., "multi_box": {"flap_pose_source": "nominal"}}
    path = tmp_path / "executed.hdf5"
    with h5py.File(path, "w") as output:
        output.attrs.update(format="kuavo_v2_grasp_sac_transitions", format_version=1,
                            manifest_json=json.dumps(meta))
        episode = output.create_group("episodes/attempt")
        episode.attrs["success"] = True
        initial = episode.create_group("initial_state")
        initial.attrs.update(schema="v2_physical_seed_v1", capture_timing="before_first_recorded_action")
        for key in ("scene", "drive_targets", "action_terms", "logical_boxes", "observations"):
            initial.create_group(key)
        initial["observations"].create_dataset("policy", data=actor[0])
        data = {"actor_obs": actor, "next_actor_obs": next_actor, "critic_obs": critic,
                "next_critic_obs": next_critic, "action": np.zeros((3, 24), dtype=np.float32),
                "reward": np.array([0., .25, 1.], dtype=np.float32),
                "terminated": np.array([False, False, True]),
                "success": np.array([False, False, True]), "unsafe": np.zeros(3, dtype=bool),
                "truncated": np.zeros(3, dtype=bool), "sim_time_s": np.arange(3)/30}
        transitions = episode.create_group("transitions")
        for key, value in data.items():
            transitions.create_dataset(key, data=value)
    return path, contract


def test_full_executed_approach_and_terminal_success_are_preserved(tmp_path):
    path, contract = _record(tmp_path)
    replay, audit = read_executed_successes(path, contract)
    assert replay["actor_obs"].shape == (3, 464)
    assert replay["next_actor_obs"][-1, 0] == 3
    assert replay["terminated"].tolist() == [False, False, True]
    assert replay["reward"].tolist() == [0., .25, 1.]
    assert audit["executed_rows"] == 3
    assert audit["successful_episodes"] == 1
    assert audit["recorded_old_demo_rewards_imported"] is False


def test_old_vr_rewards_and_cpu_comparisons_are_not_imported_into_gpu_q(tmp_path):
    path, contract = _record(tmp_path)
    with h5py.File(path, "r+") as output:
        meta = json.loads(output.attrs["manifest_json"])
        meta["sim_device"] = "cpu"
        output.attrs["manifest_json"] = json.dumps(meta)
    with pytest.raises(ValueError, match="current GPU"):
        read_executed_successes(path, contract)


def test_reward_or_observation_contract_change_requires_new_measured_data(tmp_path):
    path, contract = _record(tmp_path)
    contract["reward_profile"] = {"weights": {"success_event": 2}}
    with pytest.raises(ValueError, match="reward_profile"):
        read_executed_successes(path, contract)


def test_data_only_archive_cannot_be_restored_as_a_policy(tmp_path):
    from kuavo_isaaclab_scene.rl.multi_box.experiments.train_grasp_v2_sac import _compatible_checkpoint
    (_, contract) = _record(tmp_path)
    (tmp_path/'manifest.json').write_text(json.dumps(
        contract | {'artifact_type': 'executed_experience_only'}))
    with pytest.raises(ValueError, match='--experience-checkpoint'):
        _compatible_checkpoint(tmp_path/'checkpoint_00000000.pt', contract)


def test_reference_selection_never_crosses_episode_end():
    import torch
    from kuavo_isaaclab_scene.rl.multi_box.experiments.vr_reference import select_reference_episode
    batch = {'terminated': torch.tensor([False, True, False, False, True]),
             'actor_obs': torch.arange(5)[:, None]}
    assert select_reference_episode(batch, 0)['actor_obs'][:, 0].tolist() == [0, 1]
    assert select_reference_episode(batch, 1)['actor_obs'][:, 0].tolist() == [2, 3, 4]
    with pytest.raises(ValueError, match='episode index'):
        select_reference_episode(batch, 2)


def test_inferred_reference_cannot_overwrite_a_vector_training_scene():
    from types import SimpleNamespace
    from kuavo_isaaclab_scene.rl.multi_box.experiments.vr_reference import restore_inferred_scene
    with pytest.raises(ValueError, match='single-environment'):
        restore_inferred_scene(SimpleNamespace(num_envs=128), None)


@pytest.mark.parametrize("mutation,match", [
    ("gap", "observation gap"), ("terminal_reset", "terminal observation"),
    ("unsafe", "measured success"), ("missing_controller", "controller-state"),
    ("wrong_seed", "seed"),
])
def test_invalid_success_paths_are_rejected(tmp_path, mutation, match):
    path, contract = _record(tmp_path)
    with h5py.File(path, "r+") as output:
        t = output["episodes/attempt/transitions"]
        if mutation == "gap":
            t["next_actor_obs"][0, 0] = 99
            t["next_critic_obs"][0, 0] = 99
        elif mutation == "terminal_reset":
            t["next_critic_obs"][-1, 464:] = 0
        elif mutation == "unsafe":
            t["unsafe"][1] = True
        elif mutation == "missing_controller":
            t["actor_obs"][:, 439] = 0
            t["next_actor_obs"][:, 439] = 0
            t["critic_obs"][:, 439] = 0
            t["next_critic_obs"][:, 439] = 0
            output["episodes/attempt/initial_state/observations/policy"][439] = 0
        elif mutation == "wrong_seed":
            output["episodes/attempt/initial_state/observations/policy"][0] = 99
    with pytest.raises(ValueError, match=match):
        read_executed_successes(path, contract)
