"""Pack measured current-environment successes as data-only SAC experience.

Old VR rewards are not eligible. The producer must record native, executed
transitions in the current GPU environment, including its physical contract
and initial controller state. This module never fabricates missing observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from ...runners.storage import save_checkpoint


PHYSICAL_KEYS = (
    "task_family", "schema_version", "skill", "robot_model", "gripper",
    "actions", "action_contract", "action_projection", "observations",
    "observation_contract", "critic_mapping", "contact_contract",
    "reward_profile", "discount", "terminal_contract", "self_collision",
)
WIDTHS = {"actor_obs": "actor_obs_dim", "critic_obs": "critic_obs_dim",
          "action": "action_dim", "next_actor_obs": "actor_obs_dim",
          "next_critic_obs": "critic_obs_dim"}

MEASURED_COLLECTION_SOURCES={
    'current_v2_environment_executed_vr_reference','mixed_VR_actor_DAgger',
    'pose_goal_sac_no_live_reference','learned_pose_student_BC_no_live_reference',
    'layout_reference_residual_sac','fixed_scene_reference_residual_sac',
}


def read_executed_successes(dataset: Path, contract: dict):
    """Accept complete, continuous, successful paths with the same physical MDP."""
    dataset = Path(dataset)
    parts = {name: [] for name in (*WIDTHS, "reward", "terminated")}
    accepted = 0
    with h5py.File(dataset, "r") as source:
        if source.attrs.get("format") != "kuavo_v2_grasp_sac_transitions" \
                or source.attrs.get("format_version") != 1:
            raise ValueError("Executed replay requires the native transition format")
        meta = json.loads(source.attrs["manifest_json"])
        collection_source = meta.get('collection_source')
        if collection_source not in MEASURED_COLLECTION_SOURCES \
                or meta.get("current_reward_verified_against_breakdown") is not True \
                or not str(meta.get("sim_device", "")).startswith("cuda:") \
                or meta.get("old_demo_rewards_used") is not False:
            raise ValueError("Only measured current GPU replay is eligible; old VR rewards are excluded")
        if collection_source == 'mixed_VR_actor_DAgger':
            fraction = meta.get('actor_reference_mix')
            if not isinstance(fraction, (int, float)) or not 0 <= fraction <= .2:
                raise ValueError('Mixed replay requires a declared bounded actor fraction')
        recorded = meta.get("training_contract", {})
        for key in PHYSICAL_KEYS:
            if key not in contract or recorded.get(key) != contract[key]:
                raise ValueError(f"Executed replay {key} differs from the training contract")
        if meta.get("multi_box", {}).get("flap_pose_source", "nominal") \
                != contract.get("flap_pose_source", "nominal"):
            raise ValueError("Executed replay flap source differs from training")
        actor_dim = contract["observations"]["policy"][0]
        dims = {"actor_obs_dim": actor_dim,
                "critic_obs_dim": actor_dim + contract["observations"]["critic"][0],
                "action_dim": sum(contract["actions"].values())}
        if any(meta.get(key) != value for key, value in dims.items()):
            raise ValueError("Executed replay observation/action dimensions differ")
        dt = float(meta.get("control_dt", 0))
        if not np.isclose(dt, 1/30, atol=1e-6, rtol=0) \
                or meta.get("episode_seconds") != 30.0:
            raise ValueError("Executed replay requires the current 30 Hz / 30 s horizon")
        for episode in source.get("episodes", {}).values():
            if not bool(episode.attrs.get("success", False)):
                continue
            initial = episode.get("initial_state")
            if initial is None or initial.attrs.get("schema") != "v2_physical_seed_v1" \
                    or initial.attrs.get("capture_timing") != "before_first_recorded_action":
                raise ValueError("Executed replay requires its actual initial physical/controller state")
            if any(key not in initial for key in (
                    "scene", "drive_targets", "action_terms", "logical_boxes", "observations")):
                raise ValueError("Executed replay initial physical/controller state is incomplete")
            transitions = episode["transitions"]
            values = {name: transitions[name][:] for name in (*parts, "success", "unsafe", "truncated", "sim_time_s")}
            n = len(values["reward"])
            if not 1 <= n <= 900 or any(len(v) != n for v in values.values()):
                raise ValueError("Executed replay must contain one complete aligned attempt")
            if any(values[key].shape != (n, dims[field]) for key, field in WIDTHS.items()) \
                    or any(values[key].shape != (n,) for key in ("reward", "terminated", "success", "unsafe", "truncated", "sim_time_s")):
                raise ValueError("Executed replay transition shapes differ")
            if any(not np.isfinite(v).all() for v in values.values()) \
                    or (np.abs(values["action"]) > 1.00001).any():
                raise ValueError("Executed replay contains invalid values or unnormalized actions")
            if not bool(values["success"][-1]) or not bool(values["terminated"][-1]) \
                    or values["unsafe"].any() or values["truncated"].any() \
                    or values["terminated"][:-1].any() or values["success"][:-1].any():
                raise ValueError("Executed replay must end in measured success without earlier termination")
            if not np.allclose(np.diff(values["sim_time_s"]), dt, atol=1e-5, rtol=0):
                raise ValueError("Executed replay control timestamps are discontinuous")
            for prefix in ("", "next_"):
                if not np.array_equal(values[prefix+"critic_obs"][:, :actor_dim], values[prefix+"actor_obs"]):
                    raise ValueError("Executed replay actor/critic views do not agree")
            for name in ("actor_obs", "critic_obs"):
                if not np.array_equal(values["next_"+name][:-1], values[name][1:]):
                    raise ValueError("Executed replay contains an observation gap or reset seam")
            terminal_critic = values["next_critic_obs"][-1, actor_dim:]
            if terminal_critic.shape != (66,) or not (terminal_critic[35:37] > .5).all() \
                    or terminal_critic[54] <= .5:
                raise ValueError("Executed replay terminal observation lacks the measured bilateral success; check auto-reset timing")
            if not np.array_equal(initial["observations/policy"][:], values["actor_obs"][0]):
                raise ValueError("Executed replay seed does not describe the first pre-action observation")
            # Current 464-D contract: controller availability is just before
            # the 24 previous-action features. Missing old PD targets cannot
            # silently become protected Q transitions.
            if actor_dim != 464 or dims["action_dim"] != 24 \
                    or not (values["actor_obs"][:, 439] > .5).all():
                raise ValueError("Executed replay requires native live controller-state observations")
            for name in parts:
                dtype = torch.bool if name == "terminated" else torch.float32
                parts[name].append(torch.as_tensor(values[name], dtype=dtype))
            accepted += 1
    if not accepted:
        raise ValueError("Executed replay has no measured successful attempt")
    return {name: torch.cat(rows) for name, rows in parts.items()}, {
        "source_dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "successful_episodes": accepted, "executed_rows": sum(len(p) for p in parts["reward"]),
        "source": f"measured_current_gpu_{collection_source}",
        "neural_controller_without_live_reference":collection_source in {
            'pose_goal_sac_no_live_reference','learned_pose_student_BC_no_live_reference'},
        "recorded_old_demo_rewards_imported": False,
        "hypothetical_correction_labels_imported": False,
        "initial_poses": meta.get("initial_poses"),
    }


def merge_executed_successes(datasets, contract):
    """Validate each complete attempt independently before joining Q rows.

    Only native `action`/reward/next-state fields enter Q. The separately
    exported `teacher_imitation` proposal archive is never read here.
    """
    sources = [read_executed_successes(path, contract) for path in datasets]
    if not sources:
        raise ValueError('At least one executed dataset is required')
    if len(sources) == 1:
        return sources[0]
    replay = {key: torch.cat([rows[key] for rows, _ in sources]) for key in sources[0][0]}
    audit = dict(sources=[entry for _, entry in sources],
                 successful_episodes=sum(entry['successful_episodes'] for _, entry in sources),
                 executed_rows=len(replay['reward']), recorded_old_demo_rewards_imported=False,
                 hypothetical_correction_labels_imported=False)
    audit['source_dataset_sha256']=hashlib.sha256(json.dumps(
        [entry['source_dataset_sha256'] for _,entry in sources],separators=(',',':')).encode()).hexdigest()
    return replay, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action='append', required=True,
                        help='Repeat for multiple independently measured current successes.')
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.training_manifest.read_text())
    replay, audit = merge_executed_successes(args.dataset, contract)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = contract | {"artifact_type": "executed_experience_only", "executed_replay_audit": audit}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    state = {"algorithm": "asymmetric_sac", "experience_only": True,
             "actor_obs_dim": replay["actor_obs"].shape[1],
             "critic_obs_dim": replay["critic_obs"].shape[1],
             "action_dim": replay["action"].shape[1],
             "success_replay": replay, "executed_replay_audit": audit}
    path = save_checkpoint(args.output_dir, state, 0)
    print(json.dumps({"experience_archive": str(path), **audit}, indent=2))


if __name__ == "__main__":
    main()
