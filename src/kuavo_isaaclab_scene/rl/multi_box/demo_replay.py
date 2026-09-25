"""Convert the original Quest v2 demonstrations to the current SAC observation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import torch
from torch.nn import functional as F

from .geometry.grasp import (
    GRASP_APPROACH_REWARD_SCALE_M,
    closest_flap_surface,
    nominal_flap_geometry,
    opposing_flap_reach_assignment,
)
from .observations import flat_actor_observation_dim
from .spec import MAX_BOXES


LEGACY_ACTOR_DIM = 403
LEGACY_CRITIC_DIM = 469
DEMO_FORMAT = "kuavo_v2_grasp_sac_transitions"
_OLD_BOX_TOKENS_START = 68 + 2 * 9
_OLD_BOX_MASK_START = _OLD_BOX_TOKENS_START + MAX_BOXES * 22
_OLD_TARGET_START = _OLD_BOX_MASK_START + MAX_BOXES


def _rotation_matrix(rotation6: torch.Tensor) -> torch.Tensor:
    first = F.normalize(rotation6[..., :3], dim=-1)
    second = rotation6[..., 3:] - (first * rotation6[..., 3:]).sum(-1, keepdim=True) * first
    second = F.normalize(second, dim=-1)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def convert_legacy_actor_observation(actor_obs: torch.Tensor) -> torch.Tensor:
    """Rebuild the 38 hand/flap features from recorded deployable 403-D inputs.

    Both TCP and box poses were recorded relative to the same robot base, so
    no simulator replay, world pose, contact truth or stale reward is needed.
    """
    if actor_obs.ndim != 2 or actor_obs.shape[1] != LEGACY_ACTOR_DIM:
        raise ValueError("Legacy v2 actor observation must be [steps, 403]")
    if not bool(torch.isfinite(actor_obs).all()):
        raise ValueError("Legacy actor observation contains non-finite values")
    n = len(actor_obs)
    tcp_pose9 = actor_obs[:, 50:68].reshape(n, 2, 9)
    tokens = actor_obs[:, _OLD_BOX_TOKENS_START:_OLD_BOX_MASK_START].reshape(n, MAX_BOXES, 22)
    mask = actor_obs[:, _OLD_BOX_MASK_START:_OLD_TARGET_START]
    target = actor_obs[:, _OLD_TARGET_START:_OLD_TARGET_START + MAX_BOXES]
    target_ids = target.argmax(-1)
    rows = torch.arange(n, device=actor_obs.device)
    selected = tokens[rows, target_ids]
    valid = (target.sum(-1) > 0.5) & (mask[rows, target_ids] > 0.5) \
        & (selected[:, 21] > 0)
    box_pose9 = selected[:, 12:21]
    box_rotation = _rotation_matrix(box_pose9[:, 3:])
    tcp_rotation = _rotation_matrix(tcp_pose9[..., 3:])
    centers, halves, axes = nominal_flap_geometry(
        selected[:, 5:8], selected[:, 3:5].argmax(-1))
    tcp_in_box = torch.matmul(
        box_rotation.transpose(-1, -2)[:, None],
        (tcp_pose9[..., :3] - box_pose9[:, None, :3])[..., None],
    ).squeeze(-1)
    surface = closest_flap_surface(
        tcp_in_box[:, :, None], centers[:, None], halves[:, None], axes[:, None])
    distances = (tcp_in_box[:, :, None] - surface).norm(dim=-1)
    center_in_base = box_pose9[:, None, None, :3] + torch.matmul(
        box_rotation[:, None, None], centers[:, None, :, :, None]
    ).squeeze(-1)
    relative_position = torch.matmul(
        tcp_rotation.transpose(-1, -2)[:, :, None],
        (center_in_base - tcp_pose9[:, :, None, :3])[..., None],
    ).squeeze(-1)
    relative_rotation = torch.matmul(
        tcp_rotation.transpose(-1, -2)[:, :, None],
        box_rotation[:, None, None].expand(-1, 2, 2, -1, -1),
    )
    relative_rotation6 = torch.cat((
        relative_rotation[..., :, 0], relative_rotation[..., :, 1]), dim=-1)
    relations = torch.cat((relative_position, relative_rotation6), dim=-1)
    relations *= valid[:, None, None, None]
    _, assignment = opposing_flap_reach_assignment(
        distances, GRASP_APPROACH_REWARD_SCALE_M)
    assignment_one_hot = F.one_hot(assignment[:, 0], 2).to(actor_obs.dtype) * valid[:, None]
    result = torch.cat((
        actor_obs[:, :_OLD_BOX_MASK_START], relations.flatten(1),
        assignment_one_hot, actor_obs[:, _OLD_BOX_MASK_START:],
    ), dim=-1)
    if result.shape[1] != flat_actor_observation_dim(25):
        raise RuntimeError("Converted v2 actor observation has the wrong dimension")
    return result


def load_v2_grasp_demonstrations(
    path: str | Path, *, self_collision_enabled: bool,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Load successful episodes and convert both current and terminal-next views."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    batches: dict[str, list[torch.Tensor]] = {key: [] for key in (
        "actor_obs", "critic_obs", "action", "reward", "next_actor_obs",
        "next_critic_obs", "terminated",
    )}
    with h5py.File(path) as source:
        if source.attrs.get("format") != DEMO_FORMAT or source.attrs.get("format_version") != 1:
            raise ValueError("Unsupported v2 grasp demonstration format")
        manifest = json.loads(source.attrs["manifest_json"])
        required = {
            "task_family": "multi_box_v2", "skill": "grasp",
            "robot_model": "s63", "gripper": "leju-twofinger",
            "rack_rollers": True, "action_dim": 25,
            "actor_obs_dim": LEGACY_ACTOR_DIM,
            "critic_obs_dim": LEGACY_CRITIC_DIM,
            "controller_mapping": "scaled",
        }
        for name, expected in required.items():
            if manifest.get(name) != expected:
                raise ValueError(f"Demonstration {name} differs from current v2 grasp contract")
        if manifest.get("multi_box", {}).get("self_collision_enabled") != self_collision_enabled:
            raise ValueError("Demonstration self-collision setting differs from training")
        if abs(float(manifest.get("control_dt", 0)) - 1 / 30) > 1e-6:
            raise ValueError("Demonstration control rate differs from v2 grasp training")
        episodes = source.get("episodes")
        if episodes is None:
            raise ValueError("Demonstration file has no episodes")
        accepted = 0
        for episode in episodes.values():
            if not bool(episode.attrs.get("success", False)):
                continue
            transitions = episode["transitions"]
            current = torch.from_numpy(transitions["actor_obs"][:])
            next_actor = torch.from_numpy(transitions["next_actor_obs"][:])
            critic = torch.from_numpy(transitions["critic_obs"][:])
            next_critic = torch.from_numpy(transitions["next_critic_obs"][:])
            if critic.shape != (len(current), LEGACY_CRITIC_DIM) \
                    or next_critic.shape != (len(current), LEGACY_CRITIC_DIM) \
                    or next_actor.shape != current.shape \
                    or not torch.equal(critic[:, :LEGACY_ACTOR_DIM], current) \
                    or not torch.equal(next_critic[:, :LEGACY_ACTOR_DIM], next_actor):
                raise ValueError("Demonstration critic views do not match actor views")
            actor_new = convert_legacy_actor_observation(current)
            next_new = convert_legacy_actor_observation(next_actor)
            raw = {
                "actor_obs": actor_new,
                "critic_obs": torch.cat((actor_new, critic[:, LEGACY_ACTOR_DIM:]), dim=-1),
                "action": torch.from_numpy(transitions["action"][:]),
                "reward": torch.from_numpy(transitions["reward"][:]).float(),
                "next_actor_obs": next_new,
                "next_critic_obs": torch.cat((next_new, next_critic[:, LEGACY_ACTOR_DIM:]), dim=-1),
                "terminated": torch.from_numpy(transitions["terminated"][:]),
            }
            if raw["action"].shape != (len(current), 25) \
                    or any(len(value) != len(current) for value in raw.values()) \
                    or any(not bool(torch.isfinite(value).all()) for value in raw.values()):
                raise ValueError("Demonstration contains invalid transitions")
            if not bool(transitions["success"][-1]) or not bool(raw["terminated"][-1]):
                raise ValueError("Successful demonstration lacks a terminal success step")
            for name, value in raw.items():
                batches[name].append(value)
            accepted += 1
    if not accepted:
        raise ValueError("Demonstration file has no successful episodes")
    result = {name: torch.cat(parts) for name, parts in batches.items()}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return result, {
        "path": str(path), "sha256": digest,
        "episodes": accepted, "transitions": len(result["reward"]),
        "action_terms": manifest.get("action_terms"),
        "source_actor_dim": LEGACY_ACTOR_DIM,
        "converted_actor_dim": result["actor_obs"].shape[-1],
        "converted_critic_dim": result["critic_obs"].shape[-1],
        "observation_conversion": "nominal_flap_center_v1",
    }
