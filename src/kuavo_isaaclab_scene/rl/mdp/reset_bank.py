"""Versioned JSON reset states, recorded before automatic environment reset.

Only state tensors are loaded, never pickle/code. Banks are tied to a robot,
ordered box set, geometry and layout to prevent silently invalid grasp resets.
"""

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import torch
from isaaclab.managers import RecorderTerm
from ...robots.robot_model import resolve_robot_model
from ...robots.gripper_config import resolve_gripper_settings
from ..tasks.specs import PREDECESSOR, PHASES


def contract(env):
    cfg = env.cfg
    spec = cfg.task
    names = ["robot", "rack", "button_station", "conveyor_surface", *spec.box_names]
    return {"robot": resolve_robot_model().name, "gripper": resolve_gripper_settings().name,
        "box_names": list(spec.box_names), "cargo_per_box": spec.cargo_per_box,
        "cargo_radius": spec.cargo_radius, "prefill_count": spec.prefill_count,
        "slot_count": spec.slot_count, "slot_pitch": spec.slot_pitch,
        "finger_bodies": list(spec.finger_bodies), "tool_bodies": list(spec.tool_bodies),
        "tool_offset": list(spec.tool_offset),
        "layout": {n: {"pos": list(getattr(cfg.scene, n).init_state.pos),
                       "rot": list(getattr(cfg.scene, n).init_state.rot),
                       "scale": list(getattr(getattr(cfg.scene, n).spawn, "scale", None) or (1, 1, 1))}
                   for n in names},
        "joints": {name: list(asset.joint_names) for name, asset in env.scene.articulations.items()},
        "geometry": {n: {"center": list(g.center), "half_size": list(g.half_size), "body_path": g.body_path}
                     for n, g in cfg.commands.workcell.geometry.items()}}


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_bank(env):
    if hasattr(env, "_rl_bank"):
        return env._rl_bank
    path = Path(env.cfg.task.reset_bank).expanduser().resolve()
    paths = sorted(path.glob("*.json")) if path.is_dir() else [path]
    records = []
    expected = signature(contract(env))
    for file in paths:
        record = json.loads(file.read_text())
        if record.get("version") != 1 or record.get("contract_hash") != expected:
            raise ValueError(f"Reset bank schema/robot/box/layout mismatch: {file}")
        if record.get("source_task") != PREDECESSOR.get(env.cfg.task.name):
            raise ValueError(f"Wrong predecessor task in {file}; expected {PREDECESSOR.get(env.cfg.task.name)}")
        if not record.get("success"):
            raise ValueError(f"Refusing an unsuccessful reset state: {file}")
        command = record.get("command", {})
        if not 0 <= command.get("active_box", -1) < len(env.cfg.task.box_names):
            raise ValueError(f"Invalid active box in reset state: {file}")
        heights = torch.tensor(command.get("initial_z", []))
        if heights.shape != (len(env.cfg.task.box_names),) or not torch.isfinite(heights).all():
            raise ValueError(f"Invalid initial box heights: {file}")
        records.append(record)
    if not records:
        raise ValueError(f"No successful reset snapshots found at {path}")
    env._rl_bank = records
    return records


def restore_bank(env, env_ids):
    bank = load_bank(env)
    picks = torch.randint(len(bank), (len(env_ids),), device=env.device).cpu().tolist()
    records = [bank[i] for i in picks]
    template = env.scene.get_state(is_relative=True)
    state = {}
    for category, assets in template.items():
        state[category] = {}
        for name, fields in assets.items():
            state[category][name] = {}
            for key, current in fields.items():
                values = torch.tensor([r["state"][category][name][key] for r in records], device=env.device, dtype=current.dtype)
                if values.shape != current[env_ids].shape or not torch.isfinite(values).all():
                    raise ValueError(f"Invalid reset tensor: {category}/{name}/{key}")
                if key == "root_pose" and not torch.allclose(
                        values[:, 3:7].norm(dim=-1), torch.ones(len(env_ids), device=env.device), atol=1e-3):
                    raise ValueError(f"Reset quaternion is not normalized: {name}")
                state[category][name][key] = values
    env.scene.reset_to(state, env_ids=env_ids, is_relative=True)
    env._rl_reset_metadata = getattr(env, "_rl_reset_metadata", {})
    for env_id, record in zip(env_ids.cpu().tolist(), records):
        env._rl_reset_metadata[env_id] = record["command"]


class SuccessSnapshotRecorder(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.saved = 0

    def record_pre_reset(self, env_ids):
        env = self._env
        spec = env.cfg.task
        if not hasattr(env, "termination_manager"):
            return None, None
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else torch.as_tensor(env_ids, device=env.device)
        # A bounded latest-step record survives auto-reset for rollout metrics.
        env._rl_last_outcomes = {"step": env.common_step_counter, "episodes": [
            {"env_id": i, "success": bool(env.termination_manager.get_term("success")[i]),
             "unsafe": bool(env.termination_manager.get_term("unsafe")[i]),
             "seconds": float(env.episode_length_buf[i]) * env.step_dt}
            for i in ids.tolist() if env.episode_length_buf[i] > 0]}
        if (not spec.snapshot_dir or self.saved >= spec.max_snapshots
                or spec.name not in ("approach_rack", "pick", "carry", "place")):
            return None, None
        ids = ids[env.termination_manager.get_term("success")[ids]]
        self._save(ids, spec.name)
        return None, None

    def record_post_step(self):
        env = self._env
        spec = env.cfg.task
        if spec.name != "full" or not spec.snapshot_dir:
            return None, None
        command = env.command_manager.get_term("workcell")
        for phase in (0, 1, 2, 3):
            selected = command.transition & (command.reward_phase == phase) & ~command.failure
            if phase == 3:
                # Only final placement is a valid entry state for press_button.
                selected &= command.supported.all(-1)
            self._save(selected.nonzero().flatten(), PHASES[phase])
        return None, None

    def _save(self, ids, source_task):
        env = self._env
        spec = env.cfg.task
        if not spec.snapshot_dir or self.saved >= spec.max_snapshots:
            return
        if not len(ids):
            return
        directory = Path(spec.snapshot_dir).expanduser().resolve()
        if spec.name == "full":
            directory /= source_task
        directory.mkdir(parents=True, exist_ok=True)
        state = env.scene.get_state(is_relative=True)
        cmd = env.command_manager.get_term("workcell")
        metadata = contract(env)
        for env_id in ids[:spec.max_snapshots - self.saved].tolist():
            selected = {cat: {name: {key: tensor[env_id].detach().cpu().tolist() for key, tensor in fields.items()}
                              for name, fields in assets.items()} for cat, assets in state.items()}
            record = {"version": 1, "source_task": source_task, "success": True,
                "contract": metadata, "contract_hash": signature(metadata), "state": selected,
                "command": {"active_box": int(cmd.reward_box[env_id]), "initial_z": cmd.initial_z[env_id].cpu().tolist()}}
            # Unique, exclusive files: never replace an earlier demonstration.
            with (directory / f"{source_task}_{uuid4().hex}.json").open("x") as stream:
                json.dump(record, stream, allow_nan=False)
            self.saved += 1
