"""JSON success banks captured before auto-reset; no object attachments."""
import json
from pathlib import Path
from uuid import uuid4
import torch
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, DatasetExportMode
from isaaclab.utils import configclass
from .spec import PREDECESSOR


def read_bank(path, contract, skill):
    records = []
    for file in sorted(Path(path).glob("*.json")):
        item = json.loads(file.read_text())
        if item.get("version") != 1 or item.get("contract") != contract:
            raise ValueError(f"Reset-bank contract mismatch: {file}")
        if item.get("skill") != PREDECESSOR[skill] or not item.get("success"):
            raise ValueError(f"Expected successful {PREDECESSOR[skill]} state: {file}")
        records.append(item)
    if not records:
        raise ValueError(f"Empty reset bank: {path}; collect predecessor successes first.")
    return records


def restore(env, ids):
    bank = env._multi_box_bank
    choices = torch.randint(len(bank), (len(ids),), device=env.device).tolist()
    pending = getattr(env, "_multi_box_restore", {})
    for index, choice in zip(ids.tolist(), choices):
        record = bank[choice]
        selected = torch.tensor([index], device=env.device)
        for name, data in record["assets"].items():
            asset = env.scene[name]
            if data["joint_names"] != asset.joint_names:
                raise ValueError(f"Reset bank joint ordering differs: {name}")
            root = torch.tensor([data["root"]], device=env.device)
            root[:, :3] += env.scene.env_origins[selected]
            asset.write_root_state_to_sim(root, env_ids=selected)
            q = torch.tensor([data["q"]], device=env.device)
            v = torch.tensor([data["v"]], device=env.device)
            asset.write_joint_state_to_sim(q, v, env_ids=selected)
            asset.set_joint_position_target(q, env_ids=selected)
        pending[index] = record
    env._multi_box_restore = pending


class OutcomeRecorder(RecorderTerm):
    def record_pre_reset(self, env_ids):
        env = self._env
        if not hasattr(env, "command_manager"):
            return None, None
        t = env.command_manager.get_term("workcell")
        entries = []
        for i in env_ids.tolist():
            if env.episode_length_buf[i] == 0:
                continue
            entries.append(dict(success=bool(t.success[i]), failed=bool(t.failure[i]),
                placed=int(t.complete[i].sum()), seconds=float(t.elapsed[i]),
                base_distance=float(t.base_distance[i]), dual_carry_seconds=float(t.dual_time[i])))
            if not t.success[i] or t.spec.skill == "full" or not t.spec.snapshot_dir:
                continue
            directory = Path(t.spec.snapshot_dir)
            directory.mkdir(parents=True, exist_ok=True)
            # Runner requires a unique bank/output directory for concurrent jobs.
            count = getattr(env, "_multi_box_saved", len(list(directory.glob("*.json"))))
            if count >= t.spec.max_snapshots:
                continue
            assets = {}
            for name in ("robot", *t.spec.box_names):
                asset = env.scene[name]
                root = asset.data.root_state_w[i].clone()
                root[:3] -= env.scene.env_origins[i]
                assets[name] = dict(root=root.tolist(), q=asset.data.joint_pos[i].tolist(),
                                    v=asset.data.joint_vel[i].tolist(), joint_names=asset.joint_names)
            targets = {}
            for name in env.action_manager.active_terms:
                term = env.action_manager.get_term(name)
                targets[name] = {key: getattr(term, key)[i].tolist() for key in
                    ("_targets", "_processed_actions", "_signed_target", "_velocity") if hasattr(term, key)}
            record = dict(version=1, contract=env.cfg.experiment_contract, skill=t.spec.skill, success=True,
                target=int(t.target[i]), initial_centers=t.initial_centers[i].tolist(),
                paid=t.paid[i].tolist(), assets=assets, targets=targets)
            path = directory / f"{uuid4().hex}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, allow_nan=False))
            temporary.replace(path)
            env._multi_box_saved = count + 1
        if entries:
            env._multi_box_outcomes = {"step": env.common_step_counter, "episodes": entries}
        return None, None


@configclass
class RecordersCfg(RecorderManagerBaseCfg):
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False
    outcomes = RecorderTermCfg(class_type=OutcomeRecorder)
