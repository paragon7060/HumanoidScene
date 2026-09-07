"""Named, JSON-only stationary reset states. Safe to import in a VR debug console.

Pose order is xyz + quaternion wxyz; translations are relative to env origin.
No simulator imports occur until capture/reset/configuration is requested.
"""

from copy import deepcopy
import json
import math
from pathlib import Path

from ..core.paths import CONFIG_DIR


def default_state_path():
    return CONFIG_DIR / "initial_states.json"


def validate_state(state):
    if not isinstance(state, dict):
        raise ValueError("An initial state must be an object.")
    if state.get("coordinate_frame") != "env-origin":
        raise ValueError("Initial state coordinate_frame must be 'env-origin'.")
    for field in ("robot_model", "gripper"):
        if not isinstance(state.get(field), str) or not state[field]:
            raise ValueError(f"Initial state needs {field}.")
    assets = state.get("assets")
    if not isinstance(assets, dict) or "robot" not in assets:
        raise ValueError("Initial state needs assets.robot.")
    for name, record in assets.items():
        if not isinstance(name, str) or not name.isidentifier() or not isinstance(record, dict):
            raise ValueError("Initial state asset keys must be scene names.")
        if set(record) - {"joint_positions", "root_pose"} or not record:
            raise ValueError(f"Unsupported/empty fields for {name}; use joint_positions and/or root_pose.")
        joints = record.get("joint_positions", {})
        if not isinstance(joints, dict):
            raise ValueError(f"{name}.joint_positions must be a name/value dictionary.")
        for joint, value in joints.items():
            if not isinstance(joint, str) or not joint or not _finite(value):
                raise ValueError(f"Invalid joint name/value in {name}: {joint}")
        if "root_pose" in record:
            pose = record["root_pose"]
            if not isinstance(pose, list) or len(pose) != 7 or not all(_finite(v) for v in pose):
                raise ValueError(f"{name}.root_pose must contain seven finite numbers.")
            if abs(math.sqrt(sum(v*v for v in pose[3:])) - 1.0) > 1e-3:
                raise ValueError(f"{name}.root_pose quaternion must have unit length (wxyz).")
    return state


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_states(path=None):
    path = Path(path or default_state_path()).expanduser().resolve()
    bank = json.loads(path.read_text())
    if not isinstance(bank, dict) or bank.get("version") != 1 or not isinstance(bank.get("states"), dict):
        raise ValueError(f"Unsupported initial-state file: {path}")
    for name, state in bank["states"].items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("State names must be nonempty strings.")
        validate_state(state)
    return bank


def load_initial_state(name, path=None, *, robot_model=None, gripper=None):
    bank = read_states(path)
    if name not in bank["states"]:
        raise ValueError(f"Unknown initial state {name!r}; available: {', '.join(bank['states'])}")
    state = deepcopy(bank["states"][name])
    for field, expected in (("robot_model", robot_model), ("gripper", gripper)):
        if expected is not None and state[field] != expected:
            raise ValueError(f"Initial state {name!r} uses {field}={state[field]}, selected={expected}.")
    return state


def save_initial_state(name, state, path=None, *, overwrite=False):
    """Merge one preset atomically; an existing name is protected by default."""
    import fcntl
    import os
    import tempfile

    if not isinstance(name, str) or not name.strip():
        raise ValueError("State name must not be empty.")
    validate_state(state)
    path = Path(path or default_state_path()).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        bank = read_states(path) if path.exists() else {"version": 1, "states": {}}
        if name in bank["states"] and not overwrite:
            raise FileExistsError(f"State {name!r} already exists; choose another name or overwrite=True.")
        bank["states"][name] = deepcopy(state)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".initial-state-", delete=False) as file:
                temporary = Path(file.name)
                json.dump(bank, file, indent=2, ensure_ascii=False, allow_nan=False)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return str(path)


def capture_initial_state(env, name, path=None, *, env_index=0, objects=(), description="", overwrite=False):
    """Save actual q and base pose, including external hands and optional boxes.

    This is a stationary initialization, not a resumable physics/contact snapshot.
    Pause following and let motion settle before calling from Debug Console.
    """
    from .robot_model import resolve_robot_model
    from .gripper_config import resolve_gripper_settings

    env = env.unwrapped
    if not 0 <= env_index < env.num_envs:
        raise ValueError("env_index is out of range.")
    hand = resolve_gripper_settings()
    if isinstance(objects, str):
        raise ValueError("objects must be a sequence of scene names, e.g. ('medium_box_0',).")
    names = list(dict.fromkeys(["robot", *[hand.asset_name_for(s) for s in hand.active_sides], *objects]))
    state = {"robot_model": resolve_robot_model().name, "gripper": hand.name,
             "coordinate_frame": "env-origin", "description": description, "assets": {}}
    for asset_name in names:
        asset = env.scene[asset_name]
        pose = asset.data.root_pose_w[env_index].detach().clone()
        pose[:3] -= env.scene.env_origins[env_index]
        record = {"root_pose": pose.cpu().tolist()}
        if hasattr(asset, "joint_names"):
            record["joint_positions"] = dict(zip(asset.joint_names,
                asset.data.joint_pos[env_index].detach().cpu().tolist(), strict=True))
        state["assets"][asset_name] = record
    saved = save_initial_state(name, state, path, overwrite=overwrite)
    print(f"[INITIAL STATE] Saved {name!r} to {saved}", flush=True)
    return saved


def apply_initial_state(env, env_ids, state, state_name=""):
    """Reset event: validate every target first, then set q, zero velocities and PD targets."""
    import torch

    ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None
           else torch.as_tensor(env_ids, dtype=torch.long, device=env.device))
    if not len(ids):
        return
    validate_state(state)
    prepared = []
    for name, record in state["assets"].items():
        asset = env.scene[name]
        pose = None
        if "root_pose" in record:
            if not hasattr(asset, "write_root_pose_to_sim"):
                raise ValueError(f"{name} is not a physical asset with a writable root pose.")
            pose = torch.tensor(record["root_pose"], dtype=asset.data.root_pose_w.dtype,
                                device=env.device).repeat(len(ids), 1)
            pose[:, 3:] /= pose[:, 3:].norm(dim=-1, keepdim=True)
            pose[:, :3] += env.scene.env_origins[ids]
        positions = None
        if "joint_positions" in record:
            if not hasattr(asset, "joint_names"):
                raise ValueError(f"{name} is not an articulation.")
            unknown = set(record["joint_positions"]) - set(asset.joint_names)
            if unknown:
                raise ValueError(f"Unknown joints for {name}: {sorted(unknown)}")
            positions = asset.data.joint_pos[ids].clone()
            for joint, value in record["joint_positions"].items():
                j = asset.joint_names.index(joint)
                limits = asset.data.joint_pos_limits[ids, j]
                # Allow only tiny contact-solver overshoot from a measured VR pose.
                if ((value < limits[:, 0] - 1e-4) | (value > limits[:, 1] + 1e-4)).any():
                    raise ValueError(f"Initial state {state_name}: {name}/{joint}={value} exceeds physical joint limits.")
                positions[:, j] = torch.full_like(positions[:, j], value).clamp(limits[:, 0], limits[:, 1])
        prepared.append((asset, pose, positions))
    for asset, pose, positions in prepared:
        if pose is not None:
            asset.write_root_pose_to_sim(pose, env_ids=ids)
        if hasattr(asset, "write_root_velocity_to_sim"):
            asset.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), env_ids=ids)
        if positions is not None:
            zeros = torch.zeros_like(positions)
            asset.write_joint_state_to_sim(positions, zeros, env_ids=ids)
            asset.set_joint_position_target(positions, env_ids=ids)
            asset.set_joint_velocity_target(zeros, env_ids=ids)


def add_initial_state_args(parser):
    parser.add_argument("--initial-state", metavar="NAME", help="Named stationary reset preset from initial_states.json.")
    parser.add_argument("--initial-states-file", type=Path, metavar="JSON", help="Override the named initial-state library.")


def configure_initial_state(cfg, args):
    """Append after existing reset events, before action/command manager resets."""
    if not args.initial_state:
        if args.initial_states_file:
            raise ValueError("--initial-states-file requires --initial-state NAME.")
        return None
    from isaaclab.managers import EventTermCfg
    from .robot_model import resolve_robot_model
    from .gripper_config import resolve_gripper_settings

    if getattr(getattr(cfg, "task", None), "reset_bank", None):
        raise ValueError("Choose --initial-state OR --reset-bank; a pose override would break a saved grasp.")
    path = Path(args.initial_states_file or default_state_path()).expanduser().resolve()
    state = load_initial_state(args.initial_state, path,
        robot_model=resolve_robot_model().name, gripper=resolve_gripper_settings().name)
    cfg.events.initial_state = EventTermCfg(func=apply_initial_state, mode="reset",
        params={"state": state, "state_name": args.initial_state})
    metadata = {"name": args.initial_state, "file": str(path), "state": state}
    cfg.initial_state_metadata = metadata
    print(f"[INITIAL STATE] {args.initial_state} ({path})", flush=True)
    if "root_pose" not in state["assets"]["robot"]:
        print("[INITIAL STATE] No robot root pose saved; retaining the environment reset base pose.", flush=True)
    return metadata
