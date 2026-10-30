"""Fixed-body GR00T evaluation using the same physical constraints as arms-only RL.

Restore native validation limits before resetting, then latch the *new* initial pose. Never
teleport joints during rollout; PhysX enforces the narrow body joint limits.
"""

import torch

from ..rl.mdp.body_lock import FixedBody


def reset_scene_with_body_unlock(env, env_ids, reset_func, reset_params):
    lock = getattr(env, "_eval_body_lock", None)
    if lock is not None:
        asset = lock.asset
        ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None
               else torch.as_tensor(env_ids, dtype=torch.long, device=env.device))
        # Reset events run without physics steps. Restore only the validation
        # buffer here; the final latch replaces the physical limits before the
        # next step. Sending +/-inf back to PhysX after locking a continuous
        # wheel converts it to a limited revolute joint and emits an error.
        asset.data.joint_pos_limits[ids[:, None], lock.joint_ids] = lock.original_limits[ids]
        lock.ready[ids] = False
    reset_func(env, env_ids, **reset_params)


def latch_eval_body(env, env_ids):
    if not hasattr(env, "_eval_body_lock"):
        env._eval_body_lock = FixedBody(env.scene["robot"])
    env._eval_body_lock.reset(env_ids)


def configure_eval_body_lock(cfg, mode):
    """Call after all pose/reset configuration, before constructing the environment."""
    if mode == "pd":
        return
    if mode != "fixed":
        raise ValueError(f"Unknown eval body mode: {mode}")
    from isaaclab.managers import EventTermCfg

    cfg.scene.robot.spawn.articulation_props.fix_root_link = True
    reset = cfg.events.reset_all
    cfg.events.reset_all = EventTermCfg(
        func=reset_scene_with_body_unlock, mode="reset",
        params={"reset_func": reset.func, "reset_params": dict(reset.params)},
    )
    cfg.events.eval_body_lock = EventTermCfg(func=latch_eval_body, mode="reset")


def body_lock_snapshot(env):
    lock = env._eval_body_lock
    positions = lock.asset.data.joint_pos[:, lock.joint_ids]
    return {
        "joint_names": lock.joint_names,
        "target_rad": lock.targets[0].detach().cpu().tolist(),
        "position_rad": positions[0].detach().cpu().tolist(),
        "max_joint_error_rad": float((positions - lock.targets).abs().max().item()),
        "root_pose_w": lock.asset.data.root_pose_w[0].detach().cpu().tolist(),
    }
