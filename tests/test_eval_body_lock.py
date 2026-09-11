"""Eval reset ordering: unlock, restore named pose, lock, step without teleport."""

from types import SimpleNamespace
import sys
import torch

from test_rl_body_lock import Robot
from kuavo_isaaclab_scene.evaluation.body_lock import (
    latch_eval_body, reset_scene_with_body_unlock, configure_eval_body_lock,
)


def test_named_pose_can_change_between_episodes_without_corrupting_defaults():
    robot = Robot()
    env = SimpleNamespace(scene={"robot": robot}, num_envs=2, device="cpu")
    latch_eval_body(env, None)
    lock = env._eval_body_lock
    original = lock.original_limits.clone()

    def reset_scene(env, ids, value):
        # The named initializer validates against limits, so native limits must
        # already be restored when the ordinary reset / pose events execute.
        torch.testing.assert_close(robot.data.joint_pos_limits[ids][:, lock.joint_ids], original[ids])
        assert not robot.data.default_joint_pos.any()
        robot.data.joint_pos[ids[:, None], lock.joint_ids] = value

    first_targets = lock.targets[0].clone()
    ids = torch.tensor([1])
    reset_scene_with_body_unlock(env, ids, reset_scene, {"value": -0.5})
    assert lock.ready.tolist() == [True, False]
    latch_eval_body(env, ids)
    assert lock.ready.all()
    torch.testing.assert_close(lock.targets[0], first_targets)
    assert (lock.targets[1] == -0.5).all()
    assert not robot.data.default_joint_pos.any()
    # Neither driven nor passive hand joints nor arm joints are constrained.
    assert (robot.data.joint_pos_limits[:, [0, 6, 7], 0] == -2).all()


def test_first_reset_without_lock_delegates_unchanged():
    seen = []
    env = SimpleNamespace()
    reset_scene_with_body_unlock(env, None, lambda e, ids, **kw: seen.append((e, ids, kw)), {"a": 3})
    assert seen == [(env, None, {"a": 3})]


def test_continuous_wheel_unlock_does_not_send_infinite_limits_to_physx():
    robot = Robot()
    robot.data.joint_pos_limits[:, 5] = torch.tensor([-torch.inf, torch.inf])
    env = SimpleNamespace(scene={"robot": robot}, num_envs=2, device="cpu")
    latch_eval_body(env, None)
    calls = []
    write_limits = robot.write_joint_position_limit_to_sim

    def checked_write(limits, **kwargs):
        assert torch.isfinite(limits).all()
        calls.append(limits.clone())
        write_limits(limits, **kwargs)

    robot.write_joint_position_limit_to_sim = checked_write
    reset_scene_with_body_unlock(env, None, lambda *_: None, {})
    assert not calls
    assert torch.isinf(robot.data.joint_pos_limits[:, 5]).all()
    latch_eval_body(env, None)
    assert len(calls) == 1


def test_configuration_preserves_reset_params_and_appends_lock_last(monkeypatch):
    monkeypatch.setitem(sys.modules, "isaaclab.managers", SimpleNamespace(EventTermCfg=SimpleNamespace))
    reset_func = lambda *_: None
    cfg = SimpleNamespace(
        scene=SimpleNamespace(robot=SimpleNamespace(spawn=SimpleNamespace(
            articulation_props=SimpleNamespace(fix_root_link=None)))),
        events=SimpleNamespace(reset_all=SimpleNamespace(func=reset_func, params={"x": 1}),
                               initial_state=object()),
    )
    configure_eval_body_lock(cfg, "pd")
    assert cfg.scene.robot.spawn.articulation_props.fix_root_link is None
    assert cfg.events.reset_all.func is reset_func
    configure_eval_body_lock(cfg, "fixed")
    assert cfg.scene.robot.spawn.articulation_props.fix_root_link is True
    assert cfg.events.reset_all.params == {"reset_func": reset_func, "reset_params": {"x": 1}}
    assert list(vars(cfg.events))[-2:] == ["initial_state", "eval_body_lock"]
