"""Quest collection against the actual V2 staged-grasp SAC environment."""

from __future__ import annotations

from dataclasses import asdict, replace
import logging
import math
import time

import numpy as np
import torch
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import combine_frame_transforms, convert_camera_frame_orientation_convention

from ...recording.rl_transition_recorder import RlTransitionRecorder
from ...robots.robot_model import resolve_robot_model
from ...teleop.quest_openxr import RawQuestOpenXRDevice, start_quest_xr_session
from ...teleop.urdf_arm_ik import ArmJointLimitError
from ...workcell.workcell_layout import offset as layout_offset, rotation as layout_rotation
from ..envs.terminal_observation import TerminalObservationMixin
from ..multi_box.scene.reset_settling import reset_settling_step
from ..multi_box.training_env_cfg import MultiBoxGraspAssemblyEnvCfg
from .quest_control import QuestRLControl
from .quest_multi_box import _tracked


class _TransitionEnv(TerminalObservationMixin, ManagerBasedRLEnv):
    pass


def _vector(value):
    return value[0].detach().cpu().numpy().copy()


def _transition(pre, action, reward, terminated, truncated, info, env):
    """Use the pre-reset terminal observation, never an auto-reset observation."""
    terminal = info["transition_next_observations"]
    terms = env.termination_manager
    sample = {
        "actor_obs": _vector(pre["policy"]),
        "critic_obs": _vector(torch.cat((pre["policy"], pre["critic"]), -1)),
        "action": _vector(action),
        "reward": float(reward[0].item()),
        "next_actor_obs": _vector(terminal["policy"]),
        "next_critic_obs": _vector(torch.cat((terminal["policy"], terminal["critic"]), -1)),
        "terminated": bool(terminated[0].item()),
        "truncated": bool(truncated[0].item()),
        "success": bool(terms.get_term("success")[0].item()),
        "unsafe": bool(terms.get_term("unsafe")[0].item()),
        "sim_time_s": float(env.common_step_counter * env.step_dt),
    }
    for name in ("actor_obs", "critic_obs", "action", "next_actor_obs", "next_critic_obs"):
        if not np.isfinite(sample[name]).all():
            raise RuntimeError(f"Non-finite V2 demonstration {name}")
    if not math.isfinite(sample["reward"]):
        raise RuntimeError("Non-finite V2 demonstration reward")
    breakdown = env._multi_box_grasp_reward_breakdown
    if not np.isclose(sample["reward"], float(breakdown.total[0].item()), atol=1e-5):
        raise RuntimeError("V2 demonstration reward differs from training breakdown")
    return sample


def run(args, app):
    env = recorder = None
    try:
        model = resolve_robot_model()
        cfg = MultiBoxGraspAssemblyEnvCfg(num_envs=1)
        cfg.multi_box = replace(
            cfg.multi_box, self_collision_enabled=bool(args.rl_demo_self_collision))
        cfg.seed = args.seed
        cfg.sim.device = args.device
        cfg.xr = XrCfg(
            anchor_pos=layout_offset("robot", (0.0, 0.0, 0.55)),
            anchor_rot=layout_rotation("robot"), near_plane=0.08,
        )
        quality = args.render_quality == "quality"
        cfg.sim.render.antialiasing_mode = "DLSS"
        cfg.sim.render.dlss_mode = 2 if quality else 0
        cfg.sim.render.enable_reflections = quality
        cfg.sim.render.enable_translucency = quality
        cfg.sim.render.enable_global_illumination = quality
        cfg.sim.render.enable_ambient_occlusion = quality
        cfg.sim.render.samples_per_pixel = 2 if quality else 1
        # Preserve the training action gate; only the user-selected close force
        # may differ and is written into the manifest below.
        from ...robots.claw_assets.vr import configure_binary_gripper_control
        configure_binary_gripper_control(
            cfg, args.gripper_close_force, contact_feedback=False,
            command_gate="multi_box_reset",
        )
        env = _TransitionEnv(cfg)
        obs, _ = env.reset(seed=args.seed)
        xr = RawQuestOpenXRDevice(
            OpenXRDeviceCfg(xr_cfg=cfg.xr, sim_device=env.device), input_mode="controllers")
        control = QuestRLControl(env, model, args, xr)
        manifest = {
            "task_family": "multi_box_v2", "skill": "grasp",
            "strategy": "staged", "robot_model": model.name,
            "gripper": "leju-twofinger", "seed": args.seed,
            "control_dt": float(env.step_dt),
            "action_encoding": "v2_all_joints_normalized_increment_binary_grippers",
            "action_dim": env.action_manager.total_action_dim,
            "action_terms": [
                [name, env.action_manager.get_term(name).action_dim]
                for name in env.action_manager.active_terms
            ],
            "actor_obs_dim": int(obs["policy"].shape[-1]),
            "critic_obs_dim": int(obs["policy"].shape[-1] + obs["critic"].shape[-1]),
            "critic_mapping": "policy_plus_privileged",
            "gripper_close_force_n": args.gripper_close_force,
            "rack_rollers": bool(args.rack_rollers),
            "controller_mapping": args.controller_mapping,
            "reward_source": "MultiBoxGraspAssemblyEnvCfg",
            "multi_box": asdict(cfg.multi_box),
            "task": asdict(cfg.task),
        }
        recorder = RlTransitionRecorder(args.rl_demo_dataset, manifest)
        start_quest_xr_session(
            app, enable_ui=False, resolution_scale=args.xr_resolution_scale,
            render_quality=args.render_quality,
        )
        keyboard = Se3Keyboard(Se3KeyboardCfg(
            pos_sensitivity=0.0, rot_sensitivity=0.0, sim_device=env.device))
        requests = {name: False for name in ("toggle", "reset", "recenter")}

        def request(name):
            requests[name] = True

        for hand, button, name in (
            ("right", "a", "toggle"), ("right", "b", "reset"),
            ("left", "x", "recenter"),
        ):
            xr.bind_button(hand, button, lambda name=name: request(name))
        for key, name in (("T", "toggle"), ("R", "reset"), ("C", "recenter")):
            keyboard.add_callback(key, lambda name=name: request(name))

        robot = env.scene["robot"]
        head_id = robot.find_bodies(model.head_camera_body)[0][0]
        mount = model.head_camera_mount
        mount_pos = torch.tensor([mount.pos], device=env.device)
        mount_rot = convert_camera_frame_orientation_convention(
            torch.tensor([mount.rot], device=env.device), origin="ros", target="opengl")

        def head_pose():
            position, rotation = combine_frame_transforms(
                robot.data.body_pos_w[:, head_id], robot.data.body_quat_w[:, head_id],
                mount_pos, mount_rot)
            return (position[0].detach().cpu().numpy(),
                    rotation[0].detach().cpu().numpy())

        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport is not None:
            viewport.updates_enabled = True
            if not args.desktop_render:
                viewport.fill_frame = False
                viewport.resolution = (160, 90)

        running = view_ready = False
        zero_action = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
        print(f"[V2 DEMO] Recording SAC transitions to {recorder.path}", flush=True)
        print("[V2 DEMO] X/C recenter; A/T run or pause; B/R discard current attempt and randomize.", flush=True)
        print("[V2 DEMO] Success, unsafe, and timeout end an episode and pause control.", flush=True)
        while app.is_running():
            started = time.monotonic()
            raw = xr.advance()
            packets = {
                side: raw.get(getattr(xr.TrackingTarget, "CONTROLLER_" + side.upper()))
                for side in ("left", "right")
            }
            head = raw.get(xr.TrackingTarget.HEAD)
            tracked = _tracked(head, packets, control.sides)
            if requests["reset"]:
                requests["reset"] = False
                name = recorder.finish_episode(success=False, reason="operator_reset")
                if name is not None:
                    print(f"[V2 DEMO] {name}: operator_reset", flush=True)
                if args.max_episodes and recorder.finished >= args.max_episodes:
                    print("[V2 DEMO] Requested episode count reached.", flush=True)
                    break
                obs, _ = env.reset()
                control.reset()
                running = False
            if requests["recenter"] or (not view_ready and tracked):
                requests["recenter"] = False
                if tracked:
                    xr.recenter_view(*head_pose())
                    app.update()
                    xr.reset()
                    control.reset()
                    view_ready = True
                running = False
            if requests["toggle"]:
                requests["toggle"] = False
                if tracked and view_ready:
                    running = not running
                    control.reset()
                    print(f"[V2 DEMO] {'RUN' if running else 'PAUSED'}", flush=True)
            if running and not tracked:
                running = False
                control.reset()
                print("[V2 DEMO] Tracking lost; paused. Press A/T to resume.", flush=True)
            if running:
                xr.pin_view_position(head_pose()[0])
                ready = bool(reset_settling_step(env).ready[0].item())
                if not ready:
                    with torch.no_grad():
                        obs, _, settling_terminated, settling_truncated, _ = env.step(zero_action)
                    if bool((settling_terminated | settling_truncated)[0].item()):
                        control.reset()
                else:
                    try:
                        action = control.action(packets)
                    except ArmJointLimitError as exc:
                        running = False
                        control.reset()
                        logging.getLogger(__name__).warning("V2 Quest control paused: %s", exc)
                        continue
                    # ObservationManager may reuse its buffers during step.
                    # Freeze obs_t before physics and automatic reset can run.
                    pre_obs = {name: value.clone() for name, value in obs.items()}
                    applied_action = action.clone()
                    with torch.no_grad():
                        next_obs, reward, terminated, truncated, info = env.step(action)
                    if bool(env.termination_manager.get_term("invalid_reset")[0].item()):
                        recorder.finish_episode(success=False, reason="invalid_reset")
                        obs = next_obs
                        running = False
                        control.reset()
                        print("[V2 DEMO] Invalid reset; no transition recorded.", flush=True)
                        continue
                    sample = _transition(pre_obs, applied_action, reward, terminated, truncated, info, env)
                    if not recorder.recording:
                        recorder.start_episode()
                    recorder.append(sample)
                    obs = next_obs
                    if sample["terminated"] or sample["truncated"]:
                        terms = env.termination_manager
                        reason = next((name for name in terms.active_terms
                                       if bool(terms.get_term(name)[0].item())), "terminated")
                        name = recorder.finish_episode(success=sample["success"], reason=reason)
                        print(f"[V2 DEMO] {name}: {reason}, success={sample['success']}", flush=True)
                        running = False
                        control.reset()
                        if args.max_episodes and recorder.finished >= args.max_episodes:
                            print("[V2 DEMO] Requested episode count reached.", flush=True)
                            break
            else:
                env.sim.render()
            remaining = env.step_dt - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        try:
            if recorder is not None:
                recorder.close()
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                app.close()
