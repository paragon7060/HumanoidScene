"""Optional branch of the existing Quest collector; real RL rewards, no dataset writer."""

from argparse import Namespace
from pathlib import Path
import time

import numpy as np
import torch
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RecorderTermCfg
from isaaclab.utils.math import combine_frame_transforms, convert_camera_frame_orientation_convention

from ...core.paths import CONFIG_DIR
from ...display.xr_reward_panel import QuestRewardPanel
from ...robots.robot_model import resolve_robot_model
from ...teleop.quest_openxr import RawQuestOpenXRDevice, start_quest_xr_session
from ..runners.common import build_configs
from .quest_control import QuestRLControl
from .reward_recorder import RewardProbe
from .reward_report import format_report
from .stationary_surface import StationarySurface


def _config(args):
    config = Path(args.rl_config or CONFIG_DIR / "rl_pick_arms_only.py").expanduser().resolve()
    if not config.is_file():
        raise ValueError(f"Missing RL config: {config}")
    # Use the same experiment builder as training, including named pose and physics.
    rl = Namespace(task="pick", boxes="medium_box_0", control_mode="arms-only",
        config=config, reset_bank=None, snapshot_dir=None, max_snapshots=1,
        prefill=0, cargo_per_box=0, slots=4, slot_pitch=.52, no_randomization=True,
        num_envs=1, env_spacing=8., enable_cameras=args.enable_cameras, seed=args.seed, device=args.device,
        max_iterations=1, save_interval=None, initial_state=None,
        initial_states_file=CONFIG_DIR / "initial_states.json")
    cfg, _ = build_configs(rl)
    if cfg.task.control_mode != "arms-only" or cfg.task.grasp_mode != "flap_top" or cfg.task.name != "pick":
        raise ValueError("Quest reward inspection currently supports arms-only flap pick only")
    cfg.xr = XrCfg(near_plane=.08)
    cfg.scene.conveyor_surface.class_type = StationarySurface
    cfg.recorders.quest_reward = RecorderTermCfg(class_type=RewardProbe)
    if args.enable_cameras:
        # Display-only RGB cameras: never added to RL policy observations.
        cfg.scene.waist_camera = None
        cfg.scene.robustness_camera.width = args.head_camera_width
        cfg.scene.robustness_camera.height = args.head_camera_height
        cfg.scene.robustness_camera.data_types = ["rgb"]
        for side in ("left", "right"):
            camera = getattr(cfg.scene, side + "_wrist_camera")
            camera.width, camera.height = args.wrist_camera_width, args.wrist_camera_height
            camera.data_types = ["rgb"]
            if not args.wrist_cameras:
                setattr(cfg.scene, side + "_wrist_camera", None)
    cfg.sim.render.antialiasing_mode = "DLSS"
    quality = args.render_quality == "quality"
    cfg.sim.render.dlss_mode = 2 if quality else 0
    cfg.sim.render.enable_reflections = quality
    cfg.sim.render.enable_translucency = quality
    cfg.sim.render.enable_global_illumination = quality
    cfg.sim.render.enable_ambient_occlusion = quality
    cfg.sim.render.samples_per_pixel = 2 if quality else 1
    return cfg


def run(args, app):
    env = hud = camera_overlay = None
    try:
        cfg = _config(args)
        env = ManagerBasedRLEnv(cfg)
        env.reset(seed=args.seed)
        model = resolve_robot_model()
        control = QuestRLControl(env, model, args)
        xr = RawQuestOpenXRDevice(OpenXRDeviceCfg(xr_cfg=cfg.xr, sim_device=env.device), input_mode="controllers")
        start_quest_xr_session(app, enable_ui=True, resolution_scale=args.xr_resolution_scale,
                              render_quality=args.render_quality)
        hud = QuestRewardPanel(forward_axis=args.xr_overlay_forward_axis)
        hud.set_visible(True)
        if args.quest_camera_overlay:
            from ...display.xr_camera_overlay import QuestCameraOverlay, QuestCameraOverlayCfg
            from ...display.camera_frames import camera_rgb
            camera_overlay = QuestCameraOverlay(
                head_resolution=(args.head_camera_width, args.head_camera_height),
                wrist_resolution=(args.wrist_camera_width, args.wrist_camera_height),
                cfg=QuestCameraOverlayCfg(distance_m=args.xr_overlay_distance,
                                         forward_axis=args.xr_overlay_forward_axis))
        if args.camera_preview:
            from ...display.camera_viewports import open_camera_viewports
            names = ["robustness_camera"]
            if args.wrist_cameras:
                names += ["left_wrist_camera", "right_wrist_camera"]
            open_camera_viewports(env.scene, names, headless=args.headless, width=240, height=180, columns=3)
        keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0., rot_sensitivity=0., sim_device=env.device))
        requests = {name: False for name in ("toggle", "reset", "recenter", "panel")}
        def request(name):
            requests[name] = True
        for hand, button, name in (("right", "a", "toggle"), ("right", "b", "reset"),
                                   ("left", "x", "recenter"), ("left", "y", "panel")):
            xr.bind_button(hand, button, lambda name=name: request(name))
        for key, name in (("T", "toggle"), ("R", "reset"), ("C", "recenter"), ("H", "panel")):
            keyboard.add_callback(key, lambda name=name: request(name))
        robot = env.scene["robot"]
        head_id = robot.find_bodies(model.head_camera_body)[0][0]
        mount = model.head_camera_mount
        mount_pos = torch.tensor([mount.pos], device=env.device)
        mount_rot = convert_camera_frame_orientation_convention(
            torch.tensor([mount.rot], device=env.device), origin="ros", target="opengl")
        def head_pose():
            p, q = combine_frame_transforms(robot.data.body_pos_w[:, head_id],
                robot.data.body_quat_w[:, head_id], mount_pos, mount_rot)
            return p[0].detach().cpu().numpy(), q[0].detach().cpu().numpy()
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport is not None:
            # Keep offscreen UI textures refreshing even without RTX cameras.
            viewport.updates_enabled = True
            if not args.desktop_render:
                viewport.fill_frame = False
                viewport.resolution = (160, 90)
        running, terminal, view_ready = False, False, False
        sample, episode_return, last_hud = None, 0., 0.
        panel_ready, next_panel_diagnostic = False, time.monotonic() + 3.
        next_grasp_diagnostic = 0.
        status = "PAUSED - X then A"
        print("[RL REWARD] No dataset recording. A/T run/pause; B/R reset; X/C recenter; Y/H panel.", flush=True)
        print(f"[RL REWARD] active_arm={cfg.task.active_arm}, actions={env.action_manager.total_action_dim}, "
              f"flaps={cfg.task.grasp_flaps}, contact_region={cfg.task.flap_contact_region}", flush=True)
        print(f"[RL REWARD] RL control rate={1/env.step_dt:g} Hz (collector --control-hz ignored); "
              f"RL drives/initial pose/body lock/rewards/terminations unchanged. Display cameras={args.enable_cameras}.", flush=True)
        print("[RL REWARD] Uses RL collision predicates, not the collector's additional self-collision guard. "
              "Simulation inspection only; not a real-robot safety controller.", flush=True)
        while app.is_running():
            start = time.monotonic()
            raw = xr.advance()
            packets = {side: raw.get(getattr(xr.TrackingTarget, "CONTROLLER_" + side.upper()))
                       for side in ("left", "right")}
            head = raw.get(xr.TrackingTarget.HEAD)
            tracked = (head is not None and np.shape(head) == (7,) and np.isfinite(head).all()
                       and np.linalg.norm(head[3:]) > .5
                       and all(v is not None and np.shape(v) == (2, 7) and np.isfinite(v).all()
                               and np.linalg.norm(v[0, 3:]) > .5 for v in (packets[s] for s in control.sides)))
            if requests["panel"]:
                requests["panel"] = False
                visible = hud.toggle_visible()
                print(f"[RL REWARD] Reward panel {'ON' if visible else 'OFF'}", flush=True)
                hud.describe()
            if requests["reset"]:
                requests["reset"] = False
                env.reset()
                control.reset()
                running = terminal = False
                sample, episode_return = None, 0.
                status = "RESET - press A"
            if requests["recenter"] or (not view_ready and tracked):
                requests["recenter"] = False
                if tracked:
                    xr.recenter_view(*head_pose())
                    # Teleport applies on a Kit update; discard pre-teleport packets.
                    app.update()
                    xr.reset()
                    control.reset()
                    view_ready = True
                running = False
                status = "PAUSED - press A"
            if requests["toggle"]:
                requests["toggle"] = False
                if not terminal and tracked and view_ready:
                    running = not running
                    control.reset()
                    status = "RUN" if running else "PAUSED (last step)"
            if running and not tracked:
                running = False
                control.reset()
                status = "TRACKING LOST - press A to resume"
            if running:
                xr.pin_view_position(head_pose()[0])
                t = env.command_manager.get_term("workcell")
                settled = t.settling is None or bool(t.settling.ready[0].item())
                with torch.no_grad():
                    if settled:
                        action = control.action(packets)
                        status = "RUN"
                    else:
                        action = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
                        control.reset()
                        status = "SETTLING"
                    env.step(action)
                sample = env._quest_reward_sample
                if start >= next_grasp_diagnostic or sample["failure"] or sample["success"] or sample["timeout"]:
                    print(f"[RL GRASP] blocked={','.join(sample['blocked_checks']) or 'none'}; "
                          f"failure={','.join(sample['failure_reasons']) or 'none'}; "
                          f"lift={sample['lift_cm']:.1f}cm hold={sample['hold']:.2f}s; "
                          + ' | '.join(sample["grasp_debug"]).replace('\n', ' '), flush=True)
                    next_grasp_diagnostic = start + 1.
                episode_return += sample["total"]
                terminal = sample["success"] or sample["failure"] or sample["timeout"]
                if terminal:
                    running = False
                    control.reset()
                    status = ("SUCCESS" if sample["success"] else "FAILURE" if sample["failure"] else "TIMEOUT")
                    status += " - last step; B to reset"
                    last_hud = 0.
            else:
                # Stop task time/physics while paused, but keep headset rendering and input alive.
                env.sim.render()
            if start - last_hud >= .1 or not running:
                ready = hud.update(format_report(sample, status, episode_return))
                if ready and not panel_ready:
                    hud.describe()
                elif not ready and start >= next_panel_diagnostic:
                    hud.describe()
                    next_panel_diagnostic = start + 10.
                panel_ready = ready
                if camera_overlay is not None:
                    frames = [camera_rgb(env.scene[name]) for name in
                              ("robustness_camera", "left_wrist_camera", "right_wrist_camera")]
                    if all(frame is not None for frame in frames):
                        camera_overlay.update(*frames)
                last_hud = start
            # Do not run simulated time faster than the human can act.
            remaining = env.step_dt - (time.monotonic() - start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        if camera_overlay is not None:
            camera_overlay.close()
        if hud is not None:
            hud.close()
        if env is not None:
            env.close()
        app.close()
