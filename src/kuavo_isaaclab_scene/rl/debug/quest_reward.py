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
from ...workcell.rack_rollers import resolve_rack_roller_settings
from ..runners.common import build_configs
from .contact_rate import configure_obstacle_contact_rate, configure_realtime_reward_debug
from .quest_control import QuestRLControl
from .reward_recorder import RewardProbe
from .reward_report import format_report, reward_summary, ReachRewardSummary
from .stationary_surface import StationarySurface


def _config(args):
    config = Path(args.rl_config or CONFIG_DIR / "rl_pick_arms_only.py").expanduser().resolve()
    if not config.is_file():
        raise ValueError(f"Missing RL config: {config}")
    # Use the same experiment builder as training, including named pose and physics.
    whole_body = getattr(args, "rl_reward_debug", 0) == 1
    rl = Namespace(task="pick", boxes="medium_box_0",
        control_mode="whole-body" if whole_body else "arms-only",
        action_space="all-joints" if whole_body else "right-arm",
        config=config, reset_bank=None, snapshot_dir=None, max_snapshots=1,
        prefill=0, cargo_per_box=0, slots=4, slot_pitch=.52, no_randomization=True,
        num_envs=1, env_spacing=8., enable_cameras=args.enable_cameras, seed=args.seed, device=args.device,
        max_iterations=1, save_interval=None, initial_state=None,
        initial_states_file=CONFIG_DIR / "initial_states.json")
    # No PPO agent is trained or read here (return value is discarded below),
    # so skip importing rsl_rl entirely; this keeps reward inspection working
    # even in an Isaac Lab environment where the training-only '.[rl]' extra
    # (rsl-rl-lib) is not installed.
    cfg, _ = build_configs(rl, include_agent=False)
    if cfg.task.grasp_mode != "flap_top" or cfg.task.name != "pick":
        raise ValueError("Quest reward inspection currently supports flap pick only")
    realtime = (
        not resolve_rack_roller_settings().enabled
        and getattr(args, "rl_obstacle_contact_hz", None) is None
    )
    if realtime:
        obstacle_sensor_count = configure_realtime_reward_debug(cfg)
    else:
        obstacle_contact_hz = getattr(args, "rl_obstacle_contact_hz", None) or 30
        obstacle_sensor_count = configure_obstacle_contact_rate(cfg.scene, obstacle_contact_hz)
    if obstacle_sensor_count == 0:
        raise RuntimeError("RL reward inspection expected obstacle contact sensors to configure.")
    cfg.xr = XrCfg(near_plane=.08)
    cfg.scene.conveyor_surface.class_type = StationarySurface
    cfg.recorders.quest_reward = RecorderTermCfg(class_type=RewardProbe)
    if args.enable_cameras:
        # Display-only RGB cameras: never added to RL policy observations.
        cfg.scene.waist_camera = None
        # Reward inspection does not record datasets. Only create the head
        # camera when a desktop camera preview explicitly requests it; the
        # Quest overlay displays the two wrist cameras.
        if args.head_camera:
            cfg.scene.robustness_camera.width = args.head_camera_width
            cfg.scene.robustness_camera.height = args.head_camera_height
            cfg.scene.robustness_camera.data_types = ["rgb"]
        else:
            cfg.scene.robustness_camera = None
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
    env = hud = camera_overlay = grasp_markers = collision_view = None
    try:
        cfg = _config(args)
        env = ManagerBasedRLEnv(cfg)
        env.reset(seed=args.seed)
        model = resolve_robot_model()
        xr = RawQuestOpenXRDevice(OpenXRDeviceCfg(xr_cfg=cfg.xr, sim_device=env.device), input_mode="controllers")
        control = QuestRLControl(env, model, args, xr)
        start_quest_xr_session(app, enable_ui=(args.rl_reward_hud or args.quest_camera_overlay),
                              resolution_scale=args.xr_resolution_scale,
                              render_quality=args.render_quality)
        if args.rl_reward_hud:
            hud = QuestRewardPanel(forward_axis=args.xr_overlay_forward_axis)
            hud.set_visible(True)
        if args.rl_collision_view:
            from .collision_overlay import CollisionOverlay
            collision_view = CollisionOverlay(env)
        if args.rl_grasp_calibration:
            from .grasp_calibration import GraspCalibration
            grasp_markers = GraspCalibration(env, args.rl_grasp_calibration_file,
                                             model.name, args.rl_grasp_marker_radius)
        elif args.rl_grasp_markers and args.rl_endeffector_centers and env.command_manager.get_term("workcell").endeffector_center.definition:
            from .endeffector_markers import EndEffectorMarkers
            grasp_markers = EndEffectorMarkers(env)
        elif args.rl_grasp_markers:
            from .grasp_markers import GraspMarkers
            grasp_markers = GraspMarkers(env, args.rl_grasp_finger_offsets,
                                         args.rl_grasp_marker_radius, args.rl_grasp_marker_flap,
                                         show_targets=args.rl_grasp_targets)
        if args.quest_camera_overlay:
            from ...display.xr_camera_overlay import QuestCameraOverlay, QuestCameraOverlayCfg
            from ...display.camera_frames import camera_rgb
            camera_overlay = QuestCameraOverlay(
                wrist_resolution=(args.wrist_camera_width, args.wrist_camera_height),
                cfg=QuestCameraOverlayCfg(distance_m=args.xr_overlay_distance,
                                         forward_axis=args.xr_overlay_forward_axis))
        if args.camera_preview:
            from ...display.camera_viewports import open_camera_viewports
            names = ["robustness_camera"] if args.head_camera else []
            if args.wrist_cameras:
                names += ["left_wrist_camera", "right_wrist_camera"]
            if names:
                open_camera_viewports(env.scene, names, headless=args.headless, width=240, height=180, columns=3)
            else:
                print("[VIEW] Camera preview requested, but no camera sensors are enabled.", flush=True)
        keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0., rot_sensitivity=0., sim_device=env.device))
        requests = {name: False for name in ("toggle", "reset", "recenter", "panel", "markers", "collisions", "save_points")}
        def request(name):
            requests[name] = True
        for hand, button, name in (("right", "a", "toggle"), ("right", "b", "reset"),
                                   ("left", "x", "recenter"), ("left", "y", "panel")):
            xr.bind_button(hand, button, lambda name=name: request(name))
        for key, name in (("T", "toggle"), ("R", "reset"), ("C", "recenter"), ("H", "panel"),
                          ("G", "markers"), ("J", "collisions"), ("K", "save_points")):
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
        sample, episode_return, last_hud, last_camera_overlay = None, 0., 0., 0.
        reach_summary = ReachRewardSummary()
        panel_ready, next_panel_diagnostic = False, time.monotonic() + 3.
        next_grasp_diagnostic = 0.
        last_collision_draw = 0.
        status = "PAUSED - X then A"
        print("[RL REWARD] No dataset recording. A/T run/pause; B/R reset; X/C recenter; Y/H panel.", flush=True)
        print(f"[RL REWARD] control={cfg.task.control_mode}, active_arm={cfg.task.active_arm}, "
              f"action_space={cfg.task.action_space}, actions={env.action_manager.total_action_dim}, "
              f"flaps={cfg.task.grasp_flaps}, contact_region={cfg.task.flap_contact_region}", flush=True)
        print(f"[RL REWARD] controller_mapping={args.controller_mapping}"
              + (f", absolute_orientation={args.absolute_orientation}" if args.controller_mapping == "absolute" else "")
              + f", arm_response={args.arm_response}", flush=True)
        print(f"[RL REWARD] RL control rate={1/env.step_dt:g} Hz (collector --control-hz ignored); "
              f"RL drives/initial pose/reward terms/terminations active. Display cameras={args.enable_cameras}.",
              flush=True)
        print("[RL REWARD] grasp/contact measurements=ON; collision constraints follow the loaded RL config; "
              "3D markers/overlay remain opt-in.", flush=True)
        realtime = not resolve_rack_roller_settings().enabled and args.rl_obstacle_contact_hz is None
        if realtime:
            print("[RL REWARD] realtime rollerless profile: physics/control=30 Hz, solver=8/2, "
                  "grasp=30 Hz, aggregate collision=30 Hz, reward=30 Hz, policy observations=OFF.", flush=True)
            if args.device != "cpu":
                print("[RL REWARD] WARNING: use --device cpu for the measured 20 Hz path; "
                      f"{args.device} is slower for this single environment.", flush=True)
        else:
            obstacle_contact_hz = args.rl_obstacle_contact_hz or 30
            print(f"[RL REWARD] fidelity profile: physics/grasp=120 Hz, obstacle={obstacle_contact_hz} Hz, "
                  "reward=30 Hz.", flush=True)
        print(f"[RL REWARD] display: desktop={'full' if args.desktop_render else 'minimal 160x90'}, "
              f"wrist_overlay={args.quest_camera_overlay}, reward_hud={args.rl_reward_hud}, "
              f"collision_overlay={args.rl_collision_view}, grasp_markers={args.rl_grasp_markers}.", flush=True)
        if cfg.task.control_mode == "whole-body":
            print("[RL REWARD] Left stick=base forward/strafe; right stick=base turn/torso lift.", flush=True)
        print("[RL REWARD] Uses RL collision predicates, not the collector's additional self-collision guard. "
              "Simulation inspection only; not a real-robot safety controller.", flush=True)
        perf_started, perf_loops, perf_steps = time.monotonic(), 0, 0
        while app.is_running():
            start = time.monotonic()
            perf_loops += 1
            if requests["save_points"]:
                requests["save_points"] = False
                if args.rl_grasp_calibration and grasp_markers is not None:
                    if running:
                        print("[GRASP CALIBRATION] Pause with A/T before saving.", flush=True)
                    else:
                        grasp_markers.save()
            if requests["collisions"]:
                requests["collisions"] = False
                if collision_view is not None:
                    collision_view.toggle()
            if collision_view is not None and start - last_collision_draw >= .1:
                collision_view.update(contact_sample_valid=sample is not None and not terminal)
                last_collision_draw = start
            if requests["markers"]:
                requests["markers"] = False
                if grasp_markers is not None:
                    grasp_markers.toggle()
            if grasp_markers is not None:
                # Use live poses before rendering, including paused/reset scenes.
                # After terminal auto-reset markers show the RESET scene, unlike the retained reward sample.
                grasp_markers.update()
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
                if hud is None:
                    print("[RL REWARD] Reward HUD was disabled by --no-rl-reward-hud.", flush=True)
                else:
                    visible = hud.toggle_visible()
                    print(f"[RL REWARD] Reward panel {'ON' if visible else 'OFF'}", flush=True)
                    hud.describe()
            if requests["reset"]:
                requests["reset"] = False
                env.reset()
                control.reset()
                running = terminal = False
                sample, episode_return = None, 0.
                reach_summary.reset()
                status = "RESET - press A"
                if collision_view is not None:
                    collision_view.update(contact_sample_valid=False)
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
                    perf_steps += 1
                sample = env._quest_reward_sample
                reach_summary.update(sample, env.step_dt)
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
                    if hud is not None:
                        hud.set_visible(True)  # a hidden panel must not hide the terminal reason
                    control.reset()
                    status = ("SUCCESS" if sample["success"] else "FAILURE" if sample["failure"] else "TIMEOUT")
                    status += " - last step; B to reset"
                    last_hud = 0.
                    if collision_view is not None:
                        collision_view.update(contact_sample_valid=False)
            else:
                # Stop task time/physics while paused, but keep headset rendering and input alive.
                env.sim.render()
            if start - last_hud >= .2:
                if hud is not None:
                    report = format_report(sample, status, episode_return)
                    if grasp_markers is not None and grasp_markers.visible:
                        report = grasp_markers.legend + "\n" + grasp_markers.info + "\n" + report
                    if collision_view is not None and collision_view.visible:
                        report = collision_view.info + "\n" + report
                    headline, checks = reward_summary(sample, status)
                    ready = hud.update(report, headline=headline, checks=checks,
                                       failure=bool(sample and sample["failure"]))
                    if ready and not panel_ready:
                        hud.describe()
                    elif not ready and start >= next_panel_diagnostic:
                        hud.describe()
                        next_panel_diagnostic = start + 10.
                    panel_ready = ready
                last_hud = start
            if camera_overlay is not None and start - last_camera_overlay >= .1:
                frames = [camera_rgb(env.scene[name]) for name in
                          ("left_wrist_camera", "right_wrist_camera")]
                if all(frame is not None for frame in frames):
                    camera_overlay.update(*frames)
                last_camera_overlay = start
            # Do not run simulated time faster than the human can act.
            remaining = env.step_dt - (time.monotonic() - start)
            if remaining > 0:
                time.sleep(remaining)
            now = time.monotonic()
            if now - perf_started >= 5.0:
                elapsed = now - perf_started
                print(f"[PERF] XR loop={perf_loops / elapsed:.1f} Hz; "
                      f"physics/control={perf_steps / elapsed:.1f}/{1 / env.step_dt:g} Hz; "
                      f"{1000 * elapsed / max(perf_loops, 1):.0f} ms/loop", flush=True)
                perf_started, perf_loops, perf_steps = now, 0, 0
    except KeyboardInterrupt:
        pass
    finally:
        if collision_view is not None:
            collision_view.close()
        if grasp_markers is not None:
            grasp_markers.close()
        if camera_overlay is not None:
            camera_overlay.close()
        if hud is not None:
            hud.close()
        if env is not None:
            env.close()
        app.close()
