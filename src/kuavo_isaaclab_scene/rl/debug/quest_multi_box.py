"""Quest teleoperation for visual and physical inspection of multi-box v2."""

import logging
import math
import sys
import time
import traceback

import numpy as np
import torch
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import combine_frame_transforms, convert_camera_frame_orientation_convention

from ...display.xr_reward_panel import QuestRewardPanel
from ...robots.robot_model import resolve_robot_model
from ...teleop.quest_openxr import RawQuestOpenXRDevice, start_quest_xr_session
from ...teleop.urdf_arm_ik import ArmJointLimitError
from ..multi_box.debug import (
    PoseShadowRewardEvaluator,
    ShadowRewardLogger,
    ShadowRewardStats,
    format_shadow_reward,
)
from ..multi_box.debug.isaac_metrics import IsaacMultiBoxMetricAdapter
from ..multi_box.scene.spawn import BOX_TYPE_IDS
from ..multi_box.teleop_env_cfg import build_quest_multi_box_cfg
from .quest_control import QuestRLControl


def _tracked(head, packets, sides):
    return (
        head is not None
        and np.shape(head) == (7,)
        and np.isfinite(head).all()
        and np.linalg.norm(head[3:]) > 0.5
        and all(
            packets[side] is not None
            and np.shape(packets[side]) == (2, 7)
            and np.isfinite(packets[side]).all()
            and np.linalg.norm(packets[side][0, 3:]) > 0.5
            for side in sides
        )
    )


def _layout_report(env, status, shadow_text=None):
    spec = env.cfg.multi_box
    if not hasattr(env, "_multi_box_counts"):
        return "MULTI BOX V2 | RESET DATA PENDING\n" + status
    count = int(env._multi_box_counts[0].item())
    active = env._multi_box_active[0]
    types = env._multi_box_box_type_ids[0]
    regions = env._multi_box_region_ids[0]
    rack_xy = env._multi_box_rack_xy_delta[0].detach().cpu().tolist()
    rack_yaw = math.degrees(float(env._multi_box_rack_yaw_delta[0].item()))
    conveyor_xy = env._multi_box_conveyor_xy_delta[0].detach().cpu().tolist()
    conveyor_yaw = math.degrees(float(env._multi_box_conveyor_yaw_delta[0].item()))
    names_by_type = {value: key for key, value in BOX_TYPE_IDS.items()}
    lines = [
        "MULTI BOX V2 | VR TELEOP",
        f"STATUS: {status}",
        f"ACTIVE: {count}/{spec.max_boxes} boxes",
        f"RACK jitter: x={rack_xy[0]:+.3f} y={rack_xy[1]:+.3f} yaw={rack_yaw:+.1f}deg",
        f"BELT jitter: x={conveyor_xy[0]:+.3f} y={conveyor_xy[1]:+.3f} yaw={conveyor_yaw:+.1f}deg",
        "B/R: randomize again | X/C: recenter | A/T: run/pause",
        "Pose shadow + raw finger/flap contacts: ON | success adapter: pending",
    ]
    if shadow_text:
        lines.extend(("", shadow_text))
    for logical_id in active.nonzero(as_tuple=False).flatten().tolist():
        type_name = names_by_type[int(types[logical_id].item())]
        region_name = spec.region_names[int(regions[logical_id].item())]
        lines.append(f"box[{logical_id:02d}] {type_name:<6} {region_name}")
    return "\n".join(lines)


def run(args, app):
    env = hud = response_probe = shadow_logger = None
    try:
        cfg = build_quest_multi_box_cfg(args)
        env = ManagerBasedRLEnv(cfg)
        env.reset(seed=args.seed)
        model = resolve_robot_model()
        metric_adapter = IsaacMultiBoxMetricAdapter(env)
        shadow_evaluator = PoseShadowRewardEvaluator(args.rl_shadow_phase)
        shadow_stats = ShadowRewardStats()
        if args.rl_shadow_log is not None:
            shadow_logger = ShadowRewardLogger(args.rl_shadow_log)
            shadow_stats = shadow_logger.stats
        shadow_snapshot = metric_adapter.measure()
        shadow_breakdown = shadow_evaluator.evaluate(
            shadow_snapshot.potentials_by_phase[shadow_evaluator.phase])
        if getattr(args, "joint_response_log", None) is not None:
            from ...recording.joint_response import JointResponseProbe
            response_probe = JointResponseProbe(env, model, args.joint_response_log, "rl_reward_debug_2")

        xr = RawQuestOpenXRDevice(
            OpenXRDeviceCfg(xr_cfg=cfg.xr, sim_device=env.device), input_mode="controllers")
        control = QuestRLControl(env, model, args, xr)
        start_quest_xr_session(
            app, enable_ui=args.rl_reward_hud,
            resolution_scale=args.xr_resolution_scale,
            render_quality=args.render_quality,
        )
        if args.rl_reward_hud:
            hud = QuestRewardPanel(forward_axis=args.xr_overlay_forward_axis)
            hud.set_visible(True)

        keyboard = Se3Keyboard(Se3KeyboardCfg(
            pos_sensitivity=0.0, rot_sensitivity=0.0, sim_device=env.device))
        requests = {name: False for name in (
            "toggle", "reset", "recenter", "panel", "target_prev", "target_next",
            "phase_grasp", "phase_carry", "phase_place",
        )}

        def request(name):
            requests[name] = True

        for hand, button, name in (
            ("right", "a", "toggle"), ("right", "b", "reset"),
            ("left", "x", "recenter"), ("left", "y", "panel"),
        ):
            xr.bind_button(hand, button, lambda name=name: request(name))
        for key, name in (("T", "toggle"), ("R", "reset"),
                          ("C", "recenter"), ("H", "panel"),
                          ("J", "target_prev"), ("L", "target_next"),
                          ("1", "phase_grasp"), ("2", "phase_carry"),
                          ("3", "phase_place")):
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
        status = "PAUSED - X then A"
        last_hud = 0.0
        panel_ready = False
        next_panel_diagnostic = time.monotonic() + 3.0
        perf_started, perf_loops, perf_steps, shadow_step = time.monotonic(), 0, 0, 0
        print("[RL MULTI BOX] Mode 2: randomized v2 scene, no dataset recording.", flush=True)
        print("[RL MULTI BOX] A/T run/pause; B/R randomize reset; X/C recenter; Y/H panel.", flush=True)
        print("[RL MULTI BOX] Shadow controls: J/L target box; 1/2/3 grasp/carry/place phase.", flush=True)
        print(f"[RL MULTI BOX] prepared_state={cfg.prepared_state_name}; boxes=1-{cfg.multi_box.max_boxes}; "
              "shelf2=small/medium, shelf3=small; rack/conveyor pose jitter=ON.", flush=True)
        print(f"[RL MULTI BOX] control=whole-body, active_arm=both, action_space=all-joints, "
              f"actions={env.action_manager.total_action_dim}, gripper_close={args.gripper_close_force:g}N.",
              flush=True)
        print("[RL MULTI BOX] pose/velocity shadow reward and raw finger/flap contacts are read-only; "
              "contact thresholds, success transitions, collisions, and regularization remain disabled.",
              flush=True)
        if shadow_logger is not None:
            print(f"[RL MULTI BOX] shadow JSONL={shadow_logger.path}", flush=True)
        print("[RL MULTI BOX] episode time limit=OFF; B/R is the only scene reset.", flush=True)

        while app.is_running():
            started = time.monotonic()
            perf_loops += 1
            raw = xr.advance()
            packets = {
                side: raw.get(getattr(xr.TrackingTarget, "CONTROLLER_" + side.upper()))
                for side in ("left", "right")
            }
            head = raw.get(xr.TrackingTarget.HEAD)
            tracked = _tracked(head, packets, control.sides)

            if requests["panel"]:
                requests["panel"] = False
                if hud is None:
                    print("[RL MULTI BOX] HUD disabled by --no-rl-reward-hud.", flush=True)
                else:
                    visible = hud.toggle_visible()
                    print(f"[RL MULTI BOX] Panel {'ON' if visible else 'OFF'}", flush=True)
                    hud.describe()
            for phase in ("grasp", "carry", "place"):
                key = "phase_" + phase
                if requests[key]:
                    requests[key] = False
                    shadow_evaluator.set_phase(phase)
                    shadow_snapshot = metric_adapter.measure()
                    shadow_breakdown = shadow_evaluator.evaluate(
                        shadow_snapshot.potentials_by_phase[phase])
                    print(f"[RL MULTI BOX] Shadow phase={phase}", flush=True)
                    last_hud = 0.0
            for key, direction in (("target_prev", -1), ("target_next", 1)):
                if requests[key]:
                    requests[key] = False
                    target = metric_adapter.cycle_target(direction)
                    shadow_evaluator.reset()
                    shadow_snapshot = metric_adapter.measure()
                    shadow_breakdown = shadow_evaluator.evaluate(
                        shadow_snapshot.potentials_by_phase[shadow_evaluator.phase])
                    print(f"[RL MULTI BOX] Shadow target=box[{target:02d}]", flush=True)
                    last_hud = 0.0
            if requests["reset"]:
                requests["reset"] = False
                if response_probe is not None:
                    response_probe.boundary("operator_reset")
                env.reset()
                control.reset()
                metric_adapter.reset()
                shadow_evaluator.reset()
                shadow_snapshot = metric_adapter.measure()
                shadow_breakdown = shadow_evaluator.evaluate(
                    shadow_snapshot.potentials_by_phase[shadow_evaluator.phase])
                shadow_stats = ShadowRewardStats() if shadow_logger is None else shadow_logger.stats
                running = False
                status = "RANDOMIZED - press A"
                print(_layout_report(env, status), flush=True)
            if requests["recenter"] or (not view_ready and tracked):
                requests["recenter"] = False
                if response_probe is not None:
                    response_probe.boundary("recenter")
                if tracked:
                    xr.recenter_view(*head_pose())
                    app.update()
                    xr.reset()
                    control.reset()
                    view_ready = True
                running = False
                status = "PAUSED - press A"
            if requests["toggle"]:
                requests["toggle"] = False
                if tracked and view_ready:
                    running = not running
                    if response_probe is not None:
                        response_probe.boundary("run" if running else "pause")
                    control.reset()
                    status = "RUN" if running else "PAUSED"
            if running and not tracked:
                if response_probe is not None:
                    response_probe.boundary("tracking_lost")
                running = False
                control.reset()
                status = "TRACKING LOST - press A"

            if running:
                xr.pin_view_position(head_pose()[0])
                try:
                    action = control.action(packets)
                except ArmJointLimitError as exc:
                    running = False
                    status = "JOINT LIMIT - B/R to reset"
                    logging.getLogger(__name__).warning("Quest multi-box control paused: %s", exc)
                    print(f"[RL MULTI BOX] {status}: {exc}", flush=True)
                    if hud is not None:
                        hud.set_visible(True)
                    last_hud = 0.0
                    env.sim.render()
                    continue
                with torch.no_grad():
                    if response_probe is not None:
                        response_probe.begin(tracked)
                    env.step(action)
                    if response_probe is not None:
                        response_probe.end(
                            tracking_valid=tracked, collision=False,
                            context={
                                "mode": "multi_box_scene_inspection",
                                "body_command": control.last_body_command,
                            })
                perf_steps += 1
                shadow_step += 1
                shadow_snapshot = metric_adapter.measure()
                phase = shadow_evaluator.phase
                shadow_breakdown = shadow_evaluator.evaluate(
                    shadow_snapshot.potentials_by_phase[phase])
                raw_metrics = shadow_snapshot.log_scalars(phase)
                if shadow_logger is not None:
                    shadow_logger.record(
                        step=shadow_step,
                        sim_time_s=shadow_step * env.step_dt,
                        phase=phase,
                        raw=raw_metrics,
                        potentials=shadow_snapshot.potentials_by_phase[phase],
                        breakdown=shadow_breakdown,
                        events={
                            "contact_adapter_available": bool(
                                shadow_snapshot.diagnostics.get("contact_adapter_available", 0.0)),
                            "success": False,
                            "collision": False,
                        },
                    )
                else:
                    shadow_stats.update(shadow_breakdown)
                radius = torch.linalg.vector_norm(
                    robot.data.root_pos_w[0, :2] - env.scene.env_origins[0, :2]).item()
                if radius > cfg.multi_box.workspace_radius:
                    running = False
                    control.reset()
                    status = f"WORKSPACE LIMIT ({radius:.2f}m) - B/R to reset"
                    print(f"[RL MULTI BOX] {status}", flush=True)
                    last_hud = 0.0
            else:
                env.sim.render()

            if started - last_hud >= 0.2:
                if hud is not None:
                    phase = shadow_evaluator.phase
                    shadow_text = (
                        f"TARGET box[{shadow_snapshot.target_logical_id:02d}] "
                        f"{shadow_snapshot.target_box_type} {shadow_snapshot.target_region}\n"
                        + format_shadow_reward(
                            phase, shadow_snapshot.raw_scalars(phase),
                            shadow_breakdown, shadow_stats)
                        + "\n" + shadow_snapshot.contact_report()
                        + "\nEVENT TERMS: disabled until thresholds/success adapters"
                    )
                    report = _layout_report(env, status, shadow_text)
                    ready = hud.update(
                        report, headline=status,
                        checks=(f"ACTIVE: {int(env._multi_box_counts[0])}/12",
                                f"SHADOW: {phase} / target {shadow_snapshot.target_logical_id:02d}"),
                        failure=status.startswith(("JOINT LIMIT", "WORKSPACE LIMIT")),
                    )
                    if ready and not panel_ready:
                        hud.describe()
                    elif not ready and started >= next_panel_diagnostic:
                        hud.describe()
                        next_panel_diagnostic = started + 10.0
                    panel_ready = ready
                last_hud = started

            remaining = env.step_dt - (time.monotonic() - started)
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
    except Exception:
        logging.getLogger(__name__).exception("Multi-box Quest inspection failed; closing simulator")
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if hud is not None:
            hud.close()
        try:
            if response_probe is not None:
                response_probe.close()
        finally:
            try:
                if shadow_logger is not None:
                    shadow_logger.close()
            finally:
                try:
                    if env is not None:
                        env.close()
                finally:
                    app.close()
