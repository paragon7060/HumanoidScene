"""Standalone desktop-only calibration. Does not import Quest or step an RL policy."""

import json
import math
import os
from pathlib import Path
import tempfile
import time


def save_points(calibration):
    """Atomically save the shared display-only format; leave the old file on error."""
    offsets = calibration.offsets()
    if not all(math.isfinite(v) and abs(v) <= .2 for xyz in offsets.values() for v in xyz):
        raise ValueError("Offsets must be finite and within +/-0.2 m of each finger.")
    payload = dict(version=1, units="m", frame="finger_link_local",
                   robot_model=calibration.model_name, offsets=offsets)
    path = calibration.path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    from ...core.paths import CONFIG_DIR
    from ..runners.common import parse_args, build_configs

    def arguments(parser):
        parser.set_defaults(task="pick", boxes="medium_box_0", control_mode="arms-only",
                            num_envs=1, prefill=0, cargo_per_box=0, no_randomization=True,
                            config=CONFIG_DIR / "rl_pick_arms_only.py")
        parser.add_argument("--points-file", type=Path, default=Path("configs/grasp_reference_points.json"))
        parser.add_argument("--point-radius", type=float, default=.004)

    args = parse_args("calibrate-grasp", add_arguments=arguments)
    if args.headless or args.xr or args.num_envs != 1 or args.enable_cameras or args.experience:
        raise ValueError("Desktop calibration requires GUI, one environment, no XR/cameras/custom experience.")
    # Only this process is changed; inherited Quest service settings must not activate XR.
    os.environ.update(XR="0", HEADLESS="0", ENABLE_CAMERAS="0", LIVESTREAM="0")
    args.livestream = 0
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    env = calibration = window = None
    try:
        import omni.ui as ui
        import omni.usd
        from pxr import UsdGeom
        from isaaclab.envs import ManagerBasedRLEnv
        from omni.kit.viewport.utility import get_active_viewport
        from ...robots.robot_model import resolve_robot_model
        from .grasp_calibration import GraspCalibration
        from .stationary_surface import StationarySurface

        cfg, _ = build_configs(args)
        # Local config instance only; no changes to the training or Quest config files.
        cfg.xr = None
        cfg.scene.conveyor_surface.class_type = StationarySurface
        env = ManagerBasedRLEnv(cfg)
        env.reset(seed=args.seed)
        env.sim.render()  # flush the initialized robot pose to the desktop renderer
        env.sim.pause()
        calibration = GraspCalibration(env, args.points_file, resolve_robot_model().name, args.point_radius)
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("No desktop viewport is available.")
        viewport.set_active_camera("/OmniverseKit_Persp")
        viewport.updates_enabled = True
        selection = omni.usd.get_context().get_selection()
        labels = {}
        window = ui.Window("Grasp reference calibration (desktop)", width=510, height=390)

        def select_tip(name):
            selection.set_selected_prim_paths([f"/Visuals/GraspCalibration/{name}/tip"], True)

        def focus_tip(name):
            tip = calibration.tips[calibration.names.index(name)]
            p = UsdGeom.XformCache().GetLocalToWorldTransform(tip.GetPrim()).ExtractTranslation()
            env.sim.set_camera_view(eye=(p[0] + .3, p[1] - .3, p[2] + .18), target=tuple(p))
            select_tip(name)

        def save():
            try:
                save_points(calibration)
            except (OSError, ValueError) as exc:
                status.text = f"NOT SAVED: {exc}"
            else:
                status.text = f"Saved all 4 points:\n{calibration.path}"
            print(f"[GRASP CALIBRATION] {status.text}", flush=True)

        with window.frame:
            with ui.VStack(spacing=6):
                ui.Label("Physics PAUSED. No XR, cameras, policy or recording.", height=22)
                ui.Label("Select tip -> move gizmo (W in viewport) -> Save all 4.", height=22)
                ui.Label("Edit tip only, not its parent or marker child. Units: metres.", height=22)
                for name in calibration.names:
                    with ui.HStack(height=26):
                        ui.Button(f"Select {name}", clicked_fn=lambda n=name: select_tip(n))
                        ui.Button("Focus", width=65, clicked_fn=lambda n=name: focus_tip(n))
                    labels[name] = ui.Label("", height=18)
                ui.Button("Save all 4 points", clicked_fn=save, height=30)
                status = ui.Label(f"Output: {calibration.path}\nUnsaved edits are lost on exit.", word_wrap=True)

        focus_tip("r_f_finger")
        print("[GRASP CALIBRATION] Desktop only; physics stays paused. Use the calibration window.", flush=True)
        while app.is_running():
            # No env.step(), sim.step() or live tracking. Parent frames remain frozen.
            # render() pumps Kit UI with physics disabled; even toolbar Play cannot run a policy.
            if env.sim.is_playing():
                env.sim.pause()
            for name, xyz in calibration.offsets().items():
                labels[name].text = "local XYZ: " + ", ".join(f"{v:.5f}" for v in xyz)
            env.sim.render()
            time.sleep(.02)
    finally:
        if window is not None:
            window.destroy()
        if calibration is not None:
            calibration.close()
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
