"""Small tiled viewport windows pegged to robot-mounted cameras (GUI mode only)."""

from __future__ import annotations


def open_camera_viewports(
    scene,
    camera_names: list[str],
    headless: bool,
    env_index: int = 0,
    width: int = 320,
    height: int = 240,
    columns: int = 2,
) -> None:
    """Open one small tiled viewport window per named camera sensor.

    Purely a visual convenience for interactive (non-headless) Isaac Sim
    sessions: each robot camera (head/waist/wrist, etc.) gets its own small
    floating window tiled in the corner of the main viewport, in addition to
    whatever the IsaacLab Camera sensor already streams to
    observations/screenshots. Skipped entirely in headless mode, since GUI
    viewport windows require an on-screen window.

    Args:
        scene: The InteractiveScene (or ManagerBasedRLEnv.scene) that
            already contains the named camera sensors.
        camera_names: Sensor names to open windows for, e.g.
            ["head_camera", "waist_camera", "left_wrist_camera",
            "right_wrist_camera"].
        headless: Skip entirely when True.
        env_index: Which cloned environment's concrete camera prim to peg to.
        width: Pixel width of each small viewport window.
        height: Pixel height of each small viewport window.
        columns: Number of viewport windows per row before wrapping.
    """
    if headless:
        return

    try:
        from isaacsim.core.utils.viewports import create_viewport_for_camera
    except ImportError:
        print("[WARN] omni.kit.viewport is unavailable; skipping camera preview windows.")
        return

    opened = 0
    for i, name in enumerate(camera_names):
        if name not in scene.sensors:
            print(f"[WARN] Camera {name!r} not found in scene; skipping its preview window.")
            continue
        camera = scene[name]
        prim_path = camera.cfg.prim_path.replace("env_.*", f"env_{env_index}")
        col = i % columns
        row = i // columns
        try:
            create_viewport_for_camera(
                viewport_name=f"{name}_view",
                camera_prim_path=prim_path,
                width=width,
                height=height,
                position_x=col * width,
                position_y=row * height,
            )
            opened += 1
        except Exception as exc:  # pragma: no cover - depends on GUI availability
            print(f"[WARN] Failed to open preview window for camera {name!r}: {exc}")

    if opened:
        print(f"[INFO] Opened {opened} camera preview viewport window(s), tiled from the top-left corner.")
