"""Explicit, trusted local scene configurations applied before env construction.

Scene files are Python code, not USD files. They must only configure ``cfg``
and must not create a SimulationApp or edit a running stage.
"""

from pathlib import Path
import runpy


def apply_scene_config(path: Path | None, cfg) -> tuple[str, ...]:
    """Apply configure(cfg); return additional objects to include in pose records."""
    if path is None:
        return ()
    path = Path(path).expanduser().resolve(strict=True)
    namespace = runpy.run_path(str(path), run_name="kuavo_selected_scene")
    configure = namespace.get("configure")
    names = namespace.get("RECORDING_OBJECTS", ())
    if not callable(configure):
        raise ValueError(f"Scene config must define configure(cfg): {path}")
    if (not isinstance(names, (tuple, list))
            or any(not isinstance(name, str) or not name.isidentifier() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("RECORDING_OBJECTS must be a list/tuple of unique scene asset names.")
    configure(cfg)
    for name in names:
        if getattr(cfg.scene, name, None) is None:
            raise ValueError(f"Scene config declared missing recording object: {name}")
    print(f"[SCENE CONFIG] {path}; extra recorded objects: {', '.join(names) or '(none)'}", flush=True)
    return tuple(names)
