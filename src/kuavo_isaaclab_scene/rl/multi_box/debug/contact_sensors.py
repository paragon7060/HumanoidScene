"""Mode-2-only filtered contact reports for raw grasp calibration."""

from __future__ import annotations

from isaaclab.sensors import ContactSensorCfg

from ...scenes.asset_geometry import box_geometry
from ..scene.spawn import physical_asset_names


FINGER_BODY_NAMES = ("l_f_finger", "l_b_finger", "r_f_finger", "r_b_finger")
FLAP_BODY_NAMES = ("flap_right", "flap_left")
CONTACT_SENSOR_NAMES = tuple(f"multi_box_finger_contact_{index}" for index in range(4))


def add_quest_contact_sensors(scene) -> None:
    """Expose finger-to-flap contact matrices without changing collision physics.

    Every sensor has one finger as its source and all 18 x 2 flap bodies as
    ordered filters.  The runtime adapter selects the current logical target's
    physical pool row.  This is intentionally added only by the one-env Quest
    config, not by the future vectorized training scene.
    """
    targets = []
    for asset_name in physical_asset_names():
        asset_cfg = getattr(scene, asset_name)
        geometry = box_geometry(asset_cfg, FLAP_BODY_NAMES)
        for flap_name in FLAP_BODY_NAMES:
            path = geometry.flaps[flap_name].body_path
            targets.append(asset_cfg.prim_path + ("/" + path if path != "." else ""))
    expected = len(physical_asset_names()) * len(FLAP_BODY_NAMES)
    if len(targets) != expected:
        raise ValueError(f"Expected {expected} ordered v2 flap contact targets.")

    for sensor_name, finger_name in zip(CONTACT_SENSOR_NAMES, FINGER_BODY_NAMES, strict=True):
        setattr(scene, sensor_name, ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Kuavo/{finger_name}",
            update_period=0.0,
            history_length=1,
            track_contact_points=True,
            max_contact_data_count_per_prim=64,
            filter_prim_paths_expr=targets,
        ))
