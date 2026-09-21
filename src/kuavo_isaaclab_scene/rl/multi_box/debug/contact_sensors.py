"""Privileged v2 finger/flap and box-body/belt contact measurements."""

from __future__ import annotations

from isaaclab.sensors import ContactSensorCfg

from ...scenes.asset_geometry import box_geometry
from ..scene.spawn import physical_asset_names


FINGER_BODY_NAMES = ("l_f_finger", "l_b_finger", "r_f_finger", "r_b_finger")
FLAP_BODY_NAMES = ("flap_right", "flap_left")
CONTACT_SENSOR_NAMES = tuple(f"multi_box_finger_contact_{index}" for index in range(4))
BELT_CONTACT_SENSOR_NAMES = tuple(
    f"multi_box_body_belt_contact_{index}" for index in range(len(physical_asset_names())))
V2_OBSTACLE_SENSOR_NAME = "multi_box_robot_obstacle_contact"


def add_multi_box_contact_sensors(scene) -> None:
    """Expose finger-to-flap contact matrices without changing collision physics.

    Every sensor has one finger as its source and all 18 x 2 flap bodies as
    ordered filters.  Runtime adapters select each environment's current
    logical target physical-pool row.  These signals are privileged and must
    never enter the deployable actor observation.
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

    belt_path = scene.conveyor_surface.prim_path
    for sensor_name, asset_name in zip(
            BELT_CONTACT_SENSOR_NAMES, physical_asset_names(), strict=True):
        asset_cfg = getattr(scene, asset_name)
        body_path = box_geometry(asset_cfg).body_path
        setattr(scene, sensor_name, ContactSensorCfg(
            prim_path=asset_cfg.prim_path + ("/" + body_path if body_path != "." else ""),
            update_period=0.0,
            history_length=1,
            filter_prim_paths_expr=[belt_path],
        ))

    # Aggregate collision-relevant body contacts. Fingers are deliberately
    # excluded because their box contacts are the grasp signal. This catches
    # arm/torso contact with the rack, conveyor, boxes, and floor without the
    # quadratic memory cost of one filtered sensor per robot link.
    setattr(scene, V2_OBSTACLE_SENSOR_NAME, ContactSensorCfg(
        prim_path=(
            "{ENV_REGEX_NS}/Kuavo/(base_link|knee_link|leg_link|waist_pitch_link|"
            "waist_yaw_link|zarm_[lr][1-7]_link|[lr]_twofinger_base)"
        ),
        update_period=0.0,
        history_length=1,
    ))


# Compatibility name for the existing mode-2 inspection config.
add_quest_contact_sensors = add_multi_box_contact_sensors
