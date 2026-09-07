"""Per-cell contact sensors and opt-in robot cameras, without display windows."""

import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg, CameraCfg
from ...robots.robot_model import resolve_robot_model


def add_contacts(scene, spec, geometry):
    for index, body in enumerate(spec.finger_bodies):
        targets = []
        for name in spec.box_names:
            geom = geometry[name]
            if spec.grasp_mode == "flap_top":
                geom = geom.flaps[spec.grasp_flaps[index // 2]]
            targets.append(getattr(scene, name).prim_path + ("/" + geom.body_path if geom.body_path != "." else ""))
        setattr(scene, f"grasp_contact_{index}", ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Kuavo/{body}", update_period=0., history_length=1,
            track_contact_points=spec.grasp_mode == "flap_top", max_contact_data_count_per_prim=32,
            filter_prim_paths_expr=targets))
    bodies = ("waist_yaw_link|zarm_[lr][1-7]_link|[lr]_twofinger_base"
              if spec.grasp_mode == "flap_top" else "waist_yaw_link|zarm_[lr][1-6]_link")
    scene.robot_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Kuavo/(" + bodies + ")",
        update_period=0., history_length=1)
    if spec.grasp_mode == "flap_top":
        from .asset_geometry import robot_rigid_body_paths
        # Filtered reports require one sensor body per environment. Include
        # every link, especially fingers omitted by the old robot_contact mask.
        # Each filter must resolve to exactly one rigid body per environment.
        obstacles = [scene.rack_visual.prim_path, scene.fence.prim_path,
                     scene.button_station.prim_path + "/Base",
                     scene.button_station.prim_path + "/Plunger",
                     scene.conveyor_surface.prim_path]
        obstacles.extend(getattr(scene, f"prefill_{i}").prim_path for i in range(spec.prefill_count))
        for index, path in enumerate(robot_rigid_body_paths(scene.robot.spawn.usd_path)):
            prim_path = scene.robot.prim_path + ("/" + path if path != "." else "")
            setattr(scene, f"obstacle_contact_{index}", ContactSensorCfg(
                prim_path=prim_path, update_period=0., history_length=4,
                filter_prim_paths_expr=list(obstacles)))


def add_cameras(scene):
    model = resolve_robot_model()
    def camera(body, name, pos, rot, focal, near, far):
        return CameraCfg(prim_path=f"{{ENV_REGEX_NS}}/Kuavo/{body}/{name}",
            update_period=1/30, height=120, width=160, data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=focal, focus_distance=1.,
                horizontal_aperture=24., clipping_range=(near, far)),
            offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="ros"))
    scene.robustness_camera = camera(model.head_camera_body, "RobustnessCamera",
        model.head_camera_mount.pos, model.head_camera_mount.rot, 18., .08, 8.)
    scene.waist_camera = camera("waist_yaw_link", "WaistCamera", (.10, 0., .05), (1., 0., 0., 0.), 18., .05, 6.)
    for side in ("left", "right"):
        mount = model.wrist_camera_mounts[side]
        setattr(scene, f"{side}_wrist_camera", camera(model.wrist_camera_bodies[side],
            side.title() + "WristCamera", mount.pos, mount.rot, 12., .03, 2.))
