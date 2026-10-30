"""Optional USD-only geometry tests; no SimulationApp or rendering.

Run in a Python environment where the USD bindings are already available.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("pxr.Usd")
from kuavo_isaaclab_scene.rl.scenes.asset_geometry import box_geometry, robot_rigid_body_paths


def test_physical_wrapper_nested_scale_and_flap_size():
    root = Path(__file__).resolve().parents[1]
    cfg = SimpleNamespace(spawn=SimpleNamespace(
        usd_path=str(root / "src/kuavo_isaaclab_scene/assets/MediumBox_atlas.usda"), scale=(1., 1., 1.)))
    g = box_geometry(cfg, ("flap_right", "flap_left"))
    assert g.half_size[:2] == pytest.approx((.16, .11), abs=1e-6)
    assert g.half_size[2] == pytest.approx(.185925 / 2, abs=1e-6)
    assert g.flaps["flap_right"].half_size[2] == pytest.approx(.055, abs=1e-6)
    assert g.flaps["flap_right"].body_path.endswith("/flap_right")
    cfg.spawn.scale = (2., 1., 1.)
    scaled = box_geometry(cfg)
    assert scaled.half_size[0] == pytest.approx(2*g.half_size[0])
    assert scaled.half_size[2] == pytest.approx(g.half_size[2])


def test_obstacle_sensors_cover_every_robot_link_and_exclude_task_box(monkeypatch):
    import importlib.util
    import re
    import sys
    from types import ModuleType
    from pxr import Usd, UsdPhysics
    from kuavo_isaaclab_scene.core.paths import ASSET_DIR
    from kuavo_isaaclab_scene.rl.tasks.specs import task_spec

    # Run the actual sensor assembly with lightweight config containers and the
    # packaged USDs. PhysX contact reporting itself still needs a simulator test.
    isaaclab = ModuleType("isaaclab")
    isaaclab.sim = ModuleType("isaaclab.sim")
    sensors = ModuleType("isaaclab.sensors")
    sensors.ContactSensorCfg = sensors.CameraCfg = SimpleNamespace
    for name, module in (("isaaclab", isaaclab), ("isaaclab.sim", isaaclab.sim), ("isaaclab.sensors", sensors)):
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[1] / "src/kuavo_isaaclab_scene/rl/scenes/sensors.py"
    loader = importlib.util.spec_from_file_location("kuavo_isaaclab_scene.rl.scenes._test_sensors", path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    robot_usd = str(ASSET_DIR / "kuavo_s200062/usd/kuavo_s200062_fixed.usd")
    scene = SimpleNamespace(robot=SimpleNamespace(prim_path="{ENV_REGEX_NS}/Kuavo", spawn=SimpleNamespace(usd_path=robot_usd)),
        rack_visual=SimpleNamespace(prim_path="{ENV_REGEX_NS}/Workcell/Racks/Rack/Visual"),
        fence=SimpleNamespace(prim_path="{ENV_REGEX_NS}/Workcell/SafetySystem/Fence/Panel"),
        button_station=SimpleNamespace(prim_path="{ENV_REGEX_NS}/Workcell/SafetySystem/ButtonStation"),
        conveyor_surface=SimpleNamespace(prim_path="{ENV_REGEX_NS}/Workcell/ConveyorSystem/Surface"),
        small_box_0=SimpleNamespace(prim_path="{ENV_REGEX_NS}/Workcell/StagingBoxes/SmallBox"))
    geometry = box_geometry(SimpleNamespace(spawn=SimpleNamespace(
        usd_path=str(ASSET_DIR / "MediumBox_atlas.usda"), scale=(1., 1., 1.))), ("flap_right", "flap_left"))
    task = task_spec("pick", control_mode="arms-only", grasp_mode="flap_top", prefill_count=0)
    module.add_contacts(scene, task, {"small_box_0": geometry})
    obstacle = [cfg for name, cfg in vars(scene).items() if name.startswith("obstacle_contact_")]
    links = robot_rigid_body_paths(robot_usd)
    assert len(obstacle) == len(links)
    paths = {cfg.prim_path.rsplit("/", 1)[-1] for cfg in obstacle}
    assert {"l_f_finger", "l_b_finger", "r_f_finger", "r_b_finger", "l_twofinger_base", "r_twofinger_base"} <= paths
    filters = obstacle[0].filter_prim_paths_expr
    assert all(cfg.history_length == 4 and cfg.update_period == 0. for cfg in obstacle)
    assert all("StagingBoxes" not in pattern for pattern in filters)
    rack = Usd.Stage.Open(str(ASSET_DIR / "Rack.usd"))
    root = rack.GetDefaultPrim().GetPath()
    colliders = [p for p in rack.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert colliders
    for prim in colliders:
        path = scene.rack_visual.prim_path + "/" + str(prim.GetPath().MakeRelativePath(root))
        assert any(path.startswith(pattern + "/") for pattern in filters), path
    assert len(filters) == 5 and all(".*" not in pattern for pattern in filters)
