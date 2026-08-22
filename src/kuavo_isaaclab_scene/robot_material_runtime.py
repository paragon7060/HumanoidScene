"""Isaac Lab runtime binding for Kuavo link-level visual materials."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Callable

from pxr import Usd, UsdGeom, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import SpawnerCfg
from isaaclab.utils import configclass

from .robot_material_config import RobotMaterialSettings


def spawn_robot_visual_materials(
    prim_path: str,
    cfg: "RobotVisualMaterialSpawnerCfg",
    translation=None,
    orientation=None,
    **kwargs,
) -> Usd.Prim:
    """Create PreviewSurface materials and bind them above instance proxies."""
    del translation, orientation, kwargs
    stage = sim_utils.get_current_stage()
    parent_expression, leaf = prim_path.rsplit("/", 1)
    env_paths = sim_utils.find_matching_prim_paths(parent_expression)
    if not env_paths:
        raise RuntimeError(f"No environment prims match {parent_expression!r}.")

    first_group = None
    for env_path in env_paths:
        group_path = f"{env_path}/{leaf}"
        group = UsdGeom.Xform.Define(stage, group_path).GetPrim()
        first_group = first_group or group
        looks_path = f"{group_path}/Looks"
        UsdGeom.Scope.Define(stage, looks_path)
        materials: dict[str, UsdShade.Material] = {}
        for name, spec in cfg.settings.materials.items():
            material_path = f"{looks_path}/{name}"
            material_cfg = sim_utils.PreviewSurfaceCfg(
                diffuse_color=spec.diffuse_color,
                roughness=spec.roughness,
                metallic=spec.metallic,
            )
            material_cfg.func(material_path, material_cfg)
            materials[name] = UsdShade.Material(stage.GetPrimAtPath(material_path))

        robot_path = f"{env_path}/{cfg.robot_prim_name}"
        robot = stage.GetPrimAtPath(robot_path)
        if not robot.IsValid():
            raise RuntimeError(f"Cannot apply Kuavo materials; robot prim is missing: {robot_path}")
        matched_links: set[str] = set()
        for link in robot.GetChildren():
            material_name = cfg.settings.material_for_link(link.GetName())
            if material_name is None:
                continue
            binding_api = UsdShade.MaterialBindingAPI.Apply(link)
            binding_api.Bind(
                materials[material_name],
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose=UsdShade.Tokens.allPurpose,
            )
            matched_links.add(link.GetName())
        if not matched_links:
            raise RuntimeError(
                f"Material preset {cfg.settings.name!r} did not match any child links below {robot_path}."
            )
    assert first_group is not None
    return first_group


@configclass
class RobotVisualMaterialSpawnerCfg(SpawnerCfg):
    func: Callable = spawn_robot_visual_materials
    settings: RobotMaterialSettings = MISSING
    robot_prim_name: str = "Kuavo"


def build_robot_visual_material_cfg(settings: RobotMaterialSettings) -> AssetBaseCfg | None:
    if not settings.enabled:
        return None
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/KuavoVisualMaterials",
        spawn=RobotVisualMaterialSpawnerCfg(settings=settings),
    )
