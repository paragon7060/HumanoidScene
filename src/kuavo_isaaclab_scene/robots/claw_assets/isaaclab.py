"""Independent-hand ArticulationCfg; import only after AppLauncher starts Kit."""

from .package import load_claw_asset
from isaaclab.sim.utils import clone


@clone
def spawn_claw(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Use the same authoritative claw physics in standalone and robot scenes."""
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from .usd import author_claw_contact
    from .package import CLAW_ASSET_DIR

    root = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    side = "left" if root.GetChild("l_twofinger_base") else "right"
    author_claw_contact(root.GetStage(), CLAW_ASSET_DIR / "config.json", side=side, root=root)
    return root


def make_claw_cfg(side: str, prim_path: str, *, fix_base: bool = True,
                  pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg

    asset = load_claw_asset(side)
    actuator = asset.metadata["actuator"]
    if not asset.usd_path.is_file():
        raise FileNotFoundError(f"Build the claw USDs first: {asset.usd_path}")
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            func=spawn_claw,
            usd_path=str(asset.usd_path),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=fix_base, enabled_self_collisions=False,
                solver_position_iteration_count=16, solver_velocity_iteration_count=4),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=pos, rot=rot, joint_pos=asset.initial_positions(), joint_vel={".*": 0.0}),
        actuators={
            "drivers": ImplicitActuatorCfg(joint_names_expr=list(asset.motor_names), **actuator),
            "passive": ImplicitActuatorCfg(
                joint_names_expr=[f"{asset.prefix}_[fb]_bar_[34]_joint"],
                effort_limit_sim=5.0, stiffness=0.0, damping=0.0, friction=0.0),
        },
        soft_joint_pos_limit_factor=1.0,
    )
