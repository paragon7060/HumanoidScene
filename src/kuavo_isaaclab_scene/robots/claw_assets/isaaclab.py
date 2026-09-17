"""Independent-hand ArticulationCfg; import only after AppLauncher starts Kit."""

from .package import CLAW_ASSET_DIR, load_claw_asset, load_claw_config
from isaaclab.sim.utils import clone


def author_integrated_claw_contact(root, finger_contact):
    """Apply the package contact model to claws embedded in a host robot USD.

    The model itself lives in `usd.py`. This function only names the links the
    host contributes and the mesh scope its donor USD actually carries.
    """
    from .usd import (author_claw_distal_pads, author_claw_jaw_contact,
                      disable_reference_colliders, resolve_finger_contact)

    contact = resolve_finger_contact(load_claw_config(), finger_contact)
    finger_links = {f"{side}_{jaw}_finger" for side in "lr" for jaw in "fb"}
    # These donors carry an empty `collisions` scope, so their visual meshes are
    # the only hand geometry available; the packaged claw uses collision meshes.
    # The wrist links ship a crude URDF cylinder that the mesh hull replaces.
    wrists = {"zarm_l7_link", "zarm_r7_link"}
    hardware = finger_links | {f"{side}_twofinger_base" for side in "lr"} | wrists
    helper_colliders = disable_reference_colliders(root)
    counts = author_claw_jaw_contact(root, contact, links=hardware, finger_links=finger_links,
                                     mesh_scope="visuals", replace_colliders=wrists)
    pad_count, pad_spring = author_claw_distal_pads(root, contact, sides="lr")
    pads = (f"{pad_count} rigid distal pads" if pad_spring is None else
            f"{pad_count} soft distal pads "
            f"({pad_spring[0]:g} N/m, {pad_spring[1]:g} Ns/m per contact)")
    print(f"[CONTACT] Added {sum(counts.values())} hand collision meshes; separate jaws, "
          f"{pads}; disabled {helper_colliders} tool-frame colliders; "
          f"{contact['contact_offset_m'] * 1_000:g}mm contact offset, speculative CCD; finger friction "
          f"{contact['finger_static_friction']}/{contact['finger_dynamic_friction']} "
          f"({contact['friction_combine_mode']}); housing/wrist friction "
          f"{contact['housing_static_friction']}/{contact['housing_dynamic_friction']}.", flush=True)


@clone
def spawn_claw(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Use the same authoritative claw physics in standalone and robot scenes."""
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from .usd import author_claw_contact
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
