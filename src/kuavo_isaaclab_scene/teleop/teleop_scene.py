"""Remove unused Quest scenery without changing workcell geometry or materials."""

# Direct children verified in Isaac's Simple_Warehouse/warehouse.usd. The
# user's working rack lives under /World/envs, never under this background.
BACKGROUND_PROP_PREFIXES = (
    "SM_RackShelf_", "SM_RackFrame_", "SM_Rackshield_", "SM_PaletteA_",
    "SM_PushcartA_", "SM_CardBoxA_", "S_AisleSign", "S_Barcode",
)


def compact_factory(factory):
    from pxr import Usd

    removed = []
    prims_before = sum(1 for _ in Usd.PrimRange(factory))
    for prim in list(factory.GetChildren()):
        if prim.GetName() == "KLT_Bins" or prim.GetName().startswith(BACKGROUND_PROP_PREFIXES):
            removed.append(str(prim.GetPath()))
            prim.SetActive(False)
    prims_after = sum(1 for _ in Usd.PrimRange(factory))
    print(f"[SCENE] Background props removed={len(removed)}; factory prims={prims_before}->{prims_after}; "
          "floor/walls/ceiling/lights/materials retained.", flush=True)
    return removed


def spawn_compact_factory(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.spawners.from_files import spawn_from_usd

    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    compact_factory(prim)
    return prim


def configure_scene_detail(cfg, detail):
    if detail == "full":
        return
    if detail != "compact":
        raise ValueError(f"Unknown scene detail: {detail}")
    from ..envs.manager_env import CUSTOM_RACK_BOXES_ACTIVE

    cfg.scene.factory.spawn.func = spawn_compact_factory
    if CUSTOM_RACK_BOXES_ACTIVE:
        # Parked legacy task objects are not the local articulated rack boxes.
        # Keep them when the legacy tote task is explicitly selected.
        cfg.scene.totes = cfg.scene.cargo = None
        for name in ("tote_physics", "cargo_physics", "reset_workcell", "cargo_disturbance"):
            setattr(cfg.events, name, None)
        print("[SCENE] Omitted 9 legacy totes and 12 cargo bodies; local task boxes retained.", flush=True)
