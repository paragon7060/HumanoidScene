"""Bounded frozen comparison of per-object and replicated scene construction."""

from .reset_world_frame import PASSIVE_STATE_MODE, validate_reset_world_frame_request


def validate_independent_scene_request(waves, frame, *, enabled, reset_enabled,
                                       training, steps, other_probe=False):
    if not enabled:
        return
    if frame is None or frame.get('probe_type') != PASSIVE_STATE_MODE or other_probe:
        raise ValueError('Independent scene probe requires only matched original world and passive state')
    validate_reset_world_frame_request(waves, frame, reset_enabled=reset_enabled,
                                      training=training, steps=steps)
    if not 1 <= len(waves[0]['layouts']) <= 128:
        raise ValueError('Independent scene comparison is bounded to1..128 frozen DEV environments')


def configure_independent_scene_probe(cfg, *, enabled):
    if not enabled:
        return None
    source=dict(replicate_physics=cfg.scene.replicate_physics,
                filter_collisions=cfg.scene.filter_collisions,
                clone_in_fabric=cfg.scene.clone_in_fabric)
    if source != dict(replicate_physics=True, filter_collisions=True, clone_in_fabric=False):
        raise ValueError('Independent scene control requires the original filtered non-fabric scene')
    cfg.scene.replicate_physics=False
    return dict(name='independent_scene_construction', frozen_only=True,
                Q_import_eligible=False, original_scene_config=source,
                requested_scene_config=source | dict(replicate_physics=False),
                physics_parameters_success_safety_and_requested_layouts_unchanged=True,
                constructor_contact_history_not_bitwise_matched=True,
                independent_collision_groups_required=True)


def verify_independent_scene_probe(env, audit):
    """Read actual scene flags and authored cross-environment collision groups."""
    if audit is None:
        return None
    from pxr import UsdPhysics
    actual={key:getattr(env.scene.cfg,key) for key in audit['requested_scene_config']}
    if actual != audit['requested_scene_config']:
        raise ValueError('Actual scene construction flags differ from the frozen request')
    root=env.sim.stage.GetPrimAtPath('/World/collisions')
    if not root.IsValid():
        raise ValueError('Independent scene has no authored environment collision groups')
    physics_scene=env.sim.stage.GetPrimAtPath(env.scene.physics_scene_path)
    global_invert=bool(physics_scene.GetAttribute('physxScene:invertCollisionGroupFilter').Get())
    groups=[]
    for prim in root.GetChildren():
        if not prim.IsA(UsdPhysics.CollisionGroup):
            continue
        group=UsdPhysics.CollisionGroup(prim)
        groups.append(dict(path=str(prim.GetPath()),
            colliders=[str(p) for p in group.GetCollidersCollectionAPI().GetIncludesRel().GetTargets()],
            filtered_groups=[str(p) for p in group.GetFilteredGroupsRel().GetTargets()],
            invert_filtered_groups=bool(group.GetInvertFilteredGroupsAttr().Get())))
    env_groups={path:[g for g in groups if path in g['colliders']] for path in env.scene.env_prim_paths}
    if any(len(found)!=1 for found in env_groups.values()):
        raise ValueError('Each independent environment must have exactly one collision group')
    for path,found in env_groups.items():
        group=found[0]
        for other,other_found in env_groups.items():
            if other==path:
                continue
            filtered=other_found[0]['path'] in group['filtered_groups']
            effective_invert=global_invert or group['invert_filtered_groups']
            if filtered==effective_invert:
                raise ValueError('Independent environment collision filtering is incomplete')
    return audit | dict(actual_scene_config=actual,
                        actual_environment_collision_groups=groups,
                        actual_PhysX_global_invert_collision_group_filter=global_invert,
                        all_environment_pairs_collision_filtered=True)
