"""Opt-in uncluttered evaluation scene; no hidden or relocated spare boxes."""


def configure_rack_boxes_only(cfg, active_keys, all_keys):
    active_keys, all_keys = tuple(active_keys), tuple(all_keys)
    if not active_keys or not set(active_keys).issubset(all_keys):
        raise ValueError("--rack-boxes-only requires at least one configured rack box.")
    if any(getattr(cfg.scene, key, None) is None for key in active_keys):
        raise ValueError("A selected rack box is missing from the scene configuration.")
    removed = [key for key in all_keys if key not in active_keys]
    for key in (*removed, "totes", "cargo"):
        setattr(cfg.scene, key, None)
    # The GR00T bridge reads only robot state/cameras. These auxiliary manager
    # observations and rewards otherwise reference deleted legacy collections.
    for name in ("tote_poses", "tote_motion", "cargo_state", "cargo_retained", "prefill_count"):
        setattr(cfg.observations.policy, name, None)
    for name in ("cargo_retention", "tote_stability"):
        setattr(cfg.rewards, name, None)
    cfg.terminations.cargo_spill = None
    for name in ("tote_physics", "cargo_physics", "reset_workcell", "cargo_disturbance"):
        setattr(cfg.events, name, None)
    if cfg.events.reset_flap_friction is not None:
        cfg.events.reset_flap_friction.params["asset_names"] = active_keys
    return {
        "mode": "rack-boxes-only",
        "spawned_box_scene_keys": list(active_keys),
        "omitted_box_scene_keys": removed,
        "omitted_collections": ["totes", "cargo"],
        "removed_rewards": ["cargo_retention", "tote_stability"],
        "success_definition": "unchanged: selected boxes on conveyor and button pressed",
    }
