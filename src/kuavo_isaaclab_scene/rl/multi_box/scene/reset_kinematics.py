"""Publish coherent links after teleporting v2 GPU articulations."""


def refresh_teleported_articulations(env, assets, env_ids):
    """Refresh reset FK without stepping physics or changing the final state.

    In the installed PhysX GPU runtime, teleporting a free articulation while
    writing unchanged DOF positions can leave its child links at the old pose.
    The next solver step then snaps the root toward those stale links. A
    temporary 0.001-rad DOF change marks FK dirty; both FK passes happen at the
    same simulation timestamp and the original DOFs are restored before any
    physics/manager call. Only reset environments are written, and drive
    targets, velocities, rewards and settling criteria are untouched.

    Two global FK passes serve the whole asset batch. CPU physics already
    updates these links correctly and retains its existing reset path.
    """
    if not str(env.device).startswith("cuda") or not len(env_ids):
        return False
    saved = [(asset, asset.data.joint_pos[env_ids].clone())
             for asset in assets if asset.num_joints]
    if not saved:
        return False
    view = env.sim.physics_sim_view
    try:
        for asset, q in saved:
            # There is no physics step at this temporary configuration.
            asset.write_joint_position_to_sim(q + .001, env_ids=env_ids)
        view.update_articulations_kinematic()
    finally:
        for asset, q in saved:
            asset.write_joint_position_to_sim(q, env_ids=env_ids)
        view.update_articulations_kinematic()
    # Readers may have populated a link buffer before the final FK pass.
    for asset, _ in saved:
        for name, buffer in vars(asset.data).items():
            if name.startswith("_body_") and hasattr(buffer, "timestamp"):
                buffer.timestamp = -1.0
    return True
