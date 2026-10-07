"""Explicit frozen TRAIN comparison of floating-base attitude gains only."""


PROFILES = {'soft15_2': dict(tilt_stiffness=15., tilt_damping=2., max_tilt_acceleration=10.)}
ORIGINAL = dict(tilt_stiffness=120., tilt_damping=22., max_tilt_acceleration=10.)


def validate_base_attitude_probe(profile, *, workplace, training, num_envs):
    if profile is None:
        return None
    if (profile not in PROFILES or training or num_envs != 128 or not workplace
            or workplace.get('name') != 'CPU_PhysX_frozen_TRAIN_workplace_search_v1'
            or workplace.get('original_candidate_requests') != 128
            or workplace.get('training') is not False):
        raise ValueError('Attitude gain comparison requires complete frozen TRAIN128; no learning or DEV/FINAL')
    return dict(name='frozen_TRAIN_base_attitude_gain_comparison_v1', profile=profile,
        original_gains=ORIGINAL.copy(), configured_gains=PROFILES[profile].copy(),
        controller_parameters_changed=True, training=False, Q_import_eligible=False,
        original_physics_timestep_solver_geometry_mass_and_contacts_preserved=True,
        original_box_base_background_dynamic_flap_randomization_preserved=True,
        success_safety_and_policy_preserved=True, no_root_or_joint_state_teleport=True,
        independent_FINAL_used=False)


def configure_base_attitude_probe(cfg, contract):
    if contract is None:
        return
    if not cfg.actions.base.dynamic:
        raise ValueError('Attitude comparison requires the original unfixed dynamic base')
    drive = cfg.actions.base.drive
    if any(getattr(drive, k) != value for k, value in ORIGINAL.items()):
        raise ValueError('Do not silently combine attitude gain changes')
    for key, value in contract['configured_gains'].items():
        setattr(drive, key, value)


def verify_base_attitude_probe(env, contract):
    if contract is None:
        return None
    drive = env.action_manager.get_term('base')._drive
    if drive is None or drive._asset.is_fixed_base:
        raise ValueError('The actual comparison must use the unfixed wrench controller')
    actual = {key: getattr(drive.cfg, key) for key in ORIGINAL}
    if actual != contract['configured_gains']:
        raise ValueError('Actual drive gains disagree with the declared frozen comparison')
    return contract | dict(actual_runtime_gains=actual, actual_runtime_gains_verified=True)
