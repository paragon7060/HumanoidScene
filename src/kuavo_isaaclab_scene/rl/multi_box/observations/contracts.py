"""Shared semantics for the same-width neutral/articulated flap observations."""


def flap_observation_contract(source):
    if source not in {'nominal','articulated'}:
        raise ValueError('Flap observation source must be nominal or articulated')
    return dict(observation_contract=(
        'neutral_flap_center_controller_state_actual_base_twist_v2' if source=='nominal' else
        'perceived_articulated_flap_center_controller_state_actual_base_twist_v3'),
        flap_pose_source=source,
        flap_perception_contract=dict(source=source,simulator_pose_proxy=source=='articulated',
            real_backend='supply_panel_midpoint_xyz_wxyz_and_confidence',
            contact_force_or_success_in_actor=False,
            missing_panel_pose='zero_relations_and_assignment; closing_blocked',
            legacy_demo_actual_panel_pose='unavailable; nominal_actor_BC_prior_only'))
