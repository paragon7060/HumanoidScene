"""Physical seeds are captured before the first transition, without Isaac."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import h5py

from kuavo_isaaclab_scene.recording.rl_initial_state import capture_rl_initial_state


def test_capture_keeps_measured_angles_distinct_from_pd_targets(tmp_path):
    q = np.array([[.1, .2]])
    data = SimpleNamespace(joint_pos_target=q + .04, joint_vel_target=q * 0,
                           joint_effort_target=q * 10)
    robot = SimpleNamespace(data=data)
    class Scene:
        articulations = {"robot": robot}
        def __getitem__(self, name):
            return self.articulations[name]
        def get_state(self, *, is_relative):
            assert not is_relative
            return {"articulation": {"robot": {"joint_position": q}}}
    term = SimpleNamespace(raw_actions=q * 0, processed_actions=q + .04,
                           _targets=q + .04)
    manager = SimpleNamespace(active_terms=["upper_body"], action=q * 0,
                              prev_action=q * 0, get_term=lambda name: term)
    env = SimpleNamespace(num_envs=1, scene=Scene(), action_manager=manager,
                          common_step_counter=17, episode_length_buf=np.array([3]),
                          _multi_box_active=np.array([[True, False]]))
    state = capture_rl_initial_state(env, {"policy": q})
    q[:] = 99
    np.testing.assert_allclose(state["scene"]["articulation"]["robot"]["joint_position"], [.1, .2])
    np.testing.assert_allclose(state["drive_targets"]["robot"]["joint_position"], [.14, .24])
    np.testing.assert_allclose(state["action_terms"]["upper_body"]["targets"], [.14, .24])
    assert state["control_step"] == 17
    assert state["episode_step"] == 3
    with pytest.raises(ValueError, match="exactly one"):
        capture_rl_initial_state(SimpleNamespace(num_envs=2), {})
    # Explicit vector capture must preserve the selected environment's pending
    # command, not average environments or accidentally take environment zero.
    q.resize((2,2),refcheck=False);q[:]=[[.1,.2],[.6,.7]]
    data.joint_pos_target=q+.04;data.joint_vel_target=q*0;data.joint_effort_target=q*10
    term.raw_actions=q*0;term.processed_actions=q+.04;term._targets=q+.04
    manager.action=q*0;manager.prev_action=q*0
    env.num_envs=2;env.episode_length_buf=np.array([3,8])
    env._multi_box_active=np.array([[True,False],[False,True]])
    selected=capture_rl_initial_state(env,{'policy':q},env_index=1)
    np.testing.assert_allclose(selected['scene']['articulation']['robot']['joint_position'],[.6,.7])
    np.testing.assert_allclose(selected['drive_targets']['robot']['joint_position'],[.64,.74])
    assert selected['episode_step']==8
    with pytest.raises(ValueError,match='valid environment index'):
        capture_rl_initial_state(env,{'policy':q},env_index=2)
    # A randomized hinge's actual parameters are part of the numeric seed,
    # not reconstructed from its angle or from the profile's midpoint.
    from kuavo_isaaclab_scene.rl.multi_box.scene.flap_dynamics import firm_flap_dynamics_contract
    from kuavo_isaaclab_scene.rl.multi_box.scene.spawn import physical_asset_names
    from kuavo_isaaclab_scene.recording.rl_transition_recorder import RlTransitionRecorder
    profile=firm_flap_dynamics_contract()
    env.cfg=SimpleNamespace(flap_dynamics=profile)
    properties={name:{key:torch.tensor([[a]*4,[b]*4]) for key,a,b in (
        ('stiffness_nm_per_rad',1.6,2.2),('damping_nm_s_per_rad',.16,.22),
        ('static_friction',.5,.6),('dynamic_friction',.3,.38))} for name in physical_asset_names()}
    env._flap_dynamics_current_parameters=dict(contract=profile,initialized=torch.ones(2,dtype=torch.bool),assets=properties)
    seed=capture_rl_initial_state(env,{'policy':q},env_index=1)
    first=physical_asset_names()[0]
    properties[first]['stiffness_nm_per_rad'][1].fill_(9)
    np.testing.assert_allclose(seed['flap_joint_properties'][first]['stiffness_nm_per_rad'],[2.2]*4)
    recorder=RlTransitionRecorder(tmp_path/'physical_seed.hdf5',{'skill':'grasp'})
    recorder.start_episode(initial_state=seed)
    recorder.append(dict(actor_obs=q[1],critic_obs=q[1],action=q[1]*0,reward=0.,
        next_actor_obs=q[1],next_critic_obs=q[1],terminated=True,truncated=False,
        success=False,unsafe=False,sim_time_s=0.))
    recorder.close()
    with h5py.File(tmp_path/'physical_seed.hdf5') as source:
        initial=source['episodes/episode_000000/initial_state']
        assert initial['flap_joint_properties_schema_version'][()]==1
        np.testing.assert_allclose(initial[f'flap_joint_properties/{first}/stiffness_nm_per_rad'][:],[2.2]*4)
    del env._flap_dynamics_current_parameters
    with pytest.raises(ValueError,match='verified parameters'):capture_rl_initial_state(env,{'policy':q},env_index=1)
