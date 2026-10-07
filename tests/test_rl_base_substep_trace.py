"""A diagnostic must preserve commands/state and expose cached/raw differences."""
import json
from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.debug.base_substep_trace import (
    BaseSubstepTrace, validate_base_substep_trace,
)


def workplace():
    return dict(name='CPU_PhysX_frozen_TRAIN_workplace_search_v1',
                original_candidate_requests=128, training=False)


@pytest.mark.parametrize('values', [dict(training=True), dict(workplace=None),
    dict(num_envs=127), dict(indices=[0, 0]), dict(indices=[128]),
    dict(indices=[]), dict(indices=list(range(9)))])
def test_trace_cannot_become_training_or_reduced_scope(values):
    args=dict(indices=[0], workplace=workplace(), training=False, num_envs=128)
    args.update(values)
    with pytest.raises(ValueError):
        validate_base_substep_trace(**args)
    assert validate_base_substep_trace(None,workplace=None,training=True,num_envs=1) is None


def scene():
    q=torch.tensor([2**-.5,0.,0.,2**-.5])
    pose=torch.cat((torch.zeros(3),q)).repeat(128,1)
    cached_velocity=torch.zeros(128,6);cached_velocity[:,4]=.8
    raw_velocity=cached_velocity.clone();raw_velocity[:,4]=.82
    raw_pose=pose[:,[0,1,2,4,5,6,3]].clone()
    data=SimpleNamespace(root_pose_w=pose,root_com_pose_w=pose.clone(),
        root_lin_vel_w=cached_velocity[:,:3],root_ang_vel_w=cached_velocity[:,3:],
        _sim_timestamp=1.)
    asset=SimpleNamespace(data=data,root_physx_view=SimpleNamespace(
        get_root_transforms=lambda:raw_pose,get_root_velocities=lambda:raw_velocity))
    drive=SimpleNamespace(_asset=asset,_force_b=torch.zeros(128,1,3),
        _torque_b=torch.zeros(128,1,3),_total_mass=torch.ones(128),
        _target_xy=torch.zeros(128,2),_target_yaw=torch.zeros(128),
        _target_height=torch.zeros(128),_level_quat=q.repeat(128,1),
        _inertia=lambda:torch.ones(128,3))
    drive._torque_b[:,0,0]=2.
    calls=[];result=object()
    def original(command):calls.append(command);return result
    drive.apply=original
    env=SimpleNamespace(physics_dt=1/120,action_manager=SimpleNamespace(
        get_term=lambda name:SimpleNamespace(_drive=drive)))
    return env,drive,calls,result,raw_velocity


def test_original_apply_state_and_return_survive_with_exact_substep_record(tmp_path):
    env,drive,calls,result,raw=scene();original=drive.apply
    initial_pose=drive._asset.data.root_pose_w.clone();initial_raw=raw.clone()
    contract=validate_base_substep_trace([0,1],workplace=workplace(),training=False,num_envs=128)
    trace=BaseSubstepTrace(env,tmp_path,contract);command=torch.zeros(128,3)
    assert drive.apply(command) is result  # no controlled context: no log row
    active=torch.zeros(128,dtype=torch.bool);active[0]=True
    trace.prepare(0,70,active,[SimpleNamespace(phase='held_grasp')]*128,torch.zeros(128,24))
    assert drive.apply(command) is result
    trace.finish_step();assert drive.apply(command) is result
    trace.close();trace.close()
    assert drive.apply is original and len(calls)==3 and all(x is command for x in calls)
    assert torch.equal(initial_pose,drive._asset.data.root_pose_w) and torch.equal(initial_raw,raw)
    rows=[json.loads(x) for x in (tmp_path/'base_substep_trace.log').read_text().splitlines()]
    assert [r['kind'] for r in rows]==['contract','substep','closed']
    r=rows[1];assert r['environment']==0 and r['control_step']==70 and r['phase']=='held_grasp'
    assert r['root_velocity_world'][4]==pytest.approx(.8)
    assert r['raw_PhysX_root_COM_velocity_world'][4]==pytest.approx(.82)
    assert r['root_pose_world_wxyz']==r['raw_PhysX_root_pose_world_wxyz']
    assert r['commanded_torque_world_nm']==pytest.approx([0.,2.,0.],abs=1e-6)
    assert r['all_fields_finite'] and rows[-1]['original_drive_apply_restored']


def test_nonfinite_raw_read_is_marked_without_sanitizing_simulator(tmp_path):
    env,drive,_,_,raw=scene();raw[0,4]=float('nan')
    contract=validate_base_substep_trace([0],workplace=workplace(),training=False,num_envs=128)
    trace=BaseSubstepTrace(env,tmp_path,contract)
    trace.prepare(0,1,torch.ones(128,dtype=torch.bool),
                  [SimpleNamespace(phase='held_grasp')]*128,torch.zeros(128,24))
    drive.apply(torch.zeros(128,3));trace.close()
    r=json.loads((tmp_path/'base_substep_trace.log').read_text().splitlines()[1])
    assert not r['all_fields_finite'] and r['raw_PhysX_root_COM_velocity_world'][4] is None
    assert torch.isnan(raw[0,4])
