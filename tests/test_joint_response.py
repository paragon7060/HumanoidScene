import json
from types import SimpleNamespace

import numpy as np
import pytest

from kuavo_isaaclab_scene.recording.joint_response import JointResponseProbe, ResponseWriter


def probe(tmp_path):
    data=SimpleNamespace(joint_pos=np.zeros((1,1)),joint_vel=np.zeros((1,1)),
        root_pose_w=np.array([[0,0,0,1,0,0,0.]]),joint_pos_target=np.zeros((1,1)),
        joint_vel_target=np.zeros((1,1)),joint_pos_limits=np.array([[[-2.,2.]]]),
        joint_stiffness=np.ones((1,1)),joint_damping=np.ones((1,1)),
        computed_torque=np.zeros((1,1)),applied_torque=np.zeros((1,1)))
    data.joint_effort_limits=np.ones((1,1))*100
    robot=SimpleNamespace(joint_names=['zarm_l1_joint'],data=data,gravity_compensation_enabled=True,
        dynamics_profile='s63-arm-id',
        _joint_pos_target_sim=np.zeros((1,1)),gravity_compensation_bias=np.zeros((1,1)))
    env=SimpleNamespace(scene={'robot':robot},step_dt=.033,cfg=SimpleNamespace(sim=SimpleNamespace(dt=.011)),
                        episode_length_buf=np.array([0]))
    obj=JointResponseProbe(env,SimpleNamespace(name='s63',urdf_path='model.urdf'),tmp_path/'response.jsonl','test')
    assert json.loads((tmp_path/'response.jsonl').read_text().splitlines()[0])["dynamics_profile"] == "s63-arm-id"
    return obj,env,robot


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_log_separates_logical_goal_from_solver_bias(tmp_path):
    p,env,robot=probe(tmp_path)
    p.begin(True)
    robot.data.joint_pos_target[:]=.1;robot._joint_pos_target_sim[:]=.4
    robot.gravity_compensation_bias[:]=.3;robot.data.joint_pos[:]=.05
    env.episode_length_buf[:]=1
    p.end(tracking_valid=True);p.close()
    result=rows(tmp_path/'response.jsonl');r=next(r for r in result if r['kind']=='step')
    assert r['logical_joint_target_rad']==[.1]
    assert r['solver_joint_target_rad']==[.4]
    assert r['q_before_rad']==[0.]
    assert r['q_after_rad']==[.05]
    assert r['exportable']
    assert result[-1]=={'kind':'closed','ok':True}


def test_autoreset_cannot_be_exported(tmp_path):
    p,env,robot=probe(tmp_path);env.episode_length_buf[:]=5
    p.begin(True);env.episode_length_buf[:]=0;p.end(tracking_valid=True);p.close()
    r=next(r for r in rows(tmp_path/'response.jsonl') if r['kind']=='step')
    assert r['reset_after_step'] and not r['exportable']


def test_tracking_loss_and_idle_segments_not_exportable(tmp_path):
    p,env,robot=probe(tmp_path)
    for i,(active,tracked) in enumerate([(True,False),(False,True),(True,True)]):
        p.begin(active);env.episode_length_buf[:]=i+1;p.end(tracking_valid=tracked)
    p.close();r=[r for r in rows(tmp_path/'response.jsonl') if r['kind']=='step']
    assert [s['exportable'] for s in r]==[False,False,True]
    assert r[0]['segment']!=r[2]['segment']


def test_existing_log_preserved(tmp_path):
    path=tmp_path/'response.jsonl';path.write_text('original')
    with pytest.raises(FileExistsError):ResponseWriter(path,{})
    assert path.read_text()=='original'


def test_nonfinite_sample_is_reported_without_success_footer(tmp_path):
    w=ResponseWriter(tmp_path/'response.jsonl',{})
    w.append({'kind':'step','value':float('nan')})
    with pytest.raises(RuntimeError):w.close()
    assert not any(r.get('kind')=='closed' for r in rows(tmp_path/'response.jsonl'))
