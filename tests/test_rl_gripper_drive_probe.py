from types import SimpleNamespace
import pytest
import torch
from kuavo_isaaclab_scene.rl.multi_box.experiments.gripper_drive_probe import (
    DRIVERS,PROPERTIES,SOFT,GripperDriveProbe,validate_gripper_drive_probe,
)


def robot_fixture():
    names=['arm_joint',DRIVERS[0],'l_f_bar_3_joint',*DRIVERS[1:]]
    data=SimpleNamespace();model=SimpleNamespace(joint_indices=slice(None))
    robot=SimpleNamespace(joint_names=names,data=data,actuators={'all':model},root_physx_view=SimpleNamespace())
    for key,(prop,writer,getter) in PROPERTIES.items():
        values=torch.arange(18,dtype=torch.float32).reshape(3,6)+1
        setattr(data,prop,values.clone());setattr(model,key,values.clone())
        def write(values,*,joint_ids,property_name=prop):getattr(data,property_name)[:,joint_ids]=values
        setattr(robot,writer,write)
        setattr(robot.root_physx_view,getter,lambda property_name=prop:getattr(data,property_name).clone())
    return robot


def test_soft_motor_probe_changes_only_four_drivers_in_physics_and_actuator_models_and_restores():
    robot=robot_fixture();original={k:getattr(robot.data,p).clone() for k,(p,_,_) in PROPERTIES.items()}
    probe=GripperDriveProbe(robot);audit=probe.apply('soft_2nm')
    assert not audit['Q_import_eligible'] and audit['frozen_only']
    for key,(prop,_,_) in PROPERTIES.items():
        actual=getattr(robot.data,prop);model=getattr(robot.actuators['all'],key)
        assert (actual[:,probe.ids]==SOFT[key]).all() and torch.equal(actual,model)
        assert torch.equal(actual[:,[0,2]],original[key][:,[0,2]])
    probe.apply('original')
    for key,(prop,_,_) in PROPERTIES.items():
        assert torch.equal(getattr(robot.data,prop),original[key])
        assert torch.equal(getattr(robot.actuators['all'],key),original[key])


def test_probe_rejects_missing_or_duplicate_motor_and_failed_physics_application():
    robot=robot_fixture();robot.joint_names[-1]='unexpected'
    with pytest.raises(ValueError,match='exactly'):GripperDriveProbe(robot)
    robot=robot_fixture();probe=GripperDriveProbe(robot)
    robot.root_physx_view.get_dof_max_forces=lambda:torch.full((3,6),100.)
    with pytest.raises(ValueError,match='PhysX'):probe.apply('soft_2nm')


@pytest.mark.parametrize('enabled,training,name,split',[
    (True,True,'original','validation'),(True,False,'soft_2nm','train'),
    (True,False,'unknown','validation'),(False,False,'soft_2nm','validation')])
def test_drive_override_requires_frozen_explicit_known_candidate(enabled,training,name,split):
    with pytest.raises(ValueError):validate_gripper_drive_probe([dict(split=split,gripper_drive_probe=name)],enabled=enabled,training=training)


def test_default_waves_and_explicit_frozen_original_soft_original_are_allowed():
    validate_gripper_drive_probe([dict(split='train')],enabled=False,training=True)
    validate_gripper_drive_probe([dict(split='validation',gripper_drive_probe=n)
        for n in ['original','soft_2nm','original']],enabled=True,training=False)
