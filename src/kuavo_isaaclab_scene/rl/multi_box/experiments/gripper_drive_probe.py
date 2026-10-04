"""Frozen comparison of only the four claw motor drives, never Q training."""
import torch

DRIVERS=tuple(f'{side}_{jaw}_bar_1_joint' for side in 'lr' for jaw in 'fb')
SOFT=dict(stiffness=100.,damping=5.,effort_limit=2.)
PROPERTIES=dict(stiffness=('joint_stiffness','write_joint_stiffness_to_sim','get_dof_stiffnesses'),
    damping=('joint_damping','write_joint_damping_to_sim','get_dof_dampings'),
    effort_limit=('joint_effort_limits','write_joint_effort_limit_to_sim','get_dof_max_forces'))


def validate_gripper_drive_probe(waves,*,enabled,training):
    if enabled and training:raise ValueError('Gripper drive comparison is frozen-only, never TRAIN')
    for wave in waves:
        name=wave.get('gripper_drive_probe')
        if not enabled and name is not None:raise ValueError('Gripper drive override requires explicit diagnostic flag')
        if enabled and (wave.get('split')=='train' or name not in ('original','soft_2nm')):
            raise ValueError('Each frozen wave needs an original or soft_2nm gripper-drive candidate')


class GripperDriveProbe:
    def __init__(self,robot):
        self.robot=robot
        if any(robot.joint_names.count(name)!=1 for name in DRIVERS):
            raise ValueError('Require exactly the four authored bar_1 motor joints')
        self.ids=[robot.joint_names.index(name) for name in DRIVERS]
        self.original={key:getattr(robot.data,prop)[:,self.ids].clone() for key,(prop,_,_) in PROPERTIES.items()}
        self.models=[]
        covered=[]
        for actuator in robot.actuators.values():
            indices=actuator.joint_indices
            indices=list(range(len(robot.joint_names)))[indices] if isinstance(indices,slice) else list(indices)
            slots=[i for i,j in enumerate(indices) if j in self.ids]
            if not slots:continue
            covered.extend(int(indices[i]) for i in slots)
            original={key:getattr(actuator,key)[:,slots].clone() for key in PROPERTIES}
            self.models.append((actuator,slots,original))
        if sorted(covered)!=sorted(self.ids):raise ValueError('Every claw motor must have one matching actuator model')

    def apply(self,name):
        if name not in ('original','soft_2nm'):raise ValueError('Unknown frozen claw drive candidate')
        for key,(prop,writer,_) in PROPERTIES.items():
            values=self.original[key] if name=='original' else torch.full_like(self.original[key],SOFT[key])
            getattr(self.robot,writer)(values,joint_ids=self.ids)
        # Isaac setters update PhysX/data buffers, but not actuator-model tensors.
        for actuator,slots,original in self.models:
            for key in PROPERTIES:
                getattr(actuator,key)[:,slots]=original[key] if name=='original' else SOFT[key]
        actual={}
        for key,(prop,_,getter) in PROPERTIES.items():
            values=getattr(self.robot.root_physx_view,getter)()[:,self.ids]
            expected=self.original[key].cpu() if name=='original' else torch.full_like(values,SOFT[key])
            if not torch.allclose(values,expected,atol=1e-6,rtol=0):
                raise ValueError('Requested claw drive does not match initialized PhysX: '+key)
            actual[key]=dict(min=float(values.min()),max=float(values.max()))
        return dict(name=name,driver_joint_names=list(DRIVERS),initialized_PhysX=actual,
            frozen_only=True,Q_import_eligible=False,passive_and_body_drives_unchanged=True,
            force_feedforward_geometry_randomization_and_safety_unchanged=True)
