"""Native-success-calibrated local contact teacher, for physical diagnosis only.

The neural approach stays frozen. Only the final two-hand contact is replaced
by the existing bounded URDF IK servo and measured pinch-confirmed lift.
This is a teacher with privileged contact confirmation, not a standalone SAC.
"""
import torch

from .kinematic_exploration import (KinematicGraspExplorer, entry_geometry,
    retarget_grasp_goal, assigned_flap_rotations)


def coordinated_close(position_error, axis_error, hand_pinching, lifting):
    ready=((position_error < .015) & (axis_error < .15)).all(-1)
    return hand_pinching | ready[:,None] | lifting[:,None]


def contact_handoff_phase(position_error, mode):
    """Starting from a held base still requires a front-stage approach."""
    if mode == 'after-base-hold':
        return 0
    if mode == 'near-contact':
        return 1 if bool((position_error < .10).all()) else None
    raise ValueError('Unsupported staged contact handoff mode')


class StagedContactIKDiagnostic:
    name='native_calibrated_local_contact_IK_teacher_NOT_SAC_v1'

    def __init__(self,env,measured,audit,*,handoff_mode='near-contact'):
        from ....teleop.urdf_arm_ik import UrdfArm
        from ....teleop.teleop_servo import RESPONSIVE
        from ....robots.robot_model import resolve_robot_model
        from ....robots.end_effector import closed_closing_axes
        self.env=env
        self.guide=KinematicGraspExplorer(env,measured,grasp_goal='demo',
            lift_distance_m=.025,orientation_mode='full')
        for side,solver in zip(('left','right'),self.guide.solvers):
            solver.configure_urdf(UrdfArm(resolve_robot_model().urdf_path,side))
            solver.response=RESPONSIVE
        self.audit=audit
        contact_handoff_phase(torch.ones(1,2),handoff_mode)
        self.handoff_mode=handoff_mode
        axes=closed_closing_axes()
        self.axes=torch.tensor([axes[side] for side in ('left','right')],device=env.device)
        self.handoff_step=None
        self.position_error=self.axis_error=None

    def geometry(self,raw):
        tcp,centers,stage,outward=entry_geometry(raw,self.guide.front_y)
        panel=assigned_flap_rotations(raw)
        goals,_,_=retarget_grasp_goal(centers,stage,outward,panel,self.guide.goal_offset,'demo')
        position_error=(goals-tcp[...,:3]).norm(dim=-1)
        from ..demo_replay import _rotation_matrix
        current=_rotation_matrix(tcp[...,3:])
        target=panel@self.guide.relative_rotation[None]
        axes=self.axes.to(raw)
        a=(current@axes[None,...,None]).squeeze(-1)
        b=(target@axes[None,...,None]).squeeze(-1)
        axis_error=torch.acos((a*b).sum(-1).abs().clamp(0,1))
        return position_error,axis_error

    def act(self,raw,neural_action,step):
        error,angle=self.geometry(raw)
        self.position_error,self.axis_error=error[0].tolist(),angle[0].tolist()
        if self.handoff_step is None:
            phase=contact_handoff_phase(error,self.handoff_mode)
            if phase is not None:
                self.handoff_step=step
                self.guide.phase[:]=phase
            else:
                return neural_action
        for solver in self.guide.solvers:
            # Keep the actual near-contact posture; do not pull toward the
            # neutral pose while correcting the moving flap's geometry.
            solver._urdf_rest=solver._numpy(
                self.env.scene['robot'].data.joint_pos[0,solver._joint_ids])[solver._urdf_order].copy()
        proposed=self.guide.act(raw)
        result=neural_action.clone()
        for columns in self.guide.columns:
            indices=[self.guide.slices['upper_body'].start+c for c in columns]
            result[:,indices]=proposed[:,indices]
        pinching=self.env._multi_box_privileged_grasp_step.pinch.hand_pinching
        closing=coordinated_close(error,angle,pinching,self.guide.phase==2)
        result[:,20:22]=torch.where(closing,1.,-1.)
        return result

    def report(self):
        return dict(name=self.name,handoff_step=self.handoff_step,
            handoff_mode=self.handoff_mode,
            goal_error_m=self.position_error,closing_axis_error_rad=self.axis_error,
            phase=int(self.guide.phase[0]),source_audit=self.audit,
            privileged_pinch_used_for_teacher_lift=True,standalone_SAC=False)
