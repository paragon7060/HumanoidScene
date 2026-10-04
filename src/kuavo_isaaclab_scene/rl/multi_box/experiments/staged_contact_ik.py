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


def executed_velocity_feedforward(current, command, proposed_velocity, dt):
    """Do not drive faster/farther than the encoded, bounded position step."""
    encoded = (command-current)/dt
    same_direction = encoded*proposed_velocity > 0
    return torch.where(same_direction,
        encoded.sign()*torch.minimum(encoded.abs(),proposed_velocity.abs()),0.)


def tracked_flap_observation(raw, assignment):
    """Keep the teacher's hand/flap identity while using live panel poses."""
    if assignment is None:
        return raw
    from .guided_exploration import ASSIGNMENT_START
    result=raw.clone()
    result[:,ASSIGNMENT_START:ASSIGNMENT_START+2]=assignment
    return result


class StagedContactIKDiagnostic:
    name='native_calibrated_local_contact_IK_teacher_NOT_SAC_v1'

    def __init__(self,env,measured,audit,*,handoff_mode='near-contact',orientation_mode='full',
                 velocity_feedforward=False,lock_assignment=False):
        from ....teleop.urdf_arm_ik import UrdfArm
        from ....teleop.teleop_servo import RESPONSIVE
        from ....robots.robot_model import resolve_robot_model
        from ....robots.end_effector import closed_closing_axes
        self.env=env
        self.guide=KinematicGraspExplorer(env,measured,grasp_goal='demo',
            lift_distance_m=.025,orientation_mode=orientation_mode)
        for side,solver in zip(('left','right'),self.guide.solvers):
            solver.configure_urdf(UrdfArm(resolve_robot_model().urdf_path,side))
            solver.response=RESPONSIVE
        self.audit=audit
        contact_handoff_phase(torch.ones(1,2),handoff_mode)
        self.handoff_mode=handoff_mode
        self.orientation_mode=orientation_mode
        self.velocity_feedforward=velocity_feedforward
        self.lock_assignment=lock_assignment
        self.tracked_assignment=None
        self.servo_telemetry=[]
        axes=closed_closing_axes()
        self.axes=torch.tensor([axes[side] for side in ('left','right')],device=env.device)
        self.handoff_step=None
        self.position_error=self.axis_error=None
        self.front_error=self.full_orientation_error=None

    def geometry(self,raw):
        tcp,centers,stage,outward=entry_geometry(raw,self.guide.front_y)
        panel=assigned_flap_rotations(raw)
        goals,front,_=retarget_grasp_goal(centers,stage,outward,panel,self.guide.goal_offset,'demo')
        position_error=(goals-tcp[...,:3]).norm(dim=-1)
        self.front_error=(front-tcp[...,:3]).norm(dim=-1)[0].tolist()
        from ..demo_replay import _rotation_matrix
        current=_rotation_matrix(tcp[...,3:])
        target=panel@self.guide.relative_rotation[None]
        relative=current.transpose(-1,-2)@target
        self.full_orientation_error=torch.acos(((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1))[0].tolist()
        axes=self.axes.to(raw)
        a=(current@axes[None,...,None]).squeeze(-1)
        b=(target@axes[None,...,None]).squeeze(-1)
        axis_error=torch.acos((a*b).sum(-1).abs().clamp(0,1))
        return position_error,axis_error

    def act(self,raw,neural_action,step):
        tracked=tracked_flap_observation(raw,self.tracked_assignment)
        error,angle=self.geometry(tracked)
        self.position_error,self.axis_error=error[0].tolist(),angle[0].tolist()
        if self.handoff_step is None:
            phase=contact_handoff_phase(error,self.handoff_mode)
            if phase is not None:
                self.handoff_step=step
                self.guide.phase[:]=phase
                if self.lock_assignment:
                    from .guided_exploration import ASSIGNMENT_START
                    self.tracked_assignment=raw[:,ASSIGNMENT_START:ASSIGNMENT_START+2].clone()
            else:
                return neural_action
        for solver in self.guide.solvers:
            # Keep the actual near-contact posture; do not pull toward the
            # neutral pose while correcting the moving flap's geometry.
            solver._urdf_rest=solver._numpy(
                self.env.scene['robot'].data.joint_pos[0,solver._joint_ids])[solver._urdf_order].copy()
        proposed=self.guide.act(tracked)
        result=neural_action.clone()
        for columns in self.guide.columns:
            indices=[self.guide.slices['upper_body'].start+c for c in columns]
            result[:,indices]=proposed[:,indices]
        self.servo_telemetry=[]
        robot=self.env.scene['robot']
        for columns,solver in zip(self.guide.columns,self.guide.solvers):
            indices=[self.guide.slices['upper_body'].start+c for c in columns]
            current=robot.data.joint_pos[:,solver._joint_ids]
            command=self.guide.upper.processed_actions[:,columns]+result[:,indices]*self.guide.upper._scale[:,columns]
            velocity=executed_velocity_feedforward(current,command,solver._joint_velocity,self.env.step_dt)
            if self.velocity_feedforward:
                # The ordinary goal-SAC controller is unchanged. This opt-in
                # teacher matches teleop's paired position/velocity targets.
                robot.set_joint_velocity_target(velocity,joint_ids=solver._joint_ids)
            self.servo_telemetry.append(dict(
                measured_velocity_rad_s=robot.data.joint_vel[:,solver._joint_ids][0].tolist(),
                IK_velocity_rad_s=solver._joint_velocity[0].tolist(),
                paired_velocity_rad_s=velocity[0].tolist(),
                encoded_position_error_rad=(command-current)[0].tolist(),
                previous_drive_error_rad=(self.guide.upper.processed_actions[:,columns]-current)[0].tolist()))
        pinching=self.env._multi_box_privileged_grasp_step.pinch.hand_pinching
        closing=coordinated_close(error,angle,pinching,self.guide.phase==2)
        result[:,20:22]=torch.where(closing,1.,-1.)
        return result

    def report(self):
        return dict(name=self.name,handoff_step=self.handoff_step,
            handoff_mode=self.handoff_mode,
            orientation_mode=self.orientation_mode,front_goal_error_m=self.front_error,
            velocity_feedforward=self.velocity_feedforward,
            hand_flap_identity_locked_at_handoff=self.lock_assignment,
            tracked_assignment=self.tracked_assignment[0].tolist() if self.tracked_assignment is not None else None,
            controller='paired_position_velocity_teacher' if self.velocity_feedforward else 'position_only_teacher',
            servo_telemetry=self.servo_telemetry,
            full_orientation_error_rad=self.full_orientation_error,
            ik_status=[solver.ik_status for solver in self.guide.solvers],
            goal_error_m=self.position_error,closing_axis_error_rad=self.axis_error,
            phase=int(self.guide.phase[0]),source_audit=self.audit,
            privileged_pinch_used_for_teacher_lift=True,standalone_SAC=False)
