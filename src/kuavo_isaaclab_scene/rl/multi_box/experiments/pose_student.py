"""Learned absolute controller goals without a live demonstration path.

This is a separate action-coordinate contract. Its supervised warm start is not
SAC success, and its goal labels must never enter the ordinary delta-action Q.
Episode elapsed time is an input; recorded commands/goals are not runtime inputs.
"""
import torch

from ...algorithms.asymmetric_sac import ActorFeatures,AsymmetricSAC
from ...algorithms.sac import SACConfig
from ..demo_replay import _rotation_matrix
from ..geometry.upright_torso import planar_position,torso_links_from_urdf
from ....robots.robot_model import resolve_robot_model
from .joint_offset import JointOffsetController


class PoseGoalCoordinates:
    name='s63_upright_absolute_joint_xz_rack_pose_grippers_v1'
    projected_base_name='s63_upright_absolute_joint_xz_rack_pose_grippers_projected_base_v2'
    actor_dim=439
    action_dim=24
    def __init__(self, *, exact_projected_base=False):
        self.exact_projected_base=exact_projected_base
        if exact_projected_base:self.name=self.projected_base_name
        self.features=ActorFeatures(464,'grasp_target_no_history')
        self.joints=JointOffsetController()
        self.links=torch.tensor(torso_links_from_urdf(resolve_robot_model('s63','leju-twofinger').urdf_path))

    def observations(self,raw,index,time_harmonics=0,clock_horizon=410,*,
                     condition_on_shelf=False,clock_limit=None):
        if not isinstance(clock_horizon,int) or not 1<=clock_horizon<=900:
            raise ValueError('Pose clock horizon must be within1..900 control steps')
        if clock_limit is not None and (not isinstance(clock_limit,int) or not 0<=clock_limit<=clock_horizon):
            raise ValueError('Actor clock limit must be within0..clock horizon')
        limit=clock_horizon if clock_limit is None else clock_limit
        clock=raw.new_full((len(raw),1),min(index,limit)/clock_horizon)
        inputs=torch.cat((self.features(raw),raw[:,86:350],clock),-1)
        if time_harmonics:
            frequency=torch.arange(1,time_harmonics+1,device=raw.device,dtype=raw.dtype)[None]
            phase=2*torch.pi*clock*frequency
            inputs=torch.cat((inputs,phase.sin(),phase.cos()),-1)
        if condition_on_shelf:
            from .kinematic_exploration import target_token
            token,valid=target_token(raw)
            if not bool(valid.all()):raise ValueError('Shelf conditioning requires a perceived selected box')
            # Region order: lowerR/L, upperR/L. Side does not select a different
            # intrinsic hand program; the measured box anchor translates base XY.
            inputs=torch.cat((inputs,token[:,10:12].sum(-1,keepdim=True)),-1)
        return inputs

    def current(self,raw):
        if raw.shape[-1]!=464 or not bool((raw[:,439]>.5).all()):
            raise ValueError('Pose student requires live464-D controller telemetry')
        targets=raw[:,:20]+raw[:,416:436]
        joint=targets[:,self.joints.joint_columns]
        torso=planar_position(targets[:,:2],self.links.to(raw))
        rotation=_rotation_matrix(raw[:,71:77]).transpose(-1,-2)
        position=-(rotation@raw[:,68:71,None]).squeeze(-1)
        heading=torch.atan2(rotation[:,1,0],rotation[:,0,0])
        return joint,torso,position[:,:2],heading,rotation

    def box_anchor(self,raw):
        from .kinematic_exploration import target_token
        token,valid=target_token(raw)
        if not bool(valid.all()):raise ValueError('Student requires a perceived selected target')
        return (_rotation_matrix(raw[:,71:77]).transpose(-1,-2)@
                (token[:,12:15]-raw[:,68:71])[...,None]).squeeze(-1)[:,:2]

    def encode_physical(self,raw,physical):
        """Exact inverse labels of commands that really executed; no Q rows."""
        if physical.shape!=(len(raw),24) or not torch.isfinite(physical).all():
            raise ValueError('Expected finite24-D executed physical commands')
        joint,torso,position,heading,rotation=self.current(raw)
        goal_joint=joint+physical[:,self.joints.action_columns]*raw.new_tensor(self.joints.scales)
        goal_torso=torso+physical[:,18:20]*(.1/30)
        displacement=torch.cat((physical[:,:2]*.15/2,raw.new_zeros(len(raw),1)),-1)
        goal_position=position+(rotation@displacement[...,None]).squeeze(-1)[:,:2]
        goal_heading=heading+physical[:,2]*.5/2
        return torch.cat((goal_joint,goal_torso,goal_position,goal_heading[:,None],physical[:,20:22]),-1)

    def decode(self,raw,goal):
        if goal.shape!=(len(raw),24) or not torch.isfinite(goal).all():
            raise ValueError('Expected finite24-D absolute goals')
        joint,torso,position,heading,rotation=self.current(raw)
        action=raw.new_zeros(len(raw),24)
        action[:,self.joints.action_columns]=(goal[:,:17]-joint)/raw.new_tensor(self.joints.scales)
        action[:,18:20]=(goal[:,17:19]-torso)/(.1/30)
        displacement=torch.cat((goal[:,19:21]-position,raw.new_zeros(len(raw),1)),-1)
        if self.exact_projected_base:
            # Encode keeps rack XY of a body-plane command. With actual base
            # roll/pitch, projecting loses Z: R.T is not the inverse of R[:2,:2].
            # Solve that measured2x2 map; do not relabel old actions to fit it.
            m=rotation[:,:2,:2]
            determinant=m[:,0,0]*m[:,1,1]-m[:,0,1]*m[:,1,0]
            if not torch.isfinite(determinant).all() or (determinant.abs()<.1).any():
                raise ValueError('Base-plane projection is singular or near vertical')
            dx,dy=displacement[:,0],displacement[:,1]
            action[:,0]=(m[:,1,1]*dx-m[:,0,1]*dy)/determinant*2/.15
            action[:,1]=(m[:,0,0]*dy-m[:,1,0]*dx)/determinant*2/.15
        else:
            action[:,:2]=(rotation.transpose(-1,-2)@displacement[...,None]).squeeze(-1)[:,:2]*2/.15
        difference=goal[:,21]-heading
        action[:,2]=torch.atan2(difference.sin(),difference.cos())*2/.5
        action[:,20:22]=goal[:,22:24]
        return action.clamp(-1,1)


class PoseStudent:
    artifact_type='pose_goal_student_BC_diagnostic_NOT_SAC'
    def __init__(self,state,device='cpu'):
        if state.get('artifact_type')!=self.artifact_type or state.get('action_coordinates') not in {
                PoseGoalCoordinates.name,PoseGoalCoordinates.projected_base_name}:
            raise ValueError('This checkpoint is not a matching pose-goal student')
        self.coordinates=PoseGoalCoordinates(exact_projected_base=
            state['action_coordinates']==PoseGoalCoordinates.projected_base_name)
        self.agent=AsymmetricSAC(state['actor_obs_dim'],531,24,SACConfig(**state['config']),device)
        self.agent.restore(state,training=False)
        self.center=state['goal_center'].to(device);self.scale=state['goal_scale'].to(device)
        self.index=0
        self.state=state
        self.anchor=None

    def validate_physical_contract(self, contract):
        """Old unnamed priors are eligible only for the original travel MDP."""
        from .executed_replay import PHYSICAL_KEYS
        recorded = self.state.get('physical_contract')
        if recorded is None:
            if contract.get('action_contract') != 's63_upright_torso_xz_fixed_pitch_v1':
                raise ValueError('Legacy pose prior lacks a matching physical travel contract')
            return
        for key in (*PHYSICAL_KEYS, 'flap_pose_source'):
            if recorded.get(key) != contract.get(key):
                raise ValueError(f'Pose prior physical contract differs: {key}')

    @torch.no_grad()
    def act(self,raw,index):
        goal=self.agent.act(self.coordinates.observations(raw,index,self.state.get('time_harmonics',0),
            self.state.get('clock_horizon',410),
            condition_on_shelf=self.state.get('shelf_conditioned_clock_fit',False),
            clock_limit=self.state.get('actor_clock_limit')),deterministic=True)
        goal=self.center+self.scale*goal
        if self.state.get('initial_box_relative_goals',False):
            if self.anchor is None:self.anchor=self.coordinates.box_anchor(raw)
            goal[:,19:21]+=self.anchor
        return self.coordinates.decode(raw,goal)
