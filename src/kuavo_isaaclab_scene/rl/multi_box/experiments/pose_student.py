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
    actor_dim=439
    action_dim=24
    def __init__(self):
        self.features=ActorFeatures(464,'grasp_target_no_history')
        self.joints=JointOffsetController()
        self.links=torch.tensor(torso_links_from_urdf(resolve_robot_model('s63','leju-twofinger').urdf_path))

    def observations(self,raw,index,time_harmonics=0):
        clock=raw.new_full((len(raw),1),min(index,410)/410)
        inputs=torch.cat((self.features(raw),raw[:,86:350],clock),-1)
        if time_harmonics:
            frequency=torch.arange(1,time_harmonics+1,device=raw.device,dtype=raw.dtype)[None]
            phase=2*torch.pi*clock*frequency
            inputs=torch.cat((inputs,phase.sin(),phase.cos()),-1)
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
        action[:,:2]=(rotation.transpose(-1,-2)@displacement[...,None]).squeeze(-1)[:,:2]*2/.15
        difference=goal[:,21]-heading
        action[:,2]=torch.atan2(difference.sin(),difference.cos())*2/.5
        action[:,20:22]=goal[:,22:24]
        return action.clamp(-1,1)


class PoseStudent:
    artifact_type='pose_goal_student_BC_diagnostic_NOT_SAC'
    def __init__(self,state,device='cpu'):
        if state.get('artifact_type')!=self.artifact_type or state.get('action_coordinates')!=PoseGoalCoordinates.name:
            raise ValueError('This checkpoint is not a matching pose-goal student')
        self.coordinates=PoseGoalCoordinates()
        self.agent=AsymmetricSAC(state['actor_obs_dim'],531,24,SACConfig(**state['config']),device)
        self.agent.restore(state,training=False)
        self.center=state['goal_center'].to(device);self.scale=state['goal_scale'].to(device)
        self.index=0
        self.state=state
        self.anchor=None

    @torch.no_grad()
    def act(self,raw,index):
        goal=self.agent.act(self.coordinates.observations(raw,index,self.state.get('time_harmonics',0)),deterministic=True)
        goal=self.center+self.scale*goal
        if self.state.get('initial_box_relative_goals',False):
            if self.anchor is None:self.anchor=self.coordinates.box_anchor(raw)
            goal[:,19:21]+=self.anchor
        return self.coordinates.decode(raw,goal)
