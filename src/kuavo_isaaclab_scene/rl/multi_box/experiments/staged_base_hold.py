"""Frozen-policy probe of neutral-arm base approach then position-held grasp.

This changes the deployed controller and its clock, not the environment's
success or safety rules. It is deliberately ineligible for the old goal-SAC
replay: a learned staged controller will need explicit phase/waypoint context.
"""
import math

import torch

from .kinematic_exploration import target_token


class StagedBaseHoldDiagnostic:
    name = 'neutral_base_approach_then_held_pose_frozen_grasp_diagnostic_v1'
    collection_source = 'staged_base_hold_diagnostic_NOT_matching_goal_SAC_replay'

    def __init__(self, coordinates, templates, raw, *, unmeasured_size_probe=False):
        if type(unmeasured_size_probe) is not bool:
            raise ValueError('Unmeasured size probe must be explicit')
        if raw.shape != (1, 464) or templates.get('format') != self.name:
            raise ValueError('Staged base probe requires one native scene and explicit templates')
        self.coordinates = coordinates
        token, valid = target_token(raw)
        if not bool(valid.all()):
            raise ValueError('Staged base probe needs a perceived target')
        shelf = 'upper' if bool(token[0, 10:12].sum() > .5) else 'middle'
        template = templates['shelves'][shelf]
        regional=templates.get('region_workplaces')
        self.region_workplace=None
        if regional is not None:
            from .region_workplaces import validate_region_workplaces
            validate_region_workplaces(regional)
            if regional['source_shelf_templates']!=templates['shelves'] or token[0,8:12].sum()<.5:
                raise ValueError('Regional target requires the original shelf templates and perceived region')
            region=regional['perceived_region_names_by_id'][int(token[0,8:12].argmax())]
            self.region_workplace=dict(region=region,**regional['regions'][region])
            template=regional['regions'][region]['template']
        if template.get('source_split') != 'train' or not template.get('measured_success'):
            if self.region_workplace is None or not self.region_workplace['unproven_grasp_candidate']:
                raise ValueError('Workplace candidates require successful TRAIN or explicit safe-candidate evidence')
        size=raw.new_tensor(template['box_size_m'])
        self.unmeasured_size_workplace_probe=None
        if size.shape!=(3,):
            raise ValueError('Staged waypoint has not been measured for this box size')
        if not torch.allclose(token[0,5:8],size,atol=1e-5,rtol=0):
            if not unmeasured_size_probe:
                raise ValueError('Staged waypoint has not been measured for this box size')
            from .size_workplace_probe import unmeasured_size_candidate
            template,self.unmeasured_size_workplace_probe=unmeasured_size_candidate(template,token[0])
        offset = raw.new_tensor(template['base_minus_initial_box_xy_rack_m'])
        heading = float(template['base_yaw_rack_rad'])
        if offset.shape != (2,) or not bool(torch.isfinite(offset).all()) or not math.isfinite(heading):
            raise ValueError('Invalid measured base waypoint')
        self.target_xy = coordinates.box_anchor(raw) + offset
        self.target_yaw = heading
        self.phase = 'approach'
        self.stable_steps = 0
        self.manipulation_start = None
        self.position_error = self.yaw_error = None
        self.linear_speed = self.angular_speed = None
        self.shelf = shelf
        self.template = template
        self.templates = regional if regional is not None else templates['shelves']

    def update(self, raw, linear_velocity, angular_velocity, step):
        _, _, xy, yaw, _ = self.coordinates.current(raw)
        self.position_error = float((xy-self.target_xy).norm())
        error = yaw-self.target_yaw
        self.yaw_error = float(torch.atan2(error.sin(), error.cos()).abs()[0])
        self.linear_speed = float(linear_velocity[0, :2].norm())
        self.angular_speed = float(angular_velocity[0].norm())
        stable = (self.position_error < .008 and self.yaw_error < .02
                  and self.linear_speed < .01 and self.angular_speed < .025)
        if self.phase == 'approach':
            self.stable_steps = self.stable_steps+1 if stable else 0
            if self.stable_steps >= 15:
                self.phase = 'held_grasp'
                self.manipulation_start = step

    def manipulation_index(self, step):
        if self.manipulation_start is None:
            raise ValueError('Do not advance the manipulation policy before base settles')
        return step-self.manipulation_start

    def action(self, raw, manipulation_action=None):
        joint, torso, _, _, _ = self.coordinates.current(raw)
        heading = raw.new_full((1, 1), self.target_yaw)
        goal = torch.cat((joint, torso, self.target_xy, heading, raw.new_full((1, 2), -1.)), -1)
        base_command = self.coordinates.decode(raw, goal)[:, :3]
        if self.phase == 'approach':
            # Zero physical joint/torso deltas preserve the neutral reset pose.
            # The grippers stay open during the actual, collision-checked move.
            result = raw.new_zeros(1, 24)
            result[:, 20:22] = -1.
        else:
            if manipulation_action is None or manipulation_action.shape != (1, 24):
                raise ValueError('Held phase needs its frozen manipulation policy')
            result = manipulation_action.clone()
        result[:, :3] = base_command
        return result

    def report(self):
        result=dict(name=self.name, phase=self.phase, shelf=self.shelf,
                    base_target_xy_rack_m=self.target_xy[0].tolist(),
                    base_target_yaw_rack_rad=self.target_yaw,
                    stable_steps=self.stable_steps, manipulation_start=self.manipulation_start,
                    position_error_m=self.position_error, yaw_error_rad=self.yaw_error,
                    linear_speed_mps=self.linear_speed, angular_speed_radps=self.angular_speed,
                    template=self.template, old_goal_replay_eligible=False)
        if hasattr(self,'waypoint_probe'):result['waypoint_probe']=self.waypoint_probe
        if self.region_workplace is not None:result['region_workplace_candidate']=self.region_workplace
        if self.unmeasured_size_workplace_probe is not None:
            result['unmeasured_size_workplace_probe']=self.unmeasured_size_workplace_probe
        return result
