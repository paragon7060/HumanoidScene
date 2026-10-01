"""Diagnostic desired-joint-goal coordinates over the existing delta servo.

Physical commands retain the current joint-delta limits and all24 channels.
Only the experimental actor's joint outputs change coordinates. The analytic
pending-target subtraction prevents hiding the servo integrator in a learned
delta. This helper does not alter the environment or create Q transitions.
"""

import math
import torch


class JointOffsetController:
    def __init__(self, offset_rad=.2):
        if not math.isfinite(offset_rad) or not 0 < offset_rad <= .5:
            raise ValueError('Joint goal offset must be in(0,.5] rad')
        self.offset_rad = offset_rad
        self.action_columns = list(range(3, 18)) + [22, 23]
        self.joint_columns = list(range(3, 18)) + [18, 19]
        self.error_columns = [416 + column for column in self.joint_columns]
        self.scales = [.01] + [.02] * 14 + [.01, .01]

    def _validate(self, observations, action):
        if observations.ndim != 2 or observations.shape[1] != 464 \
                or action.shape != (len(observations), 24) \
                or not bool((observations[:, 439] > .5).all()):
            raise ValueError('Joint-goal diagnostic requires current464-D/24-D controller telemetry')

    def actor_input(self, observations):
        result = observations.clone()
        # The pose predictor sees measured motion. Actual pending targets are
        # used only in the analytic servo conversion, never replaced in Q.
        result[:, self.error_columns] = 0
        return result

    def label_coordinates(self, observations, physical_action):
        self._validate(observations, physical_action)
        scale = physical_action.new_tensor(self.scales)
        result = physical_action.clone()
        result[:, self.action_columns] = (
            observations[:, self.error_columns] + scale * physical_action[:, self.action_columns]
        ) / self.offset_rad
        if bool((result.abs() > 1.00001).any()):
            raise ValueError('Desired label lies outside this joint-offset range')
        return result

    def physical_commands(self, observations, goal_action):
        self._validate(observations, goal_action)
        scale = goal_action.new_tensor(self.scales)
        result = goal_action.clone()
        result[:, self.action_columns] = (
            self.offset_rad * goal_action[:, self.action_columns] - observations[:, self.error_columns]
        ) / scale
        return result.clamp(-1, 1)

    def label_weights(self, action):
        weights = torch.ones(24, device=action.device, dtype=action.dtype)
        weights[self.action_columns] = self.offset_rad / action.new_tensor(self.scales)
        return weights
