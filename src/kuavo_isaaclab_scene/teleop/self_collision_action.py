"""Last action term: lightweight endpoint check once per control tick."""
import numpy as np
import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .self_collision import ClearanceViolation, CollisionStop, RobotCollisionModel, SelfCollisionFilter
from .urdf_arm_ik import quat_matrix, rotation_error


class SelfCollisionAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.model = RobotCollisionModel(cfg.urdf_path, cfg.allowed_pairs or None)
        self.guard = SelfCollisionFilter(self.model, cfg.clearance)
        self._joint_ids = []
        for name in self.model.names:
            ids, _ = self._asset.find_joints(name)
            if len(ids) != 1:
                raise ValueError(f"Self-collision model joint missing/ambiguous in USD: {name}")
            self._joint_ids.append(ids[0])
        if set(self._asset.joint_names) != set(self.model.names):
            raise ValueError("USD contains joints absent from the full self-collision URDF")
        self._raw_actions = torch.zeros((self.num_envs, 0), device=self.device)
        if self.num_envs != 1:
            raise ValueError("Quest self-collision guard currently requires one environment")
        self.validated = False
        self.last_safe_target = None
        self._physics_steps = 0
        self.step_modified = False
        self.step_minimum_distance = float("inf")
        self._cached_target = None
        self._cached_velocity = None
        self._recording = False
        self._collision_event = None
        self.step_collision = False
        self._checked_this_tick = False

    @property
    def action_dim(self):
        return 0

    @property
    def raw_actions(self):
        return self._raw_actions

    @property
    def processed_actions(self):
        return self._raw_actions

    def process_actions(self, actions):
        self.step_modified = False
        self.step_minimum_distance = float("inf")
        self.step_collision = False
        self._cached_target = None
        self._checked_this_tick = False

    def set_recording(self, recording: bool) -> None:
        """Select enforce-and-stop (recording) or monitor-only behavior."""
        recording = bool(recording)
        if recording != self._recording:
            self._cached_target = None
            self._cached_velocity = None
        self._recording = recording

    def consume_collision_event(self):
        event = self._collision_event
        self._collision_event = None
        return event

    def _report_collision(self, message: str) -> None:
        self.step_collision = True
        self._collision_event = {"message": message, "recording": self._recording}

    def _hold_current(self) -> None:
        q = self._asset.data.joint_pos[:, self._joint_ids].clone()
        self._asset.set_joint_position_target(q, self._joint_ids)
        self._asset.set_joint_velocity_target(torch.zeros_like(q), self._joint_ids)

    def _array(self, tensor):
        return tensor.detach().cpu().numpy().astype(float, copy=True)

    def validate_live(self):
        if self._env.action_manager.active_terms[-1] != "self_collision":
            raise ValueError("Self-collision must be the LAST action term; later terms could bypass it")
        q = self._array(self._asset.data.joint_pos[0, self._joint_ids])
        self.model.forward(q)
        names = self._asset.body_names
        anchor = next((n for n in names if n in self.model.frames), None)
        if anchor is None:
            raise ValueError("No common URDF/USD collision model frame")
        i = names.index(anchor)
        actual = np.eye(4)
        actual[:3, :3] = quat_matrix(self._array(self._asset.data.body_quat_w[0, i]))
        actual[:3, 3] = self._array(self._asset.data.body_pos_w[0, i])
        alignment = actual @ np.linalg.inv(self.model.frames[anchor])
        for i, name in enumerate(names):
            if name not in self.model.frames:
                raise ValueError(f"USD body not covered by self-collision URDF: {name}")
            predicted = alignment @ self.model.frames[name]
            pos = self._array(self._asset.data.body_pos_w[0, i])
            rot = quat_matrix(self._array(self._asset.data.body_quat_w[0, i]))
            if np.linalg.norm(predicted[:3, 3] - pos) > .01 or np.linalg.norm(rotation_error(predicted[:3, :3], rot)) > .03:
                raise ValueError(f"Full-body URDF/USD mismatch: {name}; self-collision not enabled")
        distances, _ = self.model.distances(q, self.cfg.clearance)
        nearest = int(np.argmin(distances))
        if distances[nearest] < self.cfg.clearance:
            message = (f"Current pose violates clearance: {self.model.pair_name(nearest)} "
                       f"distance={distances[nearest]:.5f}m")
            self._report_collision(message)
            print(f"[SELF COLLISION] startup monitor: {message}; collector remains open", flush=True)
        self.last_safe_target = q.copy()
        self.validated = True
        print(f"[SELF COLLISION] validated {len(self.model.shapes)} shapes, {len(self.model.pairs)} pairs; "
              f"LIGHT/control-tick endpoints only; clearance={self.cfg.clearance*1000:.1f}mm; convex meshes + visual fallbacks; "
              f"empty reference links={self.model.empty_links}", flush=True)

    def apply_actions(self):
        if not self.validated:
            raise CollisionStop("Self-collision model not validated; refusing physics step")
        if self._checked_this_tick and not self._recording:
            return
        if self._checked_this_tick and self.step_collision:
            self._hold_current()
            return
        continuous = [e.index for e in self.model.edges if e.kind == "continuous"]
        try:
            if self._cached_target is None:
                self._checked_this_tick = True
                q = self._array(self._asset.data.joint_pos[0, self._joint_ids])
                desired = self._array(self._asset.data.joint_pos_target[0, self._joint_ids])
                desired[continuous] = q[continuous]
                safe = self.guard.filter_light(q, desired, self._env.step_dt)
                blocked = not bool(self.guard.status["scale"])
                if blocked:
                    self._report_collision("Command would violate self-collision clearance")
                if not self._recording:
                    # Outside a recording the guard is diagnostic only. The
                    # operator must remain able to move away from the pose.
                    self.last_safe_target = desired.copy()
                    return
                self._cached_target = safe.copy()
                self._cached_velocity = (safe-q) / self._env.step_dt
            safe = self._cached_target
        except ClearanceViolation as error:
            self._report_collision(str(error))
            if self._recording:
                # Fail closed for this physics step. The collector consumes
                # the event and closes only the active episode.
                self._hold_current()
            return
        except Exception as error:
            # Invalid states and implementation errors still fail closed and
            # propagate; they are not recoverable operator collisions.
            self._hold_current()
            print(f"[SELF COLLISION ERROR] {error}", flush=True)
            raise
        safe_tensor = torch.as_tensor(safe[None], device=self.device, dtype=torch.float32)
        # Preserve wheel phase/velocity on intervening physics steps. This
        # global self-collision guard does not disable planar base movement.
        safe_tensor[:, continuous] = self._asset.data.joint_pos[:, [self._joint_ids[i] for i in continuous]]
        self._asset.set_joint_position_target(safe_tensor, self._joint_ids)
        safe_velocity = torch.as_tensor(self._cached_velocity[None], device=self.device, dtype=torch.float32)
        original_velocity = self._asset.data.joint_vel_target[:, self._joint_ids].clone()
        safe_velocity[:, continuous] = original_velocity[:, continuous]
        self._asset.set_joint_velocity_target(safe_velocity, self._joint_ids)
        self.last_safe_target = safe
        self.step_modified |= self.guard.status["modified"]
        self.step_minimum_distance = min(self.step_minimum_distance, self.guard.status["minimum_distance_m"])
        # Feed the corrected command back into both persistent IK integrators:
        # otherwise they repeatedly push an accumulated rejected goal.
        index = {name: i for i, name in enumerate(self.model.names)}
        for side in ("left", "right"):
            term = self._env.action_manager.get_term(f"{side}_arm")
            ids = [index[name] for name in term._joint_names]
            term._joint_command[:] = safe_tensor[:, ids] - term._gravity_bias
            term._joint_velocity[:] = safe_velocity[:, ids]
        self._physics_steps += 1
        if self._physics_steps % 120 == 0:
            s = self.guard.status
            print(f"[SELF COLLISION] nearest={s['pair']} distance={s['minimum_distance_m']*1000:.1f}mm "
                  f"modified={s['modified']} endpoint_accepted={bool(s['scale'])}", flush=True)

    def reset(self, env_ids=None):
        self.validated = False
        self.last_safe_target = None
        self._cached_target = None
        self._cached_velocity = None
        self._recording = False
        self._collision_event = None
        self.step_collision = False
        self._checked_this_tick = False


@configclass
class SelfCollisionActionCfg(ActionTermCfg):
    class_type: type = SelfCollisionAction
    asset_name: str = "robot"
    urdf_path: str = ""
    allowed_pairs: str = ""
    clearance: float = .003
