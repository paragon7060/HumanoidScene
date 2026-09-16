"""Optional control-rate response log, independent of image/dataset writers."""
import json
from pathlib import Path
import queue
import re
import threading
import time

JOINT_PATTERN = re.compile(r"(?:knee_joint|leg_joint|leg_[lr][1-6]_joint|waist_(?:pitch|yaw)_joint|zarm_[lr][1-7]_joint)")


def vector(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


class ResponseWriter:
    def __init__(self, path, metadata):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x")
        self.stream.write(json.dumps(dict(metadata, kind="metadata", schema="kuavo_joint_response_v1"), allow_nan=False)+"\n")
        self.stream.flush()
        self.queue = queue.Queue(maxsize=256)
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        try:
            while True:
                row = self.queue.get()
                if row is None:
                    break
                self.stream.write(json.dumps(row, allow_nan=False)+"\n")
                if self.queue.empty():
                    self.stream.flush()
        except Exception as exc:
            self.error = exc

    def append(self, row):
        if self.closed or self.error is not None:
            raise RuntimeError(f"Response log unavailable: {self.error}")
        try:
            self.queue.put_nowait(row)
        except queue.Full as exc:
            raise RuntimeError("Response log queue full; no silent sample dropping") from exc

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.queue.put(None, timeout=2)
        except queue.Full as exc:
            raise RuntimeError("Response log failed to drain") from exc
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("Response log writer did not finish")
        try:
            if self.error is not None:
                raise RuntimeError(f"Response log writer failed: {self.error}")
            self.stream.write(json.dumps({"kind":"closed","ok":True})+"\n")
        finally:
            self.stream.close()


class JointResponseProbe:
    """Before/after each env.step; logical target remains separate from solver bias.

    Applied targets/torques describe the LAST physics write of that control step,
    not a motor measurement or every substep. Auto-reset samples cannot export.
    """
    def __init__(self, env, model, path, mode):
        self.env, self.robot = env, env.scene["robot"]
        self.ids = [i for i,n in enumerate(self.robot.joint_names) if JOINT_PATTERN.fullmatch(n)]
        self.names = [self.robot.joint_names[i] for i in self.ids]
        if not self.ids:
            raise ValueError("No body/arm joints for response log")
        self.writer = ResponseWriter(path, {"robot_model":model.name,"urdf_path":str(model.urdf_path),
            "robot_usd_path":str(getattr(model,"usd_path","unknown")),
            "device":str(getattr(env,"device","unknown")),
            "mode":mode,"joint_names":self.names,"control_dt_s":float(env.step_dt),
            "physics_dt_s":float(env.cfg.sim.dt),"units":"rad, rad/s, seconds",
            "dynamics_profile":str(getattr(self.robot,"dynamics_profile","unknown")),
            "gravity_compensation":bool(getattr(self.robot,"gravity_compensation_enabled",False)),
            "logical_target_unbiased":bool(getattr(self.robot,"gravity_compensation_enabled",False)),
            "target_semantics":"logical_joint_target_rad is unbiased; solver target is simulation-only",
            "torque_semantics":"Isaac computed/applied torque estimates; not real measured output",
            "sampling":"once per control step; solver fields from final physics write",
            "lower_rad":vector(self.robot.data.joint_pos_limits[0,self.ids,0]),
            "upper_rad":vector(self.robot.data.joint_pos_limits[0,self.ids,1]),
            "effort_limit_sim":vector(self.robot.data.joint_effort_limits[0,self.ids]),
            "stiffness":vector(self.robot.data.joint_stiffness[0,self.ids]),
            "damping":vector(self.robot.data.joint_damping[0,self.ids])})
        self.segment, self.seq, self.sim_time = 0, 0, 0.
        self.active = False
        self.before = None
        print(f"[RESPONSE] Joint command/response log: {self.writer.path}", flush=True)

    def boundary(self, reason):
        self.segment += 1
        self.writer.append({"kind":"boundary","segment":self.segment,"reason":reason,"sim_time_s":self.sim_time})

    def begin(self, active):
        if bool(active) != self.active:
            self.boundary("control_started" if active else "control_stopped")
        self.active = bool(active)
        self.before = {"q_before_rad":vector(self.robot.data.joint_pos[0,self.ids]),
            "v_before_rad_s":vector(self.robot.data.joint_vel[0,self.ids]),
            "root_before_w":vector(self.robot.data.root_pose_w[0]),
            "episode_length":int(self.env.episode_length_buf[0]),
            "wall_start_monotonic_s":time.monotonic(),"sim_start_s":self.sim_time}

    def end(self, *, tracking_valid, collision=False, context=None):
        pre = self.before
        reset = int(self.env.episode_length_buf[0]) <= pre["episode_length"]
        self.sim_time += float(self.env.step_dt)
        if reset:
            self.boundary("env_auto_reset")
        def data(field):
            value = getattr(self.robot.data,field,None)
            return vector(value[0,self.ids]) if value is not None else None
        sim_target = getattr(self.robot,"_joint_pos_target_sim",None)
        bias = getattr(self.robot,"gravity_compensation_bias",None)
        command_ff = getattr(self.robot,"command_feedforward_torque",None)
        total_ff = getattr(self.robot,"total_feedforward_torque",None)
        row = dict(pre, kind="step",segment=self.segment,seq=self.seq,
            sim_end_s=self.sim_time,wall_end_monotonic_s=time.monotonic(),
            logical_joint_target_rad=data("joint_pos_target"),logical_joint_velocity_rad_s=data("joint_vel_target"),
            solver_joint_target_rad=vector(sim_target[0,self.ids]) if sim_target is not None else None,
            gravity_bias_rad=vector(bias[0,self.ids]) if bias is not None else None,
            command_feedforward_torque=vector(command_ff[0,self.ids]) if command_ff is not None else None,
            total_feedforward_torque=vector(total_ff[0,self.ids]) if total_ff is not None else None,
            q_after_rad=data("joint_pos"),v_after_rad_s=data("joint_vel"),
            computed_torque_estimate=data("computed_torque"),applied_torque_estimate=data("applied_torque"),
            root_after_w=vector(self.robot.data.root_pose_w[0]),active=self.active,
            tracking_valid=bool(tracking_valid),collision=bool(collision),reset_after_step=reset,
            exportable=self.active and bool(tracking_valid) and not collision and not reset,
            context=context or {})
        self.writer.append(row)
        self.seq += 1

    def close(self):
        self.writer.close()
