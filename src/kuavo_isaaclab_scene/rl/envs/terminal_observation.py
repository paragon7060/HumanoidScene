"""Opt-in pre-reset observation capture for SAC/DPPO (no simulator imports)."""


class TerminalObservationMixin:
    """Place before ManagerBasedRLEnv in the MRO; PPO's environment is unchanged."""

    def enable_numerical_dynamics_recovery(self):
        """Opt in to per-environment respawn; corrupted transitions stay excluded."""
        import torch
        self._numerical_failure = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        self._numerical_diagnostics = {}
        self.scene["robot"].nonfinite_dynamics_recovery = self._recover_numerical_dynamics

    def _recover_numerical_dynamics(self, mask, diagnostics):
        self._numerical_failure |= mask
        for name, value in diagnostics.items():
            self._numerical_diagnostics.setdefault(name, value.clone().zero_())
            self._numerical_diagnostics[name] |= value & mask
        print("[NUMERICAL RECOVERY] env_ids=" + str(mask.nonzero().flatten()[:16].tolist())
              + "; causes=" + str({name: int((value & mask).sum())
                                    for name, value in diagnostics.items()}), flush=True)
        # Repair failed environments before the physics write finishes, using
        # ordinary reset managers. Repaired poses are not valid task transitions.
        capture = getattr(self, "_capture_terminal", False)
        self._capture_terminal = False
        try:
            self._reset_idx(mask.nonzero(as_tuple=False).squeeze(-1))
        finally:
            self._capture_terminal = capture

    def _refresh_robot_kinematics(self):
        """Publish reset FK before managers read TCPs at the same timestamp."""
        if getattr(self, "_numerical_failure", None) is None:
            return
        view = getattr(getattr(self, "sim", None), "physics_sim_view", None)
        if view is None:
            return
        view.update_articulations_kinematic()
        # Reset managers may have populated a pose buffer before PhysX FK was
        # refreshed. Advancing time is wrong here: discard only cached views.
        for name, buffer in vars(self.scene["robot"].data).items():
            if name.startswith(("_root_", "_body_")) and hasattr(buffer, "timestamp"):
                buffer.timestamp = -1.0

    def _ensure_numerical_robot_state(self):
        """Guard task geometry after the final physics substep, before pose math."""
        import torch
        if getattr(self, "_numerical_failure", None) is None:
            return
        data = self.scene["robot"].data
        poses = data.body_link_pose_w
        root = data.root_state_w
        diagnostics = {
            "robot_pose_invalid": (~torch.isfinite(poses).flatten(1).all(-1)
                | (poses[..., 3:].norm(dim=-1) < 1e-8).any(-1)),
            "root_state_nonfinite": (~torch.isfinite(root).all(-1)
                | (root[:, 3:7].norm(dim=-1) < 1e-8)),
            "joint_state_nonfinite": (~torch.isfinite(data.joint_pos).all(-1)
                | ~torch.isfinite(data.joint_vel).all(-1)),
        }
        mask = torch.stack(list(diagnostics.values())).any(0)
        if mask.any():
            self._recover_numerical_dynamics(mask, diagnostics)
            repaired = data.body_link_pose_w[mask]
            if (not torch.isfinite(repaired).all()
                    or (repaired[..., 3:].norm(dim=-1) < 1e-8).any()
                    or not torch.isfinite(data.root_state_w[mask]).all()):
                raise ValueError("Invalid robot pose after numerical recovery")

    def _grasp_geometry_snapshot(self):
        grasp = getattr(self, "_multi_box_privileged_grasp_step", None)
        if grasp is None:
            return None
        names = ("matched_flap_distance_m", "front_staging_distance_m")
        if not all(hasattr(grasp, name) for name in names):
            return None
        result = {name: getattr(grasp, name).clone() for name in names}
        if hasattr(grasp, "pinch") and hasattr(grasp.pinch, "hand_pinching"):
            result["hand_pinching"] = grasp.pinch.hand_pinching.clone()
        if hasattr(grasp, "success") and hasattr(grasp.success, "instantaneous"):
            result["instantaneous_success"] = grasp.success.instantaneous.clone()
            for name in ("opposing_flaps", "stable", "proof_lift", "hold_time_s"):
                if hasattr(grasp.success, name):
                    result[name] = getattr(grasp.success, name).clone()
        return result

    def _reset_idx(self, env_ids):
        if getattr(self, "_capture_terminal", False) and len(env_ids):
            safety = getattr(self, "_multi_box_grasp_safety_step", None)
            if safety is not None and self._terminal_safety is None:
                # The reset event invalidates and recomputes this cache for the
                # respawned state.  Preserve the just-terminated physics step
                # before any manager or observation reset can overwrite it.
                self._terminal_safety = {
                    name: value.clone()
                    for name, value in vars(safety).items()
                    if hasattr(value, "clone")
                }
            if self._terminal_grasp_geometry is None:
                self._terminal_grasp_geometry = self._grasp_geometry_snapshot()
            observations = self.observation_manager.compute(update_history=False)
            self._terminal_ids = env_ids.clone()
            self._terminal_observations = {
                name: value[env_ids].clone()
                for name, value in observations.items()
            }
            # Kept for older SAC/DPPO callers that consume only the policy group.
            self._terminal_obs = self._terminal_observations["policy"]
            if getattr(self, "_capture_task_metrics", False):
                command = self.command_manager.get_term("workcell")
                self._terminal_task_metrics = {name: value[env_ids].clone()
                                               for name, value in command.metrics.items()}
        result = super()._reset_idx(env_ids)
        self._refresh_robot_kinematics()
        return result

    def step(self, action):
        numerical = getattr(self, "_numerical_failure", None)
        if numerical is not None:
            numerical.zero_()
            self._numerical_diagnostics.clear()
        self._terminal_ids = self._terminal_obs = None
        self._terminal_observations = None
        self._terminal_task_metrics = None
        self._terminal_safety = None
        self._terminal_grasp_geometry = None
        self._capture_terminal = True
        try:
            obs, reward, terminated, truncated, extras = super().step(action)
        finally:
            self._capture_terminal = False
        next_observations = {
            name: value.clone() for name, value in obs.items()
        }
        if self._terminal_ids is not None:
            for name, value in self._terminal_observations.items():
                next_observations[name][self._terminal_ids] = value
        result = dict(extras)
        if numerical is not None:
            result["transition_numerical_failure"] = numerical.clone()
            result["transition_numerical_diagnostics"] = {
                name: value.clone() for name, value in self._numerical_diagnostics.items()}
        result["transition_next_observations"] = next_observations
        result["transition_next_obs"] = next_observations["policy"]
        if self._terminal_safety is not None:
            result["transition_safety"] = self._terminal_safety
        else:
            safety = getattr(self, "_multi_box_grasp_safety_step", None)
            if safety is not None:
                result["transition_safety"] = {
                    name: value.clone()
                    for name, value in vars(safety).items()
                    if hasattr(value, "clone")
                }
        geometry = (self._terminal_grasp_geometry if self._terminal_grasp_geometry is not None
                    else self._grasp_geometry_snapshot())
        if geometry is not None:
            result["transition_grasp_geometry"] = geometry
        if getattr(self, "_capture_task_metrics", False):
            command = self.command_manager.get_term("workcell")
            task_metrics = {name: value.clone() for name, value in command.metrics.items()}
            if self._terminal_task_metrics is not None:
                for name, value in self._terminal_task_metrics.items():
                    task_metrics[name][self._terminal_ids] = value
            result["transition_task_metrics"] = task_metrics
        return obs, reward, terminated, truncated, result
