"""Opt-in pre-reset observation capture for SAC/DPPO (no simulator imports)."""


class TerminalObservationMixin:
    """Place before ManagerBasedRLEnv in the MRO; PPO's environment is unchanged."""

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
        return super()._reset_idx(env_ids)

    def step(self, action):
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
