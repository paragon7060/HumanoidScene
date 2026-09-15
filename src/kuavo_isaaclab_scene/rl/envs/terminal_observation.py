"""Opt-in pre-reset observation capture for SAC/DPPO (no simulator imports)."""


class TerminalObservationMixin:
    """Place before ManagerBasedRLEnv in the MRO; PPO's environment is unchanged."""

    def _reset_idx(self, env_ids):
        if getattr(self, "_capture_terminal", False) and len(env_ids):
            obs = self.observation_manager.compute(update_history=False)["policy"]
            self._terminal_ids = env_ids.clone()
            self._terminal_obs = obs[env_ids].clone()
            if getattr(self, "_capture_task_metrics", False):
                command = self.command_manager.get_term("workcell")
                self._terminal_task_metrics = {name: value[env_ids].clone()
                                               for name, value in command.metrics.items()}
        return super()._reset_idx(env_ids)

    def step(self, action):
        self._terminal_ids = self._terminal_obs = None
        self._terminal_task_metrics = None
        self._capture_terminal = True
        try:
            obs, reward, terminated, truncated, extras = super().step(action)
        finally:
            self._capture_terminal = False
        next_obs = obs["policy"].clone()
        if self._terminal_ids is not None:
            next_obs[self._terminal_ids] = self._terminal_obs
        result = dict(extras)
        result["transition_next_obs"] = next_obs
        if getattr(self, "_capture_task_metrics", False):
            command = self.command_manager.get_term("workcell")
            task_metrics = {name: value.clone() for name, value in command.metrics.items()}
            if self._terminal_task_metrics is not None:
                for name, value in self._terminal_task_metrics.items():
                    task_metrics[name][self._terminal_ids] = value
            result["transition_task_metrics"] = task_metrics
        return obs, reward, terminated, truncated, result
