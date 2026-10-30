"""Opt-in pre-reset observation capture for SAC/DPPO (no simulator imports)."""


class TerminalObservationMixin:
    """Place before ManagerBasedRLEnv in the MRO; PPO's environment is unchanged."""

    def _reset_idx(self, env_ids):
        if getattr(self, "_capture_terminal", False) and len(env_ids):
            obs = self.observation_manager.compute(update_history=False)["policy"]
            self._terminal_ids = env_ids.clone()
            self._terminal_obs = obs[env_ids].clone()
        return super()._reset_idx(env_ids)

    def step(self, action):
        self._terminal_ids = self._terminal_obs = None
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
        return obs, reward, terminated, truncated, result
