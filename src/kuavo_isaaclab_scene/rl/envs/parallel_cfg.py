"""Simulator-independent configuration for homogeneous, vectorized task cells."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ParallelEnvCfg:
    num_envs: int = 8
    env_spacing: float = 8.0
    replicate_physics: bool = True
    filter_collisions: bool = True
    clone_in_fabric: bool = False

    def validate(self):
        if isinstance(self.num_envs, bool) or not isinstance(self.num_envs, int) or self.num_envs < 1:
            raise ValueError("num_envs must be a positive integer.")
        if not math.isfinite(self.env_spacing) or self.env_spacing < 5.0:
            raise ValueError("Use finite env_spacing >= 5 m for separate workcells.")
        if not self.filter_collisions:
            raise ValueError("RL cells must have inter-environment collision filtering enabled.")
        if self.clone_in_fabric:
            raise ValueError("Keep clone_in_fabric=False: custom robot/flap spawners edit USD schemas.")

    def scene_kwargs(self):
        self.validate()
        return dict(num_envs=self.num_envs, env_spacing=self.env_spacing,
                    replicate_physics=self.replicate_physics,
                    filter_collisions=self.filter_collisions, clone_in_fabric=self.clone_in_fabric)

    @property
    def ground_extent(self):
        # The cloner centers a square grid. Leave additional room around all cells.
        return max(30.0, 2.0 * self.env_spacing * math.ceil(math.sqrt(self.num_envs)))
