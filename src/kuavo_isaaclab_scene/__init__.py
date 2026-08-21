"""Gym registration for the Kuavo robustness workcell."""

import gymnasium as gym


gym.register(
    id="Isaac-Kuavo-RobustWorkcell-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "kuavo_isaaclab_scene.manager_env:KuavoRobustWorkcellEnvCfg"
        ),
    },
)

gym.register(
    id="Isaac-Kuavo-QuestTeleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "kuavo_isaaclab_scene.teleop_env:KuavoQuestTeleopEnvCfg"
        ),
    },
)
