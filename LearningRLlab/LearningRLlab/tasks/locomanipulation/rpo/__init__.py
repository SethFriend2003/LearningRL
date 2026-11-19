"""Gym registration for the LearningRL-RPO-LocoManipulation-v0 task."""

import gymnasium as gym

from . import agents

gym.register(
    id="LearningRL-RPO-LocoManipulation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_locomanip_env_cfg:LearningRLRPOLocoManipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LearningRLRPOLocoManipPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="LearningRL-RPO-LocoManipulation-v0-PLAY",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_locomanip_env_cfg:LearningRLRPOLocoManipEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LearningRLRPOLocoManipPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
