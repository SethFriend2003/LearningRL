"""Gym registration for LearningRL-RPO-ThighBench-PACE-v0 (PACE fit simulator, no agent)."""

import gymnasium as gym

gym.register(
    id="LearningRL-RPO-ThighBench-PACE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.bench_env_cfg:LearningRLRPOThighBenchEnvCfg"},
)

gym.register(
    id="LearningRL-RPO-ThighBench-PACE-v0-PLAY",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.bench_env_cfg:LearningRLRPOThighBenchEnvCfg_PLAY"},
)
