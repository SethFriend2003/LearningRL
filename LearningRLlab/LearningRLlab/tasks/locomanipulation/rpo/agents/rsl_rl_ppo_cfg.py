"""PPO runner configuration for LearningRL-RPO-LocoManipulation-v0.

LearningRL-G1-LocoManipulation-v0's (itself the G1 walk's PPO), under its own experiment name and
an RPO wandb project, per the one-project-per-robot-and-family rule in ``CLAUDE.md``.
"""

from isaaclab.utils.configclass import configclass

from LearningRLlab.tasks.locomanipulation.g1.agents.rsl_rl_ppo_cfg import LearningRLG1LocoManipPPORunnerCfg


@configclass
class LearningRLRPOLocoManipPPORunnerCfg(LearningRLG1LocoManipPPORunnerCfg):
    experiment_name = "learningrl_rpo_locomanipulation"
    run_name = "locomanip"
    wandb_project = "RPO-locomanipulation"
