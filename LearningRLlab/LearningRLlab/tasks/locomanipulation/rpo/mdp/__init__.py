"""Task-local MDP terms for LearningRL-RPO-LocoManipulation-v0.

The G1 locomanipulation task's ``mdp`` namespace (which re-exports the G1 walk's, F6), then this
task's, so the local ``HandPosCommandCfg`` (the RPO wrist variant) wins over the G1 one.

``RPOHandPosCommand`` is not exported, for the same reason ``HandPosCommand`` is not: it
imports ``CommandTerm``, which reaches ``pxr`` (F9). ``HandPosCommandCfg.class_type`` names it.
"""

from LearningRLlab.tasks.locomanipulation.g1.mdp import *  # noqa: F403

from .commands_cfg import HandPosCommandCfg
from .wrist import WRIST_OFFSET
