"""Task-local MDP terms for LearningRL-RPO-LocoManipulation-v0.

The G1 locomanipulation task's ``mdp`` namespace (which re-exports the G1 walk's, F6), then this
task's. ``RPOHandPosCommand`` is not exported, for the same reason ``HandPosCommand`` is not: it
imports ``CommandTerm``, which reaches ``pxr`` (F9). The env cfg names it by ``class_type`` string.
"""

from LearningRLlab.tasks.locomanipulation.g1.mdp import *  # noqa: F403

from .wrist import WRIST_OFFSET
