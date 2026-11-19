"""Configuration for the RPO's two-hand (wrist) position command.

Kept apart from ``commands.py`` for the G1 task's reason: the env cfg imports this module before
Kit starts, so it may not import ``CommandTerm`` (which reaches ``pxr``, F9).

The class keeps the G1 cfg's NAME, ``HandPosCommandCfg``: the sim2sim exporter writes
``type(cfg).__name__`` into ``config.yaml``, and LearningRLsim dispatches the command on that
string, so an RPO export replays through the same LearningRLsim term as a G1 one.

Only the ``class_type`` DEFAULT changes. It has to be a class-level default rather than an
assignment in the env cfg's ``__post_init__``: ``configclass`` turns a ``"module:Class"`` string
into a resolvable reference only while an instance is being constructed, so a string assigned
afterwards stays a plain ``str`` and the command manager fails calling it.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from LearningRLlab.tasks.locomanipulation.g1.mdp.commands_cfg import HandPosCommandCfg as _G1HandPosCommandCfg


@configclass
class HandPosCommandCfg(_G1HandPosCommandCfg):
    """:class:`HandPosCommandCfg` for the RPO: tracks the wrist points (see ``wrist.py``)."""

    class_type: type | str = "LearningRLlab.tasks.locomanipulation.rpo.mdp.commands:RPOHandPosCommand"
