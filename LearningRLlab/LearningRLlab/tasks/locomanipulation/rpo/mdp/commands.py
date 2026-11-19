"""The two-hand position command, tracking the RPO's wrist points instead of a hand body origin.

Loaded only through ``HandPosCommandCfg.class_type``, after Kit starts: this module imports
``HandPosCommand`` (and through it ``CommandTerm``), which reaches ``pxr`` (F9).

Everything else -- the 8-dim layout, sampling, the velocity-advanced target, ``mirror_map`` -- is
the G1 task's ``HandPosCommand`` unchanged, and the cfg class is still ``HandPosCommandCfg``, so
the exported ``config.yaml`` names the same command type LearningRLsim already implements.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils

from LearningRLlab.tasks.locomanipulation.g1.mdp.commands import HandPosCommand

from .wrist import WRIST_OFFSET


class RPOHandPosCommand(HandPosCommand):
    """:class:`HandPosCommand` whose tracked points are ``WRIST_OFFSET`` in each hand body's frame."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._wrist_offset = torch.tensor(WRIST_OFFSET, device=self.device).expand(self.num_envs, 2, 3)

    def hand_pos_w(self) -> torch.Tensor:
        """Wrist points in the world frame. Shape is (num_envs, 2, 3)."""
        body_pos_w = self.robot.data.body_pos_w.torch[:, self.body_ids]
        body_quat_w = self.robot.data.body_quat_w.torch[:, self.body_ids]  # (x, y, z, w), F8
        return body_pos_w + math_utils.quat_apply(body_quat_w, self._wrist_offset)

    def hand_pos_b(self) -> torch.Tensor:
        """Wrist points in the yaw-aligned root frame. Shape is (num_envs, 2, 3).

        The reward (``hand_pos_tracking_reward``) and the error metric both read this, so
        overriding it is enough to move every use of the hand point to the wrist.
        """
        root_pos, yaw = self._yaw_frame()
        return math_utils.quat_apply_inverse(yaw, self.hand_pos_w() - root_pos)

    def _debug_vis_callback(self, event) -> None:
        if not self.robot.is_initialized:
            return
        marker_indices = (~self.active_hand_mask).int().view(-1)
        self.goal_pos_visualizer.visualize(
            self._to_world(self.target_pos_b()).view(-1, 3), marker_indices=marker_indices
        )
        self.current_pos_visualizer.visualize(self.hand_pos_w().reshape(-1, 3), marker_indices=marker_indices)
