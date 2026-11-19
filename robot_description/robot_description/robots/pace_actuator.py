# Adapted from PACE (https://github.com/leggedrobotics/pace-sim2real @ 7ad0f45,
# pace_sim2real/utils/pace_actuator.py), (c) 2025 ETH Zurich RSL, Apache-2.0.

"""PACE actuator for Isaac Lab 3.0: delayed PD + encoder bias + actuator-side viscous friction.

Coulomb friction and armature stay on the solver via the base cfg's `friction`/`armature`
(Newton writes `friction` as an absolute N*m frictionloss). Viscous friction is applied here
because Newton has no per-joint viscous term, and doing it here makes it backend-independent.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class PaceDelayedPDActuator(DelayedPDActuator):
    """`DelayedPDActuator` whose PD sees `q - encoder_bias`, plus `-viscous_damping * qd`."""

    cfg: PaceDelayedPDActuatorCfg

    def __init__(self, cfg: PaceDelayedPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.encoder_bias = self._parse_joint_parameter(cfg.encoder_bias, 0.0)
        self.viscous_damping = self._parse_joint_parameter(cfg.viscous_damping, 0.0)
        # per-env lags pinned by set_delay(); None means DelayedPDActuator's random resample
        self._pinned_lags: torch.Tensor | None = None

    @property
    def _delay_buffers(self):
        return (
            self.positions_delay_buffer,
            self.velocities_delay_buffer,
            self.efforts_delay_buffer,
        )

    def set_delay(self, lags: torch.Tensor) -> None:
        """Pin each env's delay (physics steps, `(num_envs,)`) across resets -- used by the fit."""
        self._pinned_lags = lags.to(device=self._device, dtype=torch.int)
        for buf in self._delay_buffers:
            buf.set_time_lag(self._pinned_lags)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        if self._pinned_lags is not None:
            ids = slice(None) if env_ids is None else env_ids
            for buf in self._delay_buffers:
                buf.set_time_lag(self._pinned_lags[ids], ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        control_action = super().compute(control_action, joint_pos - self.encoder_bias, joint_vel)
        # joint-side friction, so it is not clipped by the motor's effort limit
        control_action.joint_efforts = (
            control_action.joint_efforts - self.viscous_damping * joint_vel
        )
        return control_action


@configclass
class PaceDelayedPDActuatorCfg(DelayedPDActuatorCfg):
    """`DelayedPDActuatorCfg` plus PACE's encoder bias [rad] and viscous friction [N*m*s/rad]."""

    class_type: type = PaceDelayedPDActuator
    encoder_bias: dict[str, float] | float | None = 0.0
    viscous_damping: dict[str, float] | float | None = 0.0
