# Adapted from PACE (https://github.com/leggedrobotics/pace-sim2real @ 7ad0f45,
# pace_sim2real/optim/cma_es.py), (c) 2025 ETH Zurich RSL, Apache-2.0.

"""CMA-ES actuator identification, ported from PACE to Isaac Lab 3.0 / Newton.

Same parameter vector and cost as PACE, minus its separate dynamic friction (Newton has one
Coulomb ``frictionloss``). Layout per fitted joint J: armature, friction, viscous damping,
encoder bias, then one shared integer delay. Bounds are physical; CMA-ES samples in [-1, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import cmaes
import numpy as np
import torch

PER_JOINT = ("armature", "friction", "viscous_damping", "encoder_bias")


@dataclass(frozen=True)
class PaceBounds:
    """Physical search box, shared by every fitted joint: (lower, upper) per parameter."""

    armature: tuple[float, float]  # kg*m^2
    friction: tuple[float, float]  # N*m
    viscous_damping: tuple[float, float]  # N*m*s/rad
    encoder_bias: tuple[float, float]  # rad
    delay: tuple[int, int]  # physics steps

    def tensor(self, num_joints: int) -> torch.Tensor:
        rows = [getattr(self, name) for name in PER_JOINT for _ in range(num_joints)]
        rows.append(tuple(float(d) for d in self.delay))
        return torch.tensor(rows, dtype=torch.float64)


class PaceCMAES:
    """Thin wrapper over ``cmaes.CMA`` that speaks physical parameters as ``(pop, P)`` tensors."""

    def __init__(self, bounds: PaceBounds, num_joints: int, population: int, sigma: float, device: str):
        self.num_joints = num_joints
        self.bounds = bounds.tensor(num_joints)
        self.device = device
        unit = np.tile([-1.0, 1.0], (self.bounds.shape[0], 1))
        self._cma = cmaes.CMA(
            mean=np.zeros(self.bounds.shape[0]), sigma=sigma, bounds=unit, seed=0, population_size=population
        )
        self._unit: np.ndarray | None = None

    @property
    def generation(self) -> int:
        return self._cma.generation

    def ask(self) -> torch.Tensor:
        """Sample one population in physical units, shape ``(population, P)``."""
        self._unit = np.stack([self._cma.ask() for _ in range(self._cma.population_size)])
        return self.to_physical(self._unit)

    def tell(self, scores: torch.Tensor) -> None:
        self._cma.tell(list(zip(self._unit, scores.double().cpu().numpy())))

    def mean(self) -> torch.Tensor:
        """Current distribution mean in physical units, shape ``(P,)`` -- PACE's reported result."""
        return self.to_physical(self._cma._mean[None])[0]

    def to_physical(self, unit: np.ndarray) -> torch.Tensor:
        u = torch.as_tensor(unit, dtype=torch.float64)
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return (lo + (u + 1.0) / 2.0 * (hi - lo)).to(device=self.device, dtype=torch.float32)

    def split(self, params: torch.Tensor) -> dict[str, torch.Tensor]:
        """Named views: per-joint ``(..., J)`` tensors plus ``delay`` as int ``(...,)``."""
        j = self.num_joints
        out = {name: params[..., i * j : (i + 1) * j] for i, name in enumerate(PER_JOINT)}
        out["delay"] = torch.round(params[..., 4 * j]).to(torch.int)
        return out


def apply_to_sim(articulation, actuator, joint_ids: torch.Tensor, named: dict[str, torch.Tensor], q0: torch.Tensor):
    """Write one population into the sim, one candidate per env, and start each at ``q0 + bias``.

    ``joint_ids`` index the articulation; ``actuator`` must own exactly those joints, in that order.
    """
    articulation.write_joint_armature_to_sim_index(armature=named["armature"].contiguous(), joint_ids=joint_ids)
    articulation.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=named["friction"].contiguous(), joint_ids=joint_ids
    )
    actuator.viscous_damping = named["viscous_damping"].clone()
    actuator.encoder_bias = named["encoder_bias"].clone()
    actuator.set_delay(named["delay"])
    actuator.reset(None)
    start = (q0 + named["encoder_bias"]).contiguous()
    articulation.write_joint_position_to_sim_index(position=start, joint_ids=joint_ids)
    articulation.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(start), joint_ids=joint_ids)


def trajectory_cost(sim_q: torch.Tensor, real_q: torch.Tensor, encoder_bias: torch.Tensor) -> torch.Tensor:
    """PACE's score: per-step squared error summed over joints, averaged over time. ``(pop,)``.

    ``sim_q`` is ``(pop, T, J)`` true sim angle; the encoder reads ``sim_q - bias``.
    """
    err = sim_q - encoder_bias[:, None, :] - real_q[None]
    return err.square().sum(-1).mean(-1)
