"""PACE-identified DM-J10010L parameters, applied to every RPO thigh joint by `RPO_PACE_CFG`.

Isaac-Lab-free, like `joint_groups.py`. Rewritten by
`LearningRLlab/scripts/pace/export_fit.py` from a thigh-bench fit; edit by hand only to revert.
While `FITTED` is False the values are `RPO_CFG`'s thigh model (gains, armature 0.01, 0-2 delay).
"""

from __future__ import annotations

FITTED: bool = False
# result.pt the values came from
SOURCE: str | None = None

JOINT_EXPR: tuple[str, ...] = (
    ".*_thigh_yaw_joint",
    ".*_thigh_roll_joint",
    ".*_thigh_pitch_joint",
)

ARMATURE: float = 0.01  # kg*m^2, reflected rotor inertia
FRICTION: float = 0.0  # N*m, Coulomb (Newton frictionloss)
VISCOUS_DAMPING: float = 0.0  # N*m*s/rad, applied in PaceDelayedPDActuator
DELAY: int = 1  # physics steps; training samples [DELAY - 1, DELAY + 1]
