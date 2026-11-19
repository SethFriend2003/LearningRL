"""PACE chirp-data contract: load and validate. torch only (no Isaac Lab).

A chirp file is a ``torch.save``'d dict, PACE's own format:
``time`` (T,) s, uniformly spaced at the fit's physics dt; ``dof_pos`` (T, J) measured encoder
angle in rad; ``des_dof_pos`` (T, J) commanded position in rad; ``joint_order`` (J,) names.
"""

from __future__ import annotations

from pathlib import Path

import torch

REQUIRED = ("time", "dof_pos", "des_dof_pos", "joint_order")


def load_chirp(path: str | Path, dt: float, joint_order: list[str]) -> dict:
    """Load a chirp file and fail loudly if it does not match the fit's ``dt`` and joints."""
    data = torch.load(path, map_location="cpu")
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise KeyError(f"{path}: missing keys {missing}; convert raw logs with scripts/pace/convert_log.py")
    if list(data["joint_order"]) != list(joint_order):
        raise ValueError(f"{path}: joint_order {list(data['joint_order'])} != task's {joint_order}")
    t = data["time"].double()
    step = torch.diff(t)
    if not torch.allclose(step, torch.full_like(step, dt), rtol=1e-3, atol=1e-6):
        raise ValueError(
            f"{path}: samples are {step.mean().item():.6f} s apart (min {step.min():.6f}, max {step.max():.6f}),"
            f" fit needs uniform {dt} s; re-run convert_log.py --dt {dt}"
        )
    for key in ("dof_pos", "des_dof_pos"):
        if data[key].shape != (t.shape[0], len(joint_order)):
            raise ValueError(f"{path}: {key} has shape {tuple(data[key].shape)}, want {(t.shape[0], len(joint_order))}")
    return data
