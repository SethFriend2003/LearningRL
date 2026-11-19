"""Plot a PACE fit: real vs best simulated trajectory, plus the residual. Host-runnable (no Isaac Lab)::

python plot_fit.py logs/pace/rpo_thigh_bench/<run>   # writes <run>/fit.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

parser = argparse.ArgumentParser(description="Plot a PACE fit run directory")
parser.add_argument("run_dir", type=Path)


def main() -> int:
    run = parser.parse_args().run_dir
    traj = torch.load(run / "best_trajectory.pt", map_location="cpu")
    result = torch.load(run / "result.pt", map_location="cpu")
    t, real, target, sim = traj["time"], traj["real"], traj["target"], traj["sim_best"]
    joints = result["joint_order"]
    fig, axes = plt.subplots(2 * len(joints), 1, figsize=(13, 5 * len(joints)), squeeze=False)
    for j, name in enumerate(joints):
        ax, res = axes[2 * j, 0], axes[2 * j + 1, 0]
        ax.plot(t, target[:, j], lw=0.6, color="0.6", label="commanded")
        ax.plot(t, real[:, j], lw=0.8, label="real (measured)")
        ax.plot(t, sim[:, j], lw=0.8, ls="--", label="sim (best candidate)")
        ax.set(title=f"{name}: score {result['score']:.3e}", ylabel="position [rad]")
        ax.legend(loc="upper right")
        res.plot(t, sim[:, j] - real[:, j], lw=0.6)
        res.set(xlabel="time [s]", ylabel="sim - real [rad]")
    fig.tight_layout()
    fig.savefig(run / "fit.png", dpi=120)
    print(f"[plot] {run / 'fit.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
