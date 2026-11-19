"""Convert a real chirp log (CSV) into the PACE chirp ``.pt`` that ``fit.py`` reads.

Needs only numpy + torch (no Isaac Lab, no Kit), so it runs on the host too::

    python convert_log.py chirp.csv --time-col t --cmd-col q_des --pos-col q --time-scale 1e-3 --deg

Resamples onto a uniform grid at ``--dt`` (the fit's physics dt) by linear interpolation, and
writes a PNG next to the ``.pt`` to eyeball tracking before spending GPU time on it.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser(description="Real chirp CSV -> PACE chirp .pt")
parser.add_argument("csv", type=Path)
parser.add_argument("--time-col", required=True)
parser.add_argument("--cmd-col", required=True, nargs="+", help="commanded position column(s), one per joint")
parser.add_argument("--pos-col", required=True, nargs="+", help="measured position column(s), same order")
parser.add_argument("--joint", nargs="+", default=["thigh_bench_joint"], help="sim joint name(s), same order")
parser.add_argument("--time-scale", type=float, default=1.0, help="multiply time column into seconds (ms: 1e-3)")
parser.add_argument("--deg", action="store_true", help="position columns are in degrees")
parser.add_argument("--dt", type=float, default=0.005, help="fit physics dt [s]; must equal the bench task's")
parser.add_argument("--start", type=float, default=None, help="trim: first kept time [s, log clock]")
parser.add_argument("--end", type=float, default=None, help="trim: last kept time [s, log clock]")
parser.add_argument("--out", type=Path, default=None, help="default: logs/pace/data/<csv stem>.pt")


def resample(time: np.ndarray, values: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate ``values`` (T, J) onto a uniform grid at ``dt`` starting at 0."""
    t = time - time[0]
    grid = np.arange(0.0, t[-1] + 1e-9, dt)
    return grid, np.stack([np.interp(grid, t, values[:, j]) for j in range(values.shape[1])], axis=1)


def main() -> int:
    args = parser.parse_args()
    if not (len(args.cmd_col) == len(args.pos_col) == len(args.joint)):
        parser.error("--cmd-col, --pos-col and --joint need one entry per joint")

    with args.csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        parser.error(f"{args.csv} has no data rows")
    cols = [args.time_col, *args.cmd_col, *args.pos_col]
    missing = [c for c in cols if c not in rows[0]]
    if missing:
        parser.error(f"columns {missing} not in {args.csv}; it has {list(rows[0])}")

    raw = np.array([[float(r[c]) for c in cols] for r in rows])
    t = raw[:, 0] * args.time_scale
    order = np.argsort(t, kind="stable")
    raw, t = raw[order], t[order]
    keep = np.ones_like(t, dtype=bool)
    if args.start is not None:
        keep &= t >= args.start
    if args.end is not None:
        keep &= t <= args.end
    raw, t = raw[keep], t[keep]
    _, unique = np.unique(t, return_index=True)  # drop duplicate timestamps
    raw, t = raw[unique], t[unique]

    n = len(args.joint)
    scale = np.pi / 180.0 if args.deg else 1.0
    cmd, pos = raw[:, 1 : 1 + n] * scale, raw[:, 1 + n :] * scale
    gaps = np.diff(t)
    print(
        f"[convert] {len(t)} samples over {t[-1] - t[0]:.2f} s, raw dt median {np.median(gaps) * 1e3:.2f} ms "
        f"(max gap {gaps.max() * 1e3:.1f} ms) -> {args.dt * 1e3:.1f} ms grid"
    )
    if gaps.max() > 5 * args.dt:
        print(f"[convert] WARNING: {gaps.max() * 1e3:.1f} ms gap in the log; interpolation spans it")

    grid, cmd_u = resample(t, cmd, args.dt)
    _, pos_u = resample(t, pos, args.dt)
    out = args.out or Path("logs") / "pace" / "data" / f"{args.csv.stem}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "time": torch.tensor(grid, dtype=torch.float64),
            "dof_pos": torch.tensor(pos_u, dtype=torch.float32),
            "des_dof_pos": torch.tensor(cmd_u, dtype=torch.float32),
            "joint_order": list(args.joint),
            "source": str(args.csv),
        },
        out,
    )
    rms = np.sqrt(np.mean((pos_u - cmd_u) ** 2, axis=0))
    print(f"[convert] wrote {out}  ({len(grid)} steps; tracking rms per joint [rad]: {np.round(rms, 4).tolist()})")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return 0
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), squeeze=False)
    for j, ax in enumerate(axes[:, 0]):
        ax.plot(grid, cmd_u[:, j], label="commanded", lw=0.8)
        ax.plot(grid, pos_u[:, j], label="measured", lw=0.8)
        ax.set(title=args.joint[j], xlabel="time [s]", ylabel="position [rad]")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=120)
    print(f"[convert] plot {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
