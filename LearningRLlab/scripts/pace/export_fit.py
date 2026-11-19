"""Write a PACE fit's result into ``robot_description/.../roboparty_rpo/pace_fit.py``.

torch only (no Isaac Lab), so it runs on the host too::

    python export_fit.py logs/pace/rpo_thigh_bench/<run>/result.pt [--dry-run]

``RPO_PACE_CFG`` (and so ``LearningRL-RPO-Walk-PACE-v0``) then applies the fitted armature,
friction, viscous damping and delay to all six thigh joints. Encoder bias is reported, not exported.
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import torch


def _default_pace_fit() -> Path:
    # find_spec does not execute robot_description, so no Isaac Lab import
    spec = importlib.util.find_spec("robot_description")
    root = (
        Path(spec.origin).parent
        if spec and spec.origin
        else Path(__file__).resolve().parents[3] / "robot_description" / "robot_description"
    )
    return root / "robots" / "roboparty_rpo" / "pace_fit.py"


parser = argparse.ArgumentParser(description="PACE result.pt -> roboparty_rpo/pace_fit.py")
parser.add_argument("result", type=Path)
parser.add_argument("--joint", default=None, help="which fitted joint to export (default: the only one)")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--pace-fit", type=Path, default=None, help="default: the installed robot_description's")


def _set(src: str, name: str, value: str) -> str:
    pattern = rf"^({name}: [^=]+= ).*?(\s+#.*)?$"
    out, n = re.subn(pattern, lambda m: f"{m.group(1)}{value}{m.group(2) or ''}", src, count=1, flags=re.M)
    if n != 1:
        raise ValueError(f"{name} assignment not found in pace_fit.py")
    return out


def main() -> int:
    args = parser.parse_args()
    result = torch.load(args.result, map_location="cpu")
    if result.get("tag") != "final":
        print(f"[export] WARNING: {args.result} is a '{result.get('tag')}' checkpoint, not the final result")
    joints = result["joint_order"]
    joint = args.joint or (joints[0] if len(joints) == 1 else None)
    if joint not in joints:
        parser.error(f"pick --joint from {joints}")
    j = joints.index(joint)
    p = result["params"]
    values = {
        "ARMATURE": float(p["armature"][j]),
        "FRICTION": float(p["friction"][j]),
        "VISCOUS_DAMPING": float(p["viscous_damping"][j]),
        "DELAY": int(p["delay"]),
    }
    lo_hi = result["bounds"]
    print(f"[export] {args.result} ({joint}, score {result['score']:.3e}, generation {result['generation']})")
    for name, value in values.items():
        b = lo_hi[name.lower()]
        edge = (
            "  <-- AT BOUND, widen and refit"
            if abs(value - b[0]) < 1e-3 * (b[1] - b[0]) + 1e-9 or abs(value - b[1]) < 1e-3 * (b[1] - b[0]) + 1e-9
            else ""
        )
        print(f"  {name:16s} {value:.6g}   bounds {tuple(b)}{edge}")
    print(f"  encoder_bias     {float(p['encoder_bias'][j]):.6g} rad (bench calibration, not exported)")

    args.pace_fit = args.pace_fit or _default_pace_fit()
    src = args.pace_fit.read_text()
    src = _set(src, "FITTED", "True")
    src = _set(src, "SOURCE", json.dumps(str(args.result)))
    src = _set(src, "ARMATURE", f"{values['ARMATURE']:.6g}")
    src = _set(src, "FRICTION", f"{values['FRICTION']:.6g}")
    src = _set(src, "VISCOUS_DAMPING", f"{values['VISCOUS_DAMPING']:.6g}")
    src = _set(src, "DELAY", str(values["DELAY"]))
    if args.dry_run:
        print(src)
        return 0
    args.pace_fit.write_text(src)
    print(f"[export] wrote {args.pace_fit}; train LearningRL-RPO-Walk-PACE-v0 to use it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
