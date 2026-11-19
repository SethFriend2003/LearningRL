# PACE actuator fit — RPO thigh (DM-J10010L)

[PACE](https://github.com/leggedrobotics/pace-sim2real) fits an actuator model to a real chirp
by running thousands of simulated copies of the test rig in parallel and evolving their
parameters with CMA-ES until the sim trajectory matches the real one. Here it is ported to
Isaac Lab 3.0 / Newton, so the fitted numbers mean the same thing in the walk task.

Fitted per motor: **armature** (reflected rotor inertia), **Coulomb friction**, **viscous
damping**, **delay** (physics steps of 5 ms). The result goes to all six thigh joints
(`yaw/roll/pitch`, both legs) via `RPO_PACE_CFG`.

## 1. Recording the chirp (bench)

Rig: one thigh motor, **axis vertical**, the bare thigh bracket bolted on (**no knee motor**), nothing else touching it.

| What | Requirement |
|---|---|
| Control mode | motor-side PD on position (MIT mode), **Kp = 100 N·m/rad, Kd = 3.3 N·m·s/rad** — the thigh gains the walk trains with. Different gains → change `THIGH_BENCH_CFG` to match, or the fit is wrong. |
| Signal | linear chirp 0.1 → 10 Hz over 20–30 s, starting and ending at rest. Taper the amplitude so peak speed `2π·f·A` stays under ~20 rad/s (≤ 0.3 rad at 10 Hz); don't go tiny either (backlash). |
| Log | per sample, on one clock: **time, commanded position, measured position**. ≥ 200 Hz, ideally 500–1000 Hz. CSV, one row per sample, any column names. |
| Extra runs | a second chirp with a different amplitude, kept out of the fit, to validate. |
| The bracket | weigh the bare bracket (with its bolts); from CAD, its CoM relative to the motor axis and its inertia about the CoM. |

## 2. Put the bracket into the sim

Replace `bracket_link`'s `<inertial>` in
`robot_description/robot_description/robots/rpo_thigh_bench/thigh_bench.urdf` (it is currently
a **placeholder**: the URDF thigh link *including* the knee motor, ~2.35 kg — much heavier than the bare bracket). Bench frame: joint axis +z, bracket
along +x. Inertia errors are absorbed into the fitted armature, so this matters.

## 3. Run it (from the docker repo root, container up)

```bash
# raw CSV into the logs mount: workspace/logs/pace/raw/chirp1.csv
bash hcrl.sh run python source/LearningRLlab/scripts/pace/convert_log.py logs/pace/raw/chirp1.csv \
    --time-col <t> --cmd-col <q_des> --pos-col <q> [--time-scale 1e-3] [--deg]
#   -> logs/pace/data/chirp1.pt + chirp1.png (check tracking looks sane)

bash hcrl.sh run ./isaaclab.sh -p source/LearningRLlab/scripts/pace/fit.py \
    --data logs/pace/data/chirp1.pt --viz kit --enable_cameras --headless
#   -> logs/pace/rpo_thigh_bench/<stamp>_chirp1/{result.pt,best_trajectory.pt,events...}
#   debug first: --num_envs 64 --max_iterations 3

bash hcrl.sh run python source/LearningRLlab/scripts/pace/plot_fit.py logs/pace/rpo_thigh_bench/<run>
bash hcrl.sh run python source/LearningRLlab/scripts/pace/export_fit.py logs/pace/rpo_thigh_bench/<run>/result.pt
#   -> rewrites robot_description/.../roboparty_rpo/pace_fit.py (FITTED = True)

bash hcrl.sh train --task LearningRL-RPO-Walk-PACE-v0     # A/B against LearningRL-RPO-Walk-v0
```

Search bounds are `fit.py` flags (`--armature LO HI`, `--friction`, `--viscous_damping`,
`--delay`). `export_fit.py` flags any value sitting on a bound: widen it and refit.

## Notes

- **Encoder bias is pinned to 0** (`--encoder_bias 0 0`): with a vertical axis, nothing in the
  dynamics depends on the true angle, so bias is unobservable. Verified on synthetic data.
- **Fit and train on the same backend.** The bench task is Newton/MJWarp at `dt = 0.005`, like the
  walk. Newton friction is an absolute N·m `frictionloss`; PhysX's is not comparable.
- **Friction is measured unloaded** (no gravity torque); on the robot it is likely higher.
- Knee and torso use the same motor but keep `RPO_CFG`'s model; add their patterns to
  `pace_fit.JOINT_EXPR` to extend the fit to them.
- `sim2sim`: `export.py` writes the fitted armature/friction to `config.yaml`; LearningRLsim does
  not model the delay or viscous damping.
