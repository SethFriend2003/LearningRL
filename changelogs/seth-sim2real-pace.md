# seth/sim2real — PACE actuator model for the RPO thigh (DM-J10010L)

Goal: replace RoboParty's hand-tuned thigh actuator model (armature 0.01 placeholder, no
friction, 0–2 step random delay) with one identified from a real bench chirp, using
[PACE](https://github.com/leggedrobotics/pace-sim2real) (ETH RSL, Apache-2.0). How-to:
`LearningRLlab/scripts/pace/README.md`.

## Changes

| Commit | Repo | What |
|---|---|---|
| `dca6ee5` | LearningRL | PACE ported to Isaac Lab 3.0 / Newton: `PaceDelayedPDActuator`, thigh bench rig + `LearningRL-RPO-ThighBench-PACE-v0`, CMA-ES fit, CSV converter, plot, export; `RPO_PACE_CFG` + `LearningRL-RPO-Walk-PACE-v0` |
| `d7567fe` | LearningRL | Bench load is the bare thigh bracket (no knee motor) |
| — | LearningRL | Bracket inertia from CAD: 8469.549 kg·mm² about the motor axis (was URDF thigh link placeholder, 0.0216 kg·m² at CoM + offset) |
| `1c5efff` | hcrl-isaaclab-docker | `cmaes` in `Dockerfile.hcrl` (needs `hcrl.sh rebuild`) |

Concept: CMA-ES runs one bench-rig candidate per env, replays the real chirp's position
targets, and minimises squared position error. Fitted: armature, Coulomb friction, viscous
damping, delay; applied to all six thigh joints.

Port deviations from upstream PACE:
- Newton has one Coulomb `frictionloss` (N·m) — no separate static/dynamic friction.
- Viscous friction lives in the actuator (Newton has no per-joint viscous term).
- Encoder bias pinned to 0 by default: unobservable on a vertical-axis rig.
- Fit at the walk's `dt = 0.005`, so the delay is in the walk's physics steps.

## Verification

- ruff + `pytest -m static` (LearningRLlab 9 passed; robot_description 18 passed, 1 skipped;
  `test_sensor_groups` needs Isaac Sim and errors identically on the base commit).
- Actuator groups of `RPO_PACE_CFG` partition all 23 joints (checked by hand).
- Synthetic recovery (torch toy of the same actuator law, 1 mrad noise, ms/degree CSV through
  `convert_log.py`): true armature 0.08 / friction 1.2 / viscous 0.6 / delay 3 →
  fit 0.0800 / 1.189 / 0.605 / 3 in 17 generations (pop 256). Bias not recovered → pinned.
- **Not yet run in Isaac Lab** — no image built on this machine. First container run:
  `fit.py --num_envs 64 --max_iterations 3` on a converted chirp.

## Open

- Real chirp log format and the Kp/Kd it was recorded with.

## Runs

| Run | Task | Data | Result |
|---|---|---|---|
| — | | | |
