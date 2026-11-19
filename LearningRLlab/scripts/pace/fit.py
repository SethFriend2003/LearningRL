# Adapted from PACE (https://github.com/leggedrobotics/pace-sim2real @ 7ad0f45, scripts/pace/fit.py),
# (c) 2025 ETH Zurich RSL, Apache-2.0.

"""Fit a PACE actuator model to a real chirp with CMA-ES, one candidate per env.

Run from ``/workspace/isaaclab`` (logs are CWD-relative, F3), Kit needed for the URDF (F1)::

    ./isaaclab.sh -p source/LearningRLlab/scripts/pace/fit.py --data logs/pace/data/<chirp>.pt \\
        --viz kit --enable_cameras --headless

Writes ``logs/pace/<experiment>/<timestamp>_<chirp>/``: ``result.pt`` (named parameters +
score), ``best_trajectory.pt``, ``config.pt`` and TensorBoard scalars. Next: ``export_fit.py``.
"""

import argparse
import contextlib
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

import LearningRLlab.tasks  # noqa: F401  (registers the LearningRL tasks with gymnasium)
from LearningRLlab.sysid.pace.cma_es import PER_JOINT, PaceBounds, PaceCMAES, apply_to_sim, trajectory_cost
from LearningRLlab.sysid.pace.data import load_chirp

parser = argparse.ArgumentParser(description="PACE CMA-ES actuator fit on a real chirp.")
parser.add_argument("--task", default="LearningRL-RPO-ThighBench-PACE-v0")
parser.add_argument("--data", required=True, help="chirp .pt from convert_log.py (CWD-relative)")
parser.add_argument("--num_envs", type=int, default=4096, help="CMA-ES population size")
parser.add_argument("--max_iterations", type=int, default=200)
parser.add_argument("--sigma", type=float, default=0.5, help="initial step size in the [-1, 1] box")
parser.add_argument("--epsilon", type=float, default=1e-2, help="stop when (max-min)/min score < epsilon")
parser.add_argument("--save_interval", type=int, default=10)
parser.add_argument("--experiment", default="rpo_thigh_bench")
# Search box (PACE best practice: physically plausible, not extreme). Override per fit.
parser.add_argument("--armature", type=float, nargs=2, default=(1e-5, 0.5), metavar=("LO", "HI"))
parser.add_argument("--friction", type=float, nargs=2, default=(0.0, 5.0), metavar=("LO", "HI"))
parser.add_argument("--viscous_damping", type=float, nargs=2, default=(0.0, 5.0), metavar=("LO", "HI"))
# Pinned to 0: with a vertical axis (no gravity torque) bias is unobservable -- the PD tracks in
# encoder coordinates. Open it (e.g. -0.1 0.1) only for a gravity-loaded rig.
parser.add_argument("--encoder_bias", type=float, nargs=2, default=(0.0, 0.0), metavar=("LO", "HI"))
parser.add_argument("--delay", type=int, nargs=2, default=(0, 8), metavar=("LO", "HI"))
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def _fitted_actuator(articulation, joint_order: list[str]):
    for actuator in articulation.actuators.values():
        if list(actuator.joint_names) == joint_order:
            if not hasattr(actuator, "set_delay"):
                raise TypeError(f"actuator over {joint_order} is {type(actuator).__name__}, need PaceDelayedPDActuator")
            return actuator
    raise ValueError(f"no single actuator group owns exactly {joint_order}")


def main():
    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env = gym.make(args_cli.task, cfg=env_cfg)
        unwrapped = env.unwrapped
        device = unwrapped.device
        dt = unwrapped.physics_dt
        if unwrapped.cfg.decimation != 1:
            raise ValueError(f"{args_cli.task}: decimation must be 1 (one env.step per chirp sample)")

        articulation = unwrapped.scene["robot"]
        data_path = Path(args_cli.data)
        joint_order = list(torch.load(data_path, map_location="cpu")["joint_order"])
        data = load_chirp(data_path, dt, joint_order)
        joint_ids = torch.tensor([articulation.joint_names.index(n) for n in joint_order], device=device)
        actuator = _fitted_actuator(articulation, joint_order)
        if args_cli.delay[1] > actuator.cfg.max_delay:
            raise ValueError(f"--delay upper {args_cli.delay[1]} > actuator max_delay {actuator.cfg.max_delay}")

        real_q = data["dof_pos"].to(device=device, dtype=torch.float32)
        target = data["des_dof_pos"].to(device=device, dtype=torch.float32)
        steps, num_joints = real_q.shape
        q0 = real_q[0].expand(args_cli.num_envs, -1)

        bounds = PaceBounds(
            armature=tuple(args_cli.armature),
            friction=tuple(args_cli.friction),
            viscous_damping=tuple(args_cli.viscous_damping),
            encoder_bias=tuple(args_cli.encoder_bias),
            delay=tuple(args_cli.delay),
        )
        opt = PaceCMAES(bounds, num_joints, args_cli.num_envs, args_cli.sigma, device)

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = Path("logs") / "pace" / args_cli.experiment / f"{stamp}_{data_path.stem}"
        log_dir.mkdir(parents=True, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(log_dir))
        meta = {
            "task": args_cli.task,
            "data": str(data_path),
            "dt": dt,
            "joint_order": joint_order,
            "bounds": bounds.__dict__,
            "population": args_cli.num_envs,
            "sigma": args_cli.sigma,
        }
        torch.save(meta, log_dir / "config.pt")
        print(f"[PACE] {steps} steps x {num_joints} joints at dt={dt}, population {args_cli.num_envs} -> {log_dir}")

        actions = torch.zeros(env.action_space.shape, device=device)
        sim_q = torch.zeros(args_cli.num_envs, steps, num_joints, device=device)
        best_score = float("inf")

        def save(tag: str, score: float, best_traj: torch.Tensor):
            mean = opt.split(opt.mean())
            params = {k: v.tolist() if v.dim() else int(v) for k, v in mean.items()}
            torch.save(
                {**meta, "params": params, "score": score, "generation": opt.generation, "tag": tag},
                log_dir / "result.pt",
            )
            torch.save(
                {
                    "time": data["time"],
                    "real": data["dof_pos"],
                    "target": data["des_dof_pos"],
                    "sim_best": best_traj.cpu(),
                },
                log_dir / "best_trajectory.pt",
            )

        with torch.inference_mode():
            for it in range(args_cli.max_iterations):
                params = opt.ask()
                named = opt.split(params)
                env.reset()
                apply_to_sim(articulation, actuator, joint_ids, named, q0)
                for t in range(steps):
                    sim_q[:, t] = articulation.data.joint_pos.torch[:, joint_ids]
                    actions[:, joint_ids] = target[t]
                    env.step(actions)
                scores = trajectory_cost(sim_q, real_q, named["encoder_bias"])
                opt.tell(scores)

                lo, hi = scores.min().item(), scores.max().item()
                i_best = int(scores.argmin())
                best_traj = sim_q[i_best] - named["encoder_bias"][i_best]
                writer.add_scalar("score/best", lo, it)
                writer.add_scalar("score/spread", (hi - lo) / max(lo, 1e-12), it)
                mean = opt.split(opt.mean())
                for name in PER_JOINT:
                    for j, joint in enumerate(joint_order):
                        writer.add_scalar(f"mean/{name}/{joint}", mean[name][j].item(), it)
                        writer.add_histogram(f"population/{name}/{joint}", named[name][:, j], it)
                writer.add_scalar("mean/delay", int(mean["delay"]), it)
                writer.add_histogram("population/delay", named["delay"].float(), it)
                print(
                    f"[PACE] it {it:3d}  best {lo:.3e}  rms {lo**0.5:.4f} rad  spread {(hi - lo) / lo:.3f}  "
                    + "  ".join(
                        f"{k}={[round(x, 4) for x in v.tolist()] if v.dim() else int(v)}" for k, v in mean.items()
                    )
                )

                best_score = min(best_score, lo)
                if args_cli.save_interval > 0 and it % args_cli.save_interval == 0:
                    save("checkpoint", lo, best_traj)
                if (hi - lo) / lo < args_cli.epsilon:
                    print(f"[PACE] converged: score spread below {args_cli.epsilon}")
                    break
        save("final", lo, best_traj)
        writer.close()
        print(f"[PACE] done -> {log_dir / 'result.pt'}  (best score {best_score:.3e})")
        env.close()


if __name__ == "__main__":
    main()
