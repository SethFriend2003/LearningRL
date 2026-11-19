"""Isaac Lab `ArticulationCfg` for the PACE thigh-actuator bench rig (`thigh_bench.urdf`).

One fixed DM-J10010L, axis vertical, driving `bracket_link`. Gains and limits are the RPO
thigh joints' (`roboparty_rpo/joint_groups.py`): the real chirp must run with the same Kp/Kd.
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.assets import ArticulationCfg

from robot_description import paths
from robot_description.robots.pace_actuator import PaceDelayedPDActuatorCfg
from robot_description.robots.roboparty_rpo import joint_groups as jg
from robot_description.robots.urdf_common import RobotUrdfFileCfg

BENCH_JOINT: str = "thigh_bench_joint"
BENCH_URDF: Path = Path(__file__).resolve().parent / "thigh_bench.urdf"
# Delay-buffer length; must cover the fit's upper delay bound (physics steps).
BENCH_MAX_DELAY: int = 8

_THIGH = ".*_thigh_pitch_joint"  # yaw/roll/pitch share one limit and gain class

THIGH_BENCH_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=RobotUrdfFileCfg(
        asset_path=str(BENCH_URDF),
        usd_dir=str(paths.robot_build_dir("rpo_thigh_bench")),
        fix_base=True,
        self_collision=False,
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5), joint_pos={BENCH_JOINT: 0.0}),
    actuators={
        "thigh": PaceDelayedPDActuatorCfg(
            joint_names_expr=[BENCH_JOINT],
            effort_limit=jg.EFFORT_LIMIT[_THIGH],
            effort_limit_sim=jg.EFFORT_LIMIT[_THIGH],
            velocity_limit=jg.VELOCITY_LIMIT[_THIGH],
            velocity_limit_sim=jg.VELOCITY_LIMIT[_THIGH],
            stiffness=jg.STIFFNESS[_THIGH],
            damping=jg.DAMPING[_THIGH],
            armature=jg.ARMATURE[_THIGH],
            friction=0.0,
            min_delay=0,
            max_delay=BENCH_MAX_DELAY,
        )
    },
)
"""Bench rig with RoboParty's thigh baseline; the fit overwrites armature, friction, bias, delay."""
