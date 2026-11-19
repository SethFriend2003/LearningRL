"""Environment configuration for LearningRL-RPO-ThighBench-PACE-v0: the PACE fit's simulator.

One env per CMA-ES candidate, each a fixed DM-J10010L bench rig replaying the real chirp's
position targets. Not an RL task -- `scripts/pace/fit.py` drives it; there is no reward.
Physics is Newton/MJWarp with the walk's solver options and `sim.dt`, so the fitted
armature/friction/delay mean the same thing in `LearningRL-RPO-Walk-PACE-v0`.
"""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from robot_description.robots.rpo_thigh_bench import THIGH_BENCH_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from . import mdp

# Same as LearningRLG1WalkEnvCfg.sim.dt: one delay step here is one delay step in training.
BENCH_DT = 0.005

_NEWTON = NewtonCfg(
    solver_cfg=MJWarpSolverCfg(njmax=8, nconmax=8, cone="pyramidal", impratio=1, integrator="implicitfast"),
    num_substeps=1,
    debug_mode=False,
)


@configclass
class BenchPhysicsCfg(PresetCfg):
    # Newton either way: PhysX friction/armature semantics differ, and the walk trains on Newton.
    default = _NEWTON
    newton_mjwarp = _NEWTON


@configclass
class BenchSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = THIGH_BENCH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=750.0))


@configclass
class ActionsCfg:
    # absolute joint targets: the chirp's des_dof_pos goes straight to the PD
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class LearningRLRPOThighBenchEnvCfg(ManagerBasedRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(physics=BenchPhysicsCfg())
    viewer = ViewerCfg(eye=(0.8, 0.8, 0.9), lookat=(0.0, 0.0, 0.5))
    scene: BenchSceneCfg = BenchSceneCfg(num_envs=4096, env_spacing=0.6)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 1  # one env.step == one physics step == one chirp sample
        self.sim.dt = BENCH_DT
        self.sim.render_interval = 4
        # longer than any chirp, so the fit's rollout never auto-resets mid-trajectory
        self.episode_length_s = 3600.0


@configclass
class LearningRLRPOThighBenchEnvCfg_PLAY(LearningRLRPOThighBenchEnvCfg):
    """Playback variant of ``LearningRLRPOThighBenchEnvCfg``, per the ``-PLAY`` rule in ``CLAUDE.md``."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 100
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
