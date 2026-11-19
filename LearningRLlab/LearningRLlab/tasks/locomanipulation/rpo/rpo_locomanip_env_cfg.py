"""Environment configuration for LearningRL-RPO-LocoManipulation-v0: the RPO walk plus wrist targets.

``LearningRL-G1-LocoManipulation-v0`` with the RoboParty RPO swapped in -- the same relationship
``LearningRL-RPO-Walk-v0`` has to ``LearningRL-G1-Walk-v0``, through the same
:func:`swap_in_rpo`. One policy drives all 23 joints: it walks to the ``base_velocity`` command
and puts its wrists where ``hand_track`` asks, so arms and legs are trained together.

What differs from the G1 task, and why:

- **Tracked points.** The RPO has no hand or wrist link. ``hand_track`` names the forearms
  (``<side>_elbow_yaw_link``) and :class:`RPOHandPosCommand` tracks a point ``WRIST_OFFSET``
  along each, near the forearm tip, so the policy controls where the wrists go.
- **Target ranges.** Resized to the RPO's arm (shoulder about 0.27 m above and 0.18 m beside the
  root, about 0.43 m of reach) rather than the taller G1's. They cover holding a box between the
  forearms at table height in front of the body, and lifting it.
- **Forearm load.** ``add_forearm_mass`` adds 0-0.3 kg to each forearm at startup, so the policy
  also learns to balance with something light held in front of it.

Everything else -- rewards and weights, the hand-tracking ramp, terminations, flat ground,
physics, PPO -- is the G1 locomanipulation task's.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from LearningRLlab.tasks.locomanipulation.g1 import g1_locomanip_env_cfg as g1_locomanip
from LearningRLlab.tasks.locomotion.rpo.rpo_env_cfg import swap_in_rpo

from . import mdp

FOREARMS = ["left_elbow_yaw_link", "right_elbow_yaw_link"]


##
# MDP settings
##


@configclass
class CommandsCfg(g1_locomanip.CommandsCfg):
    """The G1 task's commands, with ``hand_track`` tracking the RPO's wrist points.

    Declared here, at class level, rather than patched in ``__post_init__``: the cfg's
    ``class_type`` string is only made resolvable while the cfg instance is constructed.
    """

    hand_track = mdp.HandPosCommandCfg(
        asset_name="robot",
        # left first, then right: HandPosCommand checks the prefixes
        body_names=list(FOREARMS),
        mode="random",
        resampling_time_range=(0.5, 2.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        # yaw-aligned root frame, metres. x: from beside the hip to a forearm's length past a
        # table edge; y: each wrist on its own side, from a 0.2 m box's faces out to arms-wide;
        # z: from arms hanging at the hips to a lifted box held at chest height.
        ranges=mdp.HandPosCommandCfg.Ranges(
            pos_x=(0.0, 0.40),
            pos_y=(-0.35, 0.35),
            pos_y_left=(0.0, 0.35),
            pos_y_right=(-0.35, 0.0),
            pos_z=(-0.15, 0.35),
        ),
    )


##
# Environment configuration
##


@configclass
class LearningRLRPOLocoManipEnvCfg(g1_locomanip.LearningRLG1LocoManipEnvCfg):
    """``LearningRLG1LocoManipEnvCfg`` with the RPO, tracking its wrist points."""

    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        swap_in_rpo(self)

        self.events.add_forearm_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=list(FOREARMS)),
                "mass_distribution_params": (0.0, 0.3),
                "operation": "add",
            },
        )


@configclass
class LearningRLRPOLocoManipEnvCfg_PLAY(LearningRLRPOLocoManipEnvCfg):
    """Playback variant of ``LearningRLRPOLocoManipEnvCfg``, per the ``-PLAY`` rule in ``CLAUDE.md``."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 100
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0

        self.observations.policy.enable_corruption = False
