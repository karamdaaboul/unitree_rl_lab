import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG
from unitree_rl_lab.tasks.locomotion import mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as manipulation_mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import ActionsCfg, RewardsCfg, EventCfg, RobotSceneCfg, TerminationsCfg

@configclass
class Go2PedipulationRewards(RewardsCfg):

    fl_foot_pos_error = RewTerm(
        func=manipulation_mdp.position_command_error,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="FL_foot"),
                "command_name": "ee_pose"},
    )
    fl_foot_pos_error_tanh = RewTerm(
        func=manipulation_mdp.position_command_error_tanh,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="FL_foot"),
                "std": 0.05,
                "command_name": "ee_pose"},
    )

@configclass
class Go2PedipulationObservations:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        ee_pose_command = ObsTerm(func=mdp.generated_commands,clip=(-100, 100), params={"command_name": "ee_pose"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))


        def __post_init__(self):
            # self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        ee_pose_command = ObsTerm(func=mdp.generated_commands,clip=(-100, 100), params={"command_name": "ee_pose"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))



@configclass
class Go2PedipulationCommands:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        )
    )
    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="FL_foot",
        resampling_time_range=(1.0, 3.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.45),
            pos_y=(0.05, 0.30),
            pos_z=(-0.20, 0.20),
            roll=(-0.1, 0.1),
            pitch=(-0.1, 0.1),
            yaw=(-0.1, 0.1),
        ),
    )

@configclass
class Go2PedipulationEvents(EventCfg):
    # Add an external force to simulate a payload being carried.
    left_foot_force = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="FL_foot"),
            "force_range": (-2.0, 2.0),
            "torque_range": (-0.1, 0.1),
        },
    )


@configclass
class Go2PedipulationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards:      Go2PedipulationRewards      = Go2PedipulationRewards()
    observations: Go2PedipulationObservations = Go2PedipulationObservations()
    commands:     Go2PedipulationCommands     = Go2PedipulationCommands()
    events:       Go2PedipulationEvents       = Go2PedipulationEvents()
    terminations:  TerminationsCfg              = TerminationsCfg()
    actions:      ActionsCfg                  = ActionsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # custom config
        self.scene.robot.actuators["legs"].stiffness = 40.0
        self.scene.robot.actuators["legs"].damping = 1.0


@configclass
class Go2PedipulationPlayEnvCfg(Go2PedipulationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        #self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
