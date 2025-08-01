import math
import torch
import numpy as np
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
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotEnvCfg, RobotSceneCfg



@configclass
class Go2VisionSceneCfg(RobotSceneCfg):
    rgbd_camera : CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/rgbd_camera",
        offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.5), rot=(-0.5, 0.5, -0.5, 0.5)),
        spawn=sim_utils.PinholeCameraCfg(horizontal_aperture=54.0),
        width=512,
        height=512,
        data_types= ["distance_to_image_plane"],#["rgb", "distance_to_image_plane"],
    )

    #tiled_camera: TiledCameraCfg = TiledCameraCfg(
    #    prim_path="{ENV_REGEX_NS}/Robot/base/rgbd_camera",
    #    offset=TiledCameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.5), rot=(-0.5, 0.5, -0.5, 0.5)),
    #    spawn=sim_utils.PinholeCameraCfg(horizontal_aperture=54.0),
    #    width=100,
    #    height=100,
    #    data_types= ["distance_to_image_plane"],#["rgb", "distance_to_image_plane"],
    #)


@configclass
class ObservationsCfg:
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
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        # def __post_init__(self):
        #     self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()

    @configclass
    class DepthObsCfg(ObsGroup):
        """Observations for visualization camera group."""
        image = ObsTerm(
            func=mdp.isaac_camera_data, params={"sensor_cfg": SceneEntityCfg("rgbd_camera"), "data_type": "distance_to_image_plane"}
        )
        def __post_init__(self):
            self.history_length = 4
            self.enable_corruption = True

    depth_obs: DepthObsCfg = DepthObsCfg()

 
@configclass
class RobotVisionEnvCfg(RobotEnvCfg):
    scene: Go2VisionSceneCfg = Go2VisionSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

@configclass
class RobotVisionPlayEnvCfg(RobotVisionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotVisionKeyboardEnvCfg(RobotVisionPlayEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        # define the keyboard control
        self.velocity_command = [0, 0, 0]
        def velocity_func(env):
            return torch.tensor(
                np.array([self.velocity_command]), device=env.device, dtype=torch.float32
            ).repeat(env.num_envs, 1)
        self.observations.policy.velocity_commands = ObsTerm(func=velocity_func)
        self.commands.base_velocity.ranges= mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(.0, .0), lin_vel_y=(.0, .0), ang_vel_z=(.0, .0)
        )
