from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from unitree_lab.managers.ui import ManagerBasedSafeRLEnvWindow


@configclass
class ManagerBasedSafeRLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for a reinforcement learning environment with the manager-based workflow."""
    ui_window_class_type: type | None = ManagerBasedSafeRLEnvWindow

    costs: object = MISSING
    """Cost settings."""
