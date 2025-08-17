# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Union
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from unitree_rl.safe_rl.algorithm_cfg import RslRlP3OAlgorithmCfg

@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

@configclass
class BaseP3ORunnerCfg(BasePPORunnerCfg):
    """Configuration for P3O (Penalized PPO) runner."""
    
    algorithm = RslRlP3OAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # P3O-specific parameters
        #kappa_init=1.0,  # Initial penalty factor
        kappa_max=100.0,  # Maximum penalty factor
        rho=1.5,  # Penalty factor multiplier (ρ > 1)
        cost_thresholds=[0.1],  # Cost threshold for constraints
        adaptive_penalty=True,  # Whether to use adaptive penalty updates
        constraint_margin=0.85,  # Constraint violation detection margin
        use_clipped_cost_loss=True,  # Whether to use clipped cost value loss
        cost_loss_coef=1.0,  # Coefficient for cost value loss
        #constraint_delay=0,  # Iterations to delay constraint enforcement
    )
    