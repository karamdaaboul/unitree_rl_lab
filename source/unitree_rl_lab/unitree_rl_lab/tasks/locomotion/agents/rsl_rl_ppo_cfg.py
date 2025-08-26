# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Union
from dataclasses import MISSING

from isaaclab.utils import configclass
from unitree_rl.safe_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoActorCriticRecurrentCfg, RslRlPpoAlgorithmCfg
from unitree_rl.safe_rl import RslRlP3oActorCriticCfg, RslRlP3oAlgorithmCfg
from unitree_rl.safe_rl import RslRlPpolPidActorCriticCfg, RslRlPpolPidAlgorithmCfg


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
class RecurrentPPORunnerCfg(BasePPORunnerCfg):
    policy = RslRlPpoActorCriticRecurrentCfg(
        class_name="ActorCriticRecurrent",
        init_noise_std=1.0,
        actor_hidden_dims=[200, 100],
        critic_hidden_dims=[200, 100],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )


@configclass
class BaseP3ORunnerCfg(BasePPORunnerCfg):
    """Configuration for P3O (Penalized PPO) runner."""
    
    algorithm = RslRlP3oAlgorithmCfg(
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
        adaptive_penalty=True,  # Whether to use adaptive penalty updates
        constraint_margin=0.85,  # Constraint violation detection margin
        use_clipped_cost_loss=True,  # Whether to use clipped cost value loss
        cost_loss_coef=1.0,  # Coefficient for cost value loss
        #constraint_delay=0,  # Iterations to delay constraint enforcement
    )

@configclass
class BasePpolPidRunnerCfg(BasePPORunnerCfg):
    """Configuration for PPOL-PID (PPO Lagrangian with PID) runner."""
    
    policy = RslRlPpolPidActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    algorithm = RslRlPpolPidAlgorithmCfg(
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
        # PPOL-PID specific parameters
        lagrangian_pid=(0.05, 0.0005, 0.1),  # (Kp, Ki, Kd)
        lambda_init=None,  # Will default to [0.1] * num_costs
        lambda_max=100.0,  # Maximum Lagrangian multiplier
        pid_scale=1.0,  # PID output scaling
        constraint_margin=0.95,  # Constraint activation margin
        use_clipped_cost_loss=True,  # Clipped cost value loss
        cost_loss_coef=1.0,  # Cost value loss coefficient
    )
    