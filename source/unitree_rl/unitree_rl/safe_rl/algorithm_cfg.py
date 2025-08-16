from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl.rl_cfg import RslRlPpoAlgorithmCfg
from typing import Union
from dataclasses import MISSING


@configclass  
class RslRlP3OAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the RSL-RL P3O (Penalized PPO) algorithm."""
    
    class_name: str = "P3O"
    
    # Add P3O-specific parameters on top of PPO base
    kappa_init: Union[float, list[float]] = MISSING
    """Initial penalty factor(s)."""
    
    kappa_max: Union[float, list[float]] = MISSING  
    """Maximum penalty factor(s)."""
    
    rho: float = 1.5
    """Penalty factor multiplier (ρ > 1)."""
    
    cost_thresholds: Union[float, list[float]] = MISSING
    """Cost threshold(s) for constraints."""
    
    adaptive_penalty: bool = True
    """Whether to use adaptive penalty updates."""
    
    constraint_margin: float = 0.85
    """Constraint violation detection margin."""
    
    # Cost-specific parameters
    use_clipped_cost_loss: bool = True
    """Whether to use clipped cost value loss."""
    
    cost_loss_coef: float = 1.0
    """Coefficient for cost value loss."""
    
    constraint_delay: int = 0
    """Iterations to delay constraint enforcement."""