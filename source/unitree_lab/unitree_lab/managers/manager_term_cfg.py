from __future__ import annotations
import torch
from collections.abc import Callable
from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.managers import ManagerTermBaseCfg

@configclass
class CostTermCfg(ManagerTermBaseCfg):
    """Configuration for a cost term."""
    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.
    This function should take the environment object and any other parameters
    as input and return the cost signals as torch float tensors of
    shape (num_envs,).
    """
    weight: float = MISSING
    """The weight of the cost term.
    This is multiplied with the cost term's value to compute the final
    cost.
    Note:
        If the weight is zero, the cost term is ignored.
    """
    cost_limit: float = 0.0
    """The cost limit for this term. Defaults to 0.0 if not specified."""