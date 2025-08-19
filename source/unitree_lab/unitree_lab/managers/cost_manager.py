
"""Cost manager for computing cost signals for a given world."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerBase, ManagerTermBase
from .manager_term_cfg import CostTermCfg

if TYPE_CHECKING:
    from unitree_lab.envs import ManagerBasedSafeRLEnvCfg


class CostManager(ManagerBase):
    """Manager for computing cost signals for a given world.

    The cost manager computes the total cost as a sum of the weighted cost terms. The cost
    terms are parsed from a nested config class containing the cost manager's settings and cost
    terms configuration.

    The cost terms are parsed from a config class containing the manager's settings and each term's
    parameters. Each cost term should instantiate the :class:`CostTermCfg` class.

    .. note::

        The cost manager multiplies the cost term's ``weight`` with the time-step interval ``dt``
        of the environment. This is done to ensure that the computed cost terms are balanced with
        respect to the chosen time-step interval in the environment.

    """

    _env: ManagerBasedSafeRLEnvCfg
    """The environment instance."""

    def __init__(self, cfg: object, env: ManagerBasedSafeRLEnvCfg):
        """Initialize the cost manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, CostTermCfg]``).
            env: The environment instance.
        """
        self._term_names: list[str] = list()
        self._term_cfgs: list[CostTermCfg] = list()
        self._class_term_cfgs: list[CostTermCfg] = list()

        super().__init__(cfg, env)
        # prepare extra info to store individual cost term information
        self._episode_sums = dict()
        for term_name in self._term_names:
            self._episode_sums[term_name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # create buffer for managing cost per environment
        self._cost_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def __str__(self) -> str:
        """Returns: A string representation for cost manager."""
        msg = f"<CostManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Cost Terms"
        table.field_names = ["Index", "Name", "Weight", "Cost Limit"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Weight"] = "r"
        table.align["Cost Limit"] = "r"
        # add info on each term
        for index, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            table.add_row([index, name, term_cfg.weight, term_cfg.cost_limit])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg
    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active cost terms."""
        return self._term_names

    @property
    def cost_limits(self) -> list[float]:
        """Extract cost limits from all active cost terms.
        
        Returns:
            List of cost limits for active cost terms (weight > 0).
        """
        cost_limits = []
        for term_cfg in self._term_cfgs:
            if term_cfg.weight > 0.0:  # Only include active terms
                cost_limits.append(term_cfg.cost_limit)
        return cost_limits
    
    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Returns the episodic sum of individual cost terms.

        Args:
            env_ids: The environment ids for which the episodic sum of
                individual cost terms is to be returned. Defaults to all the environment ids.

        Returns:
            Dictionary of episodic sum of individual cost terms.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # store information
        extras = {}
        for key in self._episode_sums.keys():
            # store information
            # c_1 + c_2 + ... + c_n
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode Cost/" + key] = episodic_sum_avg / self._env.max_episode_length_s
            # reset episodic sum
            self._episode_sums[key][env_ids] = 0.0
        # reset all the cost terms
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        # return logged information
        return extras

    def compute(self, dt: float) -> torch.Tensor:
        """Computes the cost signal as a weighted sum of individual terms.

        This function calls each cost term managed by the class and adds them to compute the net
        cost signal. It also updates the episodic sums corresponding to individual cost terms.

        Args:
            dt: The time-step interval of the environment.

        Returns:
            The net cost signal of shape (num_envs,).
        """
        # reset computation
        self._cost_buf[:] = 0.0
        # iterate over all the cost terms
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            # update total cost
            self._cost_buf += value
            # update episodic sum
            self._episode_sums[name] += value

        return self._cost_buf

    def compute_unscaled(self) -> torch.Tensor:
        """Computes the cost signal as a weighted sum of individual terms without scaling by dt.

        This function calls each cost term managed by the class and adds them to compute the net
        cost signal. It also updates the episodic sums corresponding to individual cost terms.

        Returns:
            The net cost signal of shape (num_envs,).
        """
        # reset computation
        self._cost_buf[:] = 0.0
        # iterate over all the cost terms
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight
            # update total cost
            self._cost_buf += value
            # update episodic sum
            self._episode_sums[name] += value

        return self._cost_buf
    """
    Operations - Term settings.
    """

    def set_term_cfg(self, term_name: str, cfg: CostTermCfg):
        """Sets the configuration of the specified term into the manager.

        Args:
            term_name: The name of the cost term.
            cfg: The configuration for the cost term.

        Raises:
            ValueError: If the term name is not found.
        """
        if term_name not in self._term_names:
            raise ValueError(f"Cost term '{term_name}' not found.")
        # set the configuration
        self._term_cfgs[self._term_names.index(term_name)] = cfg

    def get_term_cfg(self, term_name: str) -> CostTermCfg:
        """Gets the configuration for the specified term.

        Args:
            term_name: The name of the cost term.

        Returns:
            The configuration of the cost term.

        Raises:
            ValueError: If the term name is not found.
        """
        if term_name not in self._term_names:
            raise ValueError(f"Cost term '{term_name}' not found.")
        # return the configuration
        return self._term_cfgs[self._term_names.index(term_name)]

    """
    Helper functions.
    """

    def _prepare_terms(self):
        """Prepares a list of cost functions."""
        # parse remaining cost terms and decimate their information
        self._term_names: list[str] = list()
        self._term_cfgs: list[CostTermCfg] = list()
        self._class_term_cfgs: list[CostTermCfg] = list()

        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the terms
        for term_name, term_cfg in cfg_items:
            # check for non config
            if term_cfg is None:
                continue
            # check for valid config type
            if not isinstance(term_cfg, CostTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type CostTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # check for valid weight type
            if not isinstance(term_cfg.weight, (float, int)):
                raise TypeError(
                    f"Weight for the term '{term_name}' is not of type float or int."
                    f" Received: '{type(term_cfg.weight)}'."
                )
            # resolve common parameters
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
            # add function to list
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            # check if the term is a class
            if isinstance(term_cfg.func, ManagerTermBase):
                self._class_term_cfgs.append(term_cfg)
