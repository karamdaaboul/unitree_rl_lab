# SPDX-License-Identifier: BSD-3-Clause

import torch
from collections.abc import Sequence
from typing import Any, ClassVar

from isaacsim.core.version import get_version
from isaaclab.ui.widgets import ManagerLiveVisualizer
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv  # <-- Adjust path as needed
from unitree_lab.managers import CostManager
from unitree_lab.envs.manager_based_safe_rl_env_cfg import ManagerBasedSafeRLEnvCfg


class ManagerBasedSafeRLEnv(ManagerBasedRLEnv):
    """
    Manager-based environment that adds a CostManager for constraints / safe RL.
    Overrides only the methods where we need new logic.
    """

    is_vector_env: ClassVar[bool] = True
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }

    cfg: ManagerBasedSafeRLEnvCfg

    def __init__(self, cfg: ManagerBasedSafeRLEnvCfg, render_mode: str | None = None, **kwargs):
        print("ManagerBasedSafeRLEnv")
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def load_managers(self):
        """
        Overrides the parent to add the cost manager after super() sets up the others.
        """
        super().load_managers()

        # Add cost manager
        self.cost_manager = CostManager(self.cfg.costs, self)
        print("[INFO] Cost Manager:", self.cost_manager)

    # If your parent's `setup_manager_visualizers` doesn't know about cost_manager,
    # you can override and extend it here:
    def setup_manager_visualizers(self):
        """
        Add cost manager's live visualizer, if needed.
        """
        super().setup_manager_visualizers()
        self.manager_visualizers["cost_manager"] = ManagerLiveVisualizer(manager=self.cost_manager)

    def step(self, action: torch.Tensor):
        """
        Return a 6-tuple: (obs, reward, cost, terminated, truncated, info).
        """

        # 1) Let the parent do everything up until just before returning
        #    The parent's step typically returns a 5-tuple:
        #    (obs, reward, terminated, truncated, info).
        parent_step_output = super().step(action)
        # That is:
        # obs_buf, reward_buf, terminated_buf, truncated_buf, extras_dict

        obs_buf, reward_buf, terminated_buf, truncated_buf, extras_dict = parent_step_output

        # 2) Compute your cost if parent doesn’t already do so
        #    If your parent step calls self.cost_manager.compute(...), it’s computed.
        #    Just retrieve it from somewhere. If your parent's code doesn't do it,
        #    do it here:
        cost_buf = self.cost_manager.compute(dt=self.step_dt)
        extras_dict["cost"] = cost_buf

        # 3) Return the 6-tuple
        return obs_buf, reward_buf, terminated_buf, truncated_buf, extras_dict

    def _reset_idx(self, env_ids: Sequence[int]):
        """
        Override reset to ensure we also reset the cost manager
        if the parent method doesn't handle it.
        """
        super()._reset_idx(env_ids)
        # The parent's `_reset_idx` might NOT call cost_manager.reset(env_ids).
        # So let's do it explicitly:
        info = self.cost_manager.reset(env_ids)
        self.extras["log"].update(info)

    # We do NOT override `render` at all, because there's no difference from the parent.
    # The parent's `render(...)` method is automatically used.

    def close(self):
        if not self._is_closed:
            del self.cost_manager
            super().close()
