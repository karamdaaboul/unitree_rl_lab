
# Direct Integration with RSL-RL RolloutStorage
# This shows how to populate your existing RolloutStorage with offline data

import torch
import numpy as np
import json
import os
from safe_rl.storage.rollout_storage import RolloutStorage

class OfflineRolloutStoragePopulator:
    """
    Populates an existing RolloutStorage instance with offline data.
    Works directly with the RSL-RL RolloutStorage class.
    """

    def __init__(self, data_folder: str, device: str = "cpu"):
        self.data_folder = data_folder
        self.device = device
        self.meta_info = None

    def load_metadata(self):
        """Load metadata from meta.json file"""
        meta_path = os.path.join(self.data_folder, 'meta.json')
        with open(meta_path, 'r') as f:
            self.meta_info = json.load(f)
        return self.meta_info

    def convert_dtype(self, dtype_str: str) -> np.dtype:
        """Convert torch dtype string to numpy dtype"""
        if dtype_str == 'torch.float32':
            return np.float32
        elif dtype_str == 'torch.float64':
            return np.float64
        elif dtype_str == 'torch.uint8':
            return np.uint8
        else:
            return np.dtype(dtype_str)


    def load_memmap_data(self, data_key: str) -> np.memmap:
        """Load a specific memmap file"""
        file_path = os.path.join(self.data_folder, f'{data_key}.memmap')
        shape = tuple(self.meta_info[data_key]['shape'])
        dtype = self.convert_dtype(self.meta_info[data_key]['dtype'])
        return np.memmap(file_path, dtype=dtype, mode='r', shape=shape)



    def populate_rollout_storage(self, 
                               storage: 'RolloutStorage',
                               num_episodes_to_load: int = None,
                               add_synthetic_rewards: bool = True,
                               reward_value: float = 1.0) -> 'RolloutStorage':
        """
        Populate an existing RolloutStorage with offline data.

        Args:
            storage: The RolloutStorage instance to populate
            num_episodes_to_load: How many episodes to load (None = all)
            add_synthetic_rewards: Whether to add synthetic rewards
            reward_value: Value to use for synthetic rewards

        Returns:
            The populated storage instance
        """

        self.load_metadata()

        # Load all memmap data
        teacher_obs = self.load_memmap_data('teacher_observations') 
        vision_obs = self.load_memmap_data('vision_observations')
        actions_data = self.load_memmap_data('actions')

        # Get dimensions
        num_episodes, num_timesteps, _ = actions_data.shape
        
        # Limit episodes if requested
        if num_episodes_to_load is not None:
            num_episodes = min(num_episodes, num_episodes_to_load)

        # Verify storage can hold the data
        if storage.num_envs != num_episodes:
            raise ValueError(f"Storage num_envs ({storage.num_envs}) != num_episodes ({num_episodes})")
        if storage.num_transitions_per_env != num_timesteps:
            raise ValueError(f"Storage num_transitions_per_env ({storage.num_transitions_per_env}) != num_timesteps ({num_timesteps})")

        # Clear existing data
        storage.clear()

        # Populate step by step
        for step in range(num_timesteps):

            # Create Transition object (matching the RolloutStorage.Transition structure)
            transition = storage.Transition()

            # Extract data: [episodes, timestep, ...] -> [episodes, ...]
            teacher_step = teacher_obs[:num_episodes, step].copy() 
            vision_step = vision_obs[:num_episodes, step].copy()

            # Use vision observations as main observations and teacher observations as privileged
            transition.observations = torch.from_numpy(vision_step).to(self.device)
            
            if storage.privileged_observations is not None:
                transition.privileged_observations = torch.from_numpy(teacher_step).to(self.device)

            # Add actions
            actions_step = actions_data[:num_episodes, step].copy()
            transition.actions = torch.from_numpy(actions_step.copy()).to(self.device)

            # Add synthetic rewards if requested
            if add_synthetic_rewards:
                transition.rewards = torch.full((num_episodes,), reward_value, 
                                              device=self.device, dtype=torch.float32)
            else:
                transition.rewards = torch.zeros(num_episodes, device=self.device, dtype=torch.float32)

            # Add done flags (True only at the last step of each episode)
            transition.dones = torch.zeros(num_episodes, device=self.device, dtype=torch.bool)
            if step == num_timesteps - 1:  # Last timestep
                transition.dones[:] = True

            # For distillation training, add privileged actions
            if storage.training_type == "distillation":
                transition.privileged_actions = transition.actions.clone()

            # For RL training, add values, log_probs, etc.
            if storage.training_type == "rl":
                transition.values = torch.zeros(num_episodes, 1, device=self.device)
                transition.actions_log_prob = torch.zeros(num_episodes, 1, device=self.device) 
                transition.action_mean = transition.actions.clone()
                transition.action_sigma = torch.ones_like(transition.actions)
                transition.hidden_states = None

            # Add transition to storage
            storage.add_transitions(transition)

        return storage


def create_storage_from_offline_data(data_folder: str, 
                                   training_type: str = "distillation",
                                   device: str = "cpu",
                                   num_episodes_to_load: int = None) -> 'RolloutStorage':
    
    # Initialize populator
    populator = OfflineRolloutStoragePopulator(data_folder, device)
    
    # Load metadata to get dimensions
    meta = populator.load_metadata()

    # Get dimensions from metadata
    actions_shape = meta['actions']['shape']  # [episodes, timesteps, action_dim]
    num_episodes = actions_shape[0] if num_episodes_to_load is None else min(actions_shape[0], num_episodes_to_load)
    num_timesteps = actions_shape[1]
    action_dim = actions_shape[2]

    # Get observation shapes directly from metadata
    teacher_obs_shape = tuple(meta['teacher_observations']['shape'][2:])  # Remove batch and time dimensions
    vision_obs_shape = tuple(meta['vision_observations']['shape'][2:])    # Remove batch and time dimensions

    # Create RolloutStorage with proper observation shapes
    storage = RolloutStorage(
        training_type,
        num_envs=num_episodes,
        num_transitions_per_env=num_timesteps,
        obs_shape=vision_obs_shape,
        privileged_obs_shape=teacher_obs_shape,
        actions_shape=[action_dim],
        rnd_state_shape=None,
        device=device,
    )

    # Populate the storage
    populated_storage = populator.populate_rollout_storage(
        storage=storage,
        num_episodes_to_load=num_episodes_to_load,
        add_synthetic_rewards=True,
        reward_value=1.0
    )

    return populated_storage

    #return populator  # Return populator for now since we can't import RolloutStorage


# Example usage
if __name__ == "__main__":
    # Update this path to your data folder
    DATA_FOLDER = "./stored_transitions/"

    try:
        # Create and populate storage
        result = create_storage_from_offline_data(
            data_folder=DATA_FOLDER,
            training_type="distillation",  # or "rl"
            device="cpu",  # or "cuda"
            num_episodes_to_load=100  # Load only 10 episodes for testing
        )

        print("\n" + "="*50)
        print("SUCCESS: Offline dataset conversion ready!")
        print("="*50)
        print("To complete the integration:")
        print("1. Uncomment the RolloutStorage import")
        print("2. Uncomment the storage creation code")
        print("3. Run the script to get populated RolloutStorage")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
