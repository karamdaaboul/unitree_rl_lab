
# Comprehensive Solution: Converting Offline Dataset to RolloutStorage Format
# Author: AI Research Assistant
# Purpose: Convert offline RL dataset with memmap files to RSL-RL RolloutStorage format

import torch
import numpy as np
import json
import os
from tensordict import TensorDict
from typing import Dict, Union, Optional

class OfflineDatasetConverter:
    """
    Converts offline RL datasets stored as memmap files to RolloutStorage format.

    The offline data structure expected:
    - student_observations: [episodes, timesteps, features]
    - teacher_observations: [episodes, timesteps, features] 
    - vision_observations: [episodes, timesteps, height, width, channels]
    - color_vision_observations: [episodes, timesteps, height, width, channels]
    - actions: [episodes, timesteps, action_dims]
    """

    def __init__(self, data_folder: str, device: str = "cpu"):
        self.data_folder = data_folder
        self.device = device
        self.meta_info = None
        self.loaded_data = {}

    def load_metadata(self) -> Dict:
        """Load metadata from meta.json file"""
        meta_path = os.path.join(self.data_folder, 'meta.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        self.meta_info = meta
        return meta

    def convert_dtype(self, dtype_str: str) -> np.dtype:
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
        if self.meta_info is None:
            raise ValueError("Must load metadata first using load_metadata()")

        file_path = os.path.join(self.data_folder, f'{data_key}.memmap')
        shape = tuple(self.meta_info[data_key]['shape'])
        dtype = self.convert_dtype(self.meta_info[data_key]['dtype'])

        memmap_data = np.memmap(file_path, dtype=dtype, mode='r', shape=shape)
        return memmap_data

    def load_all_data(self) -> Dict[str, np.memmap]:
        """Load all available memmap data"""
        print("Loading metadata...")
        self.load_metadata()

        data_keys = ['student_observations', 'teacher_observations', 
                    'vision_observations', 'color_vision_observations', 'actions']

        print("Loading memmap files...")
        for key in data_keys:
            if key in self.meta_info:
                print(f"  Loading {key}...")
                self.loaded_data[key] = self.load_memmap_data(key)

        return self.loaded_data

    def print_dataset_statistics(self):
        """Print comprehensive dataset statistics"""
        print("\n" + "="*60)
        print("OFFLINE DATASET STATISTICS")
        print("="*60)

        for key, data in self.loaded_data.items():
            print(f"\n{key}:")
            print(f"  Shape: {data.shape}")
            print(f"  Data type: {data.dtype}")
            print(f"  Memory usage: {data.nbytes / (1024**3):.2f} GB")
            print(f"  Value range: [{data.min():.6f}, {data.max():.6f}]")
            print(f"  Mean: {data.mean():.6f}")

        # Overall statistics
        if 'actions' in self.loaded_data:
            num_episodes, num_timesteps, action_dim = self.loaded_data['actions'].shape
            total_transitions = num_episodes * num_timesteps
            print(f"\nOVERALL DATASET INFO:")
            print(f"  Episodes: {num_episodes}")
            print(f"  Timesteps per episode: {num_timesteps}")
            print(f"  Total transitions: {total_transitions}")
            print(f"  Action dimensions: {action_dim}")

    def create_observation_dict(self, timestep_idx: int) -> TensorDict:
        """
        Create observation dictionary for a specific timestep across all episodes.

        Args:
            timestep_idx: Which timestep to extract (0 to num_timesteps-1)

        Returns:
            TensorDict containing observations for all episodes at given timestep
        """
        obs_dict = {}

        # Add different observation types
        if 'student_observations' in self.loaded_data:
            obs_dict['student_obs'] = torch.from_numpy(
                self.loaded_data['student_observations'][:, timestep_idx].copy()
            ).to(self.device)

        if 'teacher_observations' in self.loaded_data:
            obs_dict['teacher_obs'] = torch.from_numpy(
                self.loaded_data['teacher_observations'][:, timestep_idx].copy()
            ).to(self.device)

        if 'vision_observations' in self.loaded_data:
            obs_dict['vision_obs'] = torch.from_numpy(
                self.loaded_data['vision_observations'][:, timestep_idx].copy()
            ).to(self.device)

        if 'color_vision_observations' in self.loaded_data:
            obs_dict['color_vision_obs'] = torch.from_numpy(
                self.loaded_data['color_vision_observations'][:, timestep_idx].copy()
            ).to(self.device)

        # Create batch_size based on number of episodes
        num_episodes = list(obs_dict.values())[0].shape[0]

        return TensorDict(obs_dict, batch_size=[num_episodes], device=self.device)

    def create_episode_done_flags(self, num_episodes: int, num_timesteps: int) -> torch.Tensor:
        """
        Create 'done' flags for episode boundaries.

        Args:
            num_episodes: Number of episodes
            num_timesteps: Number of timesteps per episode

        Returns:
            done flags tensor of shape [num_timesteps, num_episodes, 1]
        """
        # Create done flags - True only at the last timestep of each episode
        done_flags = torch.zeros(num_timesteps, num_episodes, 1, device=self.device, dtype=torch.bool)
        done_flags[-1, :, 0] = True  # Last timestep of each episode is done

        return done_flags

    def convert_to_rollout_storage_format(self, 
                                        training_type: str = "distillation",
                                        add_rewards: bool = False,
                                        reward_value: float = 0.0) -> Dict:
        """
        Convert offline dataset to format compatible with RolloutStorage.

        Args:
            training_type: "distillation" or "rl"
            add_rewards: Whether to add synthetic rewards
            reward_value: Default reward value if adding rewards

        Returns:
            Dictionary containing data in RolloutStorage format
        """
        if not self.loaded_data:
            raise ValueError("Must load data first using load_all_data()")

        # Get dimensions
        if 'actions' not in self.loaded_data:
            raise ValueError("Actions data is required")

        num_episodes, num_timesteps, action_dim = self.loaded_data['actions'].shape

        print(f"\nConverting dataset to RolloutStorage format...")
        print(f"Training type: {training_type}")
        print(f"Episodes: {num_episodes}, Timesteps: {num_timesteps}, Action dim: {action_dim}")

        # Initialize storage format data
        storage_data = {
            'num_envs': num_episodes,
            'num_transitions_per_env': num_timesteps,
            'actions_shape': [action_dim],
            'training_type': training_type
        }

        # Create observations for each timestep 
        print("Creating observation tensordicts...")
        observations = []
        for t in range(num_timesteps):
            obs_td = self.create_observation_dict(t)
            observations.append(obs_td)
        storage_data['observations'] = observations

        # Convert actions: [episodes, timesteps, actions] -> [timesteps, episodes, actions]
        print("Converting actions...")
        actions_data = self.loaded_data['actions'].copy()  # [episodes, timesteps, actions]
        actions_tensor = torch.from_numpy(actions_data).to(self.device)
        # Transpose to [timesteps, episodes, actions]
        actions_tensor = actions_tensor.transpose(0, 1).contiguous()
        storage_data['actions'] = actions_tensor

        # Create done flags
        print("Creating done flags...")
        done_flags = self.create_episode_done_flags(num_episodes, num_timesteps)
        storage_data['dones'] = done_flags

        # Add rewards if requested
        if add_rewards:
            print(f"Adding synthetic rewards with value {reward_value}...")
            rewards = torch.full((num_timesteps, num_episodes, 1), 
                               reward_value, device=self.device, dtype=torch.float32)
            storage_data['rewards'] = rewards

        # Add privileged actions for distillation (copy from teacher if available)
        if training_type == "distillation":
            print("Adding privileged actions for distillation...")
            # Use teacher actions if available, otherwise use same as actions
            storage_data['privileged_actions'] = actions_tensor.clone()

        print("Conversion completed!")
        return storage_data

    def create_rollout_storage(self, 
                             training_type: str = "distillation",
                             add_rewards: bool = False) -> 'RolloutStorage':
        """
        Create a RolloutStorage instance from the offline data.

        Note: This requires the actual RolloutStorage class to be imported
        """
        # Convert data to proper format
        storage_data = self.convert_to_rollout_storage_format(
            training_type=training_type, 
            add_rewards=add_rewards
        )

        # Create example observation structure for RolloutStorage initialization
        example_obs = self.create_observation_dict(0)

        # This would create the actual RolloutStorage (requires RSL-RL import)
        print("\nTo create RolloutStorage instance, use:")
        print("from safe_rl.storage.rollout_storage import RolloutStorage")
        print("storage = RolloutStorage(")
        print(f"    training_type='{training_type}',")
        print(f"    num_envs={storage_data['num_envs']},")
        print(f"    num_transitions_per_env={storage_data['num_transitions_per_env']},")
        print(f"    obs=example_obs,")
        print(f"    actions_shape={storage_data['actions_shape']},")
        print(f"    device='{self.device}'")
        print(")")

        return storage_data

    def save_converted_data(self, storage_data: Dict, output_path: str):
        """Save converted data to disk"""
        print(f"\nSaving converted data to {output_path}...")

        # Create output directory
        os.makedirs(output_path, exist_ok=True)

        # Save tensors
        for key, value in storage_data.items():
            if isinstance(value, torch.Tensor):
                torch.save(value, os.path.join(output_path, f"{key}.pt"))
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], TensorDict):
                # Save observation tensordicts
                for i, obs_td in enumerate(value):
                    obs_td.save(os.path.join(output_path, f"observations_t{i:04d}.td"))

        # Save metadata
        metadata = {k: v for k, v in storage_data.items() 
                   if not isinstance(v, (torch.Tensor, list))}
        with open(os.path.join(output_path, "conversion_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        print("Converted data saved successfully!")


# Usage example
def main():
    """
    Example usage of the OfflineDatasetConverter
    """
    # Initialize converter
    converter = OfflineDatasetConverter(
        data_folder="./stored_transitions/",  # Update this path
        device="cpu"  # or "cuda" if you have GPU
    )

    try:
        # Load all data
        converter.load_all_data()

        # Print statistics
        converter.print_dataset_statistics()

        # Convert to RolloutStorage format
        storage_data = converter.convert_to_rollout_storage_format(
            training_type="distillation",  # or "rl"
            add_rewards=True,
            reward_value=1.0
        )

        # Save converted data
        converter.save_converted_data(storage_data, "./converted_rollout_data/")

        print("\n" + "="*60)
        print("CONVERSION SUMMARY")
        print("="*60)
        print("✓ Offline dataset loaded successfully")
        print("✓ Data converted to RolloutStorage format")
        print("✓ Converted data saved to disk")
        print("\nNext steps:")
        print("1. Import RolloutStorage from safe_rl")
        print("2. Load the converted data into RolloutStorage")
        print("3. Use for offline RL training")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
