from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase



# Define voxel grid parameters
voxel_size_xy = 0.06  # Voxel size in the x and y dimensions
range_x = [-0.8, 0.2+1e-9]
# range_y = [-1.0 + 0.05, 1.0 + 0.05]
range_y = [-0.8, 0.8+1e-9]
range_z = [0.0, 5.0]

from collections import deque
# Create a deque with a maximum length of 10
prev_height_maps = deque(maxlen=10)

def height_map_lidar(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # global prev_height_maps

    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    hit_vec = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    hit_vec[torch.isinf(hit_vec)] = 0.0
    hit_vec[torch.isnan(hit_vec)] = 0.0
    
    hit_vec_shape = hit_vec.shape
    hit_vec = hit_vec.view(-1, hit_vec.shape[-1])
    robot_base_quat_w = env.scene["robot"].data.root_quat_w
    sensor_quat_default = torch.tensor([-0.131, 0.0, -0.991, 0.0], device=robot_base_quat_w.device).unsqueeze(0).repeat(hit_vec_shape[0], 1)
    sensor_quat_w = math_utils.quat_mul(robot_base_quat_w, sensor_quat_default)
    quat_w_dup = (sensor_quat_w.unsqueeze(1).repeat(1, hit_vec_shape[1], 1)).view(-1, sensor_quat_w.shape[-1])
    hit_vec_lidar_frame = math_utils.quat_rotate_inverse(quat_w_dup, hit_vec)
    hit_vec_lidar_frame = hit_vec_lidar_frame.view(hit_vec_shape[0], hit_vec_shape[1], hit_vec_lidar_frame.shape[-1])

    num_envs = hit_vec_lidar_frame.shape[0]

    # Calculate the number of voxels in each dimension
    x_bins = torch.arange(range_x[0], range_x[1], voxel_size_xy, device=hit_vec_lidar_frame.device)
    y_bins = torch.arange(range_y[0], range_y[1], voxel_size_xy, device=hit_vec_lidar_frame.device)

    x = hit_vec_lidar_frame[..., 0]
    y = hit_vec_lidar_frame[..., 1]
    z = hit_vec_lidar_frame[..., 2]
    
    valid_indices = (x > range_x[0]) & (x <= range_x[1]) & \
                    (y > range_y[0]) & (y <= range_y[1]) & \
                    (z >= range_z[0]) & (z <= range_z[1])

    x_filtered = x[valid_indices]
    y_filtered = y[valid_indices]
    z_filtered = z[valid_indices]

    x_indices = torch.bucketize(x_filtered, x_bins) - 1
    y_indices = torch.bucketize(y_filtered, y_bins) - 1

    env_indices = torch.arange(num_envs, device=hit_vec_lidar_frame.device).unsqueeze(1).expand_as(valid_indices)
    flat_env_indices = env_indices[valid_indices]

    map_2_5D = torch.full((num_envs, len(x_bins), len(y_bins)), float('inf'), device=hit_vec_lidar_frame.device)
    linear_indices = flat_env_indices * len(x_bins) * len(y_bins) + x_indices * len(y_bins) + y_indices

    # Subtract the offset and apply dropout
    # if torch.any(linear_indices < 0) or torch.any(linear_indices >= map_2_5D.view(-1).size(0)):
    #     print("Index out of bounds")
    #     print("linear_indices: ", linear_indices)
    #     print("map_2_5D: ", map_2_5D)
    #     import pdb; pdb.set_trace()
    # assert torch.all(linear_indices >= 0) and torch.all(linear_indices < map_2_5D.view(-1).size(0)), "Index out of bounds"
    map_2_5D = map_2_5D.view(-1).scatter_reduce_(0, linear_indices, z_filtered, reduce="amin") - offset
    map_2_5D = torch.where(map_2_5D < 0.05, torch.tensor(0.0, device=map_2_5D.device), map_2_5D)

    # # Append the cloned map to the deque
    # prev_height_maps.append(map_2_5D.clone())
    # height_maps_hist = list(prev_height_maps)

    # Calculate the maximum value for each pixel across the last ten frames
    map_2_5D = torch.where(torch.isinf(map_2_5D), torch.tensor(0.0, device=map_2_5D.device), map_2_5D)
    # Apply maximum pooling with a kernel size of 3
    # if len(map_2_5D.shape) == 2:
    #     map_2_5D = map_2_5D.unsqueeze(0)
    # import pdb; pdb.set_trace()
    map_2_5D = map_2_5D.view(num_envs, len(x_bins), len(y_bins))
    max_across_frames = F.max_pool2d(map_2_5D, kernel_size=3, stride=1, padding=1).view(num_envs, -1)

    
    # # # # import pdb; pdb.set_trace()
    # # # # # Reshape map_2_5D to 2D image
    # image = map_2_5D[0].cpu().numpy().reshape(len(x_bins), len(y_bins))

    # # # Visualization (optional)
    # image = max_across_frames[0].cpu().numpy().reshape(len(x_bins), len(y_bins))

    # # image = (image * 255).astype(int)
    # image = image.astype('uint8')

    # cv2.imshow("Height Map", image)
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()

    # output = (max_across_frames * (torch.rand(map_2_5D.shape, device=map_2_5D.device) > 0.05))

    # print("output: ", output)

    return max_across_frames