
import copy
import hydra
import numpy as np
import torch
from torch import Tensor, nn
import einops
from typing import Any, Dict


MAZE_T = torch.tensor([
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)

def perturb_batch(batch):
    # print(batch)
    traj = batch["action"]
    batch_c = copy.deepcopy(batch)
    
    collisions = traj.clone()
    collisions_free = traj.clone()
    
    batch_size = traj.shape[0]
    for i in range(batch_size):
        
        collisions[i], collisions_free[i] = perturb_trajectory(traj[i])
        
    batch_c["collision"] = collisions
    batch_c["collision_free"] = collisions_free
    
    return batch_c
    
    

def perturb_trajectory(base_trajectory: Tensor) -> Tensor:
    """
    This function expects `base_trajectory` to be a single trajectory: (T, 2)
    
    num_ic (2) one coll one non coll
    
    out:
        coll: (T, 2)
        coll_free: (T, 2)
    """
    
    # Multiple magnitudes of same pert
    base_trajectory = base_trajectory.unsqueeze(0)
    collision = base_trajectory.clone().unsqueeze(0)
    collision_free = base_trajectory.clone().unsqueeze(0)
    # perturbed_r = base_trajectory.clone().unsqueeze(0)
    # perturbed_r = perturbed_r.repeat_interleave(2, dim=0)  # (N B T 2)
    
    num_traj = base_trajectory.shape[0]
    traj_len = base_trajectory.shape[1]
            
    # Calc loc of perturbs 1 per base traj (B)
    impulse_start = (traj_len-2) * torch.rand(num_traj).to(base_trajectory.device) # np.random.randint(0, traj_len-2) 
    impulse_end = torch.mul((traj_len-1 - impulse_start+1), torch.rand(num_traj).to(base_trajectory.device)) + impulse_start+1 # np.random.randint(impulse_start+1, traj_len-1)
    impulse_start = einops.rearrange(impulse_start, "n -> n 1") # (B, 1)
    impulse_end = einops.rearrange(impulse_end, "n -> n 1") # (B, 1)
    impulse_mean = (impulse_start + impulse_end)/2
    center_index = torch.round(impulse_mean).squeeze()
    impulse_center_x = base_trajectory[torch.arange(num_traj), center_index.to(torch.int), 0] # (B, 1)?
    impulse_center_y = base_trajectory[torch.arange(num_traj), center_index.to(torch.int), 1] # (B, 1)?
        
        
    # Want traj to have pert significant enough to matter so all mags are between [-1.5,-0.5] and [0.5,1.5]
    # Randomly decide array of -0.5 or 0.25
    
    # do I care about this? 
    mag = (torch.randint(0, 2, (num_traj,2))).to(base_trajectory.device) # (B, 2)
    mag = torch.where(mag == 0, torch.tensor(-1), torch.tensor(1)).to(base_trajectory.device) # (B, 2)
    
    # Generate impulse smoothing kernel
    max_relative_dist = 10 # np.exp(-5) ~= 0.006
    kernel = torch.exp(-max_relative_dist*(torch.Tensor(range(traj_len)).to(base_trajectory.device).view(1, -1) 
                                        - impulse_mean)**2 / ((impulse_start-impulse_mean)**2))
    
    # print("mag_mul", mag_mul)
    impulse_target_x = torch.randn(num_traj).to(base_trajectory.device) * 0.2 + mag[:,0] * 1
    impulse_target_y = torch.randn(num_traj).to(base_trajectory.device) * 0.2 + mag[:,1] * 1
    
    # print("impulse_target_xy", impulses_target_x[0].item(), impulse_target_y[0].item())
        
        
    magnitude = 0.1
    max_attempts = 100
        
    for i in range(max_attempts):
        perturbed = base_trajectory.clone()
        
        impulse_target_x_i = impulse_target_x.clone() * magnitude + impulse_center_x
        impulse_target_y_i = impulse_target_y.clone() * magnitude + impulse_center_y
        
        impulse_target_x_r = einops.rearrange(impulse_target_x_i, "n -> n 1")
        impulse_target_y_r = einops.rearrange(impulse_target_y_i, "n -> n 1")
        
        perturbed[:, :, 1] += (impulse_target_y_r - perturbed[:, :, 1]) * kernel
        perturbed[:, :, 0] += (impulse_target_x_r - perturbed[:, :, 0]) * kernel
        
        if not check_collision_T(perturbed):
            
            collision_free = perturbed.clone()
        else:
            # if i == 0: 
                # print("immidiate collision")
            collision = perturbed.clone()
            break  # Stop increasing perturbation if collision is detected
        
        magnitude += 0.2  # Increase perturbation magnitude incrementally
        
        # if not check_collision_T(perturbed):
        #     collision_free.append(perturbed.clone())
    
    return collision.squeeze(), collision_free.squeeze()

# def perturb_trajectory(base_trajectory: Tensor) -> Tensor:
#     """
#     TODO: move to utils
#     Take trajectory and add a detour at a random location in the traj
#     TODO: is this the right way to sample ranomly
#     """
#     # make a variation where you have mutiple perts
    
#     num_traj = base_trajectory.shape[0]
#     traj_len = base_trajectory.shape[1]
#     # print("traj_len", traj_len)
            
#     impulse_start = (traj_len-2) * torch.rand(num_traj).to(base_trajectory.device) # np.random.randint(0, traj_len-2) 
#     impulse_end = torch.mul((traj_len-1 - impulse_start+1), torch.rand(num_traj).to(base_trajectory.device)) + impulse_start+1 # np.random.randint(impulse_start+1, traj_len-1)
#     impulse_start = einops.rearrange(impulse_start, "n -> n 1")
#     impulse_end = einops.rearrange(impulse_end, "n -> n 1")
#     impulse_mean = (impulse_start + impulse_end)/2
#     # print("impulse_mean", impulse_mean)
#     # self.gui_size = (1200, 900)
#     # print("impulse start, end", impulse_start[0], impulse_end[0])
#     center_index = torch.round(impulse_mean).squeeze()
#     impulse_center_x = base_trajectory[torch.arange(num_traj), center_index.to(torch.int), 0]
#     impulse_center_y = base_trajectory[torch.arange(num_traj), center_index.to(torch.int), 1]
#     impulse_target_x = 0.2 * torch.rand(num_traj).to(base_trajectory.device) + impulse_center_x - 0.1 # np.random.uniform(-2, 2, size=(num_traj,)) # -8, 8
#     impulse_target_x = einops.rearrange(impulse_target_x, "n -> n 1")
#     impulse_target_y = 0.2 * torch.rand(num_traj).to(base_trajectory.device) + impulse_center_y - 0.1 # np.random.uniform(-8, 8, size=(num_traj,)) # -8, 8
#     impulse_target_y = einops.rearrange(impulse_target_y, "n -> n 1")
#     max_relative_dist = 1 # np.exp(-5) ~= 0.006
#     # print("impulse target 1", impulse_target_x[0], impulse_target_y[0])
    
    
#     kernel = torch.exp(-max_relative_dist*(torch.Tensor(range(traj_len)).to(base_trajectory.device).view(1, -1) 
#                                             - impulse_mean)**2 / ((impulse_start-impulse_mean)**2))
#     # print(kernel)
#     perturbed = base_trajectory.clone()
#     perturbed[:, :, 1] += (impulse_target_y-perturbed[:, :, 1])*kernel
#     perturbed[:, :, 0] += (impulse_target_x-perturbed[:, :, 0])*kernel
    
#     return perturbed


def check_collision_T(xy_traj):
    """
    This feels wrong because I am using privilaged info? Or does this not matter 
    """
    assert xy_traj.shape[2] == 2, "Input must be a 2D array of (x, y) coordinates."
    batch_size, num_steps, _ = xy_traj.shape
    xy_traj = xy_traj.reshape(-1, 2)
    # xy_traj = np.clip(xy_traj, [0, 0], [MAZE.shape[0] - 1, MAZE.shape[1] - 1])
    xy_traj = torch.clamp(xy_traj, min=torch.tensor([0, 0]), max=torch.tensor([MAZE_T.shape[0] - 1, MAZE_T.shape[1] - 1]))

    maze_x = torch.round(xy_traj[:, 0]).long()
    maze_y = torch.round(xy_traj[:, 1]).long()

    collisions = MAZE_T[maze_x, maze_y]
    collisions = collisions.view(batch_size, num_steps)
    return torch.any(collisions.bool(), dim=1)
