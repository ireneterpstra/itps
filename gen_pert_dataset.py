import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from pprint import pformat
from threading import Lock
from typing import Any, Dict

import copy
import hydra
import numpy as np
import torch
from torch import Tensor, nn
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
# import matplotlib.colors as plt_colors
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import einops
from deepdiff import DeepDiff
from omegaconf import DictConfig, ListConfig, OmegaConf
from termcolor import colored
from torch import nn
from torch.cuda.amp import GradScaler
import datasets
from itps.common.datasets.factory import make_dataset, resolve_delta_timestamps
from itps.common.datasets.lerobot_dataset import MultiLeRobotDataset
from itps.common.datasets.online_buffer import OnlineBuffer, compute_sampler_weights
from itps.common.datasets.sampler import EpisodeAwareSampler
from itps.common.datasets.utils import cycle, load_hf_dataset, hf_transform_to_torch
from itps.common.utils.pert_utils import perturb_batch
from itps.common.envs.factory import make_env
from itps.common.logger import Logger, log_output_dir
from itps.common.policies.factory import make_policy
from itps.common.policies.policy_protocol import PolicyWithUpdate
from itps.common.policies.utils import get_device_from_parameters
from itps.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_hydra_config,
    init_logging,
    set_global_seed,
)
from itps.scripts.eval import eval_policy

# def get_dataset(root):
#     import h5py
#     import numpy as np
#     print(root)
#     with h5py.File(root, 'r') as hdf5_file:
#         # skip every 4th frame to match the original dataset
#         observations = np.array(hdf5_file['observations'])[:1000004][::4]
#         timeouts = np.array(hdf5_file['timeouts'])[:1000000][::4]

#     def create_episode_and_frame_indices(timeouts):
#         episode_endings = np.where(timeouts)[0]  # Indices where episodes end
#         episode_lengths = np.diff(np.concatenate(([0], episode_endings + 1)))  # Lengths of episodes
#         # Add the length of the last episode (from the last timeout to the end)
#         last_episode_length = len(timeouts) - episode_endings[-1] - 1
#         episode_lengths = np.append(episode_lengths, last_episode_length)
#         # Generate episode indices by repeating the episode number for each episode length
#         episode_index = np.repeat(np.arange(len(episode_lengths)), episode_lengths)
#         # Generate frame indices by concatenating ranges for each episode, restarting from 0
#         frame_index = np.concatenate([np.arange(length) for length in episode_lengths])
#         index = np.arange(len(timeouts))
#         return episode_index, frame_index, index
    
#     episode_index, frame_index, index = create_episode_and_frame_indices(timeouts)
#     data_dict = {
#         'observation.state': observations[:-1, :2],
#         'observation.environment_state': observations[:-1, :2],
#         'action': observations[1:, :2],
#         'episode_index': episode_index,
#         'frame_index': frame_index,
#         'timestamp': np.copy(frame_index)/10.0,
#         'index': index
#     }
#     hf_dataset = datasets.Dataset.from_dict(data_dict)
    
#     hf_dataset.set_transform(hf_transform_to_torch)
#     return hf_dataset

MAZE = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).astype(bool)

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

def perturb_trajectory(base_trajectory: Tensor, num_inc, mag_mul=0.4) -> Tensor:
    """
    This function expects `base_trajectory` to be a batch of trajectories: (B, T, 2)
    
    num_ic (N)
    
    out:
        perturbed (B*N T 2)
    """
    
    # Multiple magnitudes of same pert
    collision = base_trajectory.clone()
    collision_free = base_trajectory.clone()
    perturbed_r = base_trajectory.clone().unsqueeze(0)
    perturbed_r = perturbed_r.repeat_interleave(num_inc, dim=0)  # (N B T 2)
    
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
        
    # N amount of diff magnitudes
    # for i in range(num_inc):
        
    magnitude = 0.1
    max_attempts = 100
        
    for _ in range(max_attempts):
        perturbed = base_trajectory.clone()
        
        impulse_target_x_i = impulse_target_x.clone() * magnitude + impulse_center_x
        impulse_target_y_i = impulse_target_y.clone() * magnitude + impulse_center_y
        
        impulse_target_x_r = einops.rearrange(impulse_target_x_i, "n -> n 1")
        impulse_target_y_r = einops.rearrange(impulse_target_y_i, "n -> n 1")
        
        perturbed[:, :, 1] += (impulse_target_y_r - perturbed[:, :, 1]) * kernel
        perturbed[:, :, 0] += (impulse_target_x_r - perturbed[:, :, 0]) * kernel
        
        if not check_collision_T(perturbed):
            print("found non col")
            collision_free = perturbed.clone()
        else:
            collision = perturbed.clone()
            break  # Stop increasing perturbation if collision is detected
        
        magnitude += 0.2  # Increase perturbation magnitude incrementally
        
        # if not check_collision_T(perturbed):
        #     collision_free.append(perturbed.clone())
    # for i in range(num_inc):
    #     impulse_target_x_i = impulse_target_x.clone() * (0.6 + i * 0.3) + impulse_center_x
    #     impulse_target_y_i = impulse_target_y.clone() * (0.6 + i * 0.3) + impulse_center_y
        
        
        
    #     impulse_target_x_r = einops.rearrange(impulse_target_x_i, "n -> n 1")
    #     impulse_target_y_r = einops.rearrange(impulse_target_y_i, "n -> n 1")
        
        
    #     # print(kernel)
    #     perturbed = base_trajectory.clone()
    #     perturbed[:, :, 1] += (impulse_target_y_r-perturbed[:, :, 1])*kernel
    #     perturbed[:, :, 0] += (impulse_target_x_r-perturbed[:, :, 0])*kernel
        
    #     print(check_collision_T(perturbed))
        
    #     perturbed_r[i, :, :, :] = perturbed
        
    # perturbed_r = einops.rearrange(perturbed_r, "n b t c -> (n b) t c")
    
    return collision, collision_free

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

def check_collision(xy_traj):
    """
    This feels wrong because I am using privilaged info? Or does this not matter 
    """
    
    assert xy_traj.shape[2] == 2, "Input must be a 2D array of (x, y) coordinates."
    batch_size, num_steps, _ = xy_traj.shape
    xy_traj = xy_traj.reshape(-1, 2)
    xy_traj = np.clip(xy_traj, [0, 0], [MAZE.shape[0] - 1, MAZE.shape[1] - 1])
    maze_x = np.round(xy_traj[:, 0]).astype(int)
    maze_y = np.round(xy_traj[:, 1]).astype(int)
    collisions = MAZE[maze_x, maze_y]
    collisions = collisions.reshape(batch_size, num_steps)
    return np.any(collisions, axis=1)

def check_collision_i(xy_traj):
    assert xy_traj.shape[2] == 2, "Input must be a 2D array of (x, y) coordinates."
    batch_size, num_steps, _ = xy_traj.shape
    xy_traj = xy_traj.reshape(-1, 2)
    xy_traj = np.clip(xy_traj, [0, 0], [MAZE.shape[0] - 1, MAZE.shape[1] - 1])
    maze_x = np.round(xy_traj[:, 0]).astype(int)
    maze_y = np.round(xy_traj[:, 1]).astype(int)
    collisions = MAZE[maze_x, maze_y]
    collisions = collisions.reshape(batch_size, num_steps)
    # print(collisions)
    return collisions, np.any(collisions, axis=1)

def plot_energies(xy, num_inc=1, collisions=None):
    '''
    For debuggings
    '''
    
    def imshow3d(ax, array, value_direction='z', pos=0, norm=None, cmap=None):
        """
        """
        if norm is None:
            norm = Normalize()
        colors = plt.get_cmap(cmap)(norm(array))

        if value_direction == 'x':
            nz, ny = array.shape
            zi, yi = np.mgrid[0:nz + 1, 0:ny + 1]
            xi = np.full_like(yi, pos)
        elif value_direction == 'y':
            nx, nz = array.shape
            xi, zi = np.mgrid[0:nx + 1, 0:nz + 1]
            yi = np.full_like(zi, pos)
        elif value_direction == 'z':
            ny, nx = array.shape
            yi, xi = np.mgrid[0:ny + 1, 0:nx + 1]
            zi = np.full_like(xi, pos)
        else:
            raise ValueError(f"Invalid value_direction: {value_direction!r}")
        xi = xi - 0.5
        yi = yi - 0.5
        ax.plot_surface(xi, yi, zi, rstride=1, cstride=1, facecolors=colors, shade=False, alpha=0.5)

    
    num_traj = int(xy.shape[0]/(num_inc + 1))
    
    if collisions is None:
        collisions, traj_collisions = check_collision_i(xy)
    print(collisions)
    print(collisions.shape, xy.shape)
    print(traj_collisions)
    print(xy[collisions].shape, xy[collisions])
    
    # np_base = base_traj[0:3, :, :].detach().cpu().numpy()
    # np_pert = pert_traj[:, 0:3, :, :].detach().cpu().numpy()
    
    # np_pert = einops.rearrange(np_pert, "n b t c -> (n b) t c")
    
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    maze_s = np.swapaxes(MAZE, 0, 1)
    
    imshow3d(ax, maze_s, cmap="binary")
    
    # xy = torch.stack((base_traj[0:3], pert_traj[0:3])).detach().cpu().numpy()
    # xy = np.vstack((np_base, np_pert))
    # energies = np.array([base_e.item(), pert_e.item()])
    #  energies = np.concatenate((base_e, pert_e))

    
    X = xy[:, :, 0]
    Y = xy[:, :, 1]
    print("X", X.shape)
    print("Y", Y.shape)
    energies = np.repeat(np.arange(int(num_inc + 1)), num_traj) * 0.05 + 0.001
    # energies = np.array([1,2,3,1,2,3])
    # energies = np.ones(xy.shape[0])
    print(energies.shape)
    # 
    energies = einops.rearrange(energies, "n -> 1 n")
    
    Z = np.repeat(energies.T, xy.shape[1], axis=1)
    
    print("num traj, num inc", num_traj, num_inc)
    
    # colors = mpl.colormaps['tab20'].colors
    c = np.tile(np.arange(int(num_traj)), num_inc + 1)
    C = einops.rearrange(c, "n -> 1 n")
    C = np.repeat(C.T, xy.shape[1], axis=1)
    cmap = ListedColormap(["darkorange", "lightseagreen", "lawngreen", "pink"]) #"lawngreen", 
    colors = cmap(c)
    
    Xn = X[collisions]
    Yn = Y[collisions]
    Zn = Z[collisions]
    
    
    # for i in range(int(xy.shape[0])):
    for i, color in enumerate(colors):
        ax.plot(X[i], Y[i], Z[i], color=color)  # Plot contour curves
        Xyi = X[i][~collisions[i]]
        Yyi = Y[i][~collisions[i]]
        Zyi = Z[i][~collisions[i]]
        # ax.scatter(X[i], Y[i], Z[i], c=color, s=5)
        ax.scatter(Xyi, Yyi, Zyi, c=color, s=5)
            
        # ax.plot(X[i], Y[i], Z[i], color=color, alpha=1)  # Plot contour curves
    
    # print(C)
    # ax.scatter(X, Y, Z, c=C, cmap=cmap, s=4)
    ax.scatter(Xn, Yn, Zn, c='red', s=8, marker="X")
    
    
    plt.show()
        
def process_fn(example: Dict[str, Any], *extra_args) -> Dict[str, Any]:
    print(example)
    return example

@hydra.main(version_base="1.2", config_name="default", config_path="itps/configs/")
def proc(cfg: dict):
    print(cfg.dataset_repo_id)
    # print(cfg.dataset_repo_id)
    
    # root = './maze2d-large-sparse-v1.hdf5'
    # repo_id = 'maze2d'
    # split = "train"
    # CODEBASE_VERSION = "v1.6"
    
    # hf_dataset = load_hf_dataset(repo_id, CODEBASE_VERSION, root, split)
    
    # hf_dataset = hf_dataset.map(process_fn)  # process_fn is applied on all the examples of the dataset
    # print(hf_dataset[0])
    
    # print(hf_dataset)
    
    device = get_safe_torch_device(cfg.device, log=True)
    
    offline_dataset = make_dataset(cfg)
    
    if cfg.training.get("drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            offline_dataset.episode_data_index,
            drop_n_last_frames=cfg.training.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None
    dataloader = torch.utils.data.DataLoader(
        offline_dataset,
        num_workers=cfg.training.num_workers,
        batch_size=32,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)
    
    
    
    step = 0
    for i in range(10): #step, cfg.training.offline_steps
        batch = next(dl_iter)
        # print(batch['action'].shape)
        # gen perturbed traj for each action that 
        # print(batch)
        
        print("step", i)
        batch_p = perturb_batch(batch)
        
        trajectory = batch_p["action"]
        collision = batch_p["collision"]
        collision_free = batch_p["collision_free"]
        
        
        # b = copy.deepcopy(batch)
        
        # trajectory = batch["action"]
        
        # num_inc = 2
        num_view = 2
        
        # collision, collision_free = perturb_trajectory(trajectory, num_inc=num_inc)
        
        # # pert_traj
        
        i = 0
        xy_view = trajectory[i:i+num_view]
        coll_view = collision[i:i+num_view]
        coll_f_view = collision_free[i:i+num_view]
        # xy_topo = pert_traj[:, i:i+num_view]
        
        # xy_topo = einops.rearrange(xy_topo, "n b t c -> (n b) t c")
        
        # num_inc = xy_topo.shape[0]
        xy = np.vstack((xy_view, coll_f_view, coll_view))
        
        # collisions = check_collision(xy)
        # print("collisions", collisions)
            
        # print("pert traj shape", xy.shape)
        
        plot_energies(xy, num_inc=2)
        
        # b["collision_free"] = collision_free
        # b["collision"] = collision
        
    
    
    
    # 
    
    
    
# do I generate a dataset or do I want to pert live 
#  if I gen a dataset how do I save it? 


if __name__ == "__main__":
    proc()