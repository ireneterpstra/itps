# MIT License

# Copyright (c) 2024 Yanwei Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Some of this software is derived from LeRobot, which is subject to the following copyright notice:

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# Tony Z. Zhao
# and The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys, os
import numpy as np
import pygame
import random
import torch
import copy
from torch import Tensor, nn

import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import softmax

from torch.cuda.amp import GradScaler
from contextlib import nullcontext
from omegaconf import DictConfig

import einops
from pathlib import Path
from huggingface_hub import snapshot_download
from inference_itps.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from inference_itps.common.policies.act.modeling_act import ACTPolicy
from inference_itps.common.policies.rollout_wrapper import PolicyRolloutWrapper
from inference_itps.common.utils.utils import seeded_context, init_hydra_config
from inference_itps.common.policies.factory import make_policy
from inference_itps.common.policies.utils import get_device_from_parameters
from inference_itps.common.policies.policy_protocol import PolicyWithUpdate
from inference_itps.common.logger import Logger, log_output_dir
from inference_itps.common.datasets.factory import make_dataset
from inference_itps.common.utils.path_utils import perturb_traj, generate_trajs

from inference_itps.scritps.tune import TunePolicy

import time
import json

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from mpl_toolkits.mplot3d import axes3d
import pickle as pl
import datetime

def update_policy_e(
    policy,
    batch,
    optimizer,
    grad_clip_norm,
    step, 
    grad_scaler: GradScaler,
    use_amp: bool = False,
    lock=None,
    grad_accumulation_steps: int = 2
    
):
    """Returns a dictionary of items for logging."""
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        output_dict = policy.forward_e(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
        loss = output_dict["loss"]
        loss_denoise, loss_energy, loss_opt = output_dict["sub_loss"]
        
    grad_scaler.scale(loss).backward()

    # Unscale the graident of the optimzer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )


    # Gradient Acccumulation: Only update every grad_accumulation_steps 
    if (step+1)%grad_accumulation_steps == 0:
        print("acc step")
        # optimizer.step()
        # optimizer.zero_grad()
        
        # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
        with lock if lock is not None else nullcontext():
            grad_scaler.step(optimizer)
        # Updates the scale for next iteration.
        grad_scaler.update()

        optimizer.zero_grad()


        if isinstance(policy, PolicyWithUpdate):
            # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
            policy.update()

    info = {
        "loss": loss.item(),
        "loss_denoise": loss_denoise.item(), 
        "loss_energy": loss_energy.item(), 
        "loss_opt": loss_opt.item(),
        "grad_norm": float(grad_norm),
        "lr": optimizer.param_groups[0]["lr"],
        "update_s": time.perf_counter() - start_time,
        **{k: v for k, v in output_dict.items() if (k != "loss" and k != "sub_loss") },
    }
    # print(info)
    info.update({k: v for k, v in output_dict.items() if k not in info and k != "sub_loss"})
    print(info)
    return info

class MazeEnv:
    def __init__(self):
        # GUI x coord 0 -> gui_size[0] #1200
        # GUI y coord 0 
        #         |
        #         v
        #       gui_size[1] #900
        # xy is in the same coordinate system as the background
        # bkg y coord 0 -> maze_shape[1] #12
        # bkg x coord 0
        #         |
        #         v
        #       maze_shape[0] #9
        
        self.maze = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                            [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                            [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                            [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).astype(bool)
        self.gui_size = (1200, 900)
        self.fps = 10
        self.batch_size = 32        
        self.offset = 0.5 # Offset to put object in the center of the cell

        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GRAY = (128, 128, 128)
        self.agent_color = self.RED

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(self.gui_size)
        pygame.display.set_caption("Maze")
        self.clock = pygame.time.Clock()
        self.agent_gui_pos = np.array([0, 0]) # Initialize the position of the red dot
        self.running = True
        self.run_id = datetime.datetime.now().strftime('%m.%d.%H.%M.%S')

    def check_collision(self, xy_traj):
        assert xy_traj.shape[2] == 2, "Input must be a 2D array of (x, y) coordinates."
        batch_size, num_steps, _ = xy_traj.shape
        xy_traj = xy_traj.reshape(-1, 2)
        xy_traj = np.clip(xy_traj, [0, 0], [self.maze.shape[0] - 1, self.maze.shape[1] - 1])
        maze_x = np.round(xy_traj[:, 0]).astype(int)
        maze_y = np.round(xy_traj[:, 1]).astype(int)
        collisions = self.maze[maze_x, maze_y]
        collisions = collisions.reshape(batch_size, num_steps)
        return np.any(collisions, axis=1)
    
    def check_collision_i(self, xy_traj):
        assert xy_traj.shape[2] == 2, "Input must be a 2D array of (x, y) coordinates."
        batch_size, num_steps, _ = xy_traj.shape
        xy_traj = xy_traj.reshape(-1, 2)
        xy_traj = np.clip(xy_traj, [0, 0], [self.maze.shape[0] - 1, self.maze.shape[1] - 1])
        maze_x = np.round(xy_traj[:, 0]).astype(int)
        maze_y = np.round(xy_traj[:, 1]).astype(int)
        collisions = self.maze[maze_x, maze_y]
        collisions = collisions.reshape(batch_size, num_steps)
        # print(collisions)
        return collisions, np.any(collisions, axis=1)
    
    def find_first_collision_from_GUI(self, gui_traj):
        assert gui_traj.shape[1] == 2, "Input must be a 2D array"
        xy_traj = np.array([self.gui2xy(point) for point in gui_traj])
        xy_traj = np.clip(xy_traj, [0, 0], [self.maze.shape[0] - 1, self.maze.shape[1] - 1])
        maze_x = np.round(xy_traj[:, 0]).astype(int)
        maze_y = np.round(xy_traj[:, 1]).astype(int)
        collisions = self.maze[maze_x, maze_y]
        first_collision_idx = np.argmax(collisions) # find the first index of many possible collisions
        return first_collision_idx

    def blend_with_white(self, color, factor=0.5):
        white = np.array([255, 255, 255])
        blended_color = (1 - factor) * np.array(color) + factor * white
        return blended_color.astype(int)

    def report_collision_percentage(self, collisions):
        num_trajectories = collisions.shape[0]
        num_collisions = np.sum(collisions)
        collision_percentage = (num_collisions / num_trajectories) * 100
        # print(f"{num_collisions}/{num_trajectories} trajectories are in collision ({collision_percentage:.2f}%).")
        return collision_percentage

    def xy2gui(self, xy):
        xy = xy + self.offset # Adjust normalization as necessary
        x = xy[0] * self.gui_size[1] / (self.maze.shape[0])
        y = xy[1] * self.gui_size[0] / (self.maze.shape[1])
        return np.array([y, x], dtype=float)

    def gui2xy(self, gui):
        x = gui[1] / self.gui_size[1] * self.maze.shape[0] - self.offset
        y = gui[0] / self.gui_size[0] * self.maze.shape[1] - self.offset
        return np.array([x, y], dtype=float)
    
    def gui2x(self, gui):
        x = gui / self.gui_size[1] * self.maze.shape[0] - self.offset
        return x
    
    def gui2y(self, gui):
        y = gui / self.gui_size[0] * self.maze.shape[1] - self.offset
        return y

    def generate_time_color_map(self, num_steps):
        cmap = plt.get_cmap('rainbow')
        values = np.linspace(0, 1, num_steps)
        colors = cmap(values)
        return colors

    def draw_maze_background(self):
        surface = pygame.surfarray.make_surface(255 - np.swapaxes(np.repeat(self.maze[:, :, np.newaxis] * 255, 3, axis=2).astype(np.uint8), 0, 1))
        surface = pygame.transform.scale(surface, self.gui_size)
        self.screen.blit(surface, (0, 0))

    def update_screen(self, xy_pred=None, collisions=None, scores=None, keep_drawing=False, traj_in_gui_space=False):
        self.draw_maze_background()
        if xy_pred is not None:
            time_colors = self.generate_time_color_map(xy_pred.shape[1])
            if collisions is None:
                collisions = self.check_collision(xy_pred)
            # self.report_collision_percentage(collisions)
            for idx, pred in enumerate(xy_pred):
                for step_idx in range(len(pred) - 1):
                    color = (time_colors[step_idx, :3] * 255).astype(int)
                    
                    # visualize constraint violations (collisions) by tinting trajectories white
                    whiteness_factor = 0.8 if collisions[idx] else 0.0 
                    color = self.blend_with_white(color, whiteness_factor)
                    if scores is None: 
                        circle_size = 5 if collisions[idx] else 5
                    else: # when similarity scores are provided, visualizing them by changing the trajectory size
                        circle_size = int(3 + 20 * scores[idx])
                    if traj_in_gui_space:
                        start_pos = pred[step_idx]
                        end_pos = pred[step_idx + 1]
                    else:
                        start_pos = self.xy2gui(pred[step_idx])
                        end_pos = self.xy2gui(pred[step_idx + 1])
                    pygame.draw.circle(self.screen, color, start_pos, circle_size)

        pygame.draw.circle(self.screen, self.agent_color, (int(self.agent_gui_pos[0]), int(self.agent_gui_pos[1])), 20)
        if keep_drawing: # visualize the human drawing input
            for i in range(len(self.draw_traj) - 1):
                pygame.draw.line(self.screen, self.RED, self.draw_traj[i], self.draw_traj[i + 1], 10)

  
        pygame.display.flip()

    def similarity_score(self, samples, guide=None):
        # samples: (B, pred_horizon, action_dim)
        # guide: (guide_horizon, action_dim)
        if guide is None:
            return samples, None
        assert samples.shape[2] == 2 and guide.shape[1] == 2
        indices = np.linspace(0, guide.shape[0]-1, samples.shape[1], dtype=int)
        guide = np.expand_dims(guide[indices], axis=0) # (1, pred_horizon, action_dim)
        guide = np.tile(guide, (samples.shape[0], 1, 1)) # (B, pred_horizon, action_dim)
        scores = np.linalg.norm(samples[:, :] - guide[:, :], axis=2, ord=2).mean(axis=1) # (B,)
        scores = 1 - scores / (scores.max() + 1e-6) # normalize
        temperature = 20
        scores = softmax(scores*temperature)
        # normalize the score to be between 0 and 1
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        # sort the predictions based on scores, from smallest to largest, so that larger scores will be drawn on top
        sort_idx = np.argsort(scores)
        samples = samples[sort_idx]
        scores = scores[sort_idx]  
        return samples, scores
    
    def imshow3d(self, ax, array, value_direction='z', pos=0, norm=None, cmap=None):
        """
        Display a 2D array as a  color-coded 2D image embedded in 3d.

        The image will be in a plane perpendicular to the coordinate axis *value_direction*.

        Parameters
        ----------
        ax : Axes3D
            The 3D Axes to plot into.
        array : 2D numpy array
            The image values.
        value_direction : {'x', 'y', 'z'}
            The axis normal to the image plane.
        pos : float
            The numeric value on the *value_direction* axis at which the image plane is
            located.
        norm : `~matplotlib.colors.Normalize`, default: Normalize
            The normalization method used to scale scalar data. See `imshow()`.
        cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
            The Colormap instance or registered colormap name used to map scalar data
            to colors.
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

    
    
class CollisionMapper:
    def __init__(self, policy, maze, policy_tag="None"):
        super().__init__()
        self.mouse_pos = None
        self.agent_in_collision = False
        self.agent_history_xy = []
        self.policy = policy
        self.batch_size = 32    
        self.policy_tag = policy_tag
        self.coll_perc_hist = dict()
        
        self.maze = maze
        
        self.scale = 10
        self.scale_x = int((self.maze.gui_size[0]-200)/self.scale)
        self.scale_y = int((self.maze.gui_size[1]-200)/self.scale)
        
        self.coll_map = np.zeros((self.scale_y, self.scale_x))  
        self.energy_map = np.zeros_like(self.coll_map)  
        
    def infer_target(self, xy, guide=None, visualizer=None, return_energy=False):
        agent_hist_xy = xy
            
        agent_hist_xy = np.array(agent_hist_xy).reshape(1, 2)
        
        agent_hist_xy = agent_hist_xy.repeat(2, axis=0)
            
    

        obs_batch = {
            "observation.state": einops.repeat(
                torch.from_numpy(agent_hist_xy).float().cuda(), "t d -> b t d", b=self.batch_size
            )
        }
        obs_batch["observation.environment_state"] = einops.repeat(
            torch.from_numpy(agent_hist_xy).float().cuda(), "t d -> b t d", b=self.batch_size
        )
        
        if guide is not None:
            guide = torch.from_numpy(guide).float().cuda()

        with torch.autocast(device_type="cuda"), seeded_context(0):
            if return_energy:
                action, energy = self.policy.run_inference(obs_batch, guide=guide, visualizer=visualizer, return_energy=return_energy)
                a_out = action.detach().cpu().numpy()
                e_out = energy.detach().cpu().numpy()
                return a_out, e_out
            else:
                return self.policy.run_inference(obs_batch, guide=guide, visualizer=visualizer, return_energy=return_energy).cpu().numpy() # directly call the policy in order to visualize the intermediate steps

    
    def plot_coll_energy(self, coll, energy, tag="", dpi=150):
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        im = ax[0].imshow(coll, cmap='inferno_r')
        fig.colorbar(im, ax=ax[0], label='Collision Rate')
        
        im = ax[1].imshow(energy, cmap='inferno')
        fig.colorbar(im, ax=ax[1], label='Energy')
        
        fig.suptitle('Collision Energy Map ' + self.policy_tag)
        
        plt.tight_layout()
        
        # plt.show()
        
        plt.savefig(f"energy_coll_map_{self.policy_tag}_{tag}.png", dpi=dpi)
    
    def generate_collision_map(self):

        for y in range(0, self.scale_y):
            for x in range(0, self.scale_x):
 

                xi = (x+10) * self.scale
                yi = (y+10) * self.scale
                gui_xy = tuple([xi,yi])
                xy = self.maze.gui2xy(gui_xy)

                xy_pred, energy = self.infer_target(xy, return_energy=True)
                collisions = self.maze.check_collision(xy_pred)
                coll_perc = self.maze.report_collision_percentage(collisions)
                energy_c = energy.sum()
                
                # print("loc", x, y, xy)
                
                self.coll_map[y, x] = coll_perc
                self.energy_map[y, x] = energy_c # what factor do we want 
            print("y", y)
            
            if y % 5 == 0: 
                
                self.plot_coll_energy(self.coll_map, self.energy_map, tag="test")
        
        self.plot_coll_energy(self.coll_map, self.energy_map, tag="f", dpi=300)
    
    
class UnconditionalMazeTopo(MazeEnv):
    # for dragging the agent around to explore motion manifold with energy topo
    def __init__(self, policy, policy_tag=None):
        super().__init__()
        self.mouse_pos = None
        self.agent_in_collision = False
        self.agent_history_xy = []
        self.policy = policy
        self.policy_tag = policy_tag
        
    def perturb_trajectory(self, base_trajectory: Tensor):
        # make a variation where you have mutiple perts
        # print(base_trajectory.shape)
        
        num_traj = base_trajectory.shape[0]
        traj_len = base_trajectory.shape[1]
        # print("traj_len", traj_len)
                
        impulse_start = (traj_len-2) * torch.rand(num_traj).to(base_trajectory.device) # np.random.randint(0, traj_len-2) 
        impulse_end = torch.mul((traj_len-1 - impulse_start+1), torch.rand(num_traj).to(base_trajectory.device)) + impulse_start+1 # np.random.randint(impulse_start+1, traj_len-1)
        impulse_start = einops.rearrange(impulse_start, "n -> n 1")
        impulse_end = einops.rearrange(impulse_end, "n -> n 1")
        impulse_mean = (impulse_start + impulse_end)/2
        # self.gui_size = (1200, 900)
        # print("impulse start, end", impulse_start[0], impulse_end[0])
        center_index = torch.round(impulse_mean).squeeze()
        impulse_center_x = base_trajectory[torch.arange(num_traj), center_index.to(torch.int), 0]
        impulse_center_y = base_trajectory[torch.arange(num_traj), center_index.to(torch.int), 1]
        impulse_target_x = 4 * torch.rand(num_traj).to(base_trajectory.device) + impulse_center_x - 2 # np.random.uniform(-2, 2, size=(num_traj,)) # -8, 8
        impulse_target_x = einops.rearrange(impulse_target_x, "n -> n 1")
        impulse_target_y = 4 * torch.rand(num_traj).to(base_trajectory.device) + impulse_center_y - 2 # np.random.uniform(-8, 8, size=(num_traj,)) # -8, 8
        impulse_target_y = einops.rearrange(impulse_target_y, "n -> n 1")
        max_relative_dist = 1 # np.exp(-5) ~= 0.006
        # print("impulse target 1", impulse_target_x[0], impulse_target_y[0])
        
        
        kernel = torch.exp(-max_relative_dist*(torch.Tensor(range(traj_len)).to(base_trajectory.device).view(1, -1) 
                                               - impulse_mean)**2 / ((impulse_start-impulse_mean)**2))
        # print(kernel)
        perturbed = base_trajectory.clone()
        perturbed[:, :, 1] += (impulse_target_y-perturbed[:, :, 1])*kernel
        perturbed[:, :, 0] += (impulse_target_x-perturbed[:, :, 0])*kernel
        
        return perturbed
    
    def gen_perturb_energies(self, base_trajectory: Tensor, obs_batch: dict[str, Tensor]):
        perturbed_trajectory = self.perturb_trajectory(base_trajectory)
        # print(perturbed_trajectory.shape)
        action_energy = self.policy.get_energy(base_trajectory, copy.deepcopy(obs_batch))
        perturbed_energy = self.policy.get_energy(perturbed_trajectory, copy.deepcopy(obs_batch))
        return perturbed_trajectory, perturbed_energy, action_energy
        
        
    def ablate_trajectory(self, base_trajectory: Tensor, eps: float = 3):
        noise = (1+0.1*eps) * torch.randn((base_trajectory.shape[0], base_trajectory.shape[-1]), device=base_trajectory.device)
        noise_unsqueezed = noise.unsqueeze(0)
        shift = noise_unsqueezed.repeat_interleave(base_trajectory.shape[1], 0)
        shift = einops.rearrange(shift, "n b c -> b n c")
        ablated_trajectories = shift + base_trajectory
    
        return ablated_trajectories
    
    def gen_ablation_energies(self, base_trajectory: Tensor, obs_batch: dict[str, Tensor], num_eps:int = 3):
        ablated_trajectories = torch.empty((num_eps,) + base_trajectory.shape, device=base_trajectory.device)
        ablated_energies = torch.empty((num_eps,) + (base_trajectory.shape[0],1), device=base_trajectory.device)
        
        for i in range(num_eps): 
            ablated_traj = self.ablate_trajectory(base_trajectory, i+1)
            energy = self.policy.get_energy(ablated_traj, obs_batch)
            
            ablated_trajectories[i, :] = ablated_traj
            ablated_energies[i, :] = energy
        
        ablated_trajectories = einops.rearrange(ablated_trajectories, "a b ... -> (a b) ...")
        ablated_energies = einops.rearrange(ablated_energies, "a b ... -> (a b) ...")
        
        return ablated_trajectories, ablated_energies
        
        
    def generate_energy_color_map(self, energies):
        num_es = len(energies)
        cmap = plt.get_cmap('rainbow')
        energies_norm = (energies-np.min(energies))/(np.max(energies)-np.min(energies))
        # values = np.linspace(0, 1, num_es)
        # print("min - max energies", np.min(energies), np.max(energies))
        colors = cmap(energies_norm)
        return colors
        
    def update_screen_energy(self, xy_pred=None, energies=None, collisions=None, keep_drawing=False):
        self.draw_maze_background()
        print((int(self.agent_gui_pos[0]), int(self.agent_gui_pos[1])), self.gui2xy((int(self.agent_gui_pos[0]), int(self.agent_gui_pos[1])))) 
        if xy_pred is not None:
            energy_colors = self.generate_energy_color_map(energies)
            cmap = ListedColormap(["darkorange", "lightseagreen", "lawngreen", "pink"]) #"lawngreen", 
            # colors = cmap(c)
            if collisions is None:
                collisions = self.check_collision(xy_pred)
            self.report_collision_percentage(collisions)
            for idx, pred in enumerate(xy_pred):
                color = (energy_colors[idx, :3] * 255).astype(int)
                for step_idx in range(len(pred) - 1):
                    # color = (time_colors[step_idx, :3] * 255).astype(int)
                    
                    # visualize constraint violations (collisions) by tinting trajectories white
                    whiteness_factor = 0.1 if collisions[idx] else 0.0 
                    # color = self.blend_with_white(color, whiteness_factor)
                    if idx < 32: 
                        circle_size = 5
                    else: 
                        circle_size = 2
                    
                    start_pos = self.xy2gui(pred[step_idx])
                    end_pos = self.xy2gui(pred[step_idx + 1])
                    pygame.draw.circle(self.screen, color, start_pos, circle_size)

        pygame.draw.circle(self.screen, self.agent_color, (int(self.agent_gui_pos[0]), int(self.agent_gui_pos[1])), 20)
        if keep_drawing: # visualize the human drawing input
            for i in range(len(self.draw_traj) - 1):
                pygame.draw.line(self.screen, self.RED, self.draw_traj[i], self.draw_traj[i + 1], 10)

        pygame.display.flip()

    def infer_target(self, guide=None, visualizer=None, num_inc=1, return_topo=False):
        agent_hist_xy = self.agent_history_xy[-1] 
        agent_hist_xy = np.array(agent_hist_xy).reshape(1, 2)
        if self.policy_tag[:2] == 'dp':
            agent_hist_xy = agent_hist_xy.repeat(2, axis=0)

        obs_batch = {
            "observation.state": einops.repeat(
                torch.from_numpy(agent_hist_xy).float().cuda(), "t d -> b t d", b=self.batch_size
            )
        }
        obs_batch["observation.environment_state"] = einops.repeat(
            torch.from_numpy(agent_hist_xy).float().cuda(), "t d -> b t d", b=self.batch_size
        )
        
        # print("batch size", self.batch_size)
        
        if guide is not None:
            guide = torch.from_numpy(guide).float().cuda()

        with torch.autocast(device_type="cuda"), seeded_context(0):
            
            obs_i = copy.deepcopy(obs_batch)
                        
            if num_inc > 1: 
                actions, energy_action, perturbed_trajectory, energy_perturbed = self.policy.sample_increasingly_perturbed_actions(obs_i, num_inc=2, mag_mul = 0.2, guide=guide, visualizer=visualizer) # directly call the policy in order to visualize the intermediate steps
            else: 
                actions, energy_action, perturbed_trajectory, energy_perturbed = self.policy.sample_perturbed_actions(obs_i, mag_mul = 0.6, guide=guide, visualizer=visualizer) # directly call the policy in order to visualize the intermediate steps
            
            a_out = actions.detach().cpu().numpy()
            e_out = np.squeeze(energy_action.detach().cpu().numpy())
            abl_out = perturbed_trajectory.detach().cpu().numpy()
            abl_energies = np.squeeze(energy_perturbed.detach().cpu().numpy(), axis=-1)

            # all_pert = np.vstack((abl_out1, abl_out2, abl_out3))
            # energies = np.concatenate((abl_energies1, abl_energies2, abl_energies3))
        
            # print(a_out.shape, e_out.shape, abl_out.shape, abl_energies.shape)
            return a_out, e_out, abl_out, abl_energies

    def plot_energies(self, xy, energies, num_inc=1, collisions=None, name=""):
        num_traj = int(len(energies)/(num_inc + 1))
        if collisions is None:
            collisions, traj_collisions = self.check_collision_i(xy)
        print(collisions)
        print(collisions.shape, xy.shape)
        print(traj_collisions)
        print(xy[collisions].shape, xy[collisions])
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        maze = np.swapaxes(self.maze, 0, 1)
        print(maze.shape)
        self.imshow3d(ax, maze, cmap="binary")
        # X, Y, Z = axes3d.get_test_data(0.05)
        X = xy[:, :, 0]
        Y = xy[:, :, 1]
        # print(X)
        # print(Y)
        # energies = np.array([0,0,0,3,3,3])
        energies = einops.rearrange(energies, "n -> 1 n")
        # energy_colors = self.generate_energy_color_map(energies)
        
        Z = np.repeat(energies.T, xy.shape[1], axis=1)
        # print(energies)
        # print(X.shape)
        # print(Y.shape)
        # print(Z.shape)
        # cmap = plt.get/_cmap('rainbow')
        
        
        
        print("num traj, num inc", num_traj, num_inc)
        
        # colors = mpl.colormaps['tab20'].colors
        c = np.tile(np.arange(int(num_traj)), num_inc + 1)
        print(c)
        C = einops.rearrange(c, "n -> 1 n")
        C = np.repeat(C.T, xy.shape[1], axis=1)
        # print(C)
        cmap = ListedColormap(["darkorange", "lightseagreen", "lawngreen", "pink"]) #"lawngreen", 
        colors = cmap(c)
        
        # Xy = X[~collisions]
        # Yy = Y[~collisions]
        # Zy = Z[~collisions]
        # Cy = C[~collisions]
        # ax.scatter(Xy, Yy, Zy, c=Cy, cmap=cmap, s=4)
        Xn = X[collisions]
        Yn = Y[collisions]
        Zn = Z[collisions]
        
        # for i in range(int(xy.shape[0])):
        for i, color in enumerate(colors):
            print("color", color)
            # color = (energy_colors[i, :3] * 255).astype(int)
            ax.plot(X[i], Y[i], Z[i], color=color)  # Plot contour curves
            Xyi = X[i][~collisions[i]]
            Yyi = Y[i][~collisions[i]]
            Zyi = Z[i][~collisions[i]]
            ax.scatter(Xyi, Yyi, Zyi, c=color, s=5)
        
        ax.scatter(Xn, Yn, Zn, c='red', s=8, marker="X")
        
        #colors = plt.get_cmap(cmap)(self.maze)
        
        plt.show()
        
        
        # save_dir = 'plots/' + self.run_id + '/'
        # if not os.path.isdir(save_dir): 
        #     os.mkdir(save_dir) 
        # plt.savefig(save_dir + 'energy_plot_'+ name +'.png', bbox_inches='tight')
        # with open(save_dir + 'energy_plot_'+ name +'.pickle', 'wb') as f:
        #     pl.dump(fig, f)

    def update_mouse_pos(self):
        self.mouse_pos = np.array(pygame.mouse.get_pos())

    def update_agent_pos(self, new_agent_pos, history_len=1):
        self.agent_gui_pos = np.array(new_agent_pos)
        agent_xy_pos = self.gui2xy(self.agent_gui_pos)
        self.agent_in_collision = self.check_collision(agent_xy_pos.reshape(1, 1, 2))[0]
        if self.agent_in_collision:
            self.agent_color = self.blend_with_white(self.RED, 0.8)
        else:
            self.agent_color = self.RED
        self.agent_history_xy.append(agent_xy_pos)
        self.agent_history_xy = self.agent_history_xy[-history_len:]
        
    

    def run(self):
        # i = 0s
        while self.running:
            self.update_mouse_pos()
            
            num_inc = 2
            num_view = 3
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                if event.type == pygame.KEYDOWN: 
                    # press s to save the trial
                    if event.key == pygame.K_s:
                        self.plot_energies(xy, energies, num_inc=num_inc, name = str(i)) 

            self.update_agent_pos(self.mouse_pos.copy())
            xy_pred, xy_energy, xy_topo, xy_topo_energy = self.infer_target(return_topo=True, num_inc=num_inc)
            # print(xy_pred.shape, xy_energy.shape)
            # print(xy_topo.shape, xy_topo_energy.shape)
             
            i = 0
            xy_pred = xy_pred[i:i+num_view]
            xy_energy = xy_energy[i:i+num_view]
            xy_topo = xy_topo[:, i:i+num_view]
            xy_topo_energy = xy_topo_energy[:, i:i+num_view]
            
            xy_topo = einops.rearrange(xy_topo, "n b t c -> (n b) t c")
            xy_topo_energy = einops.rearrange(xy_topo_energy, "n b -> (n b)")
            
            num_inc = xy_topo.shape[0]
            
            xy = np.vstack((xy_pred, xy_topo))
            energies = np.concatenate((xy_energy, xy_topo_energy))
            
            # print(i)
            self.update_screen_energy(xy, energies)
            # if i % 20 ==0 and i > 4: 
                
            #     self.plot_energies(xy, energies)
            
            self.clock.tick(30)
            # i += 1

        pygame.quit()

class UnconditionalMaze(MazeEnv):
    # for dragging the agent around to explore motion manifold
    def __init__(self, policy, policy_tag=None):
        super().__init__()
        self.mouse_pos = None
        self.agent_in_collision = False
        self.agent_history_xy = []
        self.policy = policy
        self.policy_tag = policy_tag

    def infer_target(self, guide=None, visualizer=None):
        agent_hist_xy = self.agent_history_xy[-1] 
        agent_hist_xy = np.array(agent_hist_xy).reshape(1, 2)
        if self.policy_tag[:2] == 'dp':
            agent_hist_xy = agent_hist_xy.repeat(2, axis=0)

        obs_batch = {
            "observation.state": einops.repeat(
                torch.from_numpy(agent_hist_xy).float().cuda(), "t d -> b t d", b=self.batch_size
            )
        }
        obs_batch["observation.environment_state"] = einops.repeat(
            torch.from_numpy(agent_hist_xy).float().cuda(), "t d -> b t d", b=self.batch_size
        )
        
        if guide is not None:
            guide = torch.from_numpy(guide).float().cuda()

        with torch.autocast(device_type="cuda"), seeded_context(0):
            if self.policy_tag == 'act':
                actions = self.policy.run_inference(obs_batch).cpu().numpy()
            else:
                actions = self.policy.run_inference(obs_batch, guide=guide, visualizer=visualizer).cpu().numpy() # directly call the policy in order to visualize the intermediate steps
        return actions

    def update_mouse_pos(self):
        self.mouse_pos = np.array(pygame.mouse.get_pos())

    def update_agent_pos(self, new_agent_pos, history_len=1):
        self.agent_gui_pos = np.array(new_agent_pos)
        agent_xy_pos = self.gui2xy(self.agent_gui_pos)
        self.agent_in_collision = self.check_collision(agent_xy_pos.reshape(1, 1, 2))[0]
        if self.agent_in_collision:
            self.agent_color = self.blend_with_white(self.RED, 0.8)
        else:
            self.agent_color = self.RED   
        # print("agent_xy_pos", agent_xy_pos)     
        self.agent_history_xy.append(agent_xy_pos)
        self.agent_history_xy = self.agent_history_xy[-history_len:]

    def run(self):
        while self.running:
            self.update_mouse_pos()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break

            self.update_agent_pos(self.mouse_pos.copy())
            xy_pred = self.infer_target()
            self.update_screen(xy_pred)
            self.clock.tick(30)

        pygame.quit()
        
        
class FinetuneEnergyMaze(UnconditionalMaze):
    # for interactive guidance dataset collection
    def __init__(self, policy, savepath=None, policy_tag=None):
        super().__init__(policy, policy_tag=policy_tag)
        self.drawing = False
        self.keep_drawing = False
        #Add back in vis if neccesary
        # self.vis_dp_dynamics = vis_dp_dynamics
        self.savefile = None
        self.savepath = savepath 
        self.draw_traj = [] # gui coordinates
        self.xy_pred = None # numpy array
        self.collisions = None # boolean array
        self.scores = None # numpy array
        # self.alignment_strategy = alignment_strategy
        self.drawmode = False
        self.finetune = False
        self.picking_start_pos = True
        
        self.max_e = -np.inf
        self.min_e = np.inf
        
        
        self.tuner = TunePolicy(policy)
        
        # self.coll_mapper = CollisionMapper(self.policy, self, policy_tag="tuned_" + self.policy_tag)
        
        # self.logger = Logger(cfg, out_dir, wandb_job_name=job_name)
        
    def gen_obs_batch(self):
        agent_hist_xy = self.agent_history_xy[-1] 
        # print("gen obs agent_hist_xy", agent_hist_xy)
        agent_hist_xy = np.array(agent_hist_xy).reshape(1, 2)
        if self.policy_tag[:2] == 'dp':
            agent_hist_xy = agent_hist_xy.repeat(2, axis=0)

        obs_batch = {
            "observation.state": einops.repeat(
                torch.from_numpy(agent_hist_xy).float().cuda(), "t d -> b t d", b=self.batch_size
            )
        }
        obs_batch["observation.environment_state"] = einops.repeat(
            torch.from_numpy(agent_hist_xy).float().cuda(), "t d -> b t d", b=self.batch_size
        )
        # print("obs_batch 0", obs_batch["observation.state"][-1][0])
        return obs_batch
        
        
    def generate_energy_color_map(self, energies, global_range=False):
        num_es = len(energies)
        
        cmap = plt.get_cmap('rainbow')
        if global_range: 
            energies_norm = (energies-self.min_e)/(self.max_e-self.min_e)
        else: 
            energies_norm = (energies-np.min(energies))/(np.max(energies)-np.min(energies))
        # energisces_norm = (energies-self.min_e)/(self.max_e-self.min_e)

        # print("min norm", np.argmin(energies_norm))
        # values = np.linspace(0, 1, num_es)
        # print("min - max energies", np.min(energies), np.max(energies))
        colors = cmap(energies_norm)
        # print("colors", colors)
        return colors
    
    def update_screen_energy(self, xy_pred=None, energies=None, collisions=None, keep_drawing=False):
        self.draw_maze_background()
        # print((int(self.agent_gui_pos[0]), int(self.agent_gui_pos[1])), self.gui2xy((int(self.agent_gui_pos[0]), int(self.agent_gui_pos[1])))) 
        if xy_pred is not None:
            # print(energies.shape)
            energy_colors = self.generate_energy_color_map(energies.squeeze())
            # print("min energies", np.argmin(energies))
            time_colors = self.generate_time_color_map(xy_pred.shape[1])
            cmap = ListedColormap(["darkorange", "lightseagreen", "lawngreen", "pink"]) #"lawngreen", 
            # colors = cmap(c)
            if collisions is None:
                collisions = self.check_collision(xy_pred)
            self.report_collision_percentage(collisions)
            for idx, pred in enumerate(xy_pred):
                # print(energy_colors)
                color = (energy_colors[idx, :3] * 255).astype(int)
                
                for step_idx in range(len(pred) - 1):
                    # color = (time_colors[step_idx, :3] * 255).astype(int)
                    # color = (time_colors[step_idx, :3] * 255).astype(int)
                    
                    # visualize constraint violations (collisions) by tinting trajectories white
                    whiteness_factor = 0.1 if collisions[idx] else 0.0 
                    # color = self.blend_with_white(color, whiteness_factor)
                    if idx < 32: 
                        circle_size = 5
                    else: 
                        circle_size = 2
                        
                    # print("color", color)
                    
                    start_pos = self.xy2gui(pred[step_idx])
                    end_pos = self.xy2gui(pred[step_idx + 1])
                    pygame.draw.circle(self.screen, color, start_pos, circle_size)

        pygame.draw.circle(self.screen, self.agent_color, (int(self.agent_gui_pos[0]), int(self.agent_gui_pos[1])), 20)
        if keep_drawing: # visualize the human drawing input
            for i in range(len(self.draw_traj) - 1):
                pygame.draw.line(self.screen, self.RED, self.draw_traj[i], self.draw_traj[i + 1], 10)

        pygame.display.flip()
    
        
    def infer_target(self, guide=None):
        obs_batch = self.gen_obs_batch()

        with torch.autocast(device_type="cuda"), seeded_context(0):
            actions, energy = self.policy.run_inference(copy.deepcopy(obs_batch), guide=guide, return_energy=True) # directly call the policy in order to visualize the intermediate steps
            actions = actions.detach().cpu().numpy()
            energy = energy.detach().cpu().numpy()
        return actions, energy, obs_batch
    
    def plot_paths(self, xy, energies, save_path="", start_loc=None, guide_loc=None, dpi=150, ):
        X = xy[:, :, 0]
        Y = xy[:, :, 1]
        
        energies = einops.rearrange(energies, "n 1 -> 1 n")
        energy_colors = self.generate_energy_color_map(energies.squeeze())
        energy_colors_glob = self.generate_energy_color_map(energies.squeeze(), global_range=True)
        # print(xy.shape)
        # print(energies.shape)
        # print(energy_colors.shape)
        Z = np.repeat(energies.T, xy.shape[1], axis=1)
        
        maze = np.swapaxes(self.maze, 0, 1)
        
        
        
        # Plot flat 
        fig, ax = plt.subplots(1, 2, figsize=(15, 9))
        im0 = ax[0].imshow(maze, cmap='binary')
        im1 = ax[1].imshow(maze, cmap='binary')
        # self.imshow3d(ax[0], maze, cmap="binary")
        # self.imshow3d(ax[1], maze, cmap="binary")
        
        for i in range(int(xy.shape[0])):
            # color = (energy_colors[i, :3] * 255).astype(int)
            color = energy_colors[i, :3]
            color_g = energy_colors_glob[i, :3]
            # print(color.shape)
            ax[0].plot(X[i], Y[i], color=color)
            ax[0].scatter(X[i], Y[i], c=color, s=5)
            ax[1].plot(X[i], Y[i], color=color_g)
            ax[1].scatter(X[i], Y[i], c=color_g, s=5)
            
            
            
        fig.suptitle(f'Energy Path Map {self.policy_tag} {start_loc},{guide_loc}', fontsize=14)
        
        ax[0].set_title("Batch Scale", fontsize=10)
        ax[1].set_title("Global Scale", fontsize=10)
        
        norm = plt.Normalize(energies.min(), energies.max())
        # cax = fig.add_axes([0.94, 0.1, 0.05, 0.75])  # [left, bottom, width 5% of figure width, height 75% of figure height]
        # cax.set_title('RF regret')
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap='rainbow'), ax=ax[0], orientation='vertical', shrink=0.7)
        
        norm = plt.Normalize(self.min_e, self.max_e)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap='rainbow'), ax=ax[1], orientation='vertical', shrink=0.7)

        
        if start_loc is not None: 
            ax[0].plot(start_loc[0], start_loc[1], color="deeppink", markersize=30, marker="P")
            ax[1].plot(start_loc[0], start_loc[1], color="fuchsia", markersize=30, marker="P")
            
        if guide_loc is not None: 
            ax[0].plot(guide_loc[0], guide_loc[1], color="lime", markersize=30, marker="X")
            ax[1].plot(guide_loc[0], guide_loc[1], color="springgreen", markersize=30, marker="X")
        
        
        plt.tight_layout()
        
        plt.savefig(f"{save_path}_energy_path_map_{self.policy_tag}.png", dpi=dpi)
        
        # plt.show()
        
        
        plt.clf()
        plt.close('all')
        
        
        
        # Plot 3D
        # fig2 = plt.figure()
        # ax2 = fig2.add_subplot(projection='3d')
        
        # self.imshow3d(ax2, maze, cmap="binary")

        # for i in range(int(xy.shape[0])):
        #     # color = (energy_colors[i, :3] * 255).astype(int)
        #     color = energy_colors[i, :3]
        #     print(color.shape)
        #     ax2.plot(X[i], Y[i], Z[i], color=color)
        #     ax2.scatter(X[i], Y[i], Z[i], c=color, s=5)
        
        # plt.show()
        
        # plt.clf()
        # plt.close('all')
        
    def update_global_energy(self):
        self.max_e = max(max(self.energy), self.max_e)
        self.min_e = min(min(self.energy), self.min_e)

    def run(self):
        '''
        States: 
            inference 
            drawing
            tuning
        
        1. select start location
            press key "c"
                enter drawmode
            
        2. select preferred path
            (future note maybe hilight closest path)
            start drawing with mouse down
            press key "f" to finish
                end drawmode
                eneter finetune
            
        3. fine tune model with low energy at preferred path
            3.a find paths closest to user preference
            3.b save paths + energies to dataset
            3.c fine tune model with new data
            
            press key "r" to reset ??
                
        
        '''
        if self.savepath is not None:
            self.savefile = open(self.savepath, "a+", buffering=1)
            self.trial_idx = 0

        key_points = [
            #row 1
            [165, 155], [445, 150], [645, 160], [860, 155], [1045, 155],
            #row 2
            [160, 340], [445, 335], [640, 355], [860, 340], [1045, 330],
            #row 3
            [150, 540], [245, 560], [445, 560], [645, 550], [840, 550], [1045, 550],
            #row 4
            [150, 750], [245, 745], [445, 760], [645, 740], [840, 730], [1045, 750],
            ]
        start_loc = key_points[np.random.randint(len(key_points))]
        guide_loc = key_points[np.random.randint(len(key_points))]

        # start_loc = self.mouse_pos.copy()
        start_loc = [640, 730]
        # guide_loc = [640, 730]
        while self.running:
            self.update_mouse_pos()
            # print(self.mouse_pos)
            # print("start pos", self.start_loc)
            # print("curr obs batch", self.obs_batch["observation.state"][-1][0])

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        self.picking_start_pos = not self.picking_start_pos
                    # if event.key == pygame.K_:
                    #     self.drawmode = True
                    if event.key == pygame.K_c:
                        self.drawmode = True
                    # if event.key == pygame.K_f:
                    #     self.drawmode = False
                    if event.key == pygame.K_r:  
                        self.drawmode = False
                        
                        self.keep_drawing = False  
                        self.draw_traj = [] 
                        if self.savepath is not None:
                            self.save_trials()
                    if event.key == pygame.K_p:
                        start_loc_p = self.agent_history_xy[-1]
                        self.plot_paths(self.xy_pred, self.energy, tag=f"p_({start_loc_p[0]},{start_loc_p[1]})", start_loc=start_loc_p)
                if self.drawmode: 
                    if any(pygame.mouse.get_pressed()):  # Check if mouse button is pressed
                        if not self.drawing:
                            self.drawing = True
                            self.draw_traj = []
                        self.draw_traj.append(self.mouse_pos)
                    else: # mouse released
                        if self.drawing: 
                            self.drawing = False # finish drawing action
                            self.keep_drawing = True # keep visualizing the drawing    
                            self.finetune = True                        
            if self.finetune: 
                # print("finetune")
                if len(self.draw_traj) > 0:
                    guide = np.array([self.gui2xy(point) for point in self.draw_traj])
                    guide_f = guide[-1]
                    # self.tuner.tune_energy_with_guide(self.xy_pred, self.energy, guide, self.obs_batch)
                    # print("obs plugged in", self.obs_batch["observation.state"][-1][0])
                    start_loc_h = self.agent_history_xy[-1]
                    self.plot_paths(self.xy_pred, self.energy, tag=f"start_({start_loc_h[0]},{start_loc_h[1]})", start_loc=start_loc_h)
                    self.xy_pred, self.energy = self.tuner.tune_energy(self.xy_pred, guide, self.obs_batch)
                    self.update_global_energy()
                    
                    self.plot_paths(self.xy_pred, self.energy, tag=f"tuned_400_({start_loc_h[0]},{start_loc_h[0]})_({guide_f[0]},{guide_f[1]})", start_loc=start_loc_h, guide_loc=guide_f)
                else: 
                    raise RuntimeError("No guidance found")
                self.finetune = False
                
            elif not self.drawing and not self.drawmode: # inference mode
                # print("inference")
                if not self.keep_drawing:
                    if self.picking_start_pos: 
                        start_loc = self.mouse_pos.copy()
                    self.update_agent_pos(start_loc.copy())
                if len(self.draw_traj) > 0:
                    raise RuntimeError("Should be in tuning state")
                else:
                    guide = None
                self.xy_pred, self.energy, self.obs_batch = self.infer_target()
                self.update_global_energy()
                
                self.collisions = self.check_collision(self.xy_pred)

            print("Glob max min e", self.max_e, self.min_e)
            self.update_screen_energy(self.xy_pred, self.energy, self.collisions, (self.keep_drawing or self.drawing))
            # if self.vis_dp_dynamics and not self.drawing and self.keep_drawing:
            #     time.sleep(1)
            self.clock.tick(30)

        pygame.quit()

    def save_trials(self):
        b, t, _ = self.xy_pred.shape
        xy_pred = self.xy_pred.reshape(b*t, 2)
        pred_gui_traj = [self.xy2gui(xy) for xy in xy_pred]
        pred_gui_traj = np.array(pred_gui_traj).reshape(b, t, 2)
        entry = {
            "trial_idx": self.trial_idx,
            "agent_pos": self.agent_gui_pos.tolist(),
            "guide": np.array(self.draw_traj).tolist(),
            # "pred_traj": pred_gui_traj.astype(int).tolist(),
            # "collisions": self.collisions.tolist()
        }
        self.savefile.write(json.dumps(entry) + "\n")
        print(f"Trial {self.trial_idx} saved to {self.savepath}.")
        self.trial_idx += 1
        
class TestFinetuneEnergyMaze(FinetuneEnergyMaze):
    def __init__(self, policy, pretrained_policy_path, savepath=None, policy_tag=None):
        super().__init__(policy, policy_tag=policy_tag)
        self.drawing = False
        self.keep_drawing = False
        #Add back in vis if neccesary
        # self.vis_dp_dynamics = vis_dp_dynamics
        self.savefile = None
        self.savepath = savepath 
        self.draw_traj = [] # gui coordinates
        self.xy_pred = None # numpy array
        self.collisions = None # boolean array
        self.scores = None # numpy array
        # self.alignment_strategy = alignment_strategy
        self.drawmode = False
        self.finetune = False
        self.picking_start_pos = True
        self.pretrained_policy_path = pretrained_policy_path
        
        self.max_e = -np.inf
        self.min_e = np.inf
        
        
        # self.tuner = TunePolicy(policy)
        
    def reset_policy(self): 
        alignment_strategy = 'post-hoc'
        policy = DiffusionPolicy.from_pretrained(self.pretrained_policy_path, alignment_strategy=alignment_strategy)
        policy.config.noise_scheduler_type = "DDIM"
        policy.diffusion.num_inference_steps = 10
        policy.config.n_action_steps = policy.config.horizon - policy.config.n_obs_steps + 1
        policy_tag = 'dp_ebm'
        policy_tag_a = args.policy
        policy.cuda()
        policy.eval()
        self.policy = policy
        self.tuner.reset_policy(self.policy)
        
        
    def run(self):
        '''
        States: 
            inference 
            drawing
            tuning
        
        1. select start location
            press key "c"
                enter drawmode
            
        2. select preferred path
            (future note maybe hilight closest path)
            start drawing with mouse down
            press key "f" to finish
                end drawmode
                eneter finetune
            
        3. fine tune model with low energy at preferred path
            3.a find paths closest to user preference
            3.b save paths + energies to dataset
            3.c fine tune model with new data
            
            press key "r" to reset ??
                
        
        '''
        

        key_points = np.array([
            #row 1
            [165, 155], [445, 150], [645, 160], [860, 155], [1045, 155],
            #row 2
            [160, 340], [445, 335], [640, 355], [860, 340], [1045, 330],
            #row 3
            [150, 540], [245, 560], [445, 560], [645, 550], [840, 550], [1045, 550],
            #row 4
            [150, 750], [245, 745], [445, 760], [645, 740], [840, 730], [1045, 750],
            ])
        
        # for i in range(len(key_points)): 
        #     for j in range(len(key_points)):
        s = 4 # np.random.randint(len(key_points))
        g = 3 # np.random.randint(len(key_points))
        if s == g: 
            pass
        else: 
            start_loc = key_points[s]
            self.update_agent_pos(start_loc)
            guide_loc = key_points[g]
            
            start_xy_pos = np.round(self.gui2xy(start_loc), 2)
            guide_xy_pos = np.round(self.gui2xy(guide_loc), 2)
            
            guide = einops.repeat(guide_loc, "n -> t n", t=2)
            guide = np.array([self.gui2xy(point) for point in guide])
            
            pos_string = f"tune_plots/{self.policy_tag}_tune_plots_16000_s2/{s}_{g}_({start_xy_pos[0]},{start_xy_pos[1]})_({guide_xy_pos[0]},{guide_xy_pos[1]})/"
            os.makedirs(pos_string, exist_ok = True)

            self.xy_pred, self.energy, self.obs_batch = self.infer_target()
            print(self.obs_batch)
            self.update_global_energy()
            self.plot_paths(self.xy_pred, self.energy, save_path=pos_string + "start", start_loc=start_xy_pos, guide_loc=guide_xy_pos)
            
            self.xy_pred, self.energy, steps = self.tuner.tune_energy(self.xy_pred, guide, self.obs_batch)
            self.update_global_energy()
            self.plot_paths(self.xy_pred, self.energy, save_path= pos_string + f"fit_{steps}", start_loc=start_xy_pos, guide_loc=guide_xy_pos)

            self.xy_pred, self.energy, self.obs_batch = self.infer_target()
            self.update_global_energy()
            self.plot_paths(self.xy_pred, self.energy, save_path=pos_string + f"tuned_{steps}", start_loc=start_xy_pos, guide_loc=guide_xy_pos)
                
                # self.reset_policy()
        pygame.quit()

class ConditionalMaze(UnconditionalMaze):
    # for interactive guidance dataset collection
    def __init__(self, policy, vis_dp_dynamics=False, savepath=None, alignment_strategy=None, policy_tag=None):
        super().__init__(policy, policy_tag=policy_tag)
        self.drawing = False
        self.keep_drawing = False
        self.vis_dp_dynamics = vis_dp_dynamics
        self.savefile = None
        self.savepath = savepath
        self.draw_traj = [] # gui coordinates
        self.xy_pred = None # numpy array
        self.collisions = None # boolean array
        self.scores = None # numpy array
        self.alignment_strategy = alignment_strategy

    def run(self):
        if self.savepath is not None:
            self.savefile = open(self.savepath, "a+", buffering=1)
            self.trial_idx = 0

        while self.running:
            self.update_mouse_pos()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                if any(pygame.mouse.get_pressed()):  # Check if mouse button is pressed
                    if not self.drawing:
                        self.drawing = True
                        self.draw_traj = []
                    self.draw_traj.append(self.mouse_pos)
                else: # mouse released
                    if self.drawing: 
                        self.drawing = False # finish drawing action
                        self.keep_drawing = True # keep visualizing the drawing
                if event.type == pygame.KEYDOWN: 
                    # press s to save the trial
                    if event.key == pygame.K_s and self.savefile is not None:
                        self.save_trials()             

            if self.keep_drawing: # visualize the human drawing input
                # Check if mouse returns to the agent's location
                if np.linalg.norm(self.mouse_pos - self.agent_gui_pos) < 20:  # Threshold distance to reactivate the agent
                    self.keep_drawing = False # delete the drawing
                    self.draw_traj = []

            if not self.drawing: # inference mode
                if not self.keep_drawing:
                    self.update_agent_pos(self.mouse_pos.copy())
                if len(self.draw_traj) > 0:
                    guide = np.array([self.gui2xy(point) for point in self.draw_traj])
                else:
                    guide = None
                self.xy_pred = self.infer_target(guide, visualizer=(self if self.vis_dp_dynamics and self.keep_drawing else None))
                self.scores = None
                if self.alignment_strategy == 'post-hoc' and guide is not None:
                    xy_pred, scores = self.similarity_score(self.xy_pred, guide)
                    self.xy_pred = xy_pred
                    self.scores = scores
                self.collisions = self.check_collision(self.xy_pred)

            self.update_screen(self.xy_pred, self.collisions, self.scores, (self.keep_drawing or self.drawing))
            if self.vis_dp_dynamics and not self.drawing and self.keep_drawing:
                time.sleep(1)
            self.clock.tick(30)

        pygame.quit()

    def save_trials(self):
        b, t, _ = self.xy_pred.shape
        xy_pred = self.xy_pred.reshape(b*t, 2)
        pred_gui_traj = [self.xy2gui(xy) for xy in xy_pred]
        pred_gui_traj = np.array(pred_gui_traj).reshape(b, t, 2)
        entry = {
            "trial_idx": self.trial_idx,
            "agent_pos": self.agent_gui_pos.tolist(),
            "guide": np.array(self.draw_traj).tolist(),
            "pred_traj": pred_gui_traj.astype(int).tolist(),
            "collisions": self.collisions.tolist()
        }
        self.savefile.write(json.dumps(entry) + "\n")
        print(f"Trial {self.trial_idx} saved to {self.savepath}.")
        self.trial_idx += 1
        
class TunedConditionalMazeXY(UnconditionalMaze):
    # for interactive guidance dataset collection
    def __init__(self, policy, savepath=None, alignment_strategy=None, policy_tag=None):
        super().__init__(policy, policy_tag=policy_tag)
        self.drawing = False
        self.keep_drawing = False
        # self.vis_dp_dynamics = vis_dp_dynamics
        self.savefile = None
        self.savepath = savepath
        self.draw_traj = [] # gui coordinates
        self.xy_pred = None # numpy array
        self.collisions = None # boolean array
        self.scores = None # numpy array
        self.alignment_strategy = alignment_strategy
        
    def generate_energy_color_map(self, energies):
        num_es = len(energies)
        cmap = plt.get_cmap('rainbow')
        energies_norm = (energies-np.min(energies))/(np.max(energies)-np.min(energies))
        # values = np.linspace(0, 1, num_es)
        # print("min - max energies", np.min(energies), np.max(energies))
        colors = cmap(energies_norm)
        return colors
        
    def update_screen_energy(self, xy_pred=None, energies=None, collisions=None, keep_drawing=False):
        self.draw_maze_background()
        print((int(self.agent_gui_pos[0]), int(self.agent_gui_pos[1])), self.gui2xy((int(self.agent_gui_pos[0]), int(self.agent_gui_pos[1])))) 
        if xy_pred is not None:
            energy_colors = self.generate_energy_color_map(energies.squeeze())
            cmap = ListedColormap(["darkorange", "lightseagreen", "lawngreen", "pink"]) #"lawngreen", 
            # colors = cmap(c)
            if collisions is None:
                collisions = self.check_collision(xy_pred)
            self.report_collision_percentage(collisions)
            for idx, pred in enumerate(xy_pred):
                color = (energy_colors[idx, :3] * 255).astype(int)
                for step_idx in range(len(pred) - 1):
                    # color = (time_colors[step_idx, :3] * 255).astype(int)
                    
                    # visualize constraint violations (collisions) by tinting trajectories white
                    whiteness_factor = 0.1 if collisions[idx] else 0.0 
                    # color = self.blend_with_white(color, whiteness_factor)
                    if idx < 32: 
                        circle_size = 5
                    else: 
                        circle_size = 2
                    
                    start_pos = self.xy2gui(pred[step_idx])
                    end_pos = self.xy2gui(pred[step_idx + 1])
                    pygame.draw.circle(self.screen, color, start_pos, circle_size)

        pygame.draw.circle(self.screen, self.agent_color, (int(self.agent_gui_pos[0]), int(self.agent_gui_pos[1])), 20)
        if keep_drawing: # visualize the human drawing input
            for i in range(len(self.draw_traj) - 1):
                pygame.draw.line(self.screen, self.RED, self.draw_traj[i], self.draw_traj[i + 1], 10)

        pygame.display.flip()
        
    def infer_target(self, guide=None):
        agent_hist_xy = self.agent_history_xy[-1] 
        agent_hist_xy = np.array(agent_hist_xy).reshape(1, 2)
        
        
        if self.policy_tag[:2] == 'dp':
            agent_hist_xy_r = agent_hist_xy.repeat(2, axis=0)

        obs_batch = {
            "observation.state": einops.repeat(
                torch.from_numpy(agent_hist_xy_r).float().cuda(), "t d -> b t d", b=self.batch_size
            )
        }
        
        if guide is not None:
            # guide = torch.from_numpy(guide).float().cuda()
            end_pos = guide[-1]
            end_pos = np.array(end_pos).reshape(1, 2)
            
            # env_state = np.concatenate((agent_hist_xy, end_pos), axis=0)
            env_state = np.concatenate((end_pos, end_pos), axis=0)
            print("env_state.shape", env_state.shape)

            obs_batch["observation.environment_state"] = einops.repeat(
                torch.from_numpy(env_state).float().cuda(), "t d -> b t d", b=self.batch_size
            )
        else: 
            obs_batch["observation.environment_state"] = einops.repeat(
                torch.from_numpy(agent_hist_xy_r).float().cuda(), "t d -> b t d", b=self.batch_size
            )
            
        with torch.autocast(device_type="cuda"), seeded_context(0):
            actions, energy = self.policy.run_inference(copy.deepcopy(obs_batch), return_energy=True) # directly call the policy in order to visualize the intermediate steps
            actions = actions.detach().cpu().numpy()
            energy = energy.detach().cpu().numpy()
        return actions, energy, obs_batch
        

    def run(self):
        if self.savepath is not None:
            self.savefile = open(self.savepath, "a+", buffering=1)
            self.trial_idx = 0

        while self.running:
            self.update_mouse_pos()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                if any(pygame.mouse.get_pressed()):  # Check if mouse button is pressed
                    if not self.drawing:
                        self.drawing = True
                        self.draw_traj = []
                    self.draw_traj.append(self.mouse_pos)
                else: # mouse released
                    if self.drawing: 
                        self.drawing = False # finish drawing action
                        self.keep_drawing = True # keep visualizing the drawing
                if event.type == pygame.KEYDOWN: 
                    # press s to save the trial
                    if event.key == pygame.K_s and self.savefile is not None:
                        self.save_trials()             

            if self.keep_drawing: # visualize the human drawing input
                # Check if mouse returns to the agent's location
                if np.linalg.norm(self.mouse_pos - self.agent_gui_pos) < 20:  # Threshold distance to reactivate the agent
                    self.keep_drawing = False # delete the drawing
                    self.draw_traj = []

            if not self.drawing: # inference mode
                if not self.keep_drawing:
                    self.update_agent_pos(self.mouse_pos.copy())
                if len(self.draw_traj) > 0:
                    guide = np.array([self.gui2xy(point) for point in self.draw_traj])
                else:
                    guide = None
                self.xy_pred, self.energy, obs_batch = self.infer_target(guide)
                self.scores = None
                
                self.collisions = self.check_collision(self.xy_pred)

            # self.update_screen(self.xy_pred, self.collisions, self.scores, (self.keep_drawing or self.drawing))
            self.update_screen_energy(self.xy_pred, self.energy, self.collisions, (self.keep_drawing or self.drawing))

            self.clock.tick(30)

        pygame.quit()

    def save_trials(self):
        b, t, _ = self.xy_pred.shape
        xy_pred = self.xy_pred.reshape(b*t, 2)
        pred_gui_traj = [self.xy2gui(xy) for xy in xy_pred]
        pred_gui_traj = np.array(pred_gui_traj).reshape(b, t, 2)
        entry = {
            "trial_idx": self.trial_idx,
            "agent_pos": self.agent_gui_pos.tolist(),
            "guide": np.array(self.draw_traj).tolist(),
            "pred_traj": pred_gui_traj.astype(int).tolist(),
            "collisions": self.collisions.tolist()
        }
        self.savefile.write(json.dumps(entry) + "\n")
        print(f"Trial {self.trial_idx} saved to {self.savepath}.")
        self.trial_idx += 1

class MazeExp(ConditionalMaze):
    # for replaying the trials and benchmarking the alignment strategies
    def __init__(self, policy, vis_dp_dynamics=False, savepath=None, alignment_strategy=None, policy_tag=None, loadpath=None):
        super().__init__(policy, vis_dp_dynamics, savepath, policy_tag=policy_tag)
        # Load saved trails
        assert loadpath is not None
        with open(args.loadpath, "r", buffering=1) as file:
            file.seek(0)
            trials = [json.loads(line) for line in file]
            # set random seed and shuffle the trials
            np.random.seed(0)
            np.random.shuffle(trials)

        self.trials = trials
        self.trial_idx = 0
        # if savepath is not None:
        #     # append loadpath to the savepath as prefix
        #     self.savepath = loadpath[:-5] + '_' + policy_tag + '_' + savepath
        #     self.savefile = open(self.savepath, "a+", buffering=1)
        #     self.trial_idx = 0
        self.alignment_strategy = alignment_strategy
        print(f"Alignment strategy: {alignment_strategy}")

    def run(self):
        if self.savepath is not None:
            self.savefile = open(savepath, "w+", buffering=1)
            self.trial_idx = 0

        while self.trial_idx < len(self.trials):
            # Load the trial
            self.draw_traj = self.trials[self.trial_idx]["guide"]
            
            # skip empty trials
            if len(self.draw_traj) == 0: 
                print(f"Skipping trial {self.trial_idx} which has no guide.")
                self.trial_idx += 1
                continue
            
            # skip trials with all collisions
            first_collision_idx = self.find_first_collision_from_GUI(np.array(self.draw_traj))
            if first_collision_idx <= 0: # no collision or all collisions
                if np.array(self.trials[self.trial_idx]["collisions"]).all():
                    print(f"Skipping trial {self.trial_idx} which has all collisions.")
                    self.trial_idx += 1
                    continue

            # initialize the agent position
            if self.alignment_strategy == 'output-perturb':
                # find the location before the first collision to initialize the agent
                if first_collision_idx <= 0: # no collision or all collisions
                    perturbed_pos = self.draw_traj[20]
                else:
                    first_collision_idx = min(first_collision_idx, 20)
                    perturbed_pos = self.draw_traj[first_collision_idx - 1]
                self.update_agent_pos(perturbed_pos)
            else:
                self.update_agent_pos(self.trials[self.trial_idx]["agent_pos"])

            # infer the target based on the guide
            if self.policy is not None:
                guide = np.array([self.gui2xy(point) for point in self.draw_traj])
                self.xy_pred = self.infer_target(guide, visualizer=(self if self.vis_dp_dynamics else None))
                if self.alignment_strategy in ['output-perturb', 'post-hoc']:
                    self.xy_pred, scores = self.similarity_score(self.xy_pred, guide)
                else:
                    scores = None
                self.collisions = self.check_collision(self.xy_pred)
                self.update_screen(self.xy_pred, self.collisions, scores=scores, keep_drawing=True, traj_in_gui_space=False)
                if self.vis_dp_dynamics:
                    time.sleep(1)
                    
                # save the experiment trial
                if self.savepath is not None:
                    self.save_trials()

            # just replay the trials without inference    
            else:
                collisions = self.trials[self.trial_idx]["collisions"]
                pred_traj = np.array(self.trials[self.trial_idx]["pred_traj"])
                if self.alignment_strategy in ['output-perturb', 'post-hoc']:
                    _, scores = self.similarity_score(pred_traj, np.array(self.trials[self.trial_idx]["guide"])) # this is a hack as both pred_traj and guide are in gui space, don't use this score for absolute statistics calculation
                else:
                    scores = None
                self.update_screen(pred_traj, collisions, scores=scores, keep_drawing=True, traj_in_gui_space=True)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    assert self.savefile is None
                    if event.key == pygame.K_n and self.savefile is None: # visualization mode rather than saving mode
                        print("manual skip to the next trial")
                        self.trial_idx += 1

            self.clock.tick(10)

        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--checkpoint", type=str, help="Path to the checkpoint")
    parser.add_argument('-p', '--policy', required=True, type=str, help="Policy name")
    parser.add_argument('-cm', '--collision', action='store_true', help="Collision Mapper")
    parser.add_argument('-u', '--unconditional', action='store_true', help="Unconditional Maze")
    parser.add_argument('-ut', '--topo', action='store_true', help="Unconditional Topo Maze")
    parser.add_argument('-eg', '--energy-guide', action='store_true', help="Guide with Energy")
    parser.add_argument('-tt', '--test-tune', action='store_true', help="Test Tune with Energy")
    parser.add_argument('-op', '--output-perturb', action='store_true', help="Output perturbation")
    parser.add_argument('-ph', '--post-hoc', action='store_true', help="Post-hoc ranking")
    parser.add_argument('-bi', '--biased-initialization', action='store_true', help="Biased initialization")
    parser.add_argument('-gd', '--guided-diffusion', action='store_true', help="Guided diffusion")
    parser.add_argument('-ss', '--stochastic-sampling', action='store_true', help="Stochastic sampling")
    parser.add_argument('-v', '--vis_dp_dynamics', action='store_true', help="Visualize dynamics in DP")
    parser.add_argument('-s', '--savepath', type=str, default=None, help="Filename to save the drawing")
    parser.add_argument('-l', '--loadpath', type=str, default=None, help="Filename to load the drawing")

    args = parser.parse_args()

    # Create and load the policy
    device = torch.device("cuda")

    alignment_strategy = 'post-hoc'
    if args.post_hoc:
        alignment_strategy = 'post-hoc'
    elif args.output_perturb:
        alignment_strategy = 'output-perturb'
    elif args.biased_initialization:
        alignment_strategy = 'biased-initialization'
    elif args.guided_diffusion:
        alignment_strategy = 'guided-diffusion'
    elif args.stochastic_sampling:
        alignment_strategy = 'stochastic-sampling'

    if args.policy in ["diffusion", "dp"]:
        checkpoint_path = 'inference_itps/weights/weights_dp'
    elif args.policy in ["dp_ebm"]:
        checkpoint_path = 'inference_itps/weights/weights_maze2d_energy_dp_100k'
    elif args.policy in ["dp_ebm_n"]:
        checkpoint_path = 'inference_itps/weights/weights_maze2d_dp_ebm_p_noise_100k'
    elif args.policy in ["dp_ebm_p"]:
        checkpoint_path = 'inference_itps/weights/weights_maze2d_dp_ebm_pert_100k'
    elif args.policy in ["dp_ebm_hp"]:
        checkpoint_path = 'inference_itps/weights/weights_maze2d_dp_ebm_half_pert_100k'
    elif args.policy in ["dp_ebm_c"]:
        checkpoint_path = 'inference_itps/weights/weights_maze2d_conf_coll_100k'
    elif args.policy in ["dp_ebm_c1"]:
        checkpoint_path = 'inference_itps/weights/weights_maze2d_conf_coll_0.1_100k'
    elif args.policy in ["dp_ebm_c3"]:
        checkpoint_path = 'inference_itps/weights/weights_maze2d_conf_coll_0.3_100k'
    elif args.policy in ["dp_no_cont"]:
        checkpoint_path = 'inference_itps/weights/weights_maze2d_dp_no_contrast1_100k'
    elif args.policy in ["dp_ebm_p_frz_film"]:
        checkpoint_path = 'inference_itps/weights/weights_maze2d_dp_ebm_frz_film_100k'
    elif args.policy in ["dp_ebm_p_iden_film"]:
        checkpoint_path = 'inference_itps/weights/weights_maze2d_dp_ebm_iden_film_100k'
    elif args.policy in ["dp_ebm_p_frz_film_tuned"]:
        checkpoint_path = 'inference_itps/tune_weights/dp_ebm_p_frz_film_2000'
    elif args.policy in ["dp_ebm_p_iden_film_tuned"]:
        checkpoint_path = 'inference_itps/tune_weights/low_p_2_env/dp_ebm_p_iden_film_3000'
    elif args.policy in ["dp_ebm_p_frz_film_low_p_2_env"]:
        checkpoint_path = 'inference_itps/tune_weights/low_p_2_env/dp_ebm_p_frz_film_2000'
    elif args.policy in ["dp_ebm_p_iden_tune_wrapped"]:
        checkpoint_path = 'inference_itps/tune_weights/wrapped_env_state/dp_ebm_p_iden_film_2000'
    else:
        raise NotImplementedError(f"Policy with name {args.policy} is not implemented.")

    if args.policy is not None:
        # Load policy
        pretrained_policy_path = Path(os.path.join(checkpoint_path, "pretrained_model"))

    if args.policy in ["diffusion", "dp"]:
        policy = DiffusionPolicy.from_pretrained(pretrained_policy_path, alignment_strategy=alignment_strategy)
        policy.config.noise_scheduler_type = "DDIM"
        policy.diffusion.num_inference_steps = 10
        policy.config.n_action_steps = policy.config.horizon - policy.config.n_obs_steps + 1
        policy_tag = 'dp'
        policy_tag_a = 'dp'
        policy.cuda()
        policy.eval()
    elif "dp" in args.policy:
        policy = DiffusionPolicy.from_pretrained(pretrained_policy_path, alignment_strategy=alignment_strategy)
        policy.config.noise_scheduler_type = "DDIM"
        policy.diffusion.num_inference_steps = 10
        policy.config.n_action_steps = policy.config.horizon - policy.config.n_obs_steps + 1
        policy_tag = 'dp_ebm'
        policy_tag_a = args.policy
        policy.cuda()
        policy.eval()
    elif args.policy in ["act"]:
        policy = ACTPolicy.from_pretrained(pretrained_policy_path)
        policy_tag = 'act'
        policy.cuda()
        policy.eval()
    else:
        policy = None
        policy_tag = None
        
    if args.collision: 
        interactiveMaze = CollisionMapper(policy, policy_tag=policy_tag_a)
    elif args.topo and policy_tag in ["dp_ebm", 'dp']:
        interactiveMaze = UnconditionalMazeTopo(policy, policy_tag=policy_tag_a)
    elif args.unconditional: 
        interactiveMaze = UnconditionalMaze(policy, policy_tag=policy_tag)
    elif args.loadpath is not None:
        if args.savepath is None:
            savepath = None
        else:
            alignment_tag = 'ph'
            if alignment_strategy == 'output-perturb':
                alignment_tag = 'op'
            elif alignment_strategy == 'biased-initialization':
                alignment_tag = 'bi'
            elif alignment_strategy == 'guided-diffusion':
                alignment_tag = 'gd'
            elif alignment_strategy == 'stochastic-sampling':
                alignment_tag = 'ss'
            savepath = f"{args.loadpath[:-5]}_{policy_tag}_{alignment_tag}{args.savepath}"
        interactiveMaze = MazeExp(policy, args.vis_dp_dynamics, savepath, alignment_strategy, policy_tag=policy_tag, loadpath=args.loadpath)
    elif args.energy_guide:
        print("eg")
        interactiveMaze = FinetuneEnergyMaze(policy, args.savepath, policy_tag=policy_tag)
    elif args.test_tune:
        print("tt")
        interactiveMaze = TunedConditionalMazeXY(policy, args.savepath, alignment_strategy, policy_tag=policy_tag_a)

    else:
        interactiveMaze = ConditionalMaze(policy, args.vis_dp_dynamics, args.savepath, alignment_strategy, policy_tag=policy_tag)
    interactiveMaze.run()
