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

import einops
from pathlib import Path
from huggingface_hub import snapshot_download
from inference_itps.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from inference_itps.common.utils.utils import seeded_context, init_hydra_config

from scipy.special import softmax
import time
import json

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from mpl_toolkits.mplot3d import axes3d
import pickle as pl
import datetime


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



class DatasetGenerator():
    # for interactive guidance dataset collection
    def __init__(self, policy, savepath=None, alignment_strategy=None, policy_tag=None):
        self.policy = policy
        self.policy_tag = policy_tag
        self.maze_env = MazeEnv()
        self.savefile = None
        self.savepath = policy_tag + "_" + savepath
        # TODO: Save a header with policy and max steps to file
        self.xy_pred = None # numpy array
        self.scores = None # numpy array
        self.alignment_strategy = alignment_strategy
        self.batch_size = 32 
    
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
    
    def find_closest_paths(self, action, guide_pos):
        assert action.shape[2] == 2 and guide_pos.shape[1] == 2
        # print(guide_pos)
        # cdist = torch.cdist(action[:, -5:, :], guide_pos, p=2)
        guide = np.expand_dims(guide_pos, axis=0) # (1, 1, 2)
        guide = np.tile(guide, (action.shape[0], 5, 1)) # (B, pred_horizon, action_dim)
        dist = np.linalg.norm(action[:, -5:] - guide[:, :], axis=2, ord=2).mean(axis=1) # (B,)
        # print("dist", dist)
        # cdist_min, cdist_min_indices = torch.min(cdist, dim=2)
        # cdist_min1, cdist_min_indices1 = torch.min(cdist_min, dim=1)

        sort_idx = np.argsort(dist)
        dist_mask_idx = np.where(dist[sort_idx]-min(dist) < 0.2, True, False)
        
        idx = sort_idx[dist_mask_idx]
        nidx = sort_idx[~dist_mask_idx]
        
        num_close = sum(dist_mask_idx)
        
        return sort_idx, num_close
    
        
    def infer_target(self, start_loc):
        agent_hist_xy = start_loc 
        # agent_hist_xy = np.array(agent_hist_xy).reshape(1, 2) # TODO: check 
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
        
        with torch.autocast(device_type="cuda"), seeded_context(0):
            actions = self.policy.run_inference(obs_batch).cpu().numpy() # directly call the policy in order to visualize the intermediate steps
        return actions

    def gen(self, num_gen = 1000):
        if self.savepath is not None:
            self.savefile = open(self.savepath, "a+", buffering=1)
            self.trial_idx = 0
        
        # to test
        # infer and plot similairy scores vs my method 
        
        
        '''
        "trial_idx": self.trial_idx, # updated in save_trials()
        "agent_pos": self.agent_gui_pos.tolist(),
        "guide": np.array(self.draw_traj).tolist(),
        "pred_traj": pred_gui_traj.astype(int).tolist(),
        "collisions": self.collisions.tolist()
        '''
        # pick random valid start location
            # I would like to pick from exsting paths but not neccecary 
        # should I cheat end loc? 
        # I am saving generated trajectries
            # should I pick closest here to save training time
                # probably 
                
        while self.trial_idx < num_gen: 
        
            start_x = np.random.uniform(0, self.maze_env.gui_size[0])
            start_y = np.random.uniform(0, self.maze_env.gui_size[1])
            
            self.start_pos = np.array([start_x, start_y])
            
            # print("start_loc", self.start_pos)
            self.start_pos = self.maze_env.gui2xy(self.start_pos).reshape(1, 2)
            
            start_valid = self.maze_env.check_collision(self.start_pos.reshape(1, 1, 2))
            
            if start_valid: 
            
                self.xy_pred = self.infer_target(self.start_pos)

                # pick valid end point
                end_points = self.xy_pred[:, -1, :]
                # print("end_points shape", end_points.shape)
                rand_i = np.random.randint(self.xy_pred.shape[0])
                self.guide_pos = end_points[rand_i].reshape(1, 2)
                # print("guide shape", self.guide_pos.shape)

                self.sort_idx, self.num_close = self.find_closest_paths(self.xy_pred, self.guide_pos)
            
            
                self.save_trials_as_individual_traj()


    def save_trials(self):
        '''
        Save in terms of xy pos
        '''
        # b, t, _ = self.xy_pred.shape
        # xy_pred = self.xy_pred.reshape(b*t, 2)
        # pred_gui_traj = [self.xy2gui(xy) for xy in xy_pred]
        # pred_gui_traj = np.array(pred_gui_traj).reshape(b, t, 2)
        entry = {
            "trial_idx": self.trial_idx,
            "start_pos": self.start_pos.tolist(),
            "end_pos": self.guide_pos.tolist(),
            "pred_traj": self.xy_pred.tolist(),
            "sort_idx": self.sort_idx.tolist(),
            "num_close": int(self.num_close),
        }
        self.savefile.write(json.dumps(entry) + "\n")
        # print(f"Trial {self.trial_idx} saved to {self.savepath}.")
        self.trial_idx += 1
        
    def save_trials_as_individual_traj(self):
        '''
        Save in terms of xy pos
        '''
        # b, t, _ = self.xy_pred.shape
        # xy_pred = self.xy_pred.reshape(b*t, 2)
        # pred_gui_traj = [self.xy2gui(xy) for xy in xy_pred]
        # pred_gui_traj = np.array(pred_gui_traj).reshape(b, t, 2)
        for i in range(self.xy_pred.shape[0]): 
            nc = int(self.num_close) + 1
            if i in self.sort_idx[:nc]: 
                high = True
            else: 
                high = False
            
            entry = {
                "trial_idx": self.trial_idx,
                "start_pos": self.start_pos.tolist(),
                "end_pos": self.guide_pos.tolist(),
                "pred_traj": self.xy_pred[i].tolist(),
                "high": high,
            }
            self.savefile.write(json.dumps(entry) + "\n")
        print(f"Trial {self.trial_idx} saved to {self.savepath}.")
        self.trial_idx += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_gen', required=True, type=int, help="Num datatpoints to generate")
    parser.add_argument('-p', '--policy', required=True, type=str, help="Policy name")
    parser.add_argument('-s', '--savepath', type=str, default=None, help="Filename to save the drawing")

    args = parser.parse_args()
    
    alignment_strategy = 'post-hoc'
    
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
    else:
        raise NotImplementedError(f"Policy with name {args.policy} is not implemented.")
    
    if args.policy is not None:
        # Load policy
        pretrained_policy_path = Path(os.path.join(checkpoint_path, "pretrained_model"))

    policy = DiffusionPolicy.from_pretrained(pretrained_policy_path, alignment_strategy=alignment_strategy)
    policy.config.noise_scheduler_type = "DDIM"
    policy.diffusion.num_inference_steps = 10
    policy.config.n_action_steps = policy.config.horizon - policy.config.n_obs_steps + 1
    policy_tag = args.policy
    policy.cuda()
    policy.eval()
        
    interactiveMaze = DatasetGenerator(policy, args.savepath, alignment_strategy, policy_tag=policy_tag)

    interactiveMaze.gen(num_gen=args.num_gen)
        
        
    
        
    