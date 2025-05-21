import torch
import time
import hydra
import copy
import os
import einops
import argparse
import numpy as np
from scipy.spatial import distance
import json


from torch.cuda.amp import GradScaler
from contextlib import nullcontext
from pathlib import Path
import matplotlib.pyplot as plt
from torch import Tensor, nn

from matplotlib import cm
from deprecated import deprecated
from torch.utils.data import Dataset, DataLoader

from inference_itps.common.policies.utils import get_device_from_parameters
from inference_itps.common.policies.policy_protocol import PolicyWithUpdate
from inference_itps.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

from inference_itps.common.datasets.utils import cycle
from inference_itps.common.utils.utils import seeded_context, init_hydra_config
from hydra import compose, initialize

from inference_itps.common.policies.policy_protocol import Policy

from inference_itps.common.datasets.sampler import EpisodeAwareSampler
from inference_itps.common.datasets.factory import make_dataset, resolve_delta_timestamps
from inference_itps.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_hydra_config,
    init_logging,
    set_global_seed,
)

from omegaconf import DictConfig, OmegaConf
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub import PyTorchModelHubMixin


def freeze_partial_policy(policy):
    # Freeze layers
    # for param in self.policy.diffusion.unet.diffusion_step_encoder.parameters():
    #     param.requires_grad = False
    # for param in self.policy.diffusion.unet.down_modules.parameters():
    #     param.requires_grad = False
    # for param in self.policy.diffusion.unet.mid_modules.parameters():
    #     param.requires_grad = False
    # action_energy_after = self.policy.get_energy(action_i, copy.deepcopy(obs_batch))
    
    
    for param in policy.parameters():
        param.requires_grad = False

    for name, layer in policy.diffusion.unet.named_children():
        # print(name)
        for name, layer in layer.named_children(): 
            # print("-", name)
            for name, layer in layer.named_children(): 
                # print("--", name)
                if name in ['cond_encoder']:
                    for param in layer.parameters():
                        param.requires_grad = True
                else: 
                    for name, layer in layer.named_children():
                        # print("---", name)
                        if name in ['cond_encoder']:
                            for param in layer.parameters():
                                param.requires_grad = True
                        # else: 
                        #     for name, layer in layer.named_children():
                        #         # print("----", name)
                        #         pass
                                
    for name, param in  policy.named_parameters():
        if param.requires_grad:
            print(f"Trainable layer: {name}")
                
class FilmWrapper(nn.Module, PyTorchModelHubMixin):
    def __init__(self, diffusion_model, input_shape, output_shape):
        super().__init__()
        
        self.device = get_device_from_parameters(diffusion_model)

        # Set up model
        self.diffusion_model = diffusion_model        
        # Freeze All Except Film Layers
        freeze_partial_policy(self.diffusion_model)
        
        # for name, param in  self.policy.named_parameters():
        #     if param.requires_grad:
        #         print(f"Trainable layer: {name}")
      
        # Encoder
        self.encoder = nn.Sequential(
                                nn.Linear(
                                    in_features=input_shape, out_features=128
                                ),
                                nn.ReLU(),
                                nn.Linear(
                                    in_features=128, out_features=output_shape
                                ),
                                nn.ReLU(),
        )
        
    def run_encoder(self, batch): 
        # Encoder
        env_input = einops.rearrange(batch["observation.environment_state"], 'b c t -> b (c t)')
        env_encoding = self.encoder(env_input)
        
        env_encoding = einops.rearrange(env_encoding, 'b (c t) -> b c t', c=2, t=2)
        batch["observation.environment_state"] = env_encoding
        return batch

    def forward(self, dl_batch, e_batch):        

        e_batch = self.run_encoder(e_batch)
        dl_batch = self.run_encoder(dl_batch)
        x = self.diffusion_model.forward_condition_end(dl_batch, e_batch)
      
        return x
    
    def get_energy(self, trajectories: Tensor, observation_batch: dict[str, Tensor]):
        observation_batch = self.run_encoder(observation_batch)
        return self.diffusion_model.get_energy(trajectories, observation_batch)
    
    def run_inference(self, observation_batch: dict[str, Tensor], guide: Tensor | None = None, visualizer=None, return_energy=False, high_energy_guide=False) -> Tensor:
        observation_batch = self.run_encoder(observation_batch)
        return self.diffusion_model.run_inference(observation_batch, guide, visualizer, return_energy, high_energy_guide)
    
    
    


def save_model(save_dir: Path, policy: Policy):
    """Save the weights of the Policy model using PyTorchModelHubMixin.
    """
    os.makedirs(f"{save_dir}/pretrained_model", exist_ok = True)
    policy.save_pretrained(save_dir)

class GuideDataset(Dataset):
    """Guide dataset."""

    def __init__(self, json_file, guide_len = 80, device="cuda"):
        """
        Arguments:
            json_file (string): Path to the json_file file with guides.
            guide_len (int): len of each path
        """
        
        # self.device =
        assert json_file is not None
        with open(json_file, "r", buffering=1) as file:
            file.seek(0)
            trials = [json.loads(line) for line in file]
            # set random seed and shuffle the trials
            np.random.seed(0)
            np.random.shuffle(trials)
            
        self.trials = trials
        self.trial_idx = 0
        self.guide_len = guide_len
        self.guides = np.zeros((len(trials), self.guide_len, 2))
 
        lens = []
        remove_idx = []
        for i in range(len(self.trials)):
            guide = np.array(self.trials[i]["guide"])
            lens.append(guide.shape[0])
            if guide.shape[0] < 20:
                remove_idx.append(i)
            n_missing = self.guide_len - guide.shape[0]
            
            if n_missing <= 0:
                extended_traj = guide[:self.guide_len]
            else:
                last_point = guide[-1]
                padding = np.tile(last_point, (n_missing, 1))
                extended_traj = np.vstack((guide, padding))
        
            self.trials[i]["guide"] = extended_traj
            self.trials[i]["pred_traj"] = np.array(self.trials[i]["pred_traj"])
            self.trials[i]["collisions"] = np.array(self.trials[i]["collisions"])
            self.trials[i]["agent_pos"] = np.array(self.trials[i]["agent_pos"])
        print(self.trials[0].keys())
        print(self.trials[0]["pred_traj"].shape)
        self.trials = [element for i, element in enumerate(self.trials) if i not in remove_idx]

        # self.guides = np.delete(self.guides, np.array(remove_idx), axis=0)
        print("min", min(lens))
        print("mean", sum(lens) / len(lens))
        print("max", max(lens))
        print("removed", remove_idx)
        print(len(self.trials))

    def __len__(self):
        # print(len(self.trials))
        return len(self.trials)

    def __getitem__(self, idx):
        return self.trials[idx]

class StartEndDataset(Dataset):
    """StartEnd dataset."""

    def __init__(self, json_file, device="cuda"):
        """
        Arguments:
            json_file (string): Path to the json_file file with guides.
            guide_len (int): len of each path
        """
        
        # self.device =
        assert json_file is not None
        with open(json_file, "r", buffering=1) as file:
            file.seek(0)
            trials = [json.loads(line) for line in file]
            # set random seed and shuffle the trials
            np.random.seed(0)
            np.random.shuffle(trials)
            
        self.trials = trials
        self.trial_idx = 0
        
        for i in range(len(trials)):
            for key in self.trials[i]:
                self.trials[i][key] = np.array(self.trials[i][key], dtype=np.float32)
                # print(self.trials[i]["num_close"])
            self.trials[i]["observation.state"] = self.trials[i]["start_pos"].repeat(2, axis=0)
            # what if env state is start + end?
            env_state = np.concatenate((self.trials[i]["start_pos"], self.trials[i]["end_pos"]), axis=0)
            
            self.trials[i]["observation.environment_state"] = env_state
            self.trials[i]["action"] = self.trials[i]["pred_traj"]
        print(self.trials[0]["start_pos"].shape)
        print(self.trials[0]["end_pos"].shape)
        print(self.trials[0]["observation.state"].shape)
        print(self.trials[0]["observation.environment_state"].shape)

        print(len(self.trials))

    def __len__(self):
        # print(len(self.trials))
        return len(self.trials)

    def __getitem__(self, idx):
        return self.trials[idx]


def plot_paths_torch(paths):
    xy = paths.detach().cpu().numpy()
    X = xy[:, :, 0]
    Y = xy[:, :, 1]
    
    print("path shape", paths.shape)
    # Plot flat 
    # fig, ax = plt.subplots(1, 2, figsize=(15, 9))
    fig = plt.figure(figsize=(4, 5))
    ax = fig.add_subplot()
    
    cmap = plt.get_cmap('rainbow')
    energies_norm = range(int(xy.shape[0]))

    colors = cmap(energies_norm)
    # for i in range(int(xy.shape[0])):
    #     color = colors[i, :3]
    #     # print(color.shape)
    #     ax.plot(X[i, :], Y[i], color=color)
    #     ax.scatter(X[i, :], Y[i], color=color, s=5)
        
    ax.plot(X[0], Y[0], color="red")
    ax.scatter(X[0], Y[0], color="red", s=5)
    
    plt.tight_layout()
    
    plt.show()
    plt.clf()
    plt.close('all')
    
def plot_paths(xy):
    # xy = paths.detach().cpu().numpy()
    X = xy[:, :, 0]
    Y = xy[:, :, 1]
    
    print("path shape", xy.shape)
    # Plot flat 
    # fig, ax = plt.subplots(1, 2, figsize=(15, 9))
    fig = plt.figure(figsize=(4, 5))
    ax = fig.add_subplot()
    
    cmap = plt.get_cmap('rainbow')
    energies_norm = range(int(xy.shape[0]))

    colors = cmap(energies_norm)
    # for i in range(int(xy.shape[0])):
    #     color = colors[i, :3]
    #     # print(color.shape)
    #     ax.plot(X[i, :], Y[i], color=color)
    #     ax.scatter(X[i, :], Y[i], color=color, s=5)
        
    ax.plot(X[0], Y[0], color="red")
    ax.scatter(X[0], Y[0], color="red", s=5)
    
    plt.tight_layout()
    
    plt.show()
    plt.clf()
    plt.close('all')
    
class MazeEnv:
    def __init__(self):
        self.maze = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                                    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                                    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                                    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                                    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=bool)
        self.gui_size = (1200, 900)
        self.offset = 0.5
        
        self.scale = 10
        self.scale_x = int((self.gui_size[0]-200)/self.scale)
        self.scale_y = int((self.gui_size[1]-200)/self.scale)
        
        
    def gui2xy(self, gui):
        x = gui[1] / self.gui_size[1] * self.maze.shape[0] - self.offset
        y = gui[0] / self.gui_size[0] * self.maze.shape[1] - self.offset
        return np.array([x, y], dtype=float)
    
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
        
class TunePolicy:
    """
    Tune model with guide traj as condition
    """
    def __init__(self, policy, guide_loadpath, policy_tag, save_path=None):
        
        # with initialize(version_base=None, config_path="conf", job_name="test_app"):
        #     cfg = compose(config_name="config", overrides=["db=mysql", "db.user=me"])
        #     print(OmegaConf.to_yaml(cfg))
        cfg = init_hydra_config("inference_itps/configs/policy/tune_config.yaml")
        self.policy_tag = policy_tag
        if save_path is None: 
            self.save_path = f"inference_itps/tune_weights/{policy_tag}"
        else: 
            self.save_path = f"{save_path}/{policy_tag}"
            
        # TODO: change input shape? 
        self.device = device = torch.device("cuda") #get_device_from_parameters(policy)
        print("policy device", self.device)
        self.policy = FilmWrapper(policy, input_shape=4, output_shape=4)
        self.policy.to(get_safe_torch_device(self.device))
        
        for name, param in self.policy.named_parameters():
            if param.requires_grad:
                print(f"Trainable layer: {name}")
        
        self.batch_size = 64
        self.dl_batch_size = 64
        
        self.step = 0
        
        self.log_freq = 20
    
        self.use_amp=False
        lr = 1.0e-6
        adam_betas = [0.95, 0.999]
        adam_eps = 1.0e-8
        adam_weight_decay = 1.0e-6
        # self.lr_scheduler
        # self.lr_warmup_steps
        self.grad_accumulation_steps =2
        self.grad_clip_norm = 10
        self.lock = None
        
        self.optimizer = torch.optim.Adam(
            self.policy.diffusion_model.diffusion.parameters(),
            lr,
            adam_betas,
            adam_eps,
            adam_weight_decay,
        )
        
        self.grad_scaler = GradScaler(enabled=self.use_amp)
        
        #### Setup DataLoader
        offline_dataset = make_dataset(cfg)
        
        shuffle = False
        sampler = EpisodeAwareSampler(
            offline_dataset.episode_data_index,
            drop_n_last_frames=7, # cfg.training.drop_n_last_frames
            shuffle=True,
        )
        
        
        
        dataloader = DataLoader(
            offline_dataset,
            num_workers= 4, # cfg.training.num_workers
            batch_size=self.dl_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=self.device.type != "cpu",
            drop_last=False,
        )
        self.dl_iter = cycle(dataloader)
    
        # Load saved guides
        # guide_dataset = GuideDataset(guide_loadpath)
        guide_dataset = StartEndDataset(guide_loadpath)
    
        guide_dataloader = DataLoader(
            guide_dataset,
            num_workers= 4, # cfg.training.num_workers
            batch_size=self.dl_batch_size,
            shuffle=True,
            # sampler=sampler,
            pin_memory=self.device.type != "cpu",
            drop_last=False,
        )
        self.guide_dl_iter = cycle(guide_dataloader)
        

    def tune_start_end(self, max_steps=400): 
                
        
        self.step = 0
        
        # for each batch 
            # identify low and high paths
                # try variation where I only pick one path
                # rip code staright from inference
                # create dataset with slected path + non selcted
            # update with dl + guide dl
        for s in range(max_steps):
            e_batch = next(self.guide_dl_iter)
            for key in e_batch:
                # print(key)
                e_batch[key] = e_batch[key].to(self.device, non_blocking=True)
            e_batch_i = copy.deepcopy(e_batch)
            # idx, nidx = self.find_closest_paths(e_batch["pred_traj"], e_batch["guide"])
            # for b in range(self.batch_size):
            #     num_close_b = e_batch["num_close"][b] + 1
            #     idx = e_batch["sort_idx"][b][:num_close_b]
            #     nidx = e_batch["sort_idx"][b][num_close_b:]
            #     self.plot_high_low(e_batch["pred_traj"][b], idx, nidx, start_pos=e_batch["start_pos"][b], end_pos=e_batch["end_pos"][b], title=f"high_low_close_w_start_end_{b}")

            dl_batch = next(self.dl_iter)
            for key in dl_batch:
                # print(key)
                dl_batch[key] = dl_batch[key].to(self.device, non_blocking=True)
            
            train_info = self.update_policy_e(
                e_batch,
                dl_batch,
                self.step,
            )
            if (self.step) % self.log_freq == 0:
                print(self.step, train_info)
            self.step += 1
            
            # Check policy converggence    
            picked = e_batch_i["high"]
            action_energy = self.policy.get_energy(e_batch_i["action"], e_batch_i)
            energy_low = action_energy[picked.bool()] # TODO: replace with idx and nidx
            energy_high = action_energy[~picked.bool()]
            
            # if s > 4 and (self.step) % 10 == 0:
            #     if len(energy_low) != 0: 
            #         if min(energy_low) < min(energy_high): 
            #             print("policy converged, steps:", s, max(energy_low), min(energy_high))
            #     else: 
            #         print("no low e")
                    
            if s > 4 and (self.step) % 1000 == 0:
                save_model(save_dir = f"{self.save_path}_{self.step}", policy=self.policy)
                    
        save_model(save_dir = f"{self.save_path}_{self.step}", policy=self.policy)
        
        return
        
    def update_policy_e(self, 
        # policy,
        e_batch,
        dl_batch, 
        # optimizer,
        # grad_clip_norm,
        step, 
        # grad_scaler: GradScaler,
        # use_amp: bool = False,
        lock=None,
        # grad_accumulation_steps: int = 2,

    ):
        """Returns a dictionary of items for logging."""
        start_time = time.perf_counter()
        # device = get_device_from_parameters(self.policy)
        
        self.policy.train()
        with torch.autocast(device_type=self.device.type) if self.use_amp else nullcontext():
            output_dict = self.policy.forward(dl_batch, e_batch)
            # TODO(rcadene): policy.unnormalize_outputs(out_dict)
            loss = output_dict["loss"]
            loss_denoise, loss_energy, loss_opt = output_dict["sub_loss"]
            
        self.grad_scaler.scale(loss).backward()

        # Unscale the graident of the optimzer's assigned params in-place **prior to gradient clipping**.
        self.grad_scaler.unscale_(self.optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.grad_clip_norm,
            error_if_nonfinite=False,
        )


        # Gradient Acccumulation: Only update every grad_accumulation_steps 
        if (step+1) % self.grad_accumulation_steps == 0:
            # print("acc step")
            # optimizer.step()
            # optimizer.zero_grad()
            
            # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
            # although it still skips optimizer.step() if the gradients contain infs or NaNs.
            with lock if lock is not None else nullcontext():
                self.grad_scaler.step(self.optimizer)
            # Updates the scale for next iteration.
            self.grad_scaler.update()

            self.optimizer.zero_grad()


            if isinstance(self.policy, PolicyWithUpdate):
                # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
                self.policy.update()

        info = {
            "loss": loss.item(),
            "loss_denoise": loss_denoise.item(), 
            "loss_energy": loss_energy.item(), 
            "loss_opt": loss_opt.item(),
            "grad_norm": float(grad_norm),
            "lr": self.optimizer.param_groups[0]["lr"],
            "update_s": time.perf_counter() - start_time,
            **{k: v for k, v in output_dict.items() if (k != "loss" and k != "sub_loss") },
        }
        info.update({k: v for k, v in output_dict.items() if k not in info and k != "sub_loss"})
        
        return info
            

    def find_closest_paths(self, action, guide):
        print(action.shape,guide.shape)
        assert action.shape[3] == 2 and guide.shape[2] == 2
        indices = torch.linspace(0, guide.shape[1]-1, action.shape[2], dtype=int)
        print(indices)
        
        guide = torch.unsqueeze(guide[:, indices, :], dim=1) # (batch, 1, pred_horizon, action_dim)
        assert guide.shape == (action.shape[0], 1, action.shape[2], action.shape[3])
        print(guide.type)
        print((action - guide).shape)
        action = action.to(torch.float64)
        guide = guide.to(torch.float64)
        dist = torch.linalg.norm(action - guide, dim=3, ord=2) # (B, B, pred_horizon)
        assert dist.shape == (action.shape[0], action.shape[1], action.shape[2])
        dist = dist.mean(dim=2) # (B, B,)
        
        print(dist)

        
        dist_min, distx_sort = torch.sort(dist, dim=1)
        
        #for now just pick one
        just_one = True
        
        if just_one:
            idx = distx_sort[:, :1]
            nidx = distx_sort[:, 1:]
            return idx, nidx
        else: 
        # print(dist_min)
        # dist_min = torch.unsqueeze(dist_min, dim=1)
        
        
            dm = dist - dist_min[:, 0:1]
            print(dm)
            # print("min cdist_min1", min(cdist_min1), cdist_min1[cdx_sort], cdist_min1[cdx_sort]-min(cdist_min1))
            cdist_mask_idx = torch.where(dm < 0.5, 1, 0)
            
            idx = distx_sort[cdist_mask_idx.bool()]
            nidx = distx_sort[~cdist_mask_idx.bool()]
            
            if (self.step+1) % self.log_freq == 0:
                print("num close found", len(idx), len(nidx))
            
            # make energies that match that guide 
            # e_choice = torch.zeros(action.size(0)).to(action.device)
            # e_choice[idx] = 1
        
            return idx, nidx #, e_choice
    
        
    
   
   
    def infer_target(self, obs_batch, guide=None, return_energy=False):
        if return_energy: 
            actions, energy = self.policy.run_inference(copy.deepcopy(obs_batch), guide=guide, return_energy=True) 
            return actions, energy
        else: 
            actions = self.policy.run_inference(copy.deepcopy(obs_batch), guide=guide, return_energy=False)
            return actions
        
    def gen_obs_batch(self, start_loc, guide_loc=None):
        start_loc = torch.tensor(start_loc.reshape(1, 2)).float().cuda()
        start_loc = einops.repeat(start_loc, "n d -> n t d", t=2)
        
        

        # can i make it more efficient here? 
        obs_batch = {
            "observation.state": einops.repeat(
                start_loc, "n t d -> (b n) t d", b=self.batch_size
            )
        }
        if guide_loc is not None: 
            guide_loc = torch.tensor(guide_loc.reshape(1, 2)).float().cuda()
            guide_loc = einops.repeat(guide_loc, "n d -> n t d", t=2)
            obs_batch["observation.environment_state"] = einops.repeat(
                guide_loc, "n t d -> (b n) t d", b=self.batch_size
            )
        else: 
            obs_batch["observation.environment_state"] = einops.repeat(
                start_loc, "n t d -> (b n) t d", b=self.batch_size
            )
        return obs_batch
    
    def plot_high_low(self, paths, idx, nidx, start_pos=None, end_pos=None, title=""):
        xy = paths.detach().cpu().numpy()
        idx_low = idx.detach().cpu().numpy()
        idx_high = nidx.detach().cpu().numpy()
        X = xy[:, :, 0]
        Y = xy[:, :, 1]        
        
        # print("path shape", paths.shape)
        # Plot flat 
        # fig, ax = plt.subplots(1, 2, figsize=(15, 9))
        fig = plt.figure(figsize=(8, 11))
        ax = fig.add_subplot()
        ax.set_ylim(12, 0)
        
        # ax.plot(X[idx_low], Y[idx_low], color="Blue")
        # ax.plot(X[idx_high], Y[idx_high], color="Red")
        # self.imshow3d(ax[0], maze, cmap="binary")
        # self.imshow3d(ax[1], maze, cmap="binary")
        cmap = plt.get_cmap('rainbow')
        energies_norm = range(int(xy.shape[0]))

        colors = cmap(energies_norm)
        for i in range(int(xy.shape[0])):
            if i in idx_low:
                ax.plot(X[i, :], Y[i], color="blue")
                ax.scatter(X[i, :], Y[i], color="blue", s=5)
            elif i in idx_high: 
                ax.plot(X[i], Y[i], color="red")
                ax.scatter(X[i], Y[i], color="red", s=5)
            else: 
                raise Exception("idx low and high should be complete set")
            
        if start_pos is not None: 
            # print("has guide")
            guide_xy = start_pos.detach().cpu().numpy()
            gX = guide_xy[:, 0]
            gY = guide_xy[:, 1]
            ax.plot(gX, gY, color="fuchsia", markersize=30, marker="P")      
            
        if end_pos is not None: 
            # print("has guide")
            guide_xy = end_pos.detach().cpu().numpy()
            gX = guide_xy[:, 0]
            gY = guide_xy[:, 1]
            ax.plot(gX, gY, color="lime", markersize=30, marker="X") 
        
        plt.tight_layout()
        
        plt.savefig(f"inference_itps/test_high_low_plots/{title}_test.png", dpi=150)
        
        plt.clf()
        plt.close('all')
    
    def plot_high_low_guide(self, paths, idx, nidx, guide=None, title=""):
        xy = paths.detach().cpu().numpy()
        idx_low = idx.detach().cpu().numpy()
        idx_high = nidx.detach().cpu().numpy()
        X = xy[:, :, 0]
        Y = xy[:, :, 1]
        
        # print("path shape", paths.shape)
        # Plot flat 
        # fig, ax = plt.subplots(1, 2, figsize=(15, 9))
        fig = plt.figure(figsize=(8, 11))
        ax = fig.add_subplot()
        # ax.set_ylim(12, 0)
        
        # ax.plot(X[idx_low], Y[idx_low], color="Blue")
        # ax.plot(X[idx_high], Y[idx_high], color="Red")
        # self.imshow3d(ax[0], maze, cmap="binary")
        # self.imshow3d(ax[1], maze, cmap="binary")
        # cmap = plt.get_cmap('rainbow')
        # energies_norm = range(int(xy.shape[0]))

        # colors = cmap(energies_norm)
        for i in range(int(xy.shape[0])):
            if i in idx_low:
                print(X[i], Y[i])
                ax.plot(X[i], Y[i], color="blue")
                ax.scatter(X[i], Y[i], color="blue", s=5)
            elif i in idx_high:
                print(X[i], Y[i])
                ax.plot(X[i], Y[i], color="red")
                ax.scatter(X[i], Y[i], color="red", s=5)
            else: 
                raise Exception("idx low and high should be complete set")
            
        if guide is not None: 
            # print("has guide")
            guide_xy = guide.detach().cpu().numpy()
            gX = guide_xy[:, 0]
            gY = guide_xy[:, 1]
            ax.plot(gX, gY, color="lime")
            ax.scatter(gX, gY, color="lime", s=40)
        
        
        plt.tight_layout()
        
        plt.savefig(f"inference_itps/{title}_test.png", dpi=150)
        
        plt.clf()
        plt.close('all')
    

class TestEnergyFineTuning():
    def __init__(self, policy, policy_tag=None, guide_loadpath=None, save_path=None):
        
            
        self.policy = policy
        self.policy_tag = policy_tag
        self.tuner = TunePolicy(self.policy, guide_loadpath, policy_tag=policy_tag, save_path=save_path)
        self.maze_env = MazeEnv()
        """
        [0        1  X  2     3     4]
        [   XXXX     X     X     X   ]
        [5       6      7  X  8     9]
        [   XXXXXXXXX     XXXXXXXX   ]
        [10 11 X 12 X  13  14      15]
        [X     X    X          XXXXXX]
        [16 17 X 18    19  20      21]
        """
        self.key_points = np.array([
            #row 1 
            [165, 155], [445, 150], [645, 160], [860, 155], [1045, 155],
            #row 2
            [160, 340], [445, 335], [640, 355], [860, 340], [1045, 330],
            #row 3
            [150, 540], [245, 560], [445, 560], [645, 550], [840, 550], [1045, 550],
            #row 4
            [150, 750], [245, 745], [445, 760], [645, 740], [840, 730], [1045, 750],
            ])
        self.max_e = -np.inf
        self.min_e = np.inf
        
    def update_global_energy(self, energy): 
        self.max_e = max(max(energy), self.max_e)
        self.min_e = min(min(energy), self.min_e)
    
    def infer_target(self, obs_batch):
        with torch.autocast(device_type="cuda"), seeded_context(0):
            actions, energy = self.policy.run_inference(copy.deepcopy(obs_batch), return_energy=True) # directly call the policy in order to visualize the intermediate steps
            actions = actions.detach().cpu().numpy()
            energy = energy.detach().cpu().numpy()
        return actions, energy
        
    
    def generate_energy_color_map(self, energies, global_range=False, include_zeros=True):
        num_es = len(energies)
        
        cmap = plt.get_cmap('rainbow')
        if global_range: 
            energies_norm = (energies-self.min_e)/(self.max_e-self.min_e)
        if include_zeros: 
            energies_norm = (energies-np.min(energies))/(np.max(energies)-np.min(energies))
        else: 
            energies_norm = (energies-np.min(energies[np.nonzero(energies)]))/(np.max(energies)-np.min(energies[np.nonzero(energies)]))

        colors = cmap(energies_norm)
        return colors
    

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
        
        maze = np.swapaxes(self.maze_env.maze, 0, 1)
        
        
        
        # Plot flat 
        fig, ax = plt.subplots(1, 2, figsize=(15, 9))
        im0 = ax[0].imshow(maze, cmap='binary')
        im1 = ax[1].imshow(maze, cmap='binary')
        # self.imshow3d(ax[0], maze, cmap="binary")
        # self.imshow3d(ax[1], maze, cmap="binary")
        
        for i in range(int(xy.shape[0])):
            color = energy_colors[i, :3]
            color_g = energy_colors_glob[i, :3]
            # print(color.shape)
            ax[0].plot(X[i], Y[i], color=color)
            ax[0].scatter(X[i], Y[i], color=color, s=5)
            ax[1].plot(X[i], Y[i], color=color_g)
            ax[1].scatter(X[i], Y[i], color=color_g, s=5)
            

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
        
        plt.clf()
        plt.close('all')
        
    def plot_maze(self, dpi=150):
        fig, ax = plt.subplots()
        maze = np.swapaxes(self.maze_env.maze, 0, 1)
        im = ax.imshow(maze, cmap='binary')
        
        plt.savefig(f"blank_plot.png", dpi=dpi)
        
        plt.clf()
        plt.close('all')
        
    def plot_avrg(self, dpi=150):
        # self.start_xy, self.start_energy
        # self.fit_xy, self.fit_energy
        # self.tune_xy, self.tune_energy
        def find_closest_paths(action, guide):
            guide = np.repeat(guide, action.shape[1], axis=1)
            guide = np.repeat(guide, action.shape[0], axis=0)
            # guide = einops.repeat(guide, 'i 2 -> b (l i) 2', b=action.shape[0], l=action.shape[1])

            # print(guide.shape)
            cdist = np.linalg.norm(action - guide, axis=2)
            # print(cdist.shape)
            # print(cdist)
            cdist_min = np.min(cdist, axis=1)
            # cdist_min1, cdist_min_indices1 = np.min(cdist_min, axis=1)
            # print(cdist_min)

            cdx_sort = np.argsort(cdist_min, axis=0)
            cdist_mask_idx = np.where(cdist_min[cdx_sort] < 0.2, True, False)
            
            idx = cdx_sort[cdist_mask_idx]
            nidx = cdx_sort[~cdist_mask_idx]
            
            # if (self.step+1) % self.log_freq == 0:
            # print("num close found", len(idx), len(nidx))
            return idx, nidx
        def scale2xy(x, y):
            xi = (x+10) * self.maze_env.scale
            yi = (y+10) * self.maze_env.scale
            gui_xy = tuple([xi,yi])
            return np.array(self.maze_env.gui2xy(gui_xy)).reshape(1, 2)
        
        start_avrg_energies = []
        start_avrg_energy_xy = []
        fit_avrg_energies = []
        fit_avrg_energy_xy = []
        tune_avrg_energies = []
        tune_avrg_energy_xy = []
        for y in range(0, self.maze_env.scale_y):
            for x in range(0, self.maze_env.scale_x):
                xy = scale2xy(x, y)
                xy_b = xy.reshape(1, 1, 2)
                if not self.maze_env.check_collision(xy_b):
                    start_idx, _ = find_closest_paths(self.start_xy, xy_b)
                    fit_idx, _ = find_closest_paths(self.fit_xy, xy_b)
                    tune_idx, _ = find_closest_paths(self.tune_xy, xy_b)
                    # print(start_idx.shape[0])
                    # print(fit_idx.shape[0])
                    # print(tune_idx.shape[0])
                    if start_idx.shape[0] != 0: 
                        start_avrg_energy = np.average(self.start_energy[start_idx])
                        print("adding", start_avrg_energy, xy.squeeze()) 
                        start_avrg_energies.append(start_avrg_energy)
                        start_avrg_energy_xy.append(xy.squeeze())
                        
                    if fit_idx.shape[0] != 0: 
                        fit_avrg_energy = np.average(self.fit_energy[fit_idx])
                        print("adding", fit_avrg_energy, xy.squeeze()) 
                        fit_avrg_energies.append(fit_avrg_energy)
                        fit_avrg_energy_xy.append(xy.squeeze())
                        
                    if tune_idx.shape[0] != 0: 
                        tune_avrg_energy = np.average(self.tune_energy[tune_idx])
                        print("adding", tune_avrg_energy, xy.squeeze()) 
                        tune_avrg_energies.append(tune_avrg_energy)
                        tune_avrg_energy_xy.append(xy.squeeze())
                        
        start_avrg_energies = np.array(start_avrg_energies)
        start_avrg_energy_xy = np.array(start_avrg_energy_xy)
        fit_avrg_energies = np.array(fit_avrg_energies)
        fit_avrg_energy_xy = np.array(fit_avrg_energy_xy)
        tune_avrg_energies = np.array(tune_avrg_energies)
        tune_avrg_energy_xy = np.array(tune_avrg_energy_xy)
        
        fig, ax = plt.subplots(nrows=1, ncols=3,  figsize=(20, 10)) #, figsize=(15, 9) #layout="compressed",
        maze = np.swapaxes(self.maze_env.maze, 0, 1)
        im0 = ax[0].imshow(maze, cmap='binary')
        im1 = ax[1].imshow(maze, cmap='binary')
        im0 = ax[2].imshow(maze, cmap='binary')
        
        # fig, ax = plt.subplots(1, 3, figsize=(10, 3))
                
        start_energy_colors = self.generate_energy_color_map(start_avrg_energies.squeeze())
        fit_energy_colors = self.generate_energy_color_map(fit_avrg_energies.squeeze())
        tune_energy_colors = self.generate_energy_color_map(tune_avrg_energies.squeeze())
        
        for i in range(start_avrg_energies.shape[0]):
            xy = start_avrg_energy_xy[i]
            color = start_energy_colors[i, :3]
            ax[0].scatter(xy[0], xy[1], color=color, s=10)
            
        for i in range(fit_avrg_energies.shape[0]):
            xy = fit_avrg_energy_xy[i]
            color = fit_energy_colors[i, :3]
            ax[1].scatter(xy[0], xy[1], color=color, s=10)
            
        for i in range(tune_avrg_energies.shape[0]):
            xy = tune_avrg_energy_xy[i]
            color = tune_energy_colors[i, :3]
            ax[2].scatter(xy[0], xy[1], color=color, s=10)
            
        for i in range(3):
            ax[i].plot(self.start_xy_pos[0], self.start_xy_pos[1], color="fuchsia", markersize=20, marker="P")
            ax[i].plot(self.guide_xy_pos[0], self.guide_xy_pos[1], color="lime", markersize=20, marker="X")
        
        norm = plt.Normalize(start_avrg_energies.min(), start_avrg_energies.max())
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap='rainbow'), ax=ax[0], label='Average Energy', orientation='vertical', shrink=0.7)
        norm = plt.Normalize(fit_avrg_energies.min(), fit_avrg_energies.max())
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap='rainbow'), ax=ax[1], label='Average Energy', orientation='vertical', shrink=0.7)
        norm = plt.Normalize(tune_avrg_energies.min(), tune_avrg_energies.max())
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap='rainbow'), ax=ax[2], label='Average Energy', orientation='vertical', shrink=0.7)
        
        fig.suptitle(f'Average Energy Map {self.policy_tag} {self.start_xy_pos},{self.guide_xy_pos}', fontsize=20)
                
        ax[0].set_title("Start", fontsize=15)
        ax[1].set_title("Fit", fontsize=15)
        ax[2].set_title("Tune", fontsize=15)
        
        plt.tight_layout()
        
        # plt.show()
            
        plt.savefig(f"{self.save_path}avrg energy map_{self.policy_tag}.png", dpi=dpi)

        
        plt.clf()
        plt.close('all')
         
        
    def plot_all(self, dpi=150):
        # self.start_xy, self.start_energy
        # self.fit_xy, self.fit_energy
        # self.tune_xy, self.tune_energy
        # start_e_idx_sort = np.argsort(self.start_energy)
        
        start_X = self.start_xy[:, :, 0]
        start_Y = self.start_xy[:, :, 1]
        
        fit_X = self.fit_xy[:, :, 0]
        fit_Y = self.fit_xy[:, :, 1]
        
        tune_X = self.tune_xy[:, :, 0]
        tune_Y = self.tune_xy[:, :, 1]
        
        
        start_energies = einops.rearrange(self.start_energy, "n 1 -> 1 n")
        
        
        fit_energies = einops.rearrange(self.fit_energy, "n 1 -> 1 n")
        tune_energies = einops.rearrange(self.tune_energy, "n 1 -> 1 n")
        
        start_energy_colors = self.generate_energy_color_map(start_energies.squeeze())
        start_energy_colors_glob = self.generate_energy_color_map(start_energies.squeeze(), global_range=True)
        
        fit_energy_colors = self.generate_energy_color_map(fit_energies.squeeze())
        fit_energy_colors_glob = self.generate_energy_color_map(fit_energies.squeeze(), global_range=True)
        
        tune_energy_colors = self.generate_energy_color_map(tune_energies.squeeze())
        tune_energy_colors_glob = self.generate_energy_color_map(tune_energies.squeeze(), global_range=True)
        
        maze = np.swapaxes(self.maze_env.maze, 0, 1)
        
        
        # Plot flat 
        fig, ax = plt.subplots(nrows=3, ncols=2, layout="compressed", figsize=(13, 20)) #, figsize=(15, 9)
        im0 = ax[0,0].imshow(maze, cmap='binary')
        im1 = ax[0,1].imshow(maze, cmap='binary')
        im0 = ax[1,0].imshow(maze, cmap='binary')
        im1 = ax[1,1].imshow(maze, cmap='binary')
        im0 = ax[2,0].imshow(maze, cmap='binary')
        im1 = ax[2,1].imshow(maze, cmap='binary')
        # self.imshow3d(ax[0], maze, cmap="binary")
        # self.imshow3d(ax[1], maze, cmap="binary")
        
        for i in range(int(self.start_xy.shape[0])):
            color = start_energy_colors[i, :3]
            color_g = start_energy_colors_glob[i, :3]
            ax[0, 0].plot(start_X[i], start_Y[i], color=color)
            ax[0, 0].scatter(start_X[i], start_Y[i], color=color, s=5)
            ax[0, 1].plot(start_X[i], start_Y[i], color=color_g)
            ax[0, 1].scatter(start_X[i], start_Y[i], color=color_g, s=5)
            
            color = fit_energy_colors[i, :3]
            color_g = fit_energy_colors_glob[i, :3]
            ax[1, 0].plot(fit_X[i], fit_Y[i], color=color)
            ax[1, 0].scatter(fit_X[i], fit_Y[i], color=color, s=5)
            ax[1, 1].plot(fit_X[i], fit_Y[i], color=color_g)
            ax[1, 1].scatter(fit_X[i], fit_Y[i], color=color_g, s=5)
            
            color = tune_energy_colors[i, :3]
            color_g = tune_energy_colors_glob[i, :3]
            ax[2, 0].plot(tune_X[i], tune_Y[i], color=color)
            ax[2, 0].scatter(tune_X[i], tune_Y[i], color=color, s=5)
            ax[2, 1].plot(tune_X[i], tune_Y[i], color=color_g)
            ax[2, 1].scatter(tune_X[i], tune_Y[i], color=color_g, s=5)

        
            

        fig.suptitle(f'Energy Path Maps {self.policy_tag} {self.start_xy_pos},{self.guide_xy_pos}', fontsize=20)
        
        ax[0, 0].set_title("Start 1 Batch Scale", fontsize=15)
        ax[0, 1].set_title("Start 1 Global Scale", fontsize=15)
        
        ax[1, 0].set_title("Fit 1 Batch Scale", fontsize=15)
        ax[1, 1].set_title("Fit 1 Global Scale", fontsize=15)
        
        ax[2, 0].set_title("Tune 2 Batch Scale", fontsize=15)
        ax[2, 1].set_title("Tune 2 Global Scale", fontsize=15)
        
        norm_s = plt.Normalize(start_energies.min(), start_energies.max())
        norm_f = plt.Normalize(fit_energies.min(), fit_energies.max())
        norm_t = plt.Normalize(tune_energies.min(), tune_energies.max())
        fig.colorbar(cm.ScalarMappable(norm=norm_s, cmap='rainbow'), ax=ax[0, 0], orientation='vertical', shrink=0.7)
        fig.colorbar(cm.ScalarMappable(norm=norm_f, cmap='rainbow'), ax=ax[1, 0], orientation='vertical', shrink=0.7)
        fig.colorbar(cm.ScalarMappable(norm=norm_t, cmap='rainbow'), ax=ax[2, 0], orientation='vertical', shrink=0.7)
        
        glob_norm = plt.Normalize(self.min_e, self.max_e)

        for i in range(3):
            fig.colorbar(cm.ScalarMappable(norm=glob_norm, cmap='rainbow'), ax=ax[i, 1], orientation='vertical', shrink=0.7)

            ax[i, 0].plot(self.start_xy_pos[0], self.start_xy_pos[1], color="fuchsia", markersize=30, marker="P")
            ax[i, 1].plot(self.start_xy_pos[0], self.start_xy_pos[1], color="fuchsia", markersize=30, marker="P")
        
            ax[i, 0].plot(self.guide_xy_pos[0], self.guide_xy_pos[1], color="lime", markersize=30, marker="X")
            ax[i, 1].plot(self.guide_xy_pos[0], self.guide_xy_pos[1], color="lime", markersize=30, marker="X")
            
            
            
        # plt.subplots(layout="constrained")
        
        plt.savefig(f"{self.save_path}all_energy_path_maps_t_{self.policy_tag}.png", dpi=dpi)
        
        plt.clf()
        plt.close('all')
        
    def test(self, s, g, max_steps=400):        
        
        # do I tune with whole batch? 
        
        start_loc_gui = self.key_points[s]
        # self.update_agent_pos(start_loc)
        guide_loc_gui = self.key_points[g]
            
        self.start_xy_pos = np.round(self.maze_env.gui2xy(start_loc_gui), 2)
        self.guide_xy_pos = np.round(self.maze_env.gui2xy(guide_loc_gui), 2)
        
        guide = einops.repeat(guide_loc_gui, "n -> t n", t=2)
        guide = np.array([self.maze_env.gui2xy(point) for point in guide])
        print("guide", guide)
        
        self.save_path = f"inference_itps/tune_plots_guide_film/{self.policy_tag}_tune_plots_{max_steps}/{s}_{g}_({self.start_xy_pos[0]},{self.start_xy_pos[1]})_({self.guide_xy_pos[0]},{self.guide_xy_pos[1]})/"
        os.makedirs(self.save_path, exist_ok = True)

        start_obs_batch = self.tuner.gen_obs_batch(self.start_xy_pos)
        self.start_xy, self.start_energy = self.infer_target(start_obs_batch)
        self.fit_xy, self.fit_energy = self.start_xy, self.start_energy
        self.update_global_energy(self.start_energy)
        # self.update_global_energy(self.fit_energy)

        # Tune        
        self.tuner.tune_start_end(max_steps=max_steps)

        guide_obs_batch = self.tuner.gen_obs_batch(self.start_xy_pos, self.guide_xy_pos)
        self.tune_xy, self.tune_energy, = self.infer_target(guide_obs_batch)
        self.update_global_energy(self.tune_energy)
        # self.plot_paths(xy_pred, energy, save_path=pos_string + f"tuned_{steps}", start_loc=start_xy_pos, guide_loc=guide_xy_pos)

        # 
        
        self.plot_avrg()
        self.plot_all()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--policy', required=True, type=str, help="Policy name")
    parser.add_argument('-sp', '--save_path', type=str, help="Model save path")
    parser.add_argument('-s', '--start', default=3, type=int, help="Start pos")
    parser.add_argument('-g', '--guide', default=4, type=int, help="Guide Pos")
    parser.add_argument('-m', '--max-steps', default=400, type=int, help="Max Steps")

    args = parser.parse_args()
    
    # Create and load the policy
    device = torch.device("cuda")
    
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
    elif args.policy in ["act"]:
        checkpoint_path = 'inference_itps/weights/weights_act'
    else:
        raise NotImplementedError(f"Policy with name {args.policy} is not implemented.")
    
    if args.policy is not None:
        pretrained_policy_path = Path(os.path.join(checkpoint_path, "pretrained_model"))
        
    if args.save_path is not None:
        save_path=f"inference_itps/tune_weights/{args.save_path}"
    else: 
        save_path="inference_itps/tune_weights/"
        
    policy = DiffusionPolicy.from_pretrained(pretrained_policy_path, alignment_strategy=alignment_strategy)
    policy.config.noise_scheduler_type = "DDIM"
    policy.diffusion.num_inference_steps = 10
    policy.config.n_action_steps = policy.config.horizon - policy.config.n_obs_steps + 1
    policy_tag = args.policy
    policy.cuda()
    policy.eval()
    
    guide_loadpath = "inference_itps/dp_ebm_p_iden_film_debug_film_dataset_i_traj_10000.json"
    # guide_loadpath = "inference_itps/a3_per_loc.json"
    
    tester = TestEnergyFineTuning(policy, policy_tag=policy_tag, guide_loadpath=guide_loadpath, save_path=save_path)
    
    s = int(args.start) # np.random.randint(len(key_points))
    g = int(args.guide) # np.random.randint(len(key_points))
    max_steps = int(args.max_steps)
    print("start guide max_steps", s, g, max_steps)
    tester.test(s, g, max_steps)
    
    
    # if s == g: 
    #     pass
    # else: 
        