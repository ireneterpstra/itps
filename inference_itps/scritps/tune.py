import torch
import time
import hydra
import copy
import os
import einops
import argparse
import numpy as np
from scipy.spatial import distance


from torch.cuda.amp import GradScaler
from contextlib import nullcontext
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
from deprecated import deprecated

from inference_itps.common.policies.utils import get_device_from_parameters
from inference_itps.common.policies.policy_protocol import PolicyWithUpdate
from inference_itps.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

from inference_itps.common.datasets.utils import cycle
from inference_itps.common.utils.utils import seeded_context, init_hydra_config
from hydra import compose, initialize

from inference_itps.common.datasets.sampler import EpisodeAwareSampler
from inference_itps.common.datasets.factory import make_dataset, resolve_delta_timestamps
from inference_itps.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_hydra_config,
    init_logging,
    set_global_seed,
)

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
    def __init__(self, policy):
        
        # with initialize(version_base=None, config_path="conf", job_name="test_app"):
        #     cfg = compose(config_name="config", overrides=["db=mysql", "db.user=me"])
        #     print(OmegaConf.to_yaml(cfg))
        cfg = init_hydra_config("inference_itps/configs/policy/tune_config.yaml")
        
        self.policy = policy 
        
        self.batch_size = 32
        self.dl_batch_size = 256
        
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
            self.policy.diffusion.parameters(),
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
        
        self.device = device = torch.device("cuda") #get_device_from_parameters(policy)
        print("policy device", self.device)
        
        dataloader = torch.utils.data.DataLoader(
            offline_dataset,
            num_workers= 4, # cfg.training.num_workers
            batch_size=self.dl_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=self.device.type != "cpu",
            drop_last=False,
        )
        self.dl_iter = cycle(dataloader)
        # 
        
    def reset_policy(self, policy): 
        self.policy = policy
        
    def tune_energy(self, xy_pred, guide, obs_batch, max_steps=400, num_path_variations = 8): 
        action_i = torch.from_numpy(xy_pred).float().cuda()
        action = torch.cat((action_i, action_i[:, -1:, :]), 1)
        guide = torch.from_numpy(guide).float().cuda()
                
        a_idx, a_nidx, a_e_choice = self.find_closest_paths(action, guide)
        self.plot_high_low(action, a_idx, a_nidx, guide=guide, title="init")
        
        # Freeze All Except Film Layers
        self.freeze_partial_policy()
        
        for name, param in  self.policy.named_parameters():
            if param.requires_grad:
                print(f"Trainable layer: {name}")
        
        self.step = 0
        
        
        
        actions_low, actions_high = self.generate_full_dataset(action, guide, copy.deepcopy(obs_batch), max_steps, num_path_variations)
        for s in range(max_steps):
            ## Sample from original dataset
            dl_batch = next(self.dl_iter)
            for key in dl_batch:
                dl_batch[key] = dl_batch[key].to(self.device, non_blocking=True)
            if dl_batch["action"].shape[0] != num_path_variations * self.batch_size: 
                print("dataloader wrong batch size:", dl_batch["action"].shape[0])
                continue
            
            ## Get genertaed high low energy paths
            action_low = actions_low[s]
            action_high = actions_high[s]
            e_batch = self.gen_batch_with_unique_energy_pairs(action_low, action_high, copy.deepcopy(obs_batch))
            all_alt_obs_batch = self.gen_all_alt_obs_batch(obs_batch, num_path_variations)
            
            e_batch["observation.state"] = all_alt_obs_batch["observation.state"]
            e_batch["observation.environment_state"] = all_alt_obs_batch["observation.environment_state"]
            
            for key in e_batch:
                e_batch[key] = e_batch[key].to(self.device, non_blocking=True)
                
            ## Assert genertaed samples match dataset batch size
            if e_batch["action_low"].shape[0] != num_path_variations * self.batch_size: 
                print("e_batch wrong batch size:", dl_batch["action"].shape[0])
                continue
            if e_batch["action_low"].shape[1] != dl_batch["action"].shape[1]: 
                es =  e_batch["action"].shape[1]
                dls =  dl_batch["action"].shape[1]
                print("dataloader wrong batch size:", dl_batch["action"].shape[0])
                raise Exception(f"action length does not match {es},{dls}")
                continue
            
            train_info = self.update_policy_e(
                e_batch,
                dl_batch, 
                self.step,
            )
            if (self.step+1) % self.log_freq == 0:
                print(self.step, train_info)
            self.step += 1
                
            
            ## Check policy converggence    
            action_energy = self.policy.get_energy(action_i, copy.deepcopy(obs_batch))
            energy_low = action_energy[a_e_choice.bool()] # TODO: replace with idx and nidx
            energy_high = action_energy[~a_e_choice.bool()]
            if s > 4 and (self.step+1) % 10 == 0:
            
                if min(energy_low) < min(energy_high): 
                    print("policy converged, steps:", s, max(energy_low), min(energy_high))
        print("policy not converged", max(energy_low), min(energy_high))
        return action_i.detach().cpu().numpy(), action_energy.detach().cpu().numpy(), self.step #, copy.deepcopy(obs_batch)
        
        
    def tune_energy_resample_at_each_policy_update(self, xy_pred, guide, obs_batch, max_steps=400):
        """
        variation on energy tuning that randomizes start loction around og point to 
        generate more trajectries to learn form 
        """
        # print("tuning obs_batch", obs_batch["observation.state"][-1][0])

        # convert from numpy array to tensor
        action_i = torch.from_numpy(xy_pred).float().cuda()
        action = torch.cat((action_i, action_i[:, -1:, :]), 1)
        guide = torch.from_numpy(guide).float().cuda()
                
        a_idx, a_nidx, a_e_choice = self.find_closest_paths(action, guide)
        self.plot_high_low(action, a_idx, a_nidx, guide=guide, title="init")
        self.freeze_partial_policy()
        
        num_path_variations = 8
        
        self.step = 0
        for s in range(max_steps):

            # new_actions = torch.zeros_like(action).to(action.device)
            # # new_actions.repeat_interleave((1, num_path_variations))
            # new_actions = einops.repeat(new_actions, 'b l d -> (n b) l d', n=num_path_variations)
            # # new_actions = torch.repeat_interleave(new_actions, num_path_variations, dim=0)
            # print(action.shape, new_actions.shape)
            
            dl_batch = next(self.dl_iter)
            for key in dl_batch:
                dl_batch[key] = dl_batch[key].to(self.device, non_blocking=True)
            if dl_batch["action"].shape[0] != num_path_variations * 32: 
                print("dataloader wrong batch size:", dl_batch["action"].shape[0])
                continue
            # else: 
            # batch size can be 256 it is 32? now so x8
            
            new_actions, new_obs_batch = self.gen_alt_paths(obs_batch, num_path_variations)

                
            idx, nidx, e_choice = self.find_closest_paths(new_actions, guide)
            # self.plot_high_low(new_actions, idx, nidx, guide=guide, title="dl")
            # print("num of close paths:", len(idx))
            # print("find_closest_paths eB", new_obs_batch["observation.state"].shape[0], new_obs_batch["observation.environment_state"].shape[0])

            all_alt_obs_batch = self.gen_all_alt_obs_batch(obs_batch, num_path_variations)
            action_low, action_high = self.gen_unique_energy_pairs(new_actions, idx, nidx)
            e_batch = self.gen_batch_with_unique_energy_pairs(action_low, action_high, all_alt_obs_batch)
            # print("gen_batch_with_energy_pairs eB", e_batch["action"].shape[0], e_batch["observation.state"].shape[0], e_batch["observation.environment_state"].shape[0])

            for key in e_batch:
                e_batch[key] = e_batch[key].to(self.device, non_blocking=True)
                
            if e_batch["action_low"].shape[0] != num_path_variations * 32: 
                print("e_batch wrong batch size:", dl_batch["action"].shape[0])
                continue
            if e_batch["action_low"].shape[1] != dl_batch["action"].shape[1]: 
                es =  e_batch["action"].shape[1]
                dls =  dl_batch["action"].shape[1]
                print("dataloader wrong batch size:", dl_batch["action"].shape[0])
                raise Exception(f"action length does not match {es},{dls}")
                continue
                
            train_info = self.update_policy_e(
                e_batch,
                dl_batch, 
                self.step,
            )
            if (self.step+1) % self.log_freq == 0:
                self.plot_high_low(new_actions, idx, nidx, guide=guide, title="dl")
                print(self.step, train_info)
            self.step += 1
                
            
            ## Check policy converggence    
            action_energy = self.policy.get_energy(action_i, copy.deepcopy(obs_batch))
            energy_low = action_energy[a_e_choice.bool()] # TODO: replace with idx and nidx
            energy_high = action_energy[~a_e_choice.bool()]
            if s > 4 and (self.step+1) % 10 == 0:
            
                if min(energy_low) < min(energy_high): 
                    print("policy converged, steps:", s, max(energy_low), min(energy_high))
                    # print("policy converged, steps:", s, energy_low, energy_high)
                    # print("obs_batch", obs_batch["observation.state"][-1][0])
                    # return action_i.detach().cpu().numpy(), action_energy.detach().cpu().numpy(), self.step #, copy.deepcopy(obs_batch)
        print("policy not converged", max(energy_low), min(energy_high))
        return action_i.detach().cpu().numpy(), action_energy.detach().cpu().numpy(), self.step #, copy.deepcopy(obs_batch)
        
        
        return
    
    @deprecated(version='0', reason="Old version of tuing without datatloader")
    def tune_energy_with_guide(self, xy_pred, energy, guide, obs_batch):
        # convert from numpy array to tensor
        action_i = torch.from_numpy(xy_pred).float().cuda()
        action = torch.cat((action_i, action_i[:, -1:, :]), 1)
        energy = torch.from_numpy(energy).float().cuda()
        guide = torch.from_numpy(guide).float().cuda()
        
        # find path closest to guide
        idx, nidx, e_choice = self.find_closest_paths(action, guide)
        
        self.freeze_partial_policy()
        
        # iterate untill model converges or max itter reached
        max_steps = 300
        for s in range(max_steps): 
            
            # gen batch
            dl_batch = next(self.dl_iter)
            
            e_batch = self.gen_batch_with_energy_pairs(action, obs_batch, idx, nidx)
            
            for key in e_batch:
                e_batch[key] = e_batch[key].to(self.device, non_blocking=True)
            for key in dl_batch:
                dl_batch[key] = dl_batch[key].to(self.device, non_blocking=True)
            
            train_info = self.update_policy_e(
                # self.policy,
                e_batch,
                dl_batch, 
                # self.optimizer,
                # self.grad_clip_norm,
                s, 
                # grad_scaler=self.grad_scaler,
                # use_amp=self.use_amp,
                # dl_batch = dl_batch
            )
            action_energy = self.policy.get_energy(action_i, copy.deepcopy(obs_batch))
            energy_low = action_energy[e_choice.bool()] 
            energy_high = action_energy[~e_choice.bool()]
            if s > 4: 
            
                if min(energy_low) < min(energy_high): 
                    print("policy converged, steps:", s, max(energy_low), min(energy_high))
                    print("policy converged, steps:", s, energy_low, energy_high)
                    break
            print("policy not converged", max(energy_low), min(energy_high))
            
            
        # logger.save_checkpont(
        #     step,
        #     policy,
        #     optimizer,
        #     lr_scheduler,
        #     identifier=step_identifier,
        # )

        return
 
    def update_policy_e(self, 
        # policy,
        batch,
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
            output_dict = self.policy.forward_e_g(dl_batch, batch)
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

    def generate_full_dataset(self, action, guide, obs_batch, max_steps, num_path_variations):
        actions_low = torch.zeros_like(action).to(action.device)
        actions_high = torch.zeros_like(action).to(action.device)
        actions_low = einops.repeat(actions_low, 'b t d -> ms (b nv) t d', ms=max_steps, nv=num_path_variations)
        actions_high = einops.repeat(actions_high, 'b t d -> ms (b nv) t d', ms=max_steps, nv=num_path_variations)
        print(actions_low.shape)
        print(actions_high.shape)
        for s in range(max_steps):
            new_actions, new_obs_batch = self.gen_alt_paths(obs_batch, num_path_variations)
            idx, nidx, e_choice = self.find_closest_paths(new_actions, guide)
            
            action_low, action_high = self.gen_unique_energy_pairs(new_actions, idx, nidx)
            if (s+1) % 50 == 0:
                print(f'num low found {len(idx)}, num high found {len(nidx)}')
                self.plot_high_low(new_actions, idx, nidx, guide=guide, title=f"dl_{s}")
            actions_low[s] = action_low
            actions_high[s] = action_high
        return actions_low, actions_high
            

    def find_closest_paths(self, action, guide):
        assert action.shape[2] == 2 and guide.shape[1] == 2
        # printsxds(guide)
        cdist = torch.cdist(action[:, -5:, :], guide, p=2)

        cdist_min, cdist_min_indices = torch.min(cdist, dim=2)
        cdist_min1, cdist_min_indices1 = torch.min(cdist_min, dim=1)

        cdx_sort = torch.argsort(cdist_min1, dim=0)
        # print("min cdist_min1", min(cdist_min1), cdist_min1[cdx_sort], cdist_min1[cdx_sort]-min(cdist_min1))
        cdist_mask_idx = torch.where(cdist_min1[cdx_sort]-min(cdist_min1) < 0.5, 1, 0)
        
        idx = cdx_sort[cdist_mask_idx.bool()]
        nidx = cdx_sort[~cdist_mask_idx.bool()]
        
        if (self.step+1) % self.log_freq == 0:
            print("num close found", len(idx), len(nidx))
            
        # make energies that match that guide 
        e_choice = torch.zeros(action.size(0)).to(action.device)
        e_choice[idx] = 1
        
        return idx, nidx, e_choice
    
    def gen_unique_energy_pairs(self, paths, idx, nidx):
        action = paths.clone()
        
        def gen_var(index, d_len): 
            i_to_pert = index[torch.randint(len(index), (d_len,))]
            ac = action[i_to_pert]
            varr = torch.rand(d_len, 2).float().cuda()
            varr = einops.repeat(varr, "d l -> d t l", t=action.shape[1])
            return ac + (varr * 0.05 - 0.025)
        
        low_shortfall = action.shape[0] - len(idx)
        low_var = gen_var(idx, low_shortfall)
        action_low = torch.cat((action[idx], low_var), 0)
        
        high_shortfall = action.shape[0] - len(nidx)
        high_var = gen_var(nidx, high_shortfall)
        action_high = torch.cat((action[nidx], high_var), 0)
        
        shuffle_low = torch.randperm(action_low.shape[0]).int().cuda()
        shuffle_high = torch.randperm(action_high.shape[0]).int().cuda()
        
        action_low = action_low[shuffle_low]
        action_high = action_high[shuffle_high]
        
        if action.shape[0] != action_low.shape[0] and action.shape[0] != action_high.shape[0]: 
            raise Exception(f'Actions not right length {action.shape[0]},{action_low.shape[0]},{action_high.shape[0]}')
        # idx_f = torch.range(0, action.shape[0]).int().cuda()
        # self.plot_high_low(action_low, idx_f, torch.range(0, 1), guide=None, title="low_high_test_low")
        # self.plot_high_low(action_high, torch.range(0, 1), idx_f, guide=None, title="low_high_test_high")
        
        return action_low, action_high
    
    def gen_batch_with_unique_energy_pairs(self, action_low, action_high, obs_batch):
                
        e_batch = copy.deepcopy(obs_batch)
            
        # e_batch["action"] = perm_paths[torch.randperm(perm_paths.shape[0])]
        e_batch["action_low"] = action_low
        e_batch["action_high"] = action_high
        
        return e_batch
    
    def gen_batch_with_energy_pairs(self, action, obs_batch, idx, nidx):
        #Shuffle batch
        # print("gen_batch_with_energy_pairs eB", obs_batch["action"].shape[0], obs_batch["observation.state"].shape[0], obs_batch["observation.environment_state"].shape[0])

        action_low = action.clone()
        action_high = action.clone()
        
        # for i in range(action.shape[0]):
            # pick random low index
        
        low_idx = idx[torch.randint(len(idx), (action.shape[0],))]
        high_idx = nidx[torch.randint(len(nidx), (action.shape[0],))]
        action_low = action[low_idx]
        action_high = action[high_idx]
        
        print("low_idx", low_idx)
        print("high_idx", high_idx)
        print("action_low shape", action_low.shape)
        print("action_high shape", action_high.shape)
        e_batch = copy.deepcopy(obs_batch)
        
        perm_a = action.clone()
    
        e_batch["action"] = perm_a[torch.randperm(perm_a.shape[0])]
        e_batch["action_low"] = action_low
        e_batch["action_high"] = action_high
        
        return e_batch
    
    def freeze_partial_policy(self):
        # Freeze layers
        # for param in self.policy.diffusion.unet.diffusion_step_encoder.parameters():
        #     param.requires_grad = False
        # for param in self.policy.diffusion.unet.down_modules.parameters():
        #     param.requires_grad = False
        # for param in self.policy.diffusion.unet.mid_modules.parameters():
        #     param.requires_grad = False
        # action_energy_after = self.policy.get_energy(action_i, copy.deepcopy(obs_batch))
        
        
        for param in self.policy.parameters():
            param.requires_grad = False
    
        for name, layer in self.policy.diffusion.unet.named_children():
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
                                    
        for name, param in  self.policy.named_parameters():
            if param.requires_grad:
                print(f"Trainable layer: {name}")
   
   
    def infer_target(self, obs_batch, guide=None, return_energy=False):
        if return_energy: 
            actions, energy = self.policy.run_inference(copy.deepcopy(obs_batch), guide=guide, return_energy=True) 
            return actions, energy
        else: 
            actions = self.policy.run_inference(copy.deepcopy(obs_batch), guide=guide, return_energy=False)
            return actions
    
    def gen_alt_paths(self, prev_obs_batch, n):
        """
        perturb start loaction and generate more paths that would be near the selected path
        """
        new_obs_batch = self.gen_alt_obs_batch(prev_obs_batch, n)
        # 
        # for n in range(num_paths): 
        action = self.infer_target(new_obs_batch)
        # print("action", action.shape)
        action = torch.cat((action, action[:, -1:, :]), 1)
        return action, new_obs_batch
    
    
    def gen_alt_obs_batch(self, prev_obs_batch, n):
        start_loc = prev_obs_batch["observation.state"][-1][0]
        # print("start_loc", start_loc)
        n_start_loc = start_loc + (torch.rand(n, 2).float().cuda() * 0.2 - 0.1)
        # print("start_loc", start_loc, n_start_loc)
        n_start_loc = einops.repeat(n_start_loc, "n d -> n t d", t=2)


        # can i make it more efficient here? 
        obs_batch = {
            "observation.state": einops.repeat(
                n_start_loc, "n t d -> (b n) t d", b=self.batch_size
            )
        }
        obs_batch["observation.environment_state"] = einops.repeat(
            n_start_loc, "n t d -> (b n) t d", b=self.batch_size
        )
        return obs_batch
    
    def gen_obs_batch(self, start_loc):
        start_loc = torch.tensor(start_loc.reshape(1, 2)).float().cuda()
        start_loc = einops.repeat(start_loc, "n d -> n t d", t=2)

        # can i make it more efficient here? 
        obs_batch = {
            "observation.state": einops.repeat(
                start_loc, "n t d -> (b n) t d", b=self.batch_size
            )
        }
        obs_batch["observation.environment_state"] = einops.repeat(
            start_loc, "n t d -> (b n) t d", b=self.batch_size
        )
        return obs_batch
    
    def gen_all_alt_obs_batch(self, prev_obs_batch, n):
        start_loc = prev_obs_batch["observation.state"][-1][0]
        start_loc = torch.tensor(start_loc.reshape(1, 2)).float().cuda()
        n_start_loc = start_loc + (torch.rand(n * self.batch_size, 2).float().cuda() * 0.1 - 0.05)
        n_start_loc = einops.repeat(n_start_loc, "a d -> a t d", t=2)

        # print("obs device", start_loc.device)
        # can i make it more efficient here? 
        obs_batch = {
            "observation.state": n_start_loc
        }
        obs_batch["observation.environment_state"] = n_start_loc
        
        return obs_batch
    
    def plot_high_low(self, paths, idx, nidx, guide=None, title=""):
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
            
        if guide is not None: 
            # print("has guide")
            guide_xy = guide.detach().cpu().numpy()
            gX = guide_xy[:, 0]
            gY = guide_xy[:, 1]  
            ax.plot(gX, gY, color="lime")
            ax.scatter(gX, gY, color="lime", s=40)
        
        
        plt.tight_layout()
        
        plt.savefig(f"inference_itps/tune_plots/{title}_test.png", dpi=150)
        
        plt.clf()
        plt.close('all')
    

class TestEnergyFineTuning():
    def __init__(self, policy, policy_tag=None):
        self.policy = policy
        self.policy_tag = policy_tag
        
        self.tuner = TunePolicy(self.policy)
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
        # with torch.autocast(device_type="cuda"), seeded_context(0):
        #     if return_energy: 
        #         xy_pred, energy = self.tuner.infer_target(copy.deepcopy(obs_batch), return_energy=True)
        #         xy_pred = xy_pred.detach().cpu().numpy()
        #         energy = energy.detach().cpu().numpy()
        #         return xy_pred, energy
        #     else: 
        #         xy_pred = self.tuner.infer_target(copy.deepcopy(obs_batch), return_energy=True)
        #         xy_pred = xy_pred.detach().cpu().numpy()
        #         return xy_pred
        
    
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
        
    def test(self, s, g, max_steps=100):
        start_loc_gui = self.key_points[s]
        # self.update_agent_pos(start_loc)
        guide_loc_gui = self.key_points[g]
            
        self.start_xy_pos = np.round(self.maze_env.gui2xy(start_loc_gui), 2)
        self.guide_xy_pos = np.round(self.maze_env.gui2xy(guide_loc_gui), 2)
        
        guide = einops.repeat(guide_loc_gui, "n -> t n", t=2)
        guide = np.array([self.maze_env.gui2xy(point) for point in guide])
        print("guide", guide)
        
        self.save_path = f"inference_itps/tune_plots_iden_film_lp0.2/{self.policy_tag}_tune_plots_{max_steps}/{s}_{g}_({self.start_xy_pos[0]},{self.start_xy_pos[1]})_({self.guide_xy_pos[0]},{self.guide_xy_pos[1]})/"
        os.makedirs(self.save_path, exist_ok = True)

        obs_batch = self.tuner.gen_obs_batch(self.start_xy_pos)
        self.start_xy, self.start_energy = self.infer_target(obs_batch)
        self.update_global_energy(self.start_energy)
        # self.plot_paths(xy_pred, energy, save_path=pos_string + "start", start_loc=start_xy_pos, guide_loc=guide_xy_pos)
        
        self.fit_xy, self.fit_energy, self.tune_steps = self.tuner.tune_energy(copy.deepcopy(self.start_xy), guide, copy.deepcopy(obs_batch), max_steps=max_steps)
        self.update_global_energy(self.fit_energy)
        # self.plot_paths(xy_pred, energy, save_path=pos_string + f"fit_{steps}", start_loc=start_xy_pos, guide_loc=guide_xy_pos)

        self.tune_xy, self.tune_energy, = self.infer_target(obs_batch)
        self.update_global_energy(self.tune_energy)
        # self.plot_paths(xy_pred, energy, save_path=pos_string + f"tuned_{steps}", start_loc=start_xy_pos, guide_loc=guide_xy_pos)

        
        self.plot_avrg()
        self.plot_all()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--policy', required=True, type=str, help="Policy name")
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
        
    policy = DiffusionPolicy.from_pretrained(pretrained_policy_path, alignment_strategy=alignment_strategy)
    policy.config.noise_scheduler_type = "DDIM"
    policy.diffusion.num_inference_steps = 10
    policy.config.n_action_steps = policy.config.horizon - policy.config.n_obs_steps + 1
    policy_tag = args.policy
    policy.cuda()
    policy.eval()
    
    tester = TestEnergyFineTuning(policy, policy_tag)
    s = int(args.start) # np.random.randint(len(key_points))
    g = int(args.guide) # np.random.randint(len(key_points))
    max_steps = int(args.max_steps)
    print("start guide max_steps", s, g, max_steps)
    tester.test(s, g, max_steps)
    # tester.plot_maze()
    
        
    
    
    
    # if s == g: 
    #     pass
    # else: 
        