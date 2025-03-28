import torch
import time
import hydra
import copy

import einops

from torch.cuda.amp import GradScaler
from contextlib import nullcontext

from common.policies.utils import get_device_from_parameters
from common.policies.policy_protocol import PolicyWithUpdate

from common.datasets.utils import cycle
from common.datasets.sampler import EpisodeAwareSampler
from common.datasets.factory import make_dataset, resolve_delta_timestamps
from common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    init_hydra_config,
    init_logging,
    set_global_seed,
)

class TunePolicy:
    def __init__(self, policy):
        cfg = init_hydra_config("./tune_config.yaml")
        
        self.policy = policy 
        
        self.batch_size = 32
        self.dl_batch_size = 256
        
        self.step = 0
    
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
        
        self.device = get_device_from_parameters(policy)
        
        # print("policy device", device)
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
        
    def tune_energy(self, xy_pred, guide, obs_batch): 
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

        self.freeze_partial_policy()
        
        num_path_variations = 8
        
        max_steps = 500
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
                
            # batch size can be 256 it is 32? now so x8
            
            # for n in range(num_path_variations):
            new_actions, new_obs_batch = self.gen_alt_paths(obs_batch, num_path_variations)
                # print("na", na.shape)
                # new_actions[n, :, :, :] = na
                
            idx, nidx, e_choice = self.find_closest_paths(new_actions, guide)
            # print("num of close paths:", len(idx))
            # print("find_closest_paths eB", new_obs_batch["observation.state"].shape[0], new_obs_batch["observation.environment_state"].shape[0])

            e_batch = self.gen_batch_with_energy_pairs(new_actions, new_obs_batch, idx, nidx)
            # print("gen_batch_with_energy_pairs eB", e_batch["action"].shape[0], e_batch["observation.state"].shape[0], e_batch["observation.environment_state"].shape[0])

            for key in e_batch:
                e_batch[key] = e_batch[key].to(self.device, non_blocking=True)
                
            train_info = self.update_policy_e(
                e_batch,
                dl_batch, 
                self.step,
            )
            if (self.step+1) % 10 == 0:
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


    def find_closest_paths(self, action, guide):
        assert action.shape[2] == 2 and guide.shape[1] == 2
        # printsxds(guide)
        cdist = torch.cdist(action[:, -5:, :], guide, p=2)

        cdist_min, cdist_min_indices = torch.min(cdist, dim=2)
        cdist_min1, cdist_min_indices1 = torch.min(cdist_min, dim=1)

        cdx_sort = torch.argsort(cdist_min1, dim=0)

        cdist_mask_idx = torch.where(cdist_min1[cdx_sort]-min(cdist_min1) < 0.2, 1, 0)
        
        idx = cdx_sort[cdist_mask_idx.bool()]
        nidx = cdx_sort[~cdist_mask_idx.bool()]
        
        if (self.step+1) % 10 == 0:
            print("num close found", len(idx))
        
        # make energies that match that guide 
        e_choice = torch.zeros(action.size(0)).to(action.device)
        e_choice[idx] = 1
        
        return idx, nidx, e_choice
    
    def gen_batch_with_energy_pairs(self, action, obs_batch, idx, nidx):
        #Shuffle batch
        # print("gen_batch_with_energy_pairs eB", obs_batch["action"].shape[0], obs_batch["observation.state"].shape[0], obs_batch["observation.environment_state"].shape[0])

        action_low = action.clone()
        action_high = action.clone()
        
        # for i in range(action.shape[0]):
            # pick random low index
        
        low_idx = idx[torch.randint(len(idx), (action.shape[0],))]
        high_idx = nidx[torch.randint(len(nidx), (1,))]
        action_low = action[low_idx]
        action_high = action[high_idx]
        
        # print(action_low)
        # print(action_high)
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
   
    def gen_alt_paths(self, prev_obs_batch, n):
        """
        perturb start loaction and generate more paths that would be near the selected path
        """
        new_obs_batch = self.gen_alt_obs_batch(prev_obs_batch, n)
        # 
        # for n in range(num_paths): 
        action = self.policy.run_inference(new_obs_batch, guide=None)
        # print("action", action.shape)
        action = torch.cat((action, action[:, -1:, :]), 1)
        return action, new_obs_batch
    
    
    def gen_alt_obs_batch(self, prev_obs_batch, n):
        start_loc = prev_obs_batch["observation.state"][-1][0]
        # print("start_loc", start_loc)
        n_start_loc = start_loc + (torch.rand(n, 2).to(self.device) * 0.2 - 0.1)
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
        


    # def imshow3d(self, ax, array, value_direction='z', pos=0, norm=None, cmap=None):
    #     """
    #     Display a 2D array as a  color-coded 2D image embedded in 3d.

    #     The image will be in a plane perpendicular to the coordinate axis *value_direction*.

    #     Parameters
    #     ----------
    #     ax : Axes3D
    #         The 3D Axes to plot into.
    #     array : 2D numpy array
    #         The image values.
    #     value_direction : {'x', 'y', 'z'}
    #         The axis normal to the image plane.
    #     pos : float
    #         The numeric value on the *value_direction* axis at which the image plane is
    #         located.
    #     norm : `~matplotlib.colors.Normalize`, default: Normalize
    #         The normalization method used to scale scalar data. See `imshow()`.
    #     cmap : str or `~matplotlib.colors.Colormap`, default: :rc:`image.cmap`
    #         The Colormap instance or registered colormap name used to map scalar data
    #         to colors.
    #     """
    #     if norm is None:
    #         norm = Normalize()
    #     colors = plt.get_cmap(cmap)(norm(array))

    #     if value_direction == 'x':
    #         nz, ny = array.shape
    #         zi, yi = np.mgrid[0:nz + 1, 0:ny + 1]
    #         xi = np.full_like(yi, pos)
    #     elif value_direction == 'y':
    #         nx, nz = array.shape
    #         xi, zi = np.mgrid[0:nx + 1, 0:nz + 1]
    #         yi = np.full_like(zi, pos)
    #     elif value_direction == 'z':
    #         ny, nx = array.shape
    #         yi, xi = np.mgrid[0:ny + 1, 0:nx + 1]
    #         zi = np.full_like(xi, pos)
    #     else:
    #         raise ValueError(f"Invalid value_direction: {value_direction!r}")
    #     # xi = self.gui2x(xi)
    #     # yi = self.gui2y(yi)
    #     xi = xi - 0.5
    #     yi = yi - 0.5
    #     # print(xi)
    #     # print(yi)
    #     # print(zi)
    #     ax.plot_surface(xi, yi, zi, rstride=1, cstride=1, facecolors=colors, shade=False, alpha=0.5)

    # def plot_energies(self, xy, energies, num_inc=1, collisions=None, name=""):
    #     num_traj = int(len(energies)/(num_inc + 1))
    #     if collisions is None:
    #         collisions, traj_collisions = self.check_collision_i(xy)
    #     print(collisions)
    #     print(collisions.shape, xy.shape)
    #     print(traj_collisions)
    #     print(xy[collisions].shape, xy[collisions])
    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     maze = np.swapaxes(self.maze, 0, 1)
    #     print(maze.shape)
    #     self.imshow3d(ax, maze, cmap="binary")
    #     # X, Y, Z = axes3d.get_test_data(0.05)
    #     X = xy[:, :, 0]
    #     Y = xy[:, :, 1]
    #     # print(X)
    #     # print(Y)
    #     # energies = np.array([0,0,0,3,3,3])
    #     energies = einops.rearrange(energies, "n -> 1 n")
    #     # energy_colors = self.generate_energy_color_map(energies)
        
    #     Z = np.repeat(energies.T, xy.shape[1], axis=1)
    #     # print(energies)
    #     # print(X.shape)
    #     # print(Y.shape)
    #     # print(Z.shape)
    #     # cmap = plt.get/_cmap('rainbow')
        
        
        
    #     print("num traj, num inc", num_traj, num_inc)
        
    #     # colors = mpl.colormaps['tab20'].colors
    #     c = np.tile(np.arange(int(num_traj)), num_inc + 1)
    #     print(c)
    #     C = einops.rearrange(c, "n -> 1 n")
    #     C = np.repeat(C.T, xy.shape[1], axis=1)
    #     # print(C)
    #     cmap = ListedColormap(["darkorange", "lightseagreen", "lawngreen", "pink"]) #"lawngreen", 
    #     colors = cmap(c)
        
    #     # Xy = X[~collisions]
    #     # Yy = Y[~collisions]
    #     # Zy = Z[~collisions]
    #     # Cy = C[~collisions]
    #     # ax.scatter(Xy, Yy, Zy, c=Cy, cmap=cmap, s=4)
    #     Xn = X[collisions]
    #     Yn = Y[collisions]
    #     Zn = Z[collisions]
        
    #     # for i in range(int(xy.shape[0])):
    #     for i, color in enumerate(colors):
    #         print("color", color)
    #         # color = (energy_colors[i, :3] * 255).astype(int)
    #         ax.plot(X[i], Y[i], Z[i], color=color)  # Plot contour curves
    #         Xyi = X[i][~collisions[i]]
    #         Yyi = Y[i][~collisions[i]]
    #         Zyi = Z[i][~collisions[i]]
    #         ax.scatter(Xyi, Yyi, Zyi, c=color, s=5)
        
    #     ax.scatter(Xn, Yn, Zn, c='red', s=8, marker="X")
        
    #     #colors = plt.get_cmap(cmap)(self.maze)
        
    #     plt.show()
        
# if __name__ == "__main__":
#     tune_cli()