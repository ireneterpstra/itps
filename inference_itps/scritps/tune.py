import torch
import time
import hydra

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

def update_policy_e(
    policy,
    batch,
    optimizer,
    grad_clip_norm,
    step, 
    grad_scaler: GradScaler,
    use_amp: bool = False,
    lock=None,
    grad_accumulation_steps: int = 2,
    dl_batch = None
    
):
    """Returns a dictionary of items for logging."""
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        output_dict = policy.forward_e_g(dl_batch, batch)
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


def tune_energy_with_guide(policy, xy_pred, energy, guide, obs_batch):
    ####Training Params
    
    # from hydra import compose, initialize

    # hydra.core.global_hydra.GlobalHydra.instance().clear()
    # initialize(config_path="./configs/")
    # cfg = compose(config_name=config_name)
    
    cfg = init_hydra_config("./tune_config.yaml")
    
    use_amp=False
    lr = 1.0e-6
    adam_betas = [0.95, 0.999]
    adam_eps = 1.0e-8
    adam_weight_decay = 1.0e-6
    # self.lr_scheduler
    # self.lr_warmup_steps
    grad_clip_norm = 10
    
    optimizer = torch.optim.Adam(
        policy.diffusion.parameters(),
        lr,
        adam_betas,
        adam_eps,
        adam_weight_decay,
    )
    
    grad_scaler = GradScaler(enabled=use_amp)
    ####
    
    
    action_i = torch.from_numpy(xy_pred).float().cuda()
    action = torch.cat((action_i, action_i[:, -1:, :]), 1)
    energy = torch.from_numpy(energy).float().cuda()
    guide = torch.from_numpy(guide).float().cuda()
    
    # find path closest to guide
    assert action.shape[2] == 2 and guide.shape[1] == 2
    indices = torch.linspace(0, guide.shape[0]-1, action.shape[1], dtype=int)
    guide = torch.unsqueeze(guide[indices], dim=0)
    dist = torch.linalg.norm(action - guide, dim=2, ord=2) # (B, pred_horizon)
    dist = dist.mean(dim=1)
    idx_sort = torch.argsort(dist, dim=0)
    print("idx_sort", idx_sort, dist[idx_sort]-min(dist))
    # return
    dist_mask_idx = torch.where(dist[idx_sort]-min(dist) < 0.25, 1, 0)
    
    idx = idx_sort[dist_mask_idx.bool()]
    nidx = idx_sort[~dist_mask_idx.bool()]
    print("close e path", idx)
    print("far / low e path", nidx)
    # return
    
    # make energies that match that guide 
    e_choice = torch.zeros(energy.size(0)).to(energy.device)
    e_choice[idx] = 1

           
    # Freeze layers
    for param in policy.diffusion.unet.diffusion_step_encoder.parameters():
        param.requires_grad = False
    for param in policy.diffusion.unet.down_modules.parameters():
        param.requires_grad = False
    for param in policy.diffusion.unet.mid_modules.parameters():
        param.requires_grad = False
    
    for name, param in  policy.named_parameters():
        if param.requires_grad:
            print(f"Trainable layer: {name}")

    
    #### Setup DataLoader
    offline_dataset = make_dataset(cfg)
    
    shuffle = False
    sampler = EpisodeAwareSampler(
        offline_dataset.episode_data_index,
        drop_n_last_frames=7, # cfg.training.drop_n_last_frames
        shuffle=True,
    )
    
    device = get_device_from_parameters(policy)
    dataloader = torch.utils.data.DataLoader(
        offline_dataset,
        num_workers= 4, # cfg.training.num_workers
        batch_size=action.shape[0],
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)
    
    # 
    batch = obs_batch
    
    # iterate untill model converges or max itter reached
    max_steps = 20
    for s in range(max_steps): 
        
        # gen batch
        dl_batch = next(dl_iter)
        
        #Shuffle batch
        action_low = action.clone()
        action_high = action.clone()
        
        for i in range(action.shape[0]):
            # pick random low index
            low_idx = idx[torch.randint(len(idx), (1,))]
            high_idx = nidx[torch.randint(len(nidx), (1,))]
            action_low[i] = action[low_idx]
            action_high[i] = action[high_idx]
            
        batch = obs_batch.clone()
        
        perm_a = action.clone()
    
        batch["action"] = perm_a[torch.randperm(perm_a.shape[0])]
        # batch["energy"] = energy
        # batch["e_choice"] = e_choice
        batch["action_low"] = action_low
        batch["action_high"] = action_high
        ###
        
        train_info = update_policy_e(
            policy,
            batch,
            optimizer,
            grad_clip_norm,
            s, 
            grad_scaler=grad_scaler,
            use_amp=use_amp,
            dl_batch = dl_batch
        )
        action_energy = policy.get_energy(action_i, obs_batch.clone())
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
    


# if __name__ == "__main__":
#     tune_cli()