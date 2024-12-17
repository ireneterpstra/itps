import os
import urllib.request
import warnings

# import gym
# from gym.utils import colorize
import h5py
from tqdm import tqdm

import torch

from pathlib import Path

import torch
import minari


from itps.common.datasets.lerobot_dataset import LeRobotDataset
from itps.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from itps.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

def get_dataset(h5path=None):
    # if h5path is None:
    #     if self._dataset_url is None:
    #         raise ValueError("Offline env not configured with a dataset URL.")
    #     h5path = download_dataset_from_url(self.dataset_url)

    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    # Run a few quick sanity checks
    for key in ['observations', 'actions', 'rewards', 'terminals']:
        assert key in data_dict, 'Dataset is missing key %s' % key
    N_samples = data_dict['observations'].shape[0]
    # if self.observation_space.shape is not None:
    #     assert data_dict['observations'].shape[1:] == self.observation_space.shape, \
    #         'Observation shape does not match env: %s vs %s' % (
    #             str(data_dict['observations'].shape[1:]), str(self.observation_space.shape))
    # assert data_dict['actions'].shape[1:] == self.action_space.shape, \
    #     'Action shape does not match env: %s vs %s' % (
    #         str(data_dict['actions'].shape[1:]), str(self.action_space.shape))
    if data_dict['rewards'].shape == (N_samples, 1):
        data_dict['rewards'] = data_dict['rewards'][:, 0]
    assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: %s' % (
        str(data_dict['rewards'].shape))
    if data_dict['terminals'].shape == (N_samples, 1):
        data_dict['terminals'] = data_dict['terminals'][:, 0]
    assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: %s' % (
        str(data_dict['rewards'].shape))
    print(N_samples)
    return data_dict

# Create a directory to store the training checkpoint.
output_directory = Path("./outputs/train/maze2d_test")
output_directory.mkdir(parents=True, exist_ok=True)

# Number of offline training steps (we'll only do offline training for this example.)
# Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
training_steps = 5000
device = torch.device("cuda")
log_freq = 10

# Set up the dataset.
delta_timestamps = {
    # Load the previous image and state at -0.1 seconds before current frame,
    # then load current image and state corresponding to 0.0 second.
    "observation.environment_state": [-0.1, 0.0],
    "observation.state": [-0.1, 0.0],
    # Load the previous action (-0.1), the next action to be executed (0.0),
    # and 14 future actions with a 0.1 seconds spacing. All these actions will be
    # used to supervise the policy.
    "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
}
    
# dataset = get_dataset(h5path='/home/clear/Documents/irene/itps/data/maze2d/train/maze2d-large-sparse-v1.hdf5')
# dataloader = torch.utils.data.DataLoader(
#     dataset,
#     num_workers=4,
#     batch_size=64,
#     shuffle=True,
#     pin_memory=device != torch.device("cpu"),
#     drop_last=True,
# )
def collate_fn(batch):
    # print(list(batch[0].observations.keys()))
    # print(type(batch))
    # for key, value in batch[0].observations.iteritems() :
    #     print(key, value)
    # print([(x.observations.keys) for x in batch])
    return {
        "id": torch.Tensor([x.id for x in batch]),
        "observations": [x.observations for x in batch],
        "actions": [torch.as_tensor(x.actions) for x in batch],
        "rewards": [torch.as_tensor(x.rewards) for x in batch],
        "terminations": [torch.as_tensor(x.terminations) for x in batch],
        "truncations": [torch.as_tensor(x.truncations) for x in batch]
    }
minari_dataset = minari.load_dataset('D4RL/pointmaze/large-v2')
print("Observation space:", minari_dataset.observation_space)
print("Action space:", minari_dataset.action_space)
print("Total episodes:", minari_dataset.total_episodes)
print("Total steps:", minari_dataset.total_steps)
dataloader = torch.utils.data.DataLoader(
    minari_dataset,
    num_workers=4,
    batch_size=64,
    shuffle=True,
    pin_memory=device != torch.device("cpu"),
    drop_last=True,
    collate_fn=collate_fn
)
# dataloader = DataLoader(minari_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)


REF_MAX_SCORE = {
    'maze2d-open-v0' : 20.66 ,
    'maze2d-umaze-v1' : 161.86 ,
    'maze2d-medium-v1' : 277.39 ,
}
REF_MIN_SCORE = {
    'maze2d-open-v0' : 0.01 ,
    'maze2d-umaze-v1' : 23.85 ,
    'maze2d-medium-v1' : 13.13 ,
    'maze2d-large-v1' : 6.7 ,
    'maze2d-large-v1' : 273.99 ,
}

def normalize(policy_id, score):
    key = policy_id + '-v0'
    min_score = infos.REF_MIN_SCORE[key]
    max_score = infos.REF_MAX_SCORE[key]
    return (score - min_score) / (max_score - min_score)

# {"mean": torch.randn(3,1,1)}, "std": torch.randn(3,1,1)}

cfg = DiffusionConfig()
policy = DiffusionPolicy(cfg, dataset_stats=minari_dataset.stats)
policy.train()
policy.to(device)

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)



# Run training loop.
step = 0
done = False
while not done:
    # print("step")
    # pbar = tqdm(dataloader, total = training_steps)
    for batch in dataloader:
        print("step", step)
        
        # batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        output_dict = policy.forward(batch)
        loss = output_dict["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % log_freq == 0:
            print(f"step: {step} loss: {loss.item():.3f}")
        step += 1
        if step >= training_steps:
            done = True
            break

# Save a policy checkpoint.
policy.save_pretrained(output_directory)