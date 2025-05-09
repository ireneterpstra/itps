"""
Run from the root directory of the project with `python lerobot/scripts/interactive_multimodality.py`.

This script:
    - Sets up a PushT environment (custom branch: https://github.com/alexander-soare/gym-pusht/tree/add_all_mode)
    - Loads up 3 policies from pretrained models on the hub: Diffusion Policy, ACT, and VQ-BeT.
        - You can switch between ACT with and without VAE (see comments below).
    - Runs the environment in a loop showing visualizations for each of the policies. You can mouse over the
      first window to control the robot.
        - You can comment in/out policies to show.
        - You can noise the observations prior to input to the policies.

NOTE about setup. You can follow the setup instructions in the main LeRobot README.
"""

import os
import cv2 # cv2 not working
import pygame

import einops
import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from pathlib import Path
import datetime
import matplotlib.pyplot as plt



from itps.common.policies.act.modeling_act import ACTPolicy
from itps.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from itps.common.policies.rollout_wrapper import PolicyRolloutWrapper
from itps.common.utils.utils import seeded_context

# self.space_size = 512






class PushtEnv:
    def __init__(self, env, vis_size):
        self.env = env
        
        self.vis_size = vis_size
        self.gui_size = (vis_size, vis_size)
        self.space_size = 512
        
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GRAY = (128, 128, 128)
        self.agent_color = self.RED
        
        pygame.init()
        self.screen = pygame.display.set_mode(self.gui_size)
        pygame.display.set_caption("Push-T")
        self.clock = pygame.time.Clock()
        self.agent_gui_pos = np.array([0, 0]) # Initialize the position of the red dot
        self.running = True
        self.run_id = datetime.datetime.now().strftime('%m.%d.%H.%M.%S')
        
        self.batch_size = 50  # visualize this many trajectories per inference
        
    def draw_state(self, img):
        # print(img)
        # img_uint8 = img.astype('uint8')
        # print(img_uint8)
        surface = pygame.surfarray.make_surface(img)
        surface = pygame.transform.scale(surface, self.gui_size)
        self.screen.blit(surface, (0, 0))
        
    def generate_time_color_map(self, num_steps):
        cmap = plt.get_cmap('rainbow')
        values = np.linspace(0, 1, num_steps)
        colors = cmap(values)
        return colors
    
    def generate_energy_color_map(self, energies):
        num_es = len(energies)
        cmap = plt.get_cmap('rainbow')
        energies_norm = (energies-np.min(energies))/(np.max(energies)-np.min(energies))
        # values = np.linspace(0, 1, num_es)
        # print("min - max energies", np.min(energies), np.max(energies))
        colors = cmap(energies_norm)
        return colors
        
    # def mouse_callback(self, event: int, x: int, y: int, flags: int = 0, *_):
    #     return np.array([x / self.vis_size * self.space_size, y / self.vis_size * self.space_size])
    
    def xy2gui(self, xy):
        # xy = xy + self.offset # Adjust normalization as necessary
        x = xy[0] * self.vis_size / (self.space_size)
        y = xy[1] * self.vis_size / (self.space_size)
        return np.array([y, x], dtype=float)
    
    def gui2xy(self, gui):
        #TODO: Why is x gui[0]?
        x = gui[1] / self.vis_size * self.space_size
        y = gui[0] / self.vis_size * self.space_size
        return np.array([x, y], dtype=float)
    
    
        
        
class TestEnv(PushtEnv):
    def __init__(self, env, policy, policy_tag=None, vis_size = 512):
        super().__init__(env, vis_size)
        self.mouse_pos = None
        self.policy = policy
        self.policy_tag = policy_tag
        self.action = None
        pygame.display.set_caption(policy_tag)
        
    def update_screen(self, xy_pred, img, energies = None):
        self.draw_state(img)
        

        # print(xy_pred.shape)
        if xy_pred is not None:
            # time_colors = self.generate_time_color_map(xy_pred.shape[1])
            energy_colors = self.generate_energy_color_map(energies.squeeze())
            for idx, pred in enumerate(xy_pred):
                color = (energy_colors[idx, :3] * 255).astype(int)
                for step_idx in range(len(pred) - 1):
                    # color = (time_colors[step_idx, :3] * 255).astype(int)
                    if idx < 32: 
                        circle_size = 5
                    else: 
                        circle_size = 2
                    start_pos = self.xy2gui(pred[step_idx])
                    end_pos = self.xy2gui(pred[step_idx + 1])
                    # print(color)
                    pygame.draw.circle(self.screen, color, start_pos, circle_size)

        # pygame.draw.circle(self.screen, self.agent_color, (int(self.agent_gui_pos[0]), int(self.agent_gui_pos[1])), 20)
        # if keep_drawing: # visualize the human drawing input
        #     for i in range(len(self.draw_traj) - 1):
        #         pygame.draw.line(self.screen, self.RED, self.draw_traj[i], self.draw_traj[i + 1], 10)

        # for b in range(policy_batch_actions.shape[1]):
            #     policy_actions = policy_batch_actions[:, b] / 512 * img_.shape[:2]  # (horizon, 2)
            #     policy_actions = np.round(policy_actions).astype(int)
            #     for k, policy_action in enumerate(policy_actions[::-1]):
            #         cv2.circle(
            #             img_,
            #             tuple(policy_action),
            #             radius=2,
            #             color=(
            #                 int(255 * k / len(policy_actions)),
            #                 0,
            #                 int(255 * (len(policy_actions) - k) / len(policy_actions)),
            #             ),
            #             thickness=1,
            #         )
        pygame.display.flip()
        
    def update_mouse_pos(self):
        self.mouse_pos = np.array(pygame.mouse.get_pos())
        
    def update_agent_pos(self, new_agent_pos):
        self.agent_gui_pos = np.array(new_agent_pos)
        self.action = self.gui2xy(self.agent_gui_pos)
        
    def run_inference(self, policy, obs, timestamp, noise_std=0, guide=None, visualizer=None, return_energy=False):
        agent_hist_xy = obs["agent_pos"]
        image_hist = obs["pixels"]
        agent_hist_xy = np.array(agent_hist_xy).reshape(1, 2)
        image_hist = einops.rearrange(np.array(image_hist), '... -> 1 ...')
        
        wrapped = False
        if self.policy_tag[:2] == 'dp' or self.policy_tag[:2] == 'DP' and not wrapped:
            agent_hist_xy = agent_hist_xy.repeat(2, axis=0)
            
            image_hist = image_hist.repeat(2, axis=0)
            image_hist = einops.rearrange(np.array(image_hist), 't a b c -> t c a b')
        else: 
            image_hist = einops.rearrange(np.array(image_hist), 'a b c -> c a b')
        
        # print(agent_hist_xy.shape)
        # print(image_hist.shape)
        
        obs_batch = {
            "observation.state": einops.repeat(
                torch.from_numpy(agent_hist_xy).float().cuda(), "... -> b ...", b=self.batch_size
            ),
        }
        if "pixels" in obs:
            obs_batch["observation.image"] = einops.repeat(
                torch.from_numpy(image_hist).float().cuda(), "... -> b ...", b=self.batch_size
            )
        else: 
            obs_batch["observation.environment_state"] = obs_batch["observation.state"]
            
        # with seeded_context(0):
        #     obs_batch["observation.state"] = (
        #         obs_batch["observation.state"] + torch.randn_like(obs_batch["observation.state"]) * noise_std
        #     )
        
        # # if "observation.image" not in self.policy.input_keys:
        # obs["observation.environment_state"] = obs["observation.state"]
        # if "environment_state" in obs:
        #     obs_batch["observation.environment_state"] = einops.repeat(
        #         torch.from_numpy(obs["environment_state"]).float().cuda(), "d -> b d", b=self.batch_size
        #     )
        #     with seeded_context(0):
        #         obs_batch["observation.environment_state"] = (
        #             obs_batch["observation.environment_state"]
        #             + torch.randn_like(obs_batch["observation.environment_state"]) * noise_std
        #         )
        
        with torch.autocast(device_type="cuda"), seeded_context(0):
            if return_energy:
                action, energy = self.policy.run_inference(obs_batch, guide=guide, visualizer=visualizer, return_energy=return_energy)
                a_out = action.detach().cpu().numpy()
                e_out = energy.detach().cpu().numpy()
                return a_out, e_out
            else:
                return self.policy.run_inference(obs_batch, guide=guide, visualizer=visualizer, return_energy=return_energy).cpu().numpy() # directly call the policy in order to visualize the intermediate steps

            actions = policy.run_inference(obs_batch, return_energy = True)
            
            # actions = policy.provide_observation_get_actions(obs_batch, timestamp, timestamp)
        
        # actions = einops.rearrange(actions.cpu().numpy(),  's b 2 -> b s 2')
        # return actions.cpu().numpy()  # (S, B, 2)
    
    def quit(self):
        pygame.quit()

    def get_mouse_pos(self): 
        return self.mouse_pos
        
    def step(self, mouse_pos=None):
        if mouse_pos is not None: 
            self.update_mouse_pos()
        else:
            self.mouse_pos = mouse_pos
            
        img = self.env.render()
        if img is None:
            print("No image returned")
        
        
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                break
            
        self.update_agent_pos(self.mouse_pos.copy())
            
        img_ = img.copy()
        # Uncomment this one (and comment out the next one) if you want to just clear the action cache but
        # not the observation cache.
        # policy_wrapped.invalidate_action_cache()
        # Uncomment this one (and comment out the last one) if you want to clear both the observation cache
        # and the action cache.
        self.policy.reset()
        
        # Set noise_std to a non-zero value to noise the observations prior to input to the policies. 4 is
        # a good value.
        policy_batch_actions, energy = self.run_inference(self.policy, obs, t, noise_std=0, return_energy=True)
        # policy_batch_actions = einops.rearrange(policy_batch_actions, "n b c -> b n c") 
        # policy_batch_actions = policy_batch_actions[:10, :, :]
        # energy = energy[:10]
        obs, *_ = env.step(self.action)
        # print(self.action)

        self.update_screen(policy_batch_actions, img_, energies=energy)
    
    def run(self):
        
        obs, _ = self.env.reset()
        action = obs["agent_pos"]
        
        
        t = 0
        
        while self.running:
            self.update_mouse_pos()
            
            img = self.env.render()
            if img is None:
                print("No image returned")
            
            
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                
            self.update_agent_pos(self.mouse_pos.copy())
                
            img_ = img.copy()
            # Uncomment this one (and comment out the next one) if you want to just clear the action cache but
            # not the observation cache.
            # policy_wrapped.invalidate_action_cache()
            # Uncomment this one (and comment out the last one) if you want to clear both the observation cache
            # and the action cache.
            self.policy.reset()
            
            # Set noise_std to a non-zero value to noise the observations prior to input to the policies. 4 is
            # a good value.
            policy_batch_actions, energy = self.run_inference(self.policy, obs, t, noise_std=0, return_energy=True)
            # policy_batch_actions = einops.rearrange(policy_batch_actions, "n b c -> b n c") 
            # policy_batch_actions = policy_batch_actions[:10, :, :]
            # energy = energy[:10]
            obs, *_ = env.step(self.action)
            # print(self.action)

            self.update_screen(policy_batch_actions, img_, energies=energy)

            self.clock.tick(30)
            t += 1 / fps

        pygame.quit()
        
        
class MultiPolicyTestEnv():
    def __init__(self, env, window_names_and_policies, vis_size = 512):
        self.policies = []
        for name, policy in window_names_and_policies: 
            p = TestEnv(env, policy, name, vis_size=vis_size)
            self.policies.append(p)
        self.running = True
        self.clock = pygame.time.Clock()
            
    def run(self):
        max = 1000000
        while self.running:
            # Need to define main 
            print("run")
            self.policies[0].step()
            mouse_pos = self.policies[0].get_mouse_pos()
            for p in self.policies[1:]: 
                p.step(mouse_pos)
            
            self.clock.tick(30)
        
            
        







if __name__ == "__main__":
    vis_size = 512
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_environment_state_agent_pos",
        visualization_width=vis_size,
        visualization_height=vis_size,
    )
    fps = env.unwrapped.metadata["render_fps"]

    checkpoint_path = 'itps/weights/weights_pusht_dp_ebm_n_200k'
    pretrained_policy_path = Path(os.path.join(checkpoint_path, "pretrained_model"))
    dp_ebm_img = DiffusionPolicy.from_pretrained(pretrained_policy_path)
    dp_ebm_img.config.noise_scheduler_type = "DDIM"
    dp_ebm_img.diffusion.num_inference_steps = 10
    dp_ebm_img.config.n_action_steps = dp_ebm_img.config.horizon - dp_ebm_img.config.n_obs_steps + 1
    dp_ebm_img.cuda()
    dp_ebm_img.eval()
    dp_ebm_img_wrapped = PolicyRolloutWrapper(dp_ebm_img, fps=fps)

    checkpoint_path = 'itps/weights/weights_pusht_dp_ebm_no_img_200k'
    pretrained_policy_path = Path(os.path.join(checkpoint_path, "pretrained_model"))
    dp_ebm_no_img = DiffusionPolicy.from_pretrained(pretrained_policy_path)
    dp_ebm_no_img.config.noise_scheduler_type = "DDIM"
    dp_ebm_no_img.diffusion.num_inference_steps = 10
    dp_ebm_no_img.config.n_action_steps = dp_ebm_no_img.config.horizon - dp_ebm_no_img.config.n_obs_steps + 1
    dp_ebm_no_img.cuda()
    dp_ebm_no_img.eval()
    dp_ebm_no_img_wrapped = PolicyRolloutWrapper(dp_ebm_no_img, fps=fps)

    checkpoint_path = 'itps/weights/weights_pusht_dp_ebm_no_img_frz_film_200k'
    pretrained_policy_path = Path(os.path.join(checkpoint_path, "pretrained_model"))
    dp_ebm_no_img_frz_film = DiffusionPolicy.from_pretrained(pretrained_policy_path)
    dp_ebm_no_img_frz_film.config.noise_scheduler_type = "DDIM"
    dp_ebm_no_img_frz_film.diffusion.num_inference_steps = 10
    dp_ebm_no_img_frz_film.config.n_action_steps = dp_ebm_no_img_frz_film.config.horizon - dp_ebm_no_img_frz_film.config.n_obs_steps + 1
    print("input keys", dp_ebm_no_img_frz_film.input_keys)
    dp_ebm_no_img_frz_film.cuda()
    dp_ebm_no_img_frz_film.eval()
    dp_ebm_no_img_frz_film_wrapped = PolicyRolloutWrapper(dp_ebm_no_img_frz_film, fps=fps)

    # Uncomment/comment pairs of policies and window names.
    ls_window_names_and_policies = [
        ("DP-EBM (image)", dp_ebm_img),
        ("DP-EBM (no img)", dp_ebm_no_img),
        ("DP-EBM (no img frz film)", dp_ebm_no_img_frz_film),
        # ("VQ-BeT", vqbet_wrapped),
    ]
    # multi_env = MultiPolicyTestEnv(env, ls_window_names_and_policies)
    test_env = TestEnv(env, ls_window_names_and_policies[0][1], ls_window_names_and_policies[0][0], vis_size=vis_size)
    test_env.run()
        
# for window_name, _ in ls_window_names_and_policies:
#     print(window_name)
#     cv2.imshow(window_name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     cv2.waitKey(0)
#     # cv2.imshow(window_name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     # cv2.imshow(window_name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# print("showed imgae")

# cv2.setMouseCallback(ls_window_names_and_policies[0][0], mouse_callback)

# quit_ = False
# t = 0

# while not quit_:
#     k = cv2.waitKey(1)
    
#     print("running")

#     if k == ord("q"):
#         quit_ = True

#     img = env.render()

#     for window_name, policy_wrapped in ls_window_names_and_policies:
#         img_ = img.copy()
#         # Uncomment this one (and comment out the next one) if you want to just clear the action cache but
#         # not the observation cache.
#         # policy_wrapped.invalidate_action_cache()
#         # Uncomment this one (and comment out the last one) if you want to clear both the observation cache
#         # and the action cache.
#         policy_wrapped.reset()
#         # Set noise_std to a non-zero value to noise the observations prior to input to the policies. 4 is
#         # a good value.
#         policy_batch_actions = run_inference(policy_wrapped, obs, t, noise_std=0)

#         obs, *_ = env.step(action)

#         for b in range(policy_batch_actions.shape[1]):
#             policy_actions = policy_batch_actions[:, b] / 512 * img_.shape[:2]  # (horizon, 2)
#             policy_actions = np.round(policy_actions).astype(int)
#             for k, policy_action in enumerate(policy_actions[::-1]):
#                 cv2.circle(
#                     img_,
#                     tuple(policy_action),
#                     radius=2,
#                     color=(
#                         int(255 * k / len(policy_actions)),
#                         0,
#                         int(255 * (len(policy_actions) - k) / len(policy_actions)),
#                     ),
#                     thickness=1,
#                 )
#         cv2.imshow(window_name, cv2.cvtColor(img_, cv2.COLOR_BGR2RGB))

#     t += 1 / fps
