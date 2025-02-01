#!/usr/bin/env python

import random
import hydra
import numpy as np
import torch

def perturb_traj(orig):
    # Symmetrical continous Gaussian perturbation
    impulse_start = random.randint(0, len(orig)-2)
    impulse_end = random.randint(impulse_start+1, len(orig)-1)
    impulse_mean = (impulse_start + impulse_end)/2
    impulse_target_x = random.uniform(-8, 8)
    impulse_target_y = random.uniform(-8, 8)
    max_relative_dist = 5 # np.exp(-5) ~= 0.006

    kernel = np.exp(-max_relative_dist*(np.array(range(len(orig))) - impulse_mean)**2 / ((impulse_start-impulse_mean)**2))
    perturbed = orig.copy()
    perturbed[:, 1] += (impulse_target_y-perturbed[:, 1])*kernel
    perturbed[:, 0] += (impulse_target_x-perturbed[:, 0])*kernel

    # succ_traj, _, _ = task.validify_traj(perturbed.T)

    return perturbed #, end_in_red


def generate_trajs(self, samples, per_SG=10):
    """
    Generates a set of trajectories.

    Params:
        samples [int] -- Number of SG trajectory samples.
        per_SG [int] -- Number of samples per SG pair.

    Returns:
        trajectory -- Deformed trajectory.
    """
    # Initialize buttons if debug mode.
    if self.debug:
        next_button = p.addUserDebugParameter("Next Sample", 1, 0, 0)
        next_num = 0

    # Collect data.
    sampling = True
    trajs = []
    while len(trajs) < samples*per_SG:
        # Operate buttons if debugging.
        if self.debug:
            next_pushes = p.readUserDebugParameter(next_button)
            if next_pushes > next_num:
                next_num = next_pushes
                if sampling == False:
                    p.removeAllUserDebugItems()
                    sampling = True

        # Sample a trajectory via deformation.
        if sampling:
            # Sample random S and G poses.
            #move_laptop(self.objectID["laptop"])
            path_length = int(self.horizon / self.timestep) + 1
            start_pos, goal_pos = random_SG_pos(self.objectID, path_length)

            # Generate multiple trajectories per SG pair.
            samples_SG = 0
            trajs_SGs = []
            while samples_SG < per_SG:
                # Get trajectory from start to goal.
                start_pose = random_legalpose(self.objectID["robot"], pos=start_pos)
                goal_pose = random_legalpose(self.objectID["robot"], pos=goal_pos)

                # Compute straight line path in configuration space.
                waypts = np.linspace(start_pose, goal_pose, path_length)
                waypts_time = np.linspace(0.0, self.horizon, path_length)
                traj = Trajectory(waypts, waypts_time)

                # Visualize base trajectory.
                if self.debug:
                    plot_trajectory(waypts_to_xyz(self.objectID["robot"], waypts), color=[0, 1, 0])

                # Perturb trajectory such that it is different enough from initial.
                traj_delta = 0
                while traj_delta < 8.0:
                    deformed_traj = copy.deepcopy(traj)
                    # Sample number of waypoints to perturb.
                    num_waypts_deform = random.randint(1, 3)
                    # Perturb trajectory.
                    for _ in range(num_waypts_deform):
                        # Choose deformation magnitude and width.
                        alpha = np.random.uniform(-0.1, 0.1)
                        n = random.randint(8, path_length - 2)

                        # Sample perturbation vector and waypoint to apply it to.
                        u = np.random.uniform(low=-math.pi, high=math.pi, size=7)
                        u = np.hstack((u, [0, 0, 0])).reshape((-1, 1))
                        waypt_idx = random.randint(1, path_length - n)

                        # Deform trajectory.
                        deformed_traj = deformed_traj.deform(u, waypt_idx * self.timestep, alpha, n)
                        print("Deformed trajectory at {} waypt, {} alpha, {} n".format(waypt_idx, alpha, n))
                    # Validate deformed trajectory.
                    traj_delta = sum(np.linalg.norm(deformed_traj.waypts - traj.waypts, axis=1))
                    traj_delta = min([traj_delta] + [sum(np.linalg.norm(traj_sg.waypts - deformed_traj.waypts, axis=1)) for traj_sg in trajs_SGs])

                # Visualize sampled trajectory if in debug mode.
                if self.debug:
                    # View trajectory.
                    plot_trajectory(waypts_to_xyz(self.objectID["robot"], deformed_traj.waypts), color=[0, 0, 1])
                    sampling = False
                # Save the trajectory.
                samples_SG += 1
                trajs.append(list(waypts_to_raw(self.objectID, deformed_traj.waypts)))
                trajs_SGs.append(copy.deepcopy(deformed_traj))
                print("--------------------- {} / {} -----------------------".format(len(trajs), samples*per_SG))

        time.sleep(0.01)
    if self.debug:
        visualize_trajset(self.objectID, trajs)
        p.removeAllUserParameters()
    return trajs