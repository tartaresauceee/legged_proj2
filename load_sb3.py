# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

import os, sys
import gymnasium as gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

# utils
from env.quadruped_gym_env import QuadrupedGymEnv
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results

LEARNING_ALG = "PPO" #"SAC"
interm_dir = "./logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '102824115106'
log_dir = interm_dir + '121225123459'

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {"motor_control_mode":"CPG", #"CARTESIAN_PD"
               "task_env": "LR_COURSE_TASK", #  "LR_COURSE_TASK", "FWD_LOCOMOTION"
               "observation_space_mode": "LR_COURSE_OBS",
               "terrain": "SLOPES"}
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False 

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
plt.show() 

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0

# [TODO] initialize arrays to save data from simulation 
fwd_velocity = []
steps = []
x_position = []
y_position = []
z_position = []

for i in range(1300):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test if the outputs make sense)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    
    if dones:
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        episode_reward = 0

    # [TODO] save data from current robot states for plots 
    # To get base position, for example: env.envs[0].env.robot.GetBasePosition()
    x_position.append(env.envs[0].env.robot.GetBasePosition()[0])
    y_position.append(env.envs[0].env.robot.GetBasePosition()[1])
    z_position.append(env.envs[0].env.robot.GetBasePosition()[2])
    fwd_velocity.append(env.envs[0].env.robot.GetBaseLinearVelocity()[0]) # x-velocity
    steps.append(i)
    
# [TODO] make plots
# Single plot with dual y-axes: Z(x) and Y(x)
fig, ax_left = plt.subplots(figsize=(8, 5))
ax_right = ax_left.twinx()
# Plot Z vs X on left axis
line_z, = ax_left.plot(x_position, z_position, color='tab:blue', label='Z vs X')
ax_left.set_xlabel('X Position (m)')
ax_left.set_ylabel('Z Position (m)', color='tab:blue')
ax_left.tick_params(axis='y', labelcolor='tab:blue')
ax_left.grid()
# Plot Y vs X on right axis
line_y, = ax_right.plot(x_position, y_position, color='tab:orange', label='Y vs X')
ax_right.set_ylabel('Y Position (m)', color='tab:orange')
ax_right.tick_params(axis='y', labelcolor='tab:orange')
# Combined legend
lines = [line_z, line_y]
labels = [l.get_label() for l in lines]
ax_left.legend(lines, labels, loc='best')

plt.figure()
plt.plot(steps, fwd_velocity, label=r'$v_x$')
mean_vel = np.mean(fwd_velocity)
plt.axhline(y=mean_vel, color='k', linestyle='--', label=fr'$\bar{{v}}_x = {mean_vel:.2f}m/s$')
plt.xlabel('Steps')
plt.ylabel('Forward Velocity (m/s)')
plt.grid()
plt.legend()
plt.show()