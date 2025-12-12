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

""" Run CPG """

import time
import numpy as np
import matplotlib

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv

ADD_CARTESIAN_PD = True
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=False,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP)

TEST_STEPS = int(10 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# [TODO] initialize data structures to save CPG and robot states
cpg_states = np.zeros((TEST_STEPS, 4, 4)) # Time x leg x state
foot_pos = np.zeros((TEST_STEPS, 3)) # Time x position (one leg)
des_foot_pos = np.zeros((TEST_STEPS, 3))
joint_angle = np.zeros((TEST_STEPS, 3))
des_joint_angle = np.zeros((TEST_STEPS, 3))
base_vel = np.zeros((TEST_STEPS, 3))
energy = 0
pos_x = np.zeros((TEST_STEPS))

############## Sample Gains
# joint PD gains
kp=np.array([100,100,100])
kd=np.array([2,2,2])

# Cartesian PD gains
kpCartesian = np.diag([500]*3)
kdCartesian = np.diag([20]*3)

for j in range(TEST_STEPS):
  # initialize torque array to send to motors
  action = np.zeros(12) 

  # get desired foot positions from CPG 
  xs,zs = cpg.update()

  # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
  q = env.robot.GetMotorAngles()
  dq = env.robot.GetMotorVelocities()

  # loop through desired foot positions and calculate torques
  for i in range(4):
    # initialize torques for legi
    tau = np.zeros(3)

    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz = np.array([xs[i], sideSign[i] * foot_y, zs[i]])

    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
    #leg_q = np.zeros(3) # [TODO] 
    leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz)

    # Add joint PD contribution to tau for leg i (Equation 4)
    #tau += np.zeros(3) # [TODO] 
    tau += kp * (leg_q - q[3*i:3*i+3]) + kd * (0 - dq[3*i:3*i+3])

    # add Cartesian PD contribution
    if ADD_CARTESIAN_PD:
      # Get desired xyz position in leg frame (use ComputeJacobianAndPosition with the joint angles you just found above)
      # [TODO] 
      _, d_pos = env.robot.ComputeJacobianAndPosition(i, leg_q)

      # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
      # [TODO] 
      J, c_pos = env.robot.ComputeJacobianAndPosition(i, q[3*i:3*i+3])

      # Get current foot velocity in leg frame (Equation 2)
      # [TODO] 
      c_vel= J @ dq[3*i : 3*i+3]

      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      # Use current foot position (c_pos) when computing PD error (target - current)
      tau += J.T @ (kpCartesian @ (d_pos - c_pos) + kdCartesian @ (0 - c_vel))

    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action)

  # [TODO] save any CPG or robot states

  # CPG States
  r = cpg.get_r().copy()
  theta = cpg.get_theta().copy()
  dr = cpg.get_dr().copy()
  dtheta = cpg.get_dtheta().copy()
  cpg_states[j] = np.stack([r, theta, dr, np.round(dtheta, decimals=5)], axis=1)

  # Foot position
  leg_plot=1
  _, foot_pos[j] = env.robot.ComputeJacobianAndPosition(leg_plot)
  des_foot_pos[j] = np.array([xs[leg_plot], sideSign[leg_plot] * foot_y, zs[leg_plot]])

  # Joint angle
  joint_angle[j] = env.robot.GetMotorAngles()[3*leg_plot:3*leg_plot+3]
  des_joint_angle[j] = env.robot.ComputeInverseKinematics(leg_plot, des_foot_pos[j])

  # Base velocity
  base_vel[j] = env.robot.GetBaseLinearVelocity()

  # Energy
  for tau,vel in zip(action,dq):
      energy += np.abs(np.dot(tau,vel)) * TIME_STEP
  pos_x[j] = env.robot.GetBasePosition()[0]
  

##################################################### 
# PLOTS
#####################################################
# [TODO] Create your plots

# 1. CPG States

leg_names = ['Front Right', 'Front Left', 'Rear Right', 'Rear Left']
state_names = ['r', 'theta', 'dr', 'dtheta']

t_start = 0
t_end = 750

fig, axs = plt.subplots(2, 2, figsize=(10, 8)) # 1 subplot per leg
axs = axs.flatten()  # Flatten to 1D array for easy iteration

for state_idx in range(4):
    ax = axs[state_idx]
    
    # Plot each state for this leg
    for leg_idx in range(4):
        # Data shape: [Time, Leg, State]
        ax.plot(t[t_start:t_end], cpg_states[t_start:t_end, leg_idx, state_idx], label=leg_names[leg_idx])
    
    ax.set_title(f'State: {state_names[state_idx]}')
    ax.set_xlabel('Time [s]')
    ax.legend()
    ax.grid()

plt.tight_layout()

# 2.Foot Position vs Desired Foot Position (for one leg)
t_start = 1000
t_end = 1500
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot 3D trajectories
ax.plot(des_foot_pos[t_start:t_end, 0], des_foot_pos[t_start:t_end, 1], des_foot_pos[t_start:t_end, 2], 
        'k--', label='Desired', linewidth=2)
ax.plot(foot_pos[t_start:t_end, 0], foot_pos[t_start:t_end, 1], foot_pos[t_start:t_end, 2], 
        'r-', label='Actual', alpha=0.8)

ax.set_title('Foot Trajectory Tracking (3D)')
ax.legend()

# Plot Axes
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
coords = ['X', 'Y', 'Z']

for i in range(3):
    axs[i].plot(t[t_start:t_end], des_foot_pos[t_start:t_end, i], 'k--', label='Desired')
    axs[i].plot(t[t_start:t_end], foot_pos[t_start:t_end, i], 'r-', label='Actual')
    axs[i].set_ylabel(f'{coords[i]} Position [m]')
    axs[i].grid(True)
    axs[i].legend()

axs[2].set_xlabel('Time [s]')
axs[0].set_title('Foot Position Tracking per Coordinate')

plt.tight_layout()

# 3. Joint Angle
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
coords = ['Hip', 'Thigh', 'Calf']

for i in range(3):
    axs[i].plot(t[t_start:t_end], des_joint_angle[t_start:t_end, i], 'k--', label='Desired')
    axs[i].plot(t[t_start:t_end], joint_angle[t_start:t_end, i], 'r-', label='Actual')
    axs[i].set_ylabel(f'{coords[i]} Angle [rad]')
    axs[i].grid(True)
    axs[i].legend()

axs[2].set_xlabel('Time [s]')
axs[0].set_title('Joint Angle Tracking')

plt.show()

# Base Velocity
print(f"Velocity [x]: \n\tmin: {np.min(base_vel[:,0])}\n\tmax: {np.max(base_vel[:,0])}\n\tmean: {np.mean(base_vel[:,0])}")
print(f"Velocity [y]: \n\tmin: {np.min(base_vel[:,1])}\n\tmax: {np.max(base_vel[:,1])}\n\tmean: {np.mean(base_vel[:,1])}")
print(f"Velocity [z]: \n\tmin: {np.min(base_vel[:,2])}\n\tmax: {np.max(base_vel[:,2])}\n\tmean: {np.mean(base_vel[:,2])}")

# CoT
mass = np.sum(env.robot.GetTotalMassFromURDF())
print("CoT: ", energy/mass/9.81/np.mean(pos_x))