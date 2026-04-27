"""
delta_robot_animator.py

This script loads a previously trained PINN model and visualizes its 
Forward Kinematics predictions by rendering a fully articulated 3D Delta Robot.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# ==========================================
# 1. THE NEURAL NETWORK ARCHITECTURE
# (Must exactly match the training script)
# ==========================================
class DeltaKinematicsPINN(nn.Module):
    def __init__(self):
        super(DeltaKinematicsPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )
        # Initialize biases to point downward
        self.net[-1].bias.data = torch.tensor([0.0, 0.0, -0.4])

    def forward(self, thetas):
        return self.net(thetas)

# ==========================================
# 2. DELTA ROBOT GEOMETRY
# ==========================================
R_base = 0.15
R_platform = 0.05
L_upper = 0.20
L_lower = 0.40
ALPHA = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])

# ==========================================
# 3. GENERATE A TRAJECTORY (Joint Space)
# ==========================================
# Simulating a smooth "Pick and Place" sweeping motion
frames = 150
t = np.linspace(0, 2 * np.pi, frames)

# Motor angles moving in sine waves (between ~11 and ~45 degrees downward)
theta1_traj = 0.2 + 0.6 * np.sin(t)
theta2_traj = 0.2 + 0.6 * np.sin(t + 1.0)
theta3_traj = 0.2 + 0.6 * np.sin(t + 2.0)

# Combine into a tensor for the PINN
thetas_tensor = torch.tensor(np.column_stack((theta1_traj, theta2_traj, theta3_traj)), dtype=torch.float32)

# ==========================================
# 4. LOAD SAVED MODEL & GET PREDICTIONS
# ==========================================
model_path = 'Projects Example/Delta Robot Kinematics Solver/delta_pinn_model.pth'

print(f"Looking for saved model: {model_path}...")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Could not find '{model_path}'. Make sure you saved it in the training script!")

# Initialize the network and load the saved weights
pinn = DeltaKinematicsPINN()
pinn.load_state_dict(torch.load(model_path))
pinn.eval() # Set to evaluation mode (disables gradients/dropout)
print("Model loaded successfully!")

# Feed the trajectory into the AI to get real 3D coordinates
with torch.no_grad():
    predicted_xyz = pinn(thetas_tensor).numpy()

# ==========================================
# 5. 3D ANIMATION SETUP
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Setup plot limits
ax.set_xlim(-0.3, 0.3)
ax.set_ylim(-0.3, 0.3)
ax.set_zlim(-0.6, 0.1)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title("PINN-Driven Delta Robot Kinematics (Loaded Model)")

# Visual Elements
base_circle, = ax.plot([], [], [], 'k-', lw=2)
platform_circle, = ax.plot([], [], [], 'r-', lw=2)
arms_upper = [ax.plot([], [], [], 'b-', lw=4)[0] for _ in range(3)]
arms_lower = [ax.plot([], [], [], 'g-', lw=2)[0] for _ in range(3)]
trajectory_line, = ax.plot([], [], [], 'k--', alpha=0.5)

# Helper function to draw circles
def get_circle(radius, z, x_center=0, y_center=0):
    theta = np.linspace(0, 2 * np.pi, 50)
    return x_center + radius * np.cos(theta), y_center + radius * np.sin(theta), np.full_like(theta, z)

base_x, base_y, base_z = get_circle(R_base, 0.0)
base_circle.set_data(base_x, base_y)
base_circle.set_3d_properties(base_z)

# Variables to store the trace of the end-effector
trace_x, trace_y, trace_z = [], [], []

def update(frame):
    theta = thetas_tensor[frame].numpy()
    ee_pos = predicted_xyz[frame]
    
    # 1. Update Platform position
    plat_x, plat_y, plat_z = get_circle(R_platform, ee_pos[2], ee_pos[0], ee_pos[1])
    platform_circle.set_data(plat_x, plat_y)
    platform_circle.set_3d_properties(plat_z)
    
    # Update trace
    trace_x.append(ee_pos[0])
    trace_y.append(ee_pos[1])
    trace_z.append(ee_pos[2])
    trajectory_line.set_data(trace_x, trace_y)
    trajectory_line.set_3d_properties(trace_z)
    
    # 2. Update Arms
    for i in range(3):
        alpha_i = ALPHA[i]
        theta_i = theta[i]
        
        # Base Joint
        B_x = R_base * np.cos(alpha_i)
        B_y = R_base * np.sin(alpha_i)
        B_z = 0.0
        
        # Elbow Joint
        E_x = (R_base + L_upper * np.cos(theta_i)) * np.cos(alpha_i)
        E_y = (R_base + L_upper * np.cos(theta_i)) * np.sin(alpha_i)
        E_z = -L_upper * np.sin(theta_i)
        
        # Platform Joint
        P_x = ee_pos[0] + R_platform * np.cos(alpha_i)
        P_y = ee_pos[1] + R_platform * np.sin(alpha_i)
        P_z = ee_pos[2]
        
        # Draw Upper Arm
        arms_upper[i].set_data([B_x, E_x], [B_y, E_y])
        arms_upper[i].set_3d_properties([B_z, E_z])
        
        # Draw Lower Arm
        arms_lower[i].set_data([E_x, P_x], [E_y, P_y])
        arms_lower[i].set_3d_properties([E_z, P_z])
        
    return [platform_circle, trajectory_line] + arms_upper + arms_lower

print("Starting animation...")
ani = animation.FuncAnimation(fig, update, frames=frames, interval=30, blit=False)
plt.show()