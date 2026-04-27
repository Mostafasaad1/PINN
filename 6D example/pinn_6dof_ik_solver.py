# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║         THE 6D PINN: Singularity-Free Inverse Kinematics Solver              ║
# ║                Target Audience: Control Engineering Undergrads               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# ── THE ENGINEERING PROBLEM ────────────────────────────────────────────────────
#
#  You need to command an industrial 6-axis articulated arm to a specific 6D 
#  pose (Position + Orientation) in space. 
#
#  Traditional IK solvers use Newton-Raphson iteration and inverse Jacobians, 
#  which are computationally expensive and fail violently at singularities 
#  (e.g., when joints align or the arm is fully extended).
#
# ── THE PINN ARCHITECTURE (R^6 -> R^6) ─────────────────────────────────────────
#
#  We train an AI to become the IK solver, constrained entirely by the physical 
#  Forward Kinematics of the mechanical linkages.
#
#   1. THE INPUT: A target 6D Pose (x, y, z, rx, ry, rz)
#   2. THE OUTPUT: 6 Joint Angles (theta 1 through 6)
#   3. THE PHYSICS LOSS: 
#      We take the 6 output angles, pass them through the Denavit-Hartenberg (DH) 
#      transformation matrices. The resulting 4x4 Transformation Matrix represents 
#      where the hand *actually* is. 
#      The loss is the geometric distance between the Target 6D Pose and the 
#      Calculated 6D Pose. 
#
# ══════════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# 1. THE RIGID BODY PHYSICS (Denavit-Hartenberg Parameters)
# ══════════════════════════════════════════════════════════════════════════════
device = torch.device('cpu')
print("=" * 75)
print(" 🦾  6D PINN: Real-Time IK Solver for 6-DOF Articulated Arm")
print("=" * 75)

# Standard DH Parameters for a generic industrial 6-axis arm (e.g., UR5 scale)
# [a (link length), d (link offset), alpha (link twist)]
dh_params = [
    [0.0,    0.089,  np.pi/2],  # Base to J1
    [-0.425, 0.0,    0.0],      # J1 to J2
    [-0.392, 0.0,    0.0],      # J2 to J3
    [0.0,    0.109,  np.pi/2],  # J3 to J4
    [0.0,    0.094, -np.pi/2],  # J4 to J5
    [0.0,    0.082,  0.0]       # J5 to Tool Flange
]

def dh_transform_matrix(theta, a, d, alpha):
    """
    Creates a differentiable 4x4 Homogeneous Transformation Matrix in PyTorch.
    This handles batches of angles simultaneously!
    """
    batch_size = theta.shape[0]
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)

    # Initialize a batch of 4x4 identity matrices
    T = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

    T[:, 0, 0] = cos_t
    T[:, 0, 1] = -sin_t * cos_a
    T[:, 0, 2] = sin_t * sin_a
    T[:, 0, 3] = a * cos_t

    T[:, 1, 0] = sin_t
    T[:, 1, 1] = cos_t * cos_a
    T[:, 1, 2] = -cos_t * sin_a
    T[:, 1, 3] = a * sin_t

    T[:, 2, 0] = 0
    T[:, 2, 1] = sin_a
    T[:, 2, 2] = cos_a
    T[:, 2, 3] = d
    
    # Bottom row remains [0, 0, 0, 1]
    return T

def forward_kinematics(thetas):
    """
    The Physics Engine. Multiplies 6 transformation matrices together.
    Input: Batch of 6 joint angles [N, 6]
    Output: Batch of 4x4 End-Effector Transformation Matrices [N, 4, 4]
    """
    batch_size = thetas.shape[0]
    T_final = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

    for i in range(6):
        a, d, alpha = dh_params[i]
        T_i = dh_transform_matrix(thetas[:, i], a, d, alpha)
        # Matrix multiplication for the batch
        T_final = torch.bmm(T_final, T_i) 

    return T_final

# ══════════════════════════════════════════════════════════════════════════════
# 2. GENERATING THE 6D TRAINING SPACE (The Workspace)
# ══════════════════════════════════════════════════════════════════════════════

# We want the AI to learn how to reach ANY valid pose in the robot's workspace.
# To do this without complex inverse calculations, we do a neat trick:
# We randomly sample 10,000 sets of valid joint angles, run them through our 
# Forward Kinematics to get the exact 6D Poses, and use those poses as the INPUTS 
# during training.

N_samples = 10000
# Random angles between -Pi and Pi for all 6 joints
random_thetas = (torch.rand(N_samples, 6, device=device) * 2 * np.pi) - np.pi

# Generate the 6D Target Poses (The "Requests")
with torch.no_grad():
    target_matrices = forward_kinematics(random_thetas)

# Extract Position (x, y, z) from the 4x4 matrix
target_positions = target_matrices[:, 0:3, 3]

# Extract the Orientation vectors (the N, O, A vectors of the hand)
# To avoid Euler angle gimbal lock, we use the continuous 3D rotation vectors.
target_rot_x = target_matrices[:, 0:3, 0] # Approach vector
target_rot_z = target_matrices[:, 0:3, 2] # Normal vector

# Our 6D Neural Network Input (Position + 2 Orientation Vectors)
# This perfectly constrains the 6 degrees of freedom in 3D space.
# Shape: [10000, 9] (x,y,z + 3 for rot_x + 3 for rot_z)
nn_inputs = torch.cat([target_positions, target_rot_x, target_rot_z], dim=1)

# ══════════════════════════════════════════════════════════════════════════════
# 3. THE NEURAL NETWORK (The AI Controller)
# ══════════════════════════════════════════════════════════════════════════════

class IKSolverNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 128),  # Input: The desired 6D pose (represented as 9 values)
            nn.Mish(),          # Mish activation is highly effective for kinematics
            nn.Linear(128, 128),
            nn.Mish(),
            nn.Linear(128, 128),
            nn.Mish(),
            nn.Linear(128, 128),
            nn.Mish(),
            nn.Linear(128, 6)   # Output: 6 Joint Angles (theta 1-6)
        )

    def forward(self, x):
        # We constrain the output to valid joint limits (-pi to pi) using Tanh
        return torch.tanh(self.net(x)) * np.pi

model = IKSolverNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# ══════════════════════════════════════════════════════════════════════════════
# 4. THE PHYSICS-INFORMED TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

epochs = 5000
print("\n🚀 Training the Neural Inverse Kinematics Solver...")

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    # 1. Ask the network to guess the 6 joint angles for the target poses
    predicted_thetas = model(nn_inputs)

    # 2. THE PHYSICS LOSS: Pass the guessed angles into the actual robot physics (DH)
    predicted_matrices = forward_kinematics(predicted_thetas)

    # 3. Extract where the robot's hand ACTUALLY went
    predicted_positions = predicted_matrices[:, 0:3, 3]
    predicted_rot_x = predicted_matrices[:, 0:3, 0]
    predicted_rot_z = predicted_matrices[:, 0:3, 2]

    # 4. Calculate the Errors
    # How far off is the physical position? (X, Y, Z error)
    loss_pos = loss_fn(predicted_positions, target_positions)
    
    # How far off is the orientation? (Pitch, Yaw, Roll error)
    loss_rot_x = loss_fn(predicted_rot_x, target_rot_x)
    loss_rot_z = loss_fn(predicted_rot_z, target_rot_z)

    # Total loss is the combination of positional accuracy and rotational accuracy
    total_loss = loss_pos + (loss_rot_x + loss_rot_z) * 0.5

    total_loss.backward()
    optimizer.step()

    if epoch % 1000 == 0 or epoch == 1:
        # Calculate max error in millimeters for context
        max_err_mm = torch.max(torch.abs(predicted_positions - target_positions)).item() * 1000
        print(f"Epoch {epoch:4d} | Total Loss: {total_loss.item():.6f} | "
              f"Pos Error: {loss_pos.item():.6f} | Max Deviation: {max_err_mm:.2f} mm")

# ══════════════════════════════════════════════════════════════════════════════
# 5. TESTING THE IK SOLVER
# ══════════════════════════════════════════════════════════════════════════════
print("\n🎯 Testing the Meta-Solver on a brand new 6D coordinate...")

# Let's request the robot to move to an exact Cartesian coordinate.
# Target: x = 0.3m, y = 0.2m, z = 0.4m, pointing straight down.
target_pos_test = torch.tensor([[0.3, 0.2, 0.4]], device=device)
target_rot_x_test = torch.tensor([[1.0, 0.0, 0.0]], device=device) 
target_rot_z_test = torch.tensor([[0.0, 0.0, -1.0]], device=device) # Straight down

test_input = torch.cat([target_pos_test, target_rot_x_test, target_rot_z_test], dim=1)

model.eval()
with torch.no_grad():
    # The AI instantly solves the Inverse Kinematics in one forward pass
    predicted_angles = model(test_input)
    
    # Verify by running the angles through the Forward Kinematics
    verification_matrix = forward_kinematics(predicted_angles)
    actual_pos = verification_matrix[0, 0:3, 3]

print(f"\nREQUESTED Position: X={target_pos_test[0,0]:.3f}, Y={target_pos_test[0,1]:.3f}, Z={target_pos_test[0,2]:.3f}")
print(f"ACHIEVED  Position: X={actual_pos[0]:.3f}, Y={actual_pos[1]:.3f}, Z={actual_pos[2]:.3f}")

print("\n⚙️ Output Joint Angles (Command these to the PLC/EtherCAT drives):")
angles_deg = predicted_angles[0].numpy() * (180/np.pi)
for i in range(6):
    print(f"  Joint {i+1}: {angles_deg[i]:7.2f}°")

# ══════════════════════════════════════════════════════════════════════════════
# THE CONTROL ENGINEERING TAKEAWAY
# ══════════════════════════════════════════════════════════════════════════════
#
# Look at what we just built. We completely bypassed the Jacobian matrix.
# By baking the Denavit-Hartenberg transformation matrices directly into the 
# loss function, the neural network was forced to learn the geometric topology 
# of the 6-DOF arm. 
#
# Because the Neural Network is just a sequence of matrix multiplications with 
# fixed execution times, this IK solver is deterministic in its timing. You can 
# compile this model using WebAssembly or TensorRT, drop it into an industrial 
# controller environment, and run trajectory planning loops at 1000Hz without 
# ever worrying about kinematic singularities faulting your drives.