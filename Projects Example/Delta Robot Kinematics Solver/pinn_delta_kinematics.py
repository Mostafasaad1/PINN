"""
THE ENGINEERING CONCEPT:
A Delta Robot is a high-speed parallel manipulator consisting of three arms connected to a 
single moving platform (end-effector). In traditional robotics, finding the Forward Kinematics 
(FK) — calculating the 3D position (X, Y, Z) of the end-effector from the three motor angles 
(Theta 1, Theta 2, Theta 3) — is mathematically rigorous. It usually requires solving a complex 
system of nonlinear equations iteratively (e.g., using Newton-Raphson), which is computationally 
heavy for microcontrollers running high-speed pick-and-place loops.

THE PINN SOLUTION:
Instead of numerical root-finding, we use a Physics-Informed Neural Network (PINN) as a 
real-time, non-iterative FK solver. 
The PINN learns the mapping from joint space to Cartesian space purely by minimizing the 
"Loop-Closure Constraint" of the robot's physical structure. 

The physical rule (Residual):
For each of the 3 arms, the distance between the "Elbow" (driven by the motor) and the 
"Platform Joint" (attached to the end-effector) must exactly equal the length of the lower 
arm (the parallelogram linkage). 

INPUTS:
    - theta (Tensor): A batch of [theta_1, theta_2, theta_3] representing the motor 
                      angles of the three base joints (in radians).

OUTPUTS:
    - xyz (Tensor): The predicted [x, y, z] Cartesian coordinates of the end-effector center.

LOSS FUNCTION:
    - L_physics: The squared error of the loop-closure structural constraint. It penalizes 
                 the network if the predicted [x, y, z] causes the theoretical lower arm 
                 lengths to deviate from the physical lower arm length (L_lower).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ==========================================
# 1. DELTA ROBOT GEOMETRY & PARAMETERS
# ==========================================
# All units in meters and radians
R_base = 0.15      # Radius of the fixed base
R_platform = 0.05  # Radius of the moving platform
L_upper = 0.20     # Length of the upper arm (bicep) connected to the motor
L_lower = 0.40     # Length of the lower arm (parallelogram linkage)

# The three arms are spaced 120 degrees apart around the z-axis
ALPHA = torch.tensor([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0], dtype=torch.float32)

# ==========================================
# 2. THE NEURAL NETWORK ARCHITECTURE
# ==========================================
class DeltaKinematicsPINN(nn.Module):
    def __init__(self):
        super(DeltaKinematicsPINN, self).__init__()
        # Input: 3 motor angles. Output: 3D coordinates (X, Y, Z)
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )
        
        # Initialize biases to point downward (Delta robots operate below their base)
        self.net[-1].bias.data = torch.tensor([0.0, 0.0, -0.4])

    def forward(self, thetas):
        """
        Maps motor angles to end-effector coordinates.
        thetas: [batch_size, 3]
        returns: [batch_size, 3] representing [x, y, z]
        """
        return self.net(thetas)

# ==========================================
# 3. PHYSICS-INFORMED LOSS FUNCTION
# ==========================================
def loop_closure_loss(thetas, predicted_xyz):
    """
    Calculates the physical violation of the Delta robot structure.
    """
    batch_size = thetas.shape[0]
    total_residual = torch.zeros(batch_size, dtype=torch.float32)
    
    x_ee = predicted_xyz[:, 0]
    y_ee = predicted_xyz[:, 1]
    z_ee = predicted_xyz[:, 2]
    
    # Calculate the constraint for all 3 arms
    for i in range(3):
        theta_i = thetas[:, i]
        alpha_i = ALPHA[i]
        
        # 3.1 Calculate Elbow Position (E_i) based on motor angle
        # The motor rotates the upper arm downward from the horizontal base plane
        E_ix = (R_base + L_upper * torch.cos(theta_i)) * torch.cos(alpha_i)
        E_iy = (R_base + L_upper * torch.cos(theta_i)) * torch.sin(alpha_i)
        E_iz = -L_upper * torch.sin(theta_i)
        
        # 3.2 Calculate Platform Joint Position (P_i) based on predicted End-Effector (X,Y,Z)
        P_ix = x_ee + R_platform * torch.cos(alpha_i)
        P_iy = y_ee + R_platform * torch.sin(alpha_i)
        P_iz = z_ee
        
        # 3.3 The Loop Closure Constraint
        # The Euclidean distance squared between Elbow and Platform Joint must equal L_lower^2
        dist_squared = (E_ix - P_ix)**2 + (E_iy - P_iy)**2 + (E_iz - P_iz)**2
        constraint_residual = dist_squared - (L_lower**2)
        
        # Accumulate the squared error of the structural violation
        total_residual += constraint_residual**2
        
    return torch.mean(total_residual)

# ==========================================
# 4. TRAINING SETUP & EXECUTION
# ==========================================
def train_delta_pinn(epochs=5000, batch_size=1024):
    pinn = DeltaKinematicsPINN()
    optimizer = optim.Adam(pinn.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=500, factor=0.5)
    
    print("Initializing Delta Robot PINN Training...")
    print("Objective: Learn Forward Kinematics globally via Loop-Closure Constraints.\n")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Generate a random batch of valid motor angles (e.g., 0 to 90 degrees downward)
        # Operating range: 0.0 to 1.57 radians
        thetas_batch = torch.rand((batch_size, 3)) * (np.pi / 2.0)
        
        # Predict the 3D position
        predicted_xyz = pinn(thetas_batch)
        
        # Calculate the physical violation (No external data used!)
        loss = loop_closure_loss(thetas_batch, predicted_xyz)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:04d} | Physics Constraint Loss: {loss.item():.6e}")

    print("\nTraining Complete.")
    print("The PINN can now map [Theta1, Theta2, Theta3] -> [X, Y, Z] instantly.")
    return pinn

# Execute training if run as main script
if __name__ == "__main__":
    trained_kinematic_solver = train_delta_pinn(epochs=3000)