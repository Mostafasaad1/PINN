"""
================================================================================
Project: Ballbot (The 3D Balancer) - Global Spatial-Temporal Trajectory Planning
File: pinn_ballbot_balancer.py

The Engineering Concept:
A Ballbot is essentially an inherently unstable 3D inverted pendulum balancing on 
a single spherical wheel. Traditional controllers (like cascaded PID or LQR) often 
rely on linearizing the system around the upright position (assuming sin(θ) ≈ θ). 
If the ballbot tilts significantly, that assumption fails and the controller struggles.

By replacing traditional numerical methods and linearized control with a PINN, we 
ask the neural network to learn the full nonlinear dynamics. The PINN acts as a 
Nonlinear Model Predictive Controller (NMPC). It discovers a continuous, mathematically 
smooth sequence of feed-forward motor torques that will drive the robot from an 
unstable initial tilt back to dead-center equilibrium, arriving with zero velocity, 
while respecting the nonlinear physics of gravity, inertia, and coupled 3D motion.
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Physical Parameters of the Ballbot
# ==========================================
g = 9.81      # Gravity (m/s^2)
m = 15.0      # Mass of the Ballbot body (kg)
l = 0.6       # Distance from ball center to Center of Mass (m)
I = m * l**2  # Moment of Inertia (Approximated as point mass for simplicity)

# Initial conditions (The Ballbot is falling!)
theta_x_0 = 0.25   # ~14 degrees tilt in X-axis (Pitch)
theta_y_0 = -0.15  # ~-8.5 degrees tilt in Y-axis (Roll)
t_final = 2.5      # Time we want the robot to be perfectly balanced (seconds)

# ==========================================
# 2. The Neural Network Architecture
# ==========================================
class BallbotPINN(nn.Module):
    """
    Physics-Informed Neural Network for Ballbot Control
    
    Inputs:
        t (tensor): A continuous time vector from t=0 to t=t_final (seconds).
        
    Outputs:
        A 4-dimensional tensor containing:
        [0] theta_x : The predicted pitch angle (X-axis tilt) in radians.
        [1] theta_y : The predicted roll angle (Y-axis tilt) in radians.
        [2] T_x     : The required control torque around the X-axis (N*m).
        [3] T_y     : The required control torque around the Y-axis (N*m).
    """
    def __init__(self):
        super(BallbotPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 4) 
        )

    def forward(self, t):
        return self.net(t)

# Instantiate the network and optimizer
pinn = BallbotPINN()
optimizer = torch.optim.Adam(pinn.parameters(), lr=0.002)

# ==========================================
# 3. Physics & Loss Formulation
# ==========================================
def calculate_loss():
    # Collocation points (Continuous time domain to enforce physics)
    t_physics = torch.linspace(0, t_final, 150).view(-1, 1).requires_grad_(True)
    
    # Forward pass
    outputs = pinn(t_physics)
    theta_x = outputs[:, 0:1]
    theta_y = outputs[:, 1:2]
    T_x     = outputs[:, 2:3]
    T_y     = outputs[:, 3:4]
    
    # Physics Gradients
    dtheta_x_dt = torch.autograd.grad(theta_x, t_physics, grad_outputs=torch.ones_like(theta_x), create_graph=True)[0]
    dtheta_y_dt = torch.autograd.grad(theta_y, t_physics, grad_outputs=torch.ones_like(theta_y), create_graph=True)[0]
    
    d2theta_x_dt2 = torch.autograd.grad(dtheta_x_dt, t_physics, grad_outputs=torch.ones_like(dtheta_x_dt), create_graph=True)[0]
    d2theta_y_dt2 = torch.autograd.grad(dtheta_y_dt, t_physics, grad_outputs=torch.ones_like(dtheta_y_dt), create_graph=True)[0]
    
    # ----------------------------------------------------
    # PHYSICS RESIDUALS
    # ----------------------------------------------------
    res_x = I * d2theta_x_dt2 - m * g * l * torch.sin(theta_x) + T_x
    res_y = I * d2theta_y_dt2 - m * g * l * torch.sin(theta_y) + T_y
    loss_physics = torch.mean(res_x**2) + torch.mean(res_y**2)
    
    # ----------------------------------------------------
    # BOUNDARY CONDITIONS (Start state) - FIXED AUTOGRAD
    # ----------------------------------------------------
    t_0 = torch.tensor([[0.0]], requires_grad=True)
    out_0 = pinn(t_0)
    theta_x_0_pred = out_0[:, 0:1]
    theta_y_0_pred = out_0[:, 1:2]
    
    # Separate the derivatives to avoid shape errors
    dtheta_x_0 = torch.autograd.grad(theta_x_0_pred, t_0, grad_outputs=torch.ones_like(theta_x_0_pred), create_graph=True)[0]
    dtheta_y_0 = torch.autograd.grad(theta_y_0_pred, t_0, grad_outputs=torch.ones_like(theta_y_0_pred), create_graph=True)[0]
    
    loss_ic = (theta_x_0_pred[0, 0] - theta_x_0)**2 + \
              (theta_y_0_pred[0, 0] - theta_y_0)**2 + \
              (dtheta_x_0[0, 0] - 0.0)**2 + \
              (dtheta_y_0[0, 0] - 0.0)**2
              
    # ----------------------------------------------------
    # TARGET EQUILIBRIUM (End state) - FIXED AUTOGRAD
    # ----------------------------------------------------
    t_end = torch.tensor([[t_final]], requires_grad=True)
    out_end = pinn(t_end)
    theta_x_end_pred = out_end[:, 0:1]
    theta_y_end_pred = out_end[:, 1:2]
    
    # Separate the derivatives here as well
    dtheta_x_end = torch.autograd.grad(theta_x_end_pred, t_end, grad_outputs=torch.ones_like(theta_x_end_pred), create_graph=True)[0]
    dtheta_y_end = torch.autograd.grad(theta_y_end_pred, t_end, grad_outputs=torch.ones_like(theta_y_end_pred), create_graph=True)[0]
    
    loss_target = (theta_x_end_pred[0, 0] - 0.0)**2 + \
                  (theta_y_end_pred[0, 0] - 0.0)**2 + \
                  (dtheta_x_end[0, 0] - 0.0)**2 + \
                  (dtheta_y_end[0, 0] - 0.0)**2
                  
    # Control Penalty
    loss_control = torch.mean(T_x**2) + torch.mean(T_y**2)
    
    # Total Objective Function
    total_loss = loss_physics + (100.0 * loss_ic) + (100.0 * loss_target) + (0.01 * loss_control)
    
    # RETURN ALL 5 VALUES FOR YOUR TRAINING LOOP
    return total_loss, loss_physics, loss_ic, loss_target, loss_control

# ==========================================
# 4. Training Loop (Residual Minimization)
# ==========================================
if __name__ == "__main__":
    print("🚀 Training Ballbot PINN Controller...")
    epochs = 3500

    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Now this unpacks perfectly!
        loss, loss_physics, loss_ic, loss_target, loss_control = calculate_loss()
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Total Loss: {loss.item():.6f} | Physics: {loss_physics.item():.6f} | IC: {loss_ic.item():.6f} | Target: {loss_target.item():.6f} | Control: {loss_control.item():.6f}")

    print("✅ Training Complete! Generating Trajectories...\n")
    # ==========================================
    # 5. Visualization (The Digital Twin Dashboard)
    # ==========================================
    t_test = torch.linspace(0, t_final, 300).view(-1, 1)
    with torch.no_grad():
        predictions = pinn(t_test)
        theta_x_pred = predictions[:, 0].numpy()
        theta_y_pred = predictions[:, 1].numpy()
        Tx_pred = predictions[:, 2].numpy()
        Ty_pred = predictions[:, 3].numpy()
    t_test = t_test.numpy()

    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('Ballbot 3D Spatial-Temporal Planning (PINN)', fontsize=16, color='cyan')

    # Plot 1: Kinematics (Angles)
    ax1.plot(t_test, np.degrees(theta_x_pred), color='lime', linewidth=2, label='Pitch Angle (X-axis)')
    ax1.plot(t_test, np.degrees(theta_y_pred), color='magenta', linewidth=2, label='Roll Angle (Y-axis)')
    ax1.axhline(0, color='white', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Tilt Angle (Degrees)')
    ax1.set_title('Learned Nonlinear State Trajectory')
    ax1.grid(True, alpha=0.2)
    ax1.legend()

    # Plot 2: Kinetics (Control Torques)
    ax2.plot(t_test, Tx_pred, color='lime', linestyle='-', linewidth=2, label='Control Torque Tx')
    ax2.plot(t_test, Ty_pred, color='magenta', linestyle='-', linewidth=2, label='Control Torque Ty')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Torque (N·m)')
    ax2.set_title('Discovered Feed-Forward Control Effort')
    ax2.grid(True, alpha=0.2)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('ballbot_pinn_control.png')
    plt.show()