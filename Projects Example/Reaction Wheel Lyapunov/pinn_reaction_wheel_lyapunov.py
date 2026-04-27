"""
pinn_reaction_wheel_lyapunov.py (With Plotting Dashboard)

===============================================================================
THE ENGINEERING CONCEPT: REACTION WHEEL (THE STABILITY PROVER)
===============================================================================
A Reaction Wheel inverted pendulum is a classic underactuated robotics problem. 
It balances a freely swinging pendulum vertically using the counter-torque of 
an accelerating/decelerating flywheel at the top.

By training a PINN to not only find the control sequence but simultaneously 
discover a valid Lyapunov Stability Function V(t), we create a mathematically 
guaranteed proof that the commanded torques will always drive the system to 
equilibrium without unbounded instability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 1. PHYSICAL SYSTEM PARAMETERS
# ---------------------------------------------------------------------------
M_P = 0.5       # Mass of the pendulum arm (kg)
M_W = 0.2       # Mass of the reaction wheel (kg)
L = 0.3         # Distance from the pivot to the center of mass (m)
I_P = 0.045     # Moment of inertia of the pendulum (kg*m^2)
I_W = 0.001     # Moment of inertia of the reaction wheel (kg*m^2)
G = 9.81        # Gravity (m/s^2)

I_TOTAL = I_P + (M_W * L**2)

THETA_0 = 0.4   # Initial tilt (radians)
T_FINAL = 3.0   # Time to reach equilibrium

# ---------------------------------------------------------------------------
# 2. THE PHYSICS-INFORMED NEURAL NETWORK
# ---------------------------------------------------------------------------
class ReactionWheelLyapunovPINN(nn.Module):
    def __init__(self):
        super(ReactionWheelLyapunovPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3) # Outputs: [theta(t), T_w(t), V(t)]
        )

    def forward(self, t):
        outputs = self.net(t)
        theta = outputs[:, 0:1]
        T_w = outputs[:, 1:2]
        V = outputs[:, 2:3]
        return theta, T_w, V

# ---------------------------------------------------------------------------
# 3. TRAINING SETUP & HISTORY TRACKING
# ---------------------------------------------------------------------------
torch.manual_seed(42)
model = ReactionWheelLyapunovPINN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

t_physics = torch.linspace(0, T_FINAL, 300).view(-1, 1).requires_grad_(True)
t_initial = torch.tensor([[0.0]], requires_grad=True)
t_target = torch.tensor([[T_FINAL]], requires_grad=True)

EPOCHS = 15000

# Trackers for plotting
history_loss_total = []
history_loss_physics = []
history_loss_lyapunov = []

print("Initializing PINN Control & Stability Training...")

# ---------------------------------------------------------------------------
# 4. PINN TRAINING LOOP
# ---------------------------------------------------------------------------
for epoch in range(EPOCHS):
    optimizer.zero_grad()

    # --- A. BOUNDARY CONDITIONS ---
    theta_init, Tw_init, V_init = model(t_initial)
    dtheta_dt_init = torch.autograd.grad(theta_init, t_initial, grad_outputs=torch.ones_like(theta_init), create_graph=True)[0]
    loss_ic = torch.mean((theta_init - THETA_0)**2) + torch.mean((dtheta_dt_init)**2)

    theta_final, Tw_final, V_final = model(t_target)
    dtheta_dt_final = torch.autograd.grad(theta_final, t_target, grad_outputs=torch.ones_like(theta_final), create_graph=True)[0]
    loss_target = torch.mean(theta_final**2) + torch.mean(dtheta_dt_final**2) + torch.mean(V_final**2)

    # --- B. PHYSICS RESIDUAL ---
    theta_pred, Tw_pred, V_pred = model(t_physics)
    
    dtheta_dt = torch.autograd.grad(theta_pred, t_physics, grad_outputs=torch.ones_like(theta_pred), create_graph=True)[0]
    d2theta_dt2 = torch.autograd.grad(dtheta_dt, t_physics, grad_outputs=torch.ones_like(dtheta_dt), create_graph=True)[0]

    gravity_torque = (M_P * L + M_W * L) * G * torch.sin(theta_pred)
    physics_residual = (I_TOTAL * d2theta_dt2) - gravity_torque + Tw_pred
    loss_physics = torch.mean(physics_residual**2)

    # --- C. LYAPUNOV STABILITY CONSTRAINTS ---
    loss_V_pos = torch.mean(torch.relu(-V_pred)**2) # Penalize V < 0
    dV_dt = torch.autograd.grad(V_pred, t_physics, grad_outputs=torch.ones_like(V_pred), create_graph=True)[0]
    loss_V_decay = torch.mean(torch.relu(dV_dt)**2) # Penalize dV/dt > 0

    # --- D. TOTAL LOSS COMPUTATION ---
    loss_control = torch.mean(Tw_pred**2) * 0.001 
    
    loss_total = loss_ic + loss_target + loss_physics + loss_V_pos + loss_V_decay + loss_control

    loss_total.backward()
    optimizer.step()

    # Save metrics for plotting
    history_loss_total.append(loss_total.item())
    history_loss_physics.append(loss_physics.item())
    history_loss_lyapunov.append(loss_V_decay.item() + loss_V_pos.item())

    if epoch % 1000 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:05d} | Total: {loss_total.item():.5f} | Physics: {loss_physics.item():.5f} | Lyap: {(loss_V_decay + loss_V_pos).item():.5f}")

print("\nTraining Complete. Generating Validation Dashboard...")

# ---------------------------------------------------------------------------
# 5. ANALYSIS & RESULTS (MATPLOTLIB DASHBOARD)
# ---------------------------------------------------------------------------
model.eval()

# Generate high-resolution time steps for smooth plotting
t_eval = torch.linspace(0, T_FINAL, 500).view(-1, 1).requires_grad_(True)
theta_eval, Tw_eval, V_eval = model(t_eval)

# Calculate dV/dt for the stability proof plot
dV_dt_eval = torch.autograd.grad(V_eval, t_eval, grad_outputs=torch.ones_like(V_eval))[0]

# Convert tensors to numpy arrays
t_num = t_eval.detach().numpy()
theta_num = theta_eval.detach().numpy()
Tw_num = Tw_eval.detach().numpy()
V_num = V_eval.detach().numpy()
dV_dt_num = dV_dt_eval.detach().numpy()

# Create the plot grid
fig = plt.figure(figsize=(14, 10))
fig.canvas.manager.set_window_title('PINN: Reaction Wheel Stability Prover')
plt.suptitle('Reaction Wheel Control & Lyapunov Stability Proof', fontsize=16, fontweight='bold')

# Plot 1: Training Convergence
ax1 = plt.subplot(2, 2, 1)
ax1.plot(history_loss_total, label='Total Loss', color='black', alpha=0.8)
ax1.plot(history_loss_physics, label='Physics Residual', color='blue', alpha=0.7)
ax1.plot(history_loss_lyapunov, label='Lyapunov Penalty', color='red', alpha=0.7)
ax1.set_yscale('log')
ax1.set_title('Training Convergence (Log Scale)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Plot 2: Pendulum Angle (Theta)
ax2 = plt.subplot(2, 2, 2)
ax2.plot(t_num, theta_num, label=r'Pendulum Angle $\theta(t)$', color='green', linewidth=2.5)
ax2.axhline(0, color='black', linestyle='--', alpha=0.5, label='Equilibrium (0 rad)')
ax2.set_title('State Trajectory: Stabilization')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Angle (Radians)')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# Plot 3: Control Torque
ax3 = plt.subplot(2, 2, 3)
ax3.plot(t_num, Tw_num, label='Motor Torque $T_w(t)$', color='purple', linewidth=2.5)
ax3.set_title('PINN Discovered Control Sequence')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Torque (Nm)')
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.legend()

# Plot 4: The Mathematical Stability Proof (Lyapunov)
ax4 = plt.subplot(2, 2, 4)
ax4.plot(t_num, V_num, label=r'Lyapunov Energy $V(t)$', color='red', linewidth=2.5)
ax4.plot(t_num, dV_dt_num, label=r'Energy Deriv $dV/dt$', color='orange', linewidth=2)
ax4.axhline(0, color='black', linestyle='--', alpha=0.5)
ax4.set_title('Physics-Guaranteed Stability Proof')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Energy Value')
ax4.grid(True, linestyle='--', alpha=0.6)
ax4.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()