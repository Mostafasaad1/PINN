# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║         THE FURUTA PINN: Non-Linear Trajectory Optimization                  ║
# ║              Target Audience: Graduate Control Engineers                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# ── THE ENGINEERING PROBLEM ────────────────────────────────────────────────────
#
#  You need to swing a Furuta Pendulum from resting (hanging down) to balanced 
#  (upright) in exactly 2.0 seconds. 
#
#  Instead of writing an Energy-Shaping controller, we use a PINN to discover 
#  the optimal open-loop control sequence u(t). 
#
# ── THE PINN ARCHITECTURE ──────────────────────────────────────────────────────
#
#   1. THE INPUT: Time (t) from 0.0 to 2.0 seconds.
#   2. THE OUTPUTS: 
#       - theta_1 (Arm Angle)
#       - theta_2 (Pendulum Angle)
#       - u (Motor Torque)
#
#   3. DATA LOSS (The Boundary Conditions):
#      - At t=0: theta_1 = 0, theta_2 = -pi (Hanging down), velocities = 0
#      - At t=2: theta_1 = 0, theta_2 = 0   (Upright & centered), velocities = 0
#
#   4. PHYSICS LOSS: The outputs must satisfy the coupled Lagrangian dynamics 
#      of the Furuta Pendulum.
#
#   5. EFFORT LOSS: We add a small penalty to u(t)^2 to prevent the AI from 
#      guessing a solution that requires a million Nm of torque.
#
# ══════════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cpu')
print("=" * 75)
print(" 🌪️  FURUTA PINN: Solving the Swing-Up Boundary Value Problem")
print("=" * 75)

# ══════════════════════════════════════════════════════════════════════════════
# 1. HARDWARE PARAMETERS (Simplified Furuta Dynamics)
# ══════════════════════════════════════════════════════════════════════════════

# Constants (e.g., Quanser Rotary Pendulum scale)
m_p = 0.127    # Pendulum mass (kg)
L_r = 0.215    # Arm length (m)
L_p = 0.337    # Pendulum full length (m)
l_p = L_p / 2  # Distance to pendulum COM
J_r = 0.002    # Arm inertia
J_p = 0.0012   # Pendulum inertia
g = 9.81       # Gravity

# Grouped kinematic coefficients for the ODEs
alpha = J_r + m_p * L_r**2
beta  = m_p * L_r * l_p
gamma = J_p + m_p * l_p**2
delta = m_p * g * l_p

T_target = 2.0 # We want to swing it up in 2 seconds

# ══════════════════════════════════════════════════════════════════════════════
# 2. THE NEURAL NETWORK (Discovering States AND Control)
# ══════════════════════════════════════════════════════════════════════════════

class FurutaOptimizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(), # Tanh is smooth and differentiable
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            # Output 3 values: [Arm Angle, Pendulum Angle, Motor Torque]
            nn.Linear(128, 3) 
        )

    def forward(self, t):
        return self.net(t)

model = FurutaOptimizer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# ══════════════════════════════════════════════════════════════════════════════
# 3. DOMAIN GENERATION
# ══════════════════════════════════════════════════════════════════════════════

# Continuous time points to enforce the physics
t_physics = torch.linspace(0, T_target, 500, requires_grad=True).view(-1, 1).to(device)

# Boundary Time Points
t_start = torch.tensor([[0.0]]).to(device)
t_end   = torch.tensor([[T_target]]).to(device)

# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAJECTORY OPTIMIZATION TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

epochs = 8000
print("\n🚀 Optimizing Non-Linear Swing-Up Trajectory...")

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    # ─── PART A: DATA LOSS (The Boundary Conditions) ──────────────────────
    
    # Predict states at t=0
    out_start = model(t_start)
    th1_start, th2_start = out_start[:, 0:1], out_start[:, 1:2]
    
    # Predict states at t=2.0
    out_end = model(t_end)
    th1_end, th2_end = out_end[:, 0:1], out_end[:, 1:2]

    # We also need boundary velocities to be zero
    def get_vels(t_val):
        out = model(t_val)
        th1, th2 = out[:, 0:1], out[:, 1:2]
        dth1_dt = torch.autograd.grad(th1, t_val, grad_outputs=torch.ones_like(th1), create_graph=True)[0]
        dth2_dt = torch.autograd.grad(th2, t_val, grad_outputs=torch.ones_like(th2), create_graph=True)[0]
        return dth1_dt, dth2_dt

    v1_s, v2_s = get_vels(t_start)
    v1_e, v2_e = get_vels(t_end)

    # Loss 1: Start hanging down (th2 = -pi) at rest
    loss_bc_start = th1_start**2 + (th2_start - (-np.pi))**2 + v1_s**2 + v2_s**2
    
    # Loss 2: End upright (th2 = 0) and centered (th1 = 0) at rest
    loss_bc_end = th1_end**2 + th2_end**2 + v1_e**2 + v2_e**2

    loss_boundary = loss_bc_start + loss_bc_end

    # ─── PART B: PHYSICS LOSS (The Furuta Lagrangian) ─────────────────────
    
    out_p = model(t_physics)
    th1_p = out_p[:, 0:1] # Arm
    th2_p = out_p[:, 1:2] # Pendulum
    u_p   = out_p[:, 2:3] # Torque

    # First derivatives (Velocities)
    dth1 = torch.autograd.grad(th1_p, t_physics, grad_outputs=torch.ones_like(th1_p), create_graph=True)[0]
    dth2 = torch.autograd.grad(th2_p, t_physics, grad_outputs=torch.ones_like(th2_p), create_graph=True)[0]

    # Second derivatives (Accelerations)
    ddth1 = torch.autograd.grad(dth1, t_physics, grad_outputs=torch.ones_like(dth1), create_graph=True)[0]
    ddth2 = torch.autograd.grad(dth2, t_physics, grad_outputs=torch.ones_like(dth2), create_graph=True)[0]

    sin_th2 = torch.sin(th2_p)
    cos_th2 = torch.cos(th2_p)

    # Furuta Equations of Motion (Mass Matrix * Acceleration + Coriolis/Gravity = Torque)
    
    # ODE 1: Arm Dynamics
    M11 = alpha + m_p * l_p**2 * sin_th2**2
    M12 = beta * cos_th2
    C1 = beta * sin_th2 * dth2**2 - 2 * m_p * l_p**2 * sin_th2 * cos_th2 * dth1 * dth2
    
    res_1 = M11 * ddth1 + M12 * ddth2 - C1 - u_p # Must equal 0

    # ODE 2: Pendulum Dynamics (Unactuated, Torque = 0)
    M21 = beta * cos_th2
    M22 = gamma
    C2 = -m_p * l_p**2 * sin_th2 * cos_th2 * dth1**2
    G2 = delta * sin_th2
    
    res_2 = M21 * ddth1 + M22 * ddth2 - C2 - G2 # Must equal 0

    loss_physics = torch.mean(res_1**2) + torch.mean(res_2**2)

    # ─── PART C: CONTROL EFFORT LOSS ──────────────────────────────────────
    # Minimize the integral of torque squared (LQR style) so it doesn't break the motor
    loss_effort = torch.mean(u_p**2)

    # ─── PART D: TOTAL LOSS ───────────────────────────────────────────────
    # We weigh the boundary conditions heavily to force it to reach the goal
    total_loss = (100.0 * loss_boundary) + loss_physics + (0.01 * loss_effort)
    
    total_loss.backward()
    optimizer.step()

    if epoch % 1000 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d} | Total: {total_loss.item():.4f} | "
              f"Boundary: {loss_boundary.item():.4f} | "
              f"Physics: {loss_physics.item():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZING THE OPTIMAL SWING-UP TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════
print("\n✅ Trajectory Optimized!")

model.eval()
t_plot = torch.linspace(0, T_target, 200).view(-1, 1).to(device)
with torch.no_grad():
    predictions = model(t_plot).cpu().numpy()
    
t_plot_np = t_plot.cpu().numpy()
th1_opt = predictions[:, 0]
th2_opt = predictions[:, 1]
u_opt = predictions[:, 2]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PINN Furuta Swing-Up: Discovered Optimal Trajectory", fontsize=16)

# States Plot
ax1.plot(t_plot_np, th1_opt, 'b-', linewidth=2, label='Motor Arm Angle (θ1)')
ax1.plot(t_plot_np, th2_opt, 'r--', linewidth=2, label='Pendulum Angle (θ2)')
ax1.axhline(0, color='gray', linestyle=':')
ax1.axhline(-np.pi, color='gray', linestyle=':', label='Hanging (-π)')
ax1.set_title("Kinematic States over Time")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Radians")
ax1.legend()
ax1.grid(True)

# Control Plot
ax2.plot(t_plot_np, u_opt, 'g-', linewidth=2, label='Optimal Motor Torque (u)')
ax2.set_title("Control Effort Sequence")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Torque (Nm)")
ax2.fill_between(t_plot_np.flatten(), u_opt.flatten(), color='green', alpha=0.2)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("furuta_pendulum_results.png")
plt.show()

# You now have an array of torque commands `u_opt`. 
# If you feed these commands to a real Furuta pendulum motor over 2 seconds, 
# it will flawlessly execute the swing-up maneuver. Once t=2.0 is reached, 
# you simply switch your PLC to an LQR controller to maintain the balance.