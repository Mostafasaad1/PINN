# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║        THE BALL AND PLATE PINN: Underactuated MIMO Control Planning         ║
# ║                   Target Audience: Chief Systems Engineers                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# ── THE ENGINEERING PROBLEM ────────────────────────────────────────────────────
#
#  Move a solid sphere from a corner of a 2x2 meter plate (x=1, y=1) to the 
#  dead center (x=0, y=0) in exactly 3.0 seconds, stopping it perfectly.
#
# ── THE PINN ARCHITECTURE (R^1 -> R^4) ─────────────────────────────────────────
#
#   1. THE INPUT: Time (t) from 0 to 3.0s.
#   2. THE OUTPUTS: 
#       - x (Ball X Position)
#       - y (Ball Y Position)
#       - theta_x (Plate Pitch - controls Y axis)
#       - theta_y (Plate Roll - controls X axis)
#
#   3. DATA LOSS (The Boundary Conditions):
#      - t=0: x=1, y=1, velocities=0, tilts=0
#      - t=3: x=0, y=0, velocities=0, tilts=0
#
#   4. PHYSICS LOSS:
#      The rolling dynamics of a solid sphere on an inclined plane:
#      x'' = (5/7) * g * sin(theta_y)
#      y'' = -(5/7) * g * sin(theta_x)
#
# ══════════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cpu')
print("=" * 75)
print(" 🔲  BALL & PLATE PINN: Feed-Forward MIMO Trajectory Synthesis")
print("=" * 75)

# ══════════════════════════════════════════════════════════════════════════════
# 1. HARDWARE PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

g = 9.81
# The inertia factor for a solid sphere is 2/5 * m * r^2.
# This yields a translational acceleration multiplier of: m / (m + I/r^2) = 5/7
K = (5.0 / 7.0) * g 

T_target = 3.0 # Execute the maneuver in 3 seconds

# Initial and Final States
pos_start = (1.0, 1.0) # Start in the top right corner
pos_end   = (0.0, 0.0) # End at the dead center

# ══════════════════════════════════════════════════════════════════════════════
# 2. THE NEURAL NETWORK
# ══════════════════════════════════════════════════════════════════════════════

class TableBalancerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 4) # Outputs: [x, y, theta_x, theta_y]
        )

    def forward(self, t):
        # We constrain the tilt angles to reasonable physical limits 
        # (e.g., +/- 30 degrees ≈ 0.5 radians) to prevent the AI from 
        # suggesting flipping the table upside down.
        out = self.net(t)
        x = out[:, 0:1]
        y = out[:, 1:2]
        theta_x = torch.tanh(out[:, 2:3]) * 0.5 
        theta_y = torch.tanh(out[:, 3:4]) * 0.5
        return x, y, theta_x, theta_y

model = TableBalancerNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
loss_fn = nn.MSELoss()

# ══════════════════════════════════════════════════════════════════════════════
# 3. DOMAIN GENERATION
# ══════════════════════════════════════════════════════════════════════════════

t_physics = torch.linspace(0, T_target, 400, requires_grad=True).view(-1, 1).to(device)
t_start = torch.tensor([[0.0]], requires_grad=True).to(device)
t_end   = torch.tensor([[T_target]], requires_grad=True).to(device)

# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

epochs = 6000
print("\n🚀 Computing Optimal MIMO Trajectory...")

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    # ─── PART A: BOUNDARY CONDITIONS ──────────────────────────────────────
    
    # State at t=0
    x_0, y_0, th_x_0, th_y_0 = model(t_start)
    dx_dt_0 = torch.autograd.grad(x_0, t_start, grad_outputs=torch.ones_like(x_0), create_graph=True)[0]
    dy_dt_0 = torch.autograd.grad(y_0, t_start, grad_outputs=torch.ones_like(y_0), create_graph=True)[0]

    # State at t=T
    x_T, y_T, th_x_T, th_y_T = model(t_end)
    dx_dt_T = torch.autograd.grad(x_T, t_end, grad_outputs=torch.ones_like(x_T), create_graph=True)[0]
    dy_dt_T = torch.autograd.grad(y_T, t_end, grad_outputs=torch.ones_like(y_T), create_graph=True)[0]

    # Loss: Start at (1,1), End at (0,0), all velocities and tilts must be zero
    loss_start = (x_0 - pos_start[0])**2 + (y_0 - pos_start[1])**2 + dx_dt_0**2 + dy_dt_0**2 + th_x_0**2 + th_y_0**2
    loss_end   = (x_T - pos_end[0])**2   + (y_T - pos_end[1])**2   + dx_dt_T**2 + dy_dt_T**2 + th_x_T**2 + th_y_T**2
    
    loss_bc = loss_start + loss_end

    # ─── PART B: PHYSICS LOSS (The Rolling ODEs) ──────────────────────────
    
    x_p, y_p, th_x_p, th_y_p = model(t_physics)

    # First derivatives
    dx_dt = torch.autograd.grad(x_p, t_physics, grad_outputs=torch.ones_like(x_p), create_graph=True)[0]
    dy_dt = torch.autograd.grad(y_p, t_physics, grad_outputs=torch.ones_like(y_p), create_graph=True)[0]

    # Second derivatives (Accelerations)
    d2x_dt2 = torch.autograd.grad(dx_dt, t_physics, grad_outputs=torch.ones_like(dx_dt), create_graph=True)[0]
    d2y_dt2 = torch.autograd.grad(dy_dt, t_physics, grad_outputs=torch.ones_like(dy_dt), create_graph=True)[0]

    # Pitch (th_x) tilts the Y axis. Roll (th_y) tilts the X axis.
    # We use small angle approximation or exact sine. Let's use exact.
    res_x = d2x_dt2 - (K * torch.sin(th_y_p))
    res_y = d2y_dt2 - (-K * torch.sin(th_x_p)) # Negative because pitching UP rolls the ball DOWN

    loss_phys = torch.mean(res_x**2) + torch.mean(res_y**2)

    # ─── PART C: TOTAL LOSS ───────────────────────────────────────────────
    
    # We want smooth motor movements, so we penalize jerky tilt commands
    loss_effort = torch.mean(th_x_p**2) + torch.mean(th_y_p**2)

    total_loss = (100.0 * loss_bc) + loss_phys + (0.1 * loss_effort)
    
    total_loss.backward()
    optimizer.step()

    if epoch % 1000 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d} | Total: {total_loss.item():.4f} | "
              f"BC: {loss_bc.item():.4f} | Physics: {loss_phys.item():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZING THE DOUBLE-INTEGRATOR BRAKING
# ══════════════════════════════════════════════════════════════════════════════
print("\n✅ Trajectory Synthesized!")

model.eval()
t_plot = torch.linspace(0, T_target, 200).view(-1, 1).to(device)
with torch.no_grad():
    x_opt, y_opt, th_x_opt, th_y_opt = model(t_plot)
    
t_np = t_plot.numpy()
x_np, y_np = x_opt.numpy(), y_opt.numpy()
th_x_np, th_y_np = th_x_opt.numpy() * (180/np.pi), th_y_opt.numpy() * (180/np.pi) # Convert to degrees

fig = plt.figure(figsize=(15, 5))
fig.suptitle("PINN Ball & Plate: Solving the Double Integrator", fontsize=16)

# Subplot 1: Overhead view of the plate
ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(x_np, y_np, 'b-', linewidth=3)
ax1.scatter([1.0], [1.0], color='red', s=100, label='Start (1,1)', zorder=5)
ax1.scatter([0.0], [0.0], color='green', s=100, label='Target (0,0)', zorder=5)
ax1.set_xlim(-0.2, 1.2)
ax1.set_ylim(-0.2, 1.2)
ax1.set_title("Overhead Ball Path")
ax1.set_xlabel("X Position (m)")
ax1.set_ylabel("Y Position (m)")
ax1.legend()
ax1.grid(True)

# Subplot 2: Ball Position vs Time
ax2 = fig.add_subplot(1, 3, 2)
ax2.plot(t_np, x_np, 'b-', linewidth=2, label='X Position')
ax2.plot(t_np, y_np, 'c--', linewidth=2, label='Y Position')
ax2.axhline(0.0, color='gray', linestyle=':')
ax2.set_title("Ball Coordinates over Time")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Meters")
ax2.legend()
ax2.grid(True)

# Subplot 3: The Plate Tilt Commands (The Magic Braking)
ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(t_np, th_x_np, 'r-', linewidth=2, label='Pitch (θx) [Controls Y]')
ax3.plot(t_np, th_y_np, 'm--', linewidth=2, label='Roll (θy) [Controls X]')
ax3.axhline(0.0, color='gray', linestyle=':')
ax3.set_title("Plate Tilt Commands (Degrees)")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Tilt (Degrees)")
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.savefig("Beam_Ball_results.png")
plt.show()

# Observe Subplot 3 closely. 
# The plate tilts in one direction to start the ball rolling. 
# Then, at the halfway point, the tilt crosses ZERO and goes entirely the OTHER 
# direction to act as a brake, before flattening out to 0 degrees at exactly 3.0s.