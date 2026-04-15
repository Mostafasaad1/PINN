# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║        THE MAGLEV PINN: Overcoming 1/z² Singularities in Control            ║
# ║               Target Audience: Post-Graduate Researchers                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# ── THE ENGINEERING PROBLEM ────────────────────────────────────────────────────
#
#  Move a steel ball from z = 20mm up to z = 5mm smoothly in 1.5 seconds.
#  The dynamics are extremely non-linear: m*z'' = m*g - C*(i/z)^2.
#
# ── THE PINN ARCHITECTURE ──────────────────────────────────────────────────────
#
#   1. THE INPUT: Time (t) from 0 to 1.5s.
#   2. THE OUTPUTS: 
#       - z (Air Gap Position)
#       - i (Coil Current)
#
#   3. THE SAFEGUARD (Architectural Prior):
#      Because a neural network initializes with random weights, an early guess 
#      might be z=0. The term (i/z)^2 would cause a division by zero, exploding 
#      the gradients (NaN). We prevent this by architecturally bounding the 
#      output of the network using Softplus, ensuring the AI mathematically 
#      cannot guess an air gap less than 1mm.
#
# ══════════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt GUI issues
import matplotlib.pyplot as plt

device = torch.device('cpu')
print("=" * 75)
print(" 🧲  MAGLEV PINN: Optimal Non-Linear Trajectory Synthesis")
print("=" * 75)

# ══════════════════════════════════════════════════════════════════════════════
# 1. HARDWARE PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

m = 0.05       # Mass of the ball (kg)
g = 9.81       # Gravity (m/s^2)
C = 0.0001     # Electromagnet force constant
z_min = 0.001  # 1mm absolute minimum gap (Physical hardstop to prevent singularity)

T_target = 1.5 # Trajectory duration
z_start = 0.020 # 20mm (resting on a pedestal)
z_end   = 0.005 # 5mm (tight levitation)

# ══════════════════════════════════════════════════════════════════════════════
# 2. THE NEURAL NETWORK WITH SAFEGUARDS
# ══════════════════════════════════════════════════════════════════════════════

class MagLevNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Mish(), # Mish handles non-linearities beautifully
            nn.Linear(64, 64),
            nn.Mish(),
            nn.Linear(64, 64),
            nn.Mish(),
            nn.Linear(64, 2) # Outputs: [Raw_Z, Raw_I]
        )
        self.softplus = nn.Softplus()

    def forward(self, t):
        out = self.net(t)
        raw_z = out[:, 0:1]
        raw_i = out[:, 1:2]
        
        # ARCHITECTURAL SAFEGUARD: 
        # We force the Z prediction to ALWAYS be strictly positive and > z_min.
        # This prevents the 1/z^2 singularity during early random training epochs!
        safe_z = self.softplus(raw_z) + z_min 
        
        # Current must also be positive (can't push with this electromagnet)
        safe_i = self.softplus(raw_i)
        
        return safe_z, safe_i

model = MagLevNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# ══════════════════════════════════════════════════════════════════════════════
# 3. DOMAIN GENERATION
# ══════════════════════════════════════════════════════════════════════════════

t_physics = torch.linspace(0, T_target, 400, requires_grad=True).view(-1, 1).to(device)
t_start = torch.tensor([[0.0]], requires_grad=True).to(device)
t_end = torch.tensor([[T_target]], requires_grad=True).to(device)

# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

epochs = 8000
print("\n🚀 Synthesizing MagLev Trajectory...")

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    # ─── PART A: BOUNDARY CONDITIONS ──────────────────────────────────────
    
    z_0, i_0 = model(t_start)
    z_T, i_T = model(t_end)

    # Get velocities at boundaries
    def get_vel(t_val):
        z_val, _ = model(t_val)
        dz_dt = torch.autograd.grad(z_val, t_val, grad_outputs=torch.ones_like(z_val), create_graph=True)[0]
        return dz_dt

    v_0 = get_vel(t_start)
    v_T = get_vel(t_end)

    # Start at 20mm, at rest
    loss_start = (z_0 - z_start)**2 + v_0**2
    # End at 5mm, at rest
    loss_end = (z_T - z_end)**2 + v_T**2
    
    loss_bc = loss_start + loss_end

    # ─── PART B: PHYSICS LOSS (The MagLev ODE) ────────────────────────────
    
    z_p, i_p = model(t_physics)

    dz_dt = torch.autograd.grad(z_p, t_physics, grad_outputs=torch.ones_like(z_p), create_graph=True)[0]
    d2z_dt2 = torch.autograd.grad(dz_dt, t_physics, grad_outputs=torch.ones_like(dz_dt), create_graph=True)[0]

    # The 1/z^2 calculation is now mathematically safe thanks to Softplus!
    magnetic_force = C * (i_p / z_p)**2
    
    # m*a = mg - Fm (Positive downward)
    residual = m * d2z_dt2 - (m * g) + magnetic_force

    loss_phys = torch.mean(residual**2)

    # ─── PART C: TOTAL LOSS ───────────────────────────────────────────────
    
    # We enforce the physics heavily, and add a small penalty to smooth the current
    total_loss = (100.0 * loss_bc) + loss_phys + (0.001 * torch.mean(i_p**2))
    
    total_loss.backward()
    optimizer.step()

    if epoch % 1000 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d} | Total: {total_loss.item():.6f} | "
              f"BC: {loss_bc.item():.6f} | Physics: {loss_phys.item():.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZING THE OPTIMAL LIFT
# ══════════════════════════════════════════════════════════════════════════════
print("\n✅ Trajectory Synthesized!")

model.eval()
t_plot = torch.linspace(0, T_target, 200).view(-1, 1).to(device)
with torch.no_grad():
    z_opt, i_opt = model(t_plot)
    
t_np = t_plot.numpy()
z_np = z_opt.numpy() * 1000 # Convert to mm for readable plot
i_np = i_opt.numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PINN MagLev: Synthesized Feed-Forward Trajectory", fontsize=16)

# Position Plot
ax1.plot(t_np, z_np, 'b-', linewidth=3, label='Air Gap z(t)')
ax1.axhline(5.0, color='g', linestyle='--', label='Target: 5mm')
ax1.axhline(20.0, color='gray', linestyle='--', label='Start: 20mm')
# Invert Y axis so visually "up" is towards the magnet (0mm)
ax1.invert_yaxis() 
ax1.set_title("Ball Lift Trajectory")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Air Gap (mm)")
ax1.legend()
ax1.grid(True)

# Current Control Plot
ax2.plot(t_np, i_np, 'r-', linewidth=3, label='Optimal Coil Current i(t)')
ax2.set_title("Required Electromagnet Current")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Current (Amps)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("maglev_results.png")
# plt.show()  # Disabled for non-interactive backend - image saved to maglev_results.png

# Observe the current curve i(t). It does not just linearly increase. 
# It spikes to pull the ball up, then sharply drops to prevent the ball from 
# crashing into the magnet due to the 1/z^2 acceleration, settling precisely 
# at the holding current required for 5mm.