# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║       THE 5D PARAMETRIC PINN: Universal Battery Thermal Management          ║
# ║               Target Audience: 1st-Year Engineering Students                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# ── THE ENGINEERING PROBLEM ────────────────────────────────────────────────────
#
#  You are designing a battery pack for an EV. A battery cell generates heat.
#  You need to surround it with a cooling material. 
#  You have a budget, and you want to test materials ranging from cheap plastics 
#  to expensive copper. 
#
#  Instead of running 100 different ANSYS/COMSOL simulations for 100 different 
#  materials, we will train ONE Neural Network to solve the continuous 5D space:
#  (X, Y, Z, Time, Material_Diffusivity).
#
# ── THE PINN ARCHITECTURE (R^5 -> R^1) ─────────────────────────────────────────
#
#   1. DATA LOSS (Boundary & Initial Conditions):
#      - Initial: Battery and material start at 20°C ambient.
#      - Boundary: The center of the block (the battery) is fixed at 80°C.
#
#   2. PHYSICS LOSS (The Parametric PDE):
#      - The PDE is the 3D Heat Equation, but notice that alpha (α) is no 
#        longer a fixed number at the top of the script. It is an INPUT TENSOR.
#        dT/dt = α_input * (d²T/dx² + d²T/dy² + d²T/dz²)
#
# ── THE RESULT ─────────────────────────────────────────────────────────────────
#   A "Surrogate Model". An AI that understands how heat flows through ANY 
#   material, instantly queryable, mesh-free, and bounded by thermodynamics.
#
# ══════════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# 1. DEFINING THE 5D DESIGN SPACE (Spacetime + Material)
# ══════════════════════════════════════════════════════════════════════════════

# Spatial Bounds (A 1m x 1m x 1m block)
L = 1.0       
# Temporal Bound (Simulating 2 seconds of heat transfer)
T_max = 2.0   
# Material Bound (The 5th Dimension: Thermal Diffusivity range)
# e.g., 0.01 (slow heat transfer like plastic) to 0.1 (fast like metal)
alpha_min = 0.01
alpha_max = 0.10

device = torch.device('cpu')
print("=" * 75)
print(" 🔋  5D PINN: Parametric Surrogate Model for Battery Thermal Design")
print("=" * 75)

# ══════════════════════════════════════════════════════════════════════════════
# 2. GENERATING 5D COLLOCATION POINTS
# ══════════════════════════════════════════════════════════════════════════════

N_physics = 8000  # Points inside the domain to enforce thermodynamics
N_bc = 2000       # Points on the boundaries
N_ic = 2000       # Points at t=0

# --- A. Physics Points (Randomly scattered in x, y, z, t, AND alpha) ---
# We track gradients for x, y, z, and t to calculate the PDE.
# We DO NOT need to track the gradient of alpha, because the derivative with 
# respect to material property isn't in the heat equation! It's just a coefficient.
x_p = (torch.rand(N_physics, 1, requires_grad=True) * L).to(device)
y_p = (torch.rand(N_physics, 1, requires_grad=True) * L).to(device)
z_p = (torch.rand(N_physics, 1, requires_grad=True) * L).to(device)
t_p = (torch.rand(N_physics, 1, requires_grad=True) * T_max).to(device)

# The 5th Dimension: Randomly sampling different materials!
alpha_p = (torch.rand(N_physics, 1) * (alpha_max - alpha_min) + alpha_min).to(device)

# --- B. Initial Condition Points (t = 0, for ALL materials) ---
x_ic = (torch.rand(N_ic, 1) * L).to(device)
y_ic = (torch.rand(N_ic, 1) * L).to(device)
z_ic = (torch.rand(N_ic, 1) * L).to(device)
t_ic = torch.zeros(N_ic, 1).to(device)
alpha_ic = (torch.rand(N_ic, 1) * (alpha_max - alpha_min) + alpha_min).to(device)
T_ic_target = torch.full((N_ic, 1), 20.0).to(device) # Ambient 20°C

# --- C. Boundary Condition Points (Hot Battery in the center) ---
# For simplicity, we just say the exact coordinate (0.5, 0.5, 0.5) is 80°C.
x_bc = torch.full((N_bc, 1), L/2).to(device)
y_bc = torch.full((N_bc, 1), L/2).to(device)
z_bc = torch.full((N_bc, 1), L/2).to(device)
t_bc = (torch.rand(N_bc, 1) * T_max).to(device)
alpha_bc = (torch.rand(N_bc, 1) * (alpha_max - alpha_min) + alpha_min).to(device)
T_bc_target = torch.full((N_bc, 1), 80.0).to(device) # Hot battery core 80°C

# ══════════════════════════════════════════════════════════════════════════════
# 3. THE NEURAL NETWORK (5 Inputs -> 1 Output)
# ══════════════════════════════════════════════════════════════════════════════

class ParametricHeatSolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64), # 5 INPUTS: (x, y, z, t, alpha)
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)  # 1 OUTPUT: Temperature
        )

    def forward(self, x, y, z, t, alpha):
        inputs = torch.cat([x, y, z, t, alpha], dim=1)
        return self.net(inputs)

model = ParametricHeatSolver().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ══════════════════════════════════════════════════════════════════════════════
# 4. THE TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

epochs = 5000
loss_fn = nn.MSELoss()

print("\n🚀 Training the 5D Meta-Solver...")

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    # ─── PART 1: DATA LOSS (Is it obeying the physical setup?) ────────────
    
    T_ic_pred = model(x_ic, y_ic, z_ic, t_ic, alpha_ic)
    loss_ic = loss_fn(T_ic_pred, T_ic_target)

    T_bc_pred = model(x_bc, y_bc, z_bc, t_bc, alpha_bc)
    loss_bc = loss_fn(T_bc_pred, T_bc_target)

    # ─── PART 2: PHYSICS LOSS (Is it obeying thermodynamics?) ─────────────
    
    T_p = model(x_p, y_p, z_p, t_p, alpha_p)

    # Calculus: Spatial and Temporal Gradients
    dT_dt = torch.autograd.grad(T_p, t_p, grad_outputs=torch.ones_like(T_p), create_graph=True)[0]
    dT_dx = torch.autograd.grad(T_p, x_p, grad_outputs=torch.ones_like(T_p), create_graph=True)[0]
    dT_dy = torch.autograd.grad(T_p, y_p, grad_outputs=torch.ones_like(T_p), create_graph=True)[0]
    dT_dz = torch.autograd.grad(T_p, z_p, grad_outputs=torch.ones_like(T_p), create_graph=True)[0]

    d2T_dx2 = torch.autograd.grad(dT_dx, x_p, grad_outputs=torch.ones_like(dT_dx), create_graph=True)[0]
    d2T_dy2 = torch.autograd.grad(dT_dy, y_p, grad_outputs=torch.ones_like(dT_dy), create_graph=True)[0]
    d2T_dz2 = torch.autograd.grad(dT_dz, z_p, grad_outputs=torch.ones_like(dT_dz), create_graph=True)[0]

    # THE MAGIC LINE: Notice that alpha_p is a TENSOR, matching the material 
    # property for that specific physical coordinate being evaluated.
    pde_residual = dT_dt - alpha_p * (d2T_dx2 + d2T_dy2 + d2T_dz2)
    
    loss_pde = loss_fn(pde_residual, torch.zeros_like(pde_residual))

    # ─── PART 3: TOTAL LOSS ───────────────────────────────────────────────
    
    total_loss = loss_ic + loss_bc + loss_pde
    total_loss.backward()
    optimizer.step()

    if epoch % 1000 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d} | Total: {total_loss.item():.4f} "
              f"| Init: {loss_ic.item():.4f} "
              f"| Bound: {loss_bc.item():.4f} "
              f"| PDE: {loss_pde.item():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZING THE 5TH DIMENSION
# ══════════════════════════════════════════════════════════════════════════════
print("\n📊 Querying the AI for two entirely different materials...")

# Let's look at a 2D slice (Z=0.5, Y=0.5) right through the center of the block.
# We will freeze Time at t=1.0 seconds.
# We will query the AI TWICE. Once for Plastic, once for Copper.

grid_res = 60
x_val = np.linspace(0, L, grid_res)
y_val = np.linspace(0, L, grid_res)
X_mesh, Y_mesh = np.meshgrid(x_val, y_val)

X_flat = torch.tensor(X_mesh.flatten(), dtype=torch.float32).unsqueeze(1)
Y_flat = torch.tensor(Y_mesh.flatten(), dtype=torch.float32).unsqueeze(1)
Z_flat = torch.full_like(X_flat, L/2)  # Slice through the center
T_flat = torch.full_like(X_flat, 1.0)  # Freeze time at 1 second

model.eval()

# ── Query 1: Plastic-like Material (Low Diffusivity) ──
Alpha_Plastic = torch.full_like(X_flat, 0.01)
with torch.no_grad():
    T_plastic = model(X_flat, Y_flat, Z_flat, T_flat, Alpha_Plastic).numpy().reshape(grid_res, grid_res)

# ── Query 2: Copper-like Material (High Diffusivity) ──
Alpha_Copper = torch.full_like(X_flat, 0.10)
with torch.no_grad():
    T_copper = model(X_flat, Y_flat, Z_flat, T_flat, Alpha_Copper).numpy().reshape(grid_res, grid_res)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("5D PINN: Temperature Profile of Two Different Materials at t=1.0s", fontsize=16)

c1 = ax1.contourf(X_mesh, Y_mesh, T_plastic, levels=40, cmap="magma", vmin=20, vmax=80)
ax1.set_title("Material A (Plastic / Low α=0.01)\nHeat stays trapped near the battery.")
ax1.set_xlabel("X coordinate")
ax1.set_ylabel("Y coordinate")
fig.colorbar(c1, ax=ax1, label="Temperature °C")

c2 = ax2.contourf(X_mesh, Y_mesh, T_copper, levels=40, cmap="magma", vmin=20, vmax=80)
ax2.set_title("Material B (Copper / High α=0.10)\nHeat dissipates rapidly through the block.")
ax2.set_xlabel("X coordinate")
ax2.set_ylabel("Y coordinate")
fig.colorbar(c2, ax=ax2, label="Temperature °C")

plt.tight_layout()
plt.savefig("pinn_5d_results.png")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# THE FRESHMAN ENGINEERING TAKEAWAY
# ══════════════════════════════════════════════════════════════════════════════
#
# Look at the plots you just generated. 
# The AI was NEVER shown what "plastic" or "copper" heat signatures look like.
# It was only shown the differential equation:
# dT/dt - alpha * (d2T/dx2 + ...) = 0
# 
# Because it learned how the *math* behaves relative to the parameter alpha, 
# you can now pass a slider from 0.01 to 0.10, and watch the heat dynamically 
# spread out in real-time. 
#
# In traditional engineering software, building this parametric sweep would take 
# hours of compute time. With a 5D PINN, inference (generating the plot) takes 
# milliseconds. This is how the next generation of engineers will design 
# rockets, batteries, and robotics.