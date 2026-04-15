# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║        THE 4D SPACETIME PINN: Solving the 3D Heat Equation over Time        ║
# ║                    Target Audience: STEM High Schoolers                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# ── THE ENGINEERING PROBLEM ────────────────────────────────────────────────────
#
#  You are designing a copper heatsink for a high-performance GPU.
#  The GPU is suddenly turned on, blasting heat into the bottom of the block.
#  We need to know the temperature (T) at any point (x, y, z) inside the 
#  block, at any time (t).
#
#  We are mapping a 4D input space (x, y, z, t) to a 1D output space (T).
#
# ── THE PINN ARCHITECTURE ──────────────────────────────────────────────────────
#
#   1. DATA LOSS (Boundary & Initial Conditions):
#      - Initial Condition (t=0): The whole block is at room temp (20°C).
#      - Boundary Condition (z=0): The bottom face is touching the GPU (100°C).
#
#   2. PHYSICS LOSS (The PDE):
#      - Inside the block, the temperature must obey the 3D Heat Equation:
#        dT/dt = alpha * (d²T/dx² + d²T/dy² + d²T/dz²)
#      - The Neural Network will use PyTorch's Autograd to calculate these 
#        partial derivatives and force the math to balance to zero.
#
# ── WHY THIS IS REVOLUTIONARY ──────────────────────────────────────────────────
#   Traditional solvers (Finite Element Analysis - FEA) require slicing the 
#   3D block into millions of tiny grid cubes (meshing) and stepping through 
#   time slowly. This PINN is "mesh-free". It learns the continuous function, 
#   allowing us to query ANY coordinate at ANY time instantly.
#
# ══════════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# 1. PLAYGROUND SETUP (4D Spacetime Domain)
# ══════════════════════════════════════════════════════════════════════════════

# Material Property: Thermal Diffusivity of our material (scaled for demo)
alpha = 0.05 

# Domain bounds
L = 1.0       # The block is 1x1x1 units in size
T_max = 2.0   # We are simulating 2 seconds of time

device = torch.device('cpu')
print("=" * 70)
print(" 🌡️  4D PINN: Spatiotemporal Heat Equation Solver")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# 2. GENERATING COLLOCATION POINTS (Where we enforce the rules)
# ══════════════════════════════════════════════════════════════════════════════

# We randomly sample points in our 4D space to train the network.
N_physics = 5000  # Points inside the block to enforce the PDE
N_bc = 1000       # Points on the boundaries (Data Loss)
N_ic = 1000       # Points at time t=0 (Data Loss)

# --- A. Physics Points (Randomly scattered in x, y, z, t) ---
# We need to track gradients for these to calculate our PDE!
x_phys = (torch.rand(N_physics, 1, requires_grad=True) * L).to(device)
y_phys = (torch.rand(N_physics, 1, requires_grad=True) * L).to(device)
z_phys = (torch.rand(N_physics, 1, requires_grad=True) * L).to(device)
t_phys = (torch.rand(N_physics, 1, requires_grad=True) * T_max).to(device)

# --- B. Initial Condition Points (t = 0) ---
x_ic = (torch.rand(N_ic, 1) * L).to(device)
y_ic = (torch.rand(N_ic, 1) * L).to(device)
z_ic = (torch.rand(N_ic, 1) * L).to(device)
t_ic = torch.zeros(N_ic, 1).to(device)
T_ic_target = torch.full((N_ic, 1), 20.0).to(device) # Room temp: 20°C

# --- C. Boundary Condition Points (GPU touching bottom: z = 0) ---
x_bc = (torch.rand(N_bc, 1) * L).to(device)
y_bc = (torch.rand(N_bc, 1) * L).to(device)
z_bc = torch.zeros(N_bc, 1).to(device) # z=0 is the bottom face
t_bc = (torch.rand(N_bc, 1) * T_max).to(device)
T_bc_target = torch.full((N_bc, 1), 100.0).to(device) # GPU temp: 100°C

# ══════════════════════════════════════════════════════════════════════════════
# 3. THE NEURAL NETWORK (The Universal Function Approximator)
# ══════════════════════════════════════════════════════════════════════════════

class HeatPDE_Solver(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), # 4 Inputs: x, y, z, t
            nn.Tanh(),        # Tanh is infinitely differentiable (crucial for PDEs!)
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)  # 1 Output: Temperature (T)
        )

    def forward(self, x, y, z, t):
        # Stack the 4 inputs together
        inputs = torch.cat([x, y, z, t], dim=1)
        return self.net(inputs)

model = HeatPDE_Solver().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# ══════════════════════════════════════════════════════════════════════════════
# 4. THE TRAINING LOOP (Enforcing the Math)
# ══════════════════════════════════════════════════════════════════════════════

epochs = 5000
loss_fn = nn.MSELoss()

print("\n🚀 Training the Spatiotemporal PDE Solver...")

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    # ─── PART 1: DATA LOSS (Initial & Boundary Conditions) ────────────────
    
    # Check Initial Condition (Is it 20°C at t=0?)
    T_ic_pred = model(x_ic, y_ic, z_ic, t_ic)
    loss_ic = loss_fn(T_ic_pred, T_ic_target)

    # Check Boundary Condition (Is it 100°C at the bottom face?)
    T_bc_pred = model(x_bc, y_bc, z_bc, t_bc)
    loss_bc = loss_fn(T_bc_pred, T_bc_target)

    # ─── PART 2: PHYSICS LOSS (The Heat Equation PDE) ─────────────────────
    
    # Predict temperature at our random 4D physics points
    T_phys = model(x_phys, y_phys, z_phys, t_phys)

    # Calculate First Derivatives (Time and Space)
    dT_dt = torch.autograd.grad(T_phys, t_phys, grad_outputs=torch.ones_like(T_phys), create_graph=True)[0]
    
    dT_dx = torch.autograd.grad(T_phys, x_phys, grad_outputs=torch.ones_like(T_phys), create_graph=True)[0]
    dT_dy = torch.autograd.grad(T_phys, y_phys, grad_outputs=torch.ones_like(T_phys), create_graph=True)[0]
    dT_dz = torch.autograd.grad(T_phys, z_phys, grad_outputs=torch.ones_like(T_phys), create_graph=True)[0]

    # Calculate Second Derivatives (Spatial Laplacian)
    d2T_dx2 = torch.autograd.grad(dT_dx, x_phys, grad_outputs=torch.ones_like(dT_dx), create_graph=True)[0]
    d2T_dy2 = torch.autograd.grad(dT_dy, y_phys, grad_outputs=torch.ones_like(dT_dy), create_graph=True)[0]
    d2T_dz2 = torch.autograd.grad(dT_dz, z_phys, grad_outputs=torch.ones_like(dT_dz), create_graph=True)[0]

    # The PDE Residual (If the math is perfect, this equals 0)
    # dT/dt - alpha * (d2T/dx2 + d2T/dy2 + d2T/dz2) = 0
    pde_residual = dT_dt - alpha * (d2T_dx2 + d2T_dy2 + d2T_dz2)
    
    # The loss is how far away from 0 the residual is
    loss_pde = loss_fn(pde_residual, torch.zeros_like(pde_residual))

    # ─── PART 3: TOTAL LOSS & BACKPROP ────────────────────────────────────
    
    # We sum the losses. Sometimes we weigh them differently, but 1:1:1 is fine here.
    total_loss = loss_ic + loss_bc + loss_pde
    
    total_loss.backward()
    optimizer.step()

    if epoch % 1000 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d} | Total: {total_loss.item():.4f} "
              f"| Init (t=0): {loss_ic.item():.4f} "
              f"| Bound (z=0): {loss_bc.item():.4f} "
              f"| PDE (Physics): {loss_pde.item():.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZATION (Slicing the 4D Space)
# ══════════════════════════════════════════════════════════════════════════════
print("\n📊 Simulating Heat Diffusion Cross-Section...")

# To visualize 4D, we have to hold two variables constant.
# We will look at a 2D vertical slice of the block (y = 0.5)
# and we will look at two different snapshots in time (t=0.2s and t=1.8s).

# Create a mesh grid for X and Z axes
grid_res = 50
x_val = np.linspace(0, L, grid_res)
z_val = np.linspace(0, L, grid_res)
X_mesh, Z_mesh = np.meshgrid(x_val, z_val)

# Flatten for neural network
X_flat = torch.tensor(X_mesh.flatten(), dtype=torch.float32).unsqueeze(1)
Z_flat = torch.tensor(Z_mesh.flatten(), dtype=torch.float32).unsqueeze(1)

# Hold Y constant at the center of the block
Y_flat = torch.full_like(X_flat, L/2)

model.eval()

def predict_slice_at_time(t_snapshot):
    T_flat = torch.full_like(X_flat, t_snapshot)
    with torch.no_grad():
        Temp_pred = model(X_flat, Y_flat, Z_flat, T_flat).numpy()
    return Temp_pred.reshape(grid_res, grid_res)

# Get predictions at early time and late time
Temp_early = predict_slice_at_time(0.2)
Temp_late  = predict_slice_at_time(1.8)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("PINN Prediction: Cross-Section of CPU Heatsink Warming Up", fontsize=16)

# Time = 0.2s
c1 = ax1.contourf(X_mesh, Z_mesh, Temp_early, levels=50, cmap="inferno", vmin=20, vmax=100)
ax1.set_title("t = 0.2 seconds (Heat just entering)")
ax1.set_xlabel("Width X")
ax1.set_ylabel("Height Z (z=0 is GPU)")
fig.colorbar(c1, ax=ax1, label="Temperature °C")

# Time = 1.8s
c2 = ax2.contourf(X_mesh, Z_mesh, Temp_late, levels=50, cmap="inferno", vmin=20, vmax=100)
ax2.set_title("t = 1.8 seconds (Heat diffusing up)")
ax2.set_xlabel("Width X")
ax2.set_ylabel("Height Z (z=0 is GPU)")
fig.colorbar(c2, ax=ax2, label="Temperature °C")

plt.tight_layout()
plt.savefig("pinn_4d_results.png")
plt.show()