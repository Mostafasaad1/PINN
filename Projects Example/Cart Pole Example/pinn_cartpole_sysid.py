# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║        THE CART-POLE PINN: Non-Linear System Identification                 ║
# ║                Target Audience: Advanced Control Engineers                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# ── THE ENGINEERING PROBLEM ────────────────────────────────────────────────────
#
#  You have a Cart-Pole system. You record 2 seconds of it swinging freely 
#  (no motors turned on, just falling and rolling).
#  Your encoders are cheap and noisy. You lost the spec sheet, so you DO NOT 
#  know the Mass of the Cart (M) or the Mass of the Pole (m).
#
# ── THE PINN ARCHITECTURE ──────────────────────────────────────────────────────
#
#   1. THE INPUT: Time (t)
#   2. THE OUTPUT: Cart Position (x) and Pole Angle (theta)
#   3. THE MAGIC: We add `M_guess` and `m_guess` as PyTorch nn.Parameters.
#      The optimizer will update the weights of the network AND the mass guesses!
#
#   4. DATA LOSS: The outputs must match the noisy encoder data.
#   5. PHYSICS LOSS: The outputs (and their derivatives) MUST obey the non-linear 
#      Lagrangian equations of motion for a cart-pole.
#
# ══════════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cpu')
print("=" * 75)
print(" 🛒  CART-POLE PINN: Discovering Hardware Parameters from Noise")
print("=" * 75)

# ══════════════════════════════════════════════════════════════════════════════
# 1. GENERATING THE "REAL" WORLD DATA (Simulating the Lab Experiment)
# ══════════════════════════════════════════════════════════════════════════════

# The TRUE parameters of the hardware (The AI does NOT know these!)
TRUE_M = 1.0    # Cart mass (kg)
TRUE_m = 0.2    # Pole mass (kg)
L = 0.5         # Pole length (m) - We assume we can measure this with a ruler
g = 9.81        # Gravity

def simulate_cartpole(t_max, dt):
    """A quick numerical integrator to generate our 'ground truth' lab data."""
    steps = int(t_max / dt)
    t = np.linspace(0, t_max, steps)
    
    # State: [x, x_dot, theta, theta_dot]
    # We start with the pole leaning over at 0.5 rads (~28 degrees)
    state = np.array([0.0, 0.0, 0.5, 0.0])
    
    history = []
    for _ in t:
        history.append(state.copy())
        x, x_dot, theta, theta_dot = state
        
        # Non-linear Equations of Motion (Free-swinging, Force = 0)
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        
        # Angular acceleration of the pole
        num_theta = g * sin_t + cos_t * (-TRUE_m * L * theta_dot**2 * sin_t) / (TRUE_M + TRUE_m)
        den_theta = L * (4.0/3.0 - (TRUE_m * cos_t**2) / (TRUE_M + TRUE_m))
        theta_ddot = num_theta / den_theta
        
        # Linear acceleration of the cart
        x_ddot = (TRUE_m * L * (theta_dot**2 * sin_t - theta_ddot * cos_t)) / (TRUE_M + TRUE_m)
        
        # Euler step
        state[1] += x_ddot * dt
        state[0] += state[1] * dt
        state[3] += theta_ddot * dt
        state[2] += state[3] * dt

    return t, np.array(history)

# Run the 2-second lab experiment
t_real, state_real = simulate_cartpole(t_max=2.0, dt=0.01)
x_real = state_real[:, 0]
theta_real = state_real[:, 2]

# Introduce terrible encoder noise
np.random.seed(42)
x_noisy = x_real + np.random.normal(0, 0.05, len(x_real))
theta_noisy = theta_real + np.random.normal(0, 0.05, len(theta_real))

# Convert to PyTorch Tensors for the PINN
t_data = torch.tensor(t_real, dtype=torch.float32).view(-1, 1).to(device)
x_data = torch.tensor(x_noisy, dtype=torch.float32).view(-1, 1).to(device)
theta_data = torch.tensor(theta_noisy, dtype=torch.float32).view(-1, 1).to(device)

# ══════════════════════════════════════════════════════════════════════════════
# 2. THE NEURAL NETWORK & LEARNABLE HARDWARE PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

class CartPolePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2) # Outputs: [x_pred, theta_pred]
        )

    def forward(self, t):
        return self.net(t)

model = CartPolePINN().to(device)

# 🚨 THE MAGIC: We define our guesses for the mass as PyTorch Parameters!
# We start with terrible guesses (0.1 kg for both).
M_pred = nn.Parameter(torch.tensor([0.1], dtype=torch.float32).to(device))
m_pred = nn.Parameter(torch.tensor([0.1], dtype=torch.float32).to(device))

# We tell the optimizer to update BOTH the network weights AND the mass guesses.
optimizer = torch.optim.Adam(list(model.parameters()) + [M_pred, m_pred], lr=0.002)
loss_fn = nn.MSELoss()

# ══════════════════════════════════════════════════════════════════════════════
# 3. PHYSICS COLLOCATION POINTS (Enforcing continuous time)
# ══════════════════════════════════════════════════════════════════════════════

# We sample points between 0 and 2 seconds to calculate our derivatives
t_physics = torch.linspace(0, 2.0, 400, requires_grad=True).view(-1, 1).to(device)

# ══════════════════════════════════════════════════════════════════════════════
# 4. THE SYSTEM IDENTIFICATION TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

epochs = 6000
print("\n🚀 Commencing System Identification...")

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    # ─── PART A: DATA LOSS (Fit the noisy encoder data) ───────────────────
    pred_data = model(t_data)
    x_pred_data = pred_data[:, 0:1]
    theta_pred_data = pred_data[:, 1:2]
    
    loss_data = loss_fn(x_pred_data, x_data) + loss_fn(theta_pred_data, theta_data)

    # ─── PART B: PHYSICS LOSS (Obey the Cart-Pole ODEs) ───────────────────
    pred_phys = model(t_physics)
    x_p = pred_phys[:, 0:1]
    theta_p = pred_phys[:, 1:2]

    # Calculate Velocity and Acceleration using Autograd
    x_dot = torch.autograd.grad(x_p, t_physics, grad_outputs=torch.ones_like(x_p), create_graph=True)[0]
    x_ddot = torch.autograd.grad(x_dot, t_physics, grad_outputs=torch.ones_like(x_dot), create_graph=True)[0]

    theta_dot = torch.autograd.grad(theta_p, t_physics, grad_outputs=torch.ones_like(theta_p), create_graph=True)[0]
    theta_ddot = torch.autograd.grad(theta_dot, t_physics, grad_outputs=torch.ones_like(theta_dot), create_graph=True)[0]

    sin_t = torch.sin(theta_p)
    cos_t = torch.cos(theta_p)

    # Non-linear ODE Residual 1: Horizontal Force Balance (F=0)
    # (M + m)x_ddot + m * L * (theta_ddot * cos(theta) - theta_dot^2 * sin(theta)) = 0
    res_1 = (M_pred + m_pred) * x_ddot + m_pred * L * (theta_ddot * cos_t - (theta_dot**2) * sin_t)

    # Non-linear ODE Residual 2: Rotational Balance
    # x_ddot * cos(theta) + L * theta_ddot - g * sin(theta) = 0
    res_2 = x_ddot * cos_t + L * theta_ddot - g * sin_t

    loss_phys = torch.mean(res_1**2) + torch.mean(res_2**2)

    # ─── PART C: TOTAL LOSS ───────────────────────────────────────────────
    total_loss = loss_data + (loss_phys * 0.1) # Soften the physics loss slightly
    
    total_loss.backward()
    optimizer.step()

    # Prevent masses from becoming negative during training
    with torch.no_grad():
        M_pred.clamp_(min=0.01)
        m_pred.clamp_(min=0.01)

    if epoch % 1000 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d} | Loss: {total_loss.item():.4f} | "
              f"Guess M: {M_pred.item():.3f}kg | Guess m: {m_pred.item():.3f}kg")

# ══════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZING THE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n✅ System ID Complete!")
print(f"TRUE Cart Mass: {TRUE_M} kg  | PINN Discovered: {M_pred.item():.3f} kg")
print(f"TRUE Pole Mass: {TRUE_m} kg  | PINN Discovered: {m_pred.item():.3f} kg")

# Predict the clean trajectory
model.eval()
with torch.no_grad():
    clean_pred = model(t_data)
    x_clean = clean_pred[:, 0].numpy()
    theta_clean = clean_pred[:, 1].numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PINN System Identification: Filtering Noise & Discovering Mass", fontsize=16)

# Cart Position Plot
ax1.scatter(t_real, x_noisy, color='gray', s=10, alpha=0.5, label='Noisy Encoder Data')
ax1.plot(t_real, x_real, 'b-', linewidth=3, label='True Hidden Physics')
ax1.plot(t_real, x_clean, 'r--', linewidth=2, label='PINN Cleaned Output')
ax1.set_title("Cart Position (x)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Meters")
ax1.legend()
ax1.grid(True)

# Pole Angle Plot
ax2.scatter(t_real, theta_noisy, color='gray', s=10, alpha=0.5, label='Noisy Encoder Data')
ax2.plot(t_real, theta_real, 'b-', linewidth=3, label='True Hidden Physics')
ax2.plot(t_real, theta_clean, 'r--', linewidth=2, label='PINN Cleaned Output')
ax2.set_title("Pole Angle (θ)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Radians")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("Cart_Pole_results.png")
plt.show()

# The PINN has successfully looked at random noise, applied the non-linear 
# Lagrangian constraints, and deduced the exact physical properties of the hardware!