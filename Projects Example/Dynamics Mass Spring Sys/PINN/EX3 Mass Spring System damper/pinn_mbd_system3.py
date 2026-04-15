import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

# ==========================================
# 1. PHYSICAL PARAMETERS & ANALYTICAL BASIS
# ==========================================
# Values derived from MBD_Systems_Analysis.md
m = 10.0   # Mass [kg]
k = 200.0  # Stiffness [N/m]
c = 15.0   # Damping [Ns/m]
g = 9.81   # Gravity [m/s^2]
x0, v0 = 1.2, 3.0  # ICs: [m], [m/s]
t_max = 5.0

# Vibration Engineering Constants:
wn = np.sqrt(k/m)          # Natural Frequency (rad/s)
zeta = c / (2 * np.sqrt(m*k)) # Damping Ratio (Underdamped if < 1)
wd = wn * np.sqrt(1 - zeta**2) # Damped Natural Frequency

print(f"--- System Dynamics Analysis ---")
print(f"Natural Freq: {wn:.2f} rad/s | Damping Ratio: {zeta:.3f} (Underdamped)")

# ==========================================
# 2. NUMERICAL BENCHMARK (RK45)
# ==========================================
def mbd_ground_truth(t, state):
    x, v = state
    # MBD Matrix Solution: x'' = (F_spring + F_damper) / m
    dxdt = v
    dvdt = (-k*x - c*v) / m
    return [dxdt, dvdt]

t_eval = np.linspace(0, t_max, 200)
sol = solve_ivp(mbd_ground_truth, [0, t_max], [x0, v0], t_eval=t_eval, method='RK45')
x_ref = sol.y[0]
v_ref = sol.y[1]

# ==========================================
# 3. PINN ARCHITECTURE
# ==========================================
class VibrationPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: t (1) -> Output: [x, y, lambda] (3)
        self.net = nn.Sequential(
            nn.Linear(1, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 3)
        )

    def forward(self, t):
        return self.net(t)

def get_physics_loss(model, t_collocation):
    t_collocation.requires_grad = True
    pred = model(t_collocation)
    x, y, lam = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]

    # Velocity and Acceleration via Automatic Differentiation
    dx_dt = torch.autograd.grad(x, t_collocation, torch.ones_like(x), create_graph=True)[0]
    d2x_dt2 = torch.autograd.grad(dx_dt, t_collocation, torch.ones_like(dx_dt), create_graph=True)[0]
    
    dy_dt = torch.autograd.grad(y, t_collocation, torch.ones_like(y), create_graph=True)[0]
    d2y_dt2 = torch.autograd.grad(dy_dt, t_collocation, torch.ones_like(dy_dt), create_graph=True)[0]

    # --- Physics Residuals (MBD Formulation) ---
    # R1: Dynamics in X (m*x'' + c*x' + k*x = 0)
    res_x = m * d2x_dt2 + c * dx_dt + k * x 
    
    # R2: Dynamics in Y (m*y'' + lambda + m*g = 0)
    res_y = m * d2y_dt2 + lam + m * g
    
    # R3: Kinematic Constraint (y = 0)
    res_phi = y 

    return torch.mean(res_x**2) + torch.mean(res_y**2) + 10 * torch.mean(res_phi**2)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
pinn = VibrationPINN()
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
t_pinn = torch.linspace(0, t_max, 600).view(-1, 1)

print("\nStarting Training (Vibration Convergence)...")
for epoch in range(12001):
    optimizer.zero_grad()
    
    # Physics Loss (Governing DAEs)
    loss_p = get_physics_loss(pinn, t_pinn)
    
    # IC Loss (t=0)
    t0 = torch.tensor([[0.0]], requires_grad=True)
    p0 = pinn(t0)
    x0_p = p0[:, 0:1]
    v0_p = torch.autograd.grad(x0_p, t0, torch.ones_like(x0_p), create_graph=True)[0]
    loss_ic = (x0_p - x0)**2 + (v0_p - v0)**2 + (p0[:, 1:2])**2
    
    total_loss = loss_p + 50 * loss_ic # Heavily weight ICs to fix the phase
    total_loss.backward()
    optimizer.step()
    
    if epoch % 3000 == 0:
        print(f"Epoch {epoch:5d} | Total Loss: {total_loss.item():.8f}")

# ==========================================
# 5. RESULTS COMPARISON TABLE
# ==========================================
t_test = torch.tensor(t_eval, dtype=torch.float32).view(-1, 1)
with torch.no_grad():
    results = pinn(t_test)
    x_pinn = results[:, 0].numpy()
    lam_pinn = results[:, 1].numpy() # Indexing lam from net output

# Slice for table display
df_comp = pd.DataFrame({
    "Time (s)": t_eval[::20],
    "RK45 x(t)": x_ref[::20],
    "PINN x(t)": x_pinn[::20],
    "Error (m)": np.abs(x_ref[::20] - x_pinn[::20])
})
print("\n--- COMPARISON: NUMERICAL VS. PINN ---")
print(df_comp.to_string(index=False))

# ==========================================
# 6. COMPREHENSIVE PLOTTING
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"System 3 Analysis: Damped Oscillator (zeta={zeta:.3f})", fontsize=16)

# Plot 1: Displacement x(t) - The Time Domain Response
axs[0, 0].plot(t_eval, x_ref, 'k--', label='Numerical (RK45)', alpha=0.5)
axs[0, 0].plot(t_eval, x_pinn, 'b-', label='PINN Trajectory')
axs[0, 0].set_title("Transient Displacement Response")
axs[0, 0].set_ylabel("Position x(t) [m]")
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot 2: Phase Portrait (State-Space)
# In Vibration engineering, this shows energy dissipation (stable spiral)
t_grad = t_test.clone().requires_grad_(True)
x_out = pinn(t_grad)[:, 0:1]
v_pinn = torch.autograd.grad(x_out, t_grad, torch.ones_like(x_out))[0].detach().numpy()

axs[0, 1].plot(x_ref, v_ref, 'k--', alpha=0.3, label='Numerical Path')
axs[0, 1].plot(x_pinn, v_pinn, 'r-', label='PINN State Path')
axs[0, 1].set_title("Phase Portrait (Stability Analysis)")
axs[0, 1].set_xlabel("Displacement [m]")
axs[0, 1].set_ylabel("Velocity [m/s]")
axs[0, 1].legend()
axs[0, 1].grid(True)

# Plot 3: Lagrange Multiplier (Constraint Force)
# Should converge to -mg = -98.1 N
axs[1, 0].plot(t_eval, results[:, 2].numpy(), color='purple', label='PINN λ (Normal Force)')
axs[1, 0].axhline(y=-m*g, color='black', linestyle=':', label='Theoretical (-mg)')
axs[1, 0].set_title("Constraint Force Identification")
axs[1, 0].set_ylabel("Force λ [N]")
axs[1, 0].set_xlabel("Time [s]")
axs[1, 0].legend()
axs[1, 0].grid(True)

# Plot 4: Absolute Error (Log Scale)
error = np.abs(x_ref - x_pinn)
axs[1, 1].semilogy(t_eval, error, color='green', label='Abs Error |Ref - PINN|')
axs[1, 1].set_title("Residual Convergence (Accuracy)")
axs[1, 1].set_ylabel("Error (log scale)")
axs[1, 1].set_xlabel("Time [s]")
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("vibration_pinn_system3.png")
plt.show()