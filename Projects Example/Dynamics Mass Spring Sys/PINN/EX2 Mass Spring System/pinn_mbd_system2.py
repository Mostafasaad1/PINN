import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# =================================================================
# 1. Physics Parameters (System 2: Mass-Spring)
# =================================================================
m = 10.0      # Mass (Inductance L analogy)
k = 50.0      # Spring Constant (1/C analogy)
g = 9.81      # Gravity
x0 = 1.0      # Initial displacement (Charge analogy)
v0 = 0.0      # Initial velocity (Current analogy)
t_max = 5.0
n_points = 500

# Analytical/Numerical Frequency: omega = sqrt(k/m)
omega = np.sqrt(k/m)

# =================================================================
# 2. Numerical Ground Truth (Benchmark)
# =================================================================
def system_dynamics(state, t):
    x, v = state
    return [v, -(k/m) * x]

t_numerical = np.linspace(0, t_max, n_points)
sol_numerical = odeint(system_dynamics, [x0, v0], t_numerical)
x_ref = sol_numerical[:, 0]
v_ref = sol_numerical[:, 1]
lambda_ref = -m * g * np.ones_like(t_numerical) # Normal force is constant mg

# =================================================================
# 3. PINN Architecture
# =================================================================
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3) # [x, y, lambda]
        )

    def forward(self, t):
        return self.net(t)

# =================================================================
# 4. Training the Physics-Informed Model
# =================================================================
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
t_physics = torch.linspace(0, t_max, n_points).view(-1, 1).requires_grad_(True)

print("--- Starting PINN Training ---")
for epoch in range(8001):
    optimizer.zero_grad()
    
    # 4.1 Forward Pass
    preds = model(t_physics)
    x, y, lam = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]

    # 4.2 Compute Derivatives (Automatic Differentiation)
    v_x = torch.autograd.grad(x, t_physics, torch.ones_like(x), create_graph=True)[0]
    a_x = torch.autograd.grad(v_x, t_physics, torch.ones_like(v_x), create_graph=True)[0]
    v_y = torch.autograd.grad(y, t_physics, torch.ones_like(y), create_graph=True)[0]
    a_y = torch.autograd.grad(v_y, t_physics, torch.ones_like(v_y), create_graph=True)[0]

    # 4.3 MBD Loss Terms (The Matrix Equivalents)
    loss_dyn_x = torch.mean((m * a_x + k * x)**2)          # m*x'' + k*x = 0
    loss_dyn_y = torch.mean((m * a_y + lam + m*g)**2)      # m*y'' + lambda + mg = 0
    loss_const = torch.mean(y**2)                          # y = 0 constraint

    # 4.4 Initial Conditions (t=0)
    t0 = torch.zeros((1, 1), requires_grad=True)
    p0 = model(t0)
    vx0 = torch.autograd.grad(p0[:, 0:1], t0, torch.ones_like(p0[:, 0:1]), create_graph=True)[0]
    loss_ic = (p0[:, 0:1] - x0)**2 + (vx0 - v0)**2 + (p0[:, 1:2] - 0.0)**2

    # Total Loss (Weighted)
    loss = loss_dyn_x + loss_dyn_y + 10*loss_const + 100*loss_ic
    loss.backward()
    optimizer.step()

    if epoch % 2000 == 0:
        print(f"Epoch {epoch:5} | Loss: {loss.item():.8f}")

# =================================================================
# 5. Output Comparison Table
# =================================================================
t_test = torch.tensor(t_numerical).float().view(-1, 1)
with torch.no_grad():
    res = model(t_test).numpy()
    x_pinn, y_pinn, lam_pinn = res[:, 0], res[:, 1], res[:, 2]

# Calculate Final Errors
mse_x = np.mean((x_ref - x_pinn)**2)
mse_lam = np.mean((lambda_ref - lam_pinn)**2)
max_y_error = np.max(np.abs(y_pinn))

print("\n" + "="*65)
print(f"{'Metric':<25} | {'Numerical':<12} | {'PINN':<12} | {'Error %':<10}")
print("-" * 65)
print(f"{'Peak Amplitude (x)':<25} | {np.max(x_ref):<12.4f} | {np.max(x_pinn):<12.4f} | {abs(np.max(x_ref)-np.max(x_pinn))/np.max(x_ref)*100:<10.2f}%")
print(f"{'Lagrange Multiplier (avg)':<25} | {np.mean(lambda_ref):<12.4f} | {np.mean(lam_pinn):<12.4f} | {abs(np.mean(lambda_ref)-np.mean(lam_pinn))/abs(np.mean(lambda_ref))*100:<10.2f}%")
print(f"{'Constraint Violation (y)':<25} | {0.0:<12.4f} | {max_y_error:<12.4f} | {max_y_error*100:<10.2f}%")
print("="*65 + "\n")

# =================================================================
# 6. Visualization
# =================================================================
plt.figure(figsize=(14, 10))

# Subplot 1: Horizontal Displacement (The Oscillation)
plt.subplot(3, 1, 1)
plt.plot(t_numerical, x_ref, 'k-', alpha=0.3, lw=5, label='Numerical (Ground Truth)')
plt.plot(t_numerical, x_pinn, 'r--', lw=2, label='PINN Prediction')
plt.title(f"System 2: Horizontal Motion ($m={m}$, $k={k}$)")
plt.ylabel("Displacement x (m)")
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Vertical Constraint & Lambda
plt.subplot(3, 1, 2)
plt.plot(t_numerical, lam_pinn, color='purple', label='PINN $\lambda$ (Normal Force)')
plt.axhline(y=-m*g, color='black', linestyle=':', label='Theoretical $-mg$')
plt.ylabel("Force (N)")
plt.title("Constraint Discovery: Lagrange Multiplier ($\lambda$) vs Gravity")
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 3: Phase Portrait (Energy State)
plt.subplot(3, 1, 3)
v_pinn = np.gradient(x_pinn, t_numerical) # Quick finite diff for plotting velocity
plt.plot(x_ref, v_ref, 'k', alpha=0.2, label='Numerical Orbit')
plt.plot(x_pinn, v_pinn, 'blue', label='PINN Orbit')
plt.xlabel("Position (x)")
plt.ylabel("Velocity (v)")
plt.title("State-Space Phase Portrait (Conservative System)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()