import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

# 1. System Parameters (From MBD_Systems_Analysis.md)
m = 10.0   # kg (Inductance L)
k = 200.0  # N/m (Elastance 1/C)
c = 15.0   # Ns/m (Resistance R)
g = 9.81   # m/s^2
x0 = 1.2   # Initial displacement
v0 = 3.0   # Initial velocity
t_max = 5.0

# 2. Numerical Method (Benchmark for comparison)
def mbd_numerical(t, state):
    x, v = state
    dxdt = v
    # Acceleration from MBD derivation: (F_ext - c*v - k*x) / m
    dvdt = (-k*x - c*v) / m
    return [dxdt, dvdt]

t_eval = np.linspace(0, t_max, 100)
sol = solve_ivp(mbd_numerical, [0, t_max], [x0, v0], t_eval=t_eval)
x_numerical = sol.y[0]

# 3. PINN Architecture
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 2) # Outputs: [x, lambda]
        )

    def forward(self, t):
        return self.net(t)

def physics_loss(model, t):
    t.requires_grad = True
    pred = model(t)
    x = pred[:, 0:1]
    lam = pred[:, 1:2]

    # Derivatives
    dx_dt = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    d2x_dt2 = torch.autograd.grad(dx_dt, t, torch.ones_like(dx_dt), create_graph=True)[0]

    # Residuals: m*x'' + c*x' + k*x = 0
    res_x = m * d2x_dt2 + c * dx_dt + k * x
    # Note: For y=0, lambda = -mg. We check if the network discovers this.
    res_lam = lam + m*g 

    return torch.mean(res_x**2) + torch.mean(res_lam**2)

# 4. Training
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
t_train = torch.linspace(0, t_max, 500).view(-1, 1)

print("Training PINN...")
for epoch in range(12001):
    optimizer.zero_grad()
    
    # Physics and IC Loss
    l_physics = physics_loss(model, t_train)
    
    t0 = torch.zeros((1, 1), requires_grad=True)
    p0 = model(t0)
    x0_p = p0[:, 0:1]
    v0_p = torch.autograd.grad(x0_p, t0, torch.ones_like(x0_p), create_graph=True)[0]
    
    l_ic = (x0_p - x0)**2 + (v0_p - v0)**2
    
    loss = l_physics + 10 * l_ic
    loss.backward()
    optimizer.step()
    
    if epoch % 4000 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

# 5. Comparison and Printing Table
t_test = torch.tensor(t_eval, dtype=torch.float32).view(-1, 1)
with torch.no_grad():
    pinn_pred = model(t_test)
    x_pinn = pinn_pred[:, 0].numpy()
    lam_pinn = pinn_pred[:, 1].numpy()

# Create Comparison Table
comparison_data = {
    "Time (s)": t_eval[::10],
    "Numerical x(t)": x_numerical[::10],
    "PINN x(t)": x_pinn[::10],
    "Error": np.abs(x_numerical[::10] - x_pinn[::10])
}
df = pd.DataFrame(comparison_data)
print("\n--- RESULTS COMPARISON TABLE ---")
print(df.to_string(index=False))

# 6. Plotting
plt.figure(figsize=(10, 6))
plt.plot(t_eval, x_numerical, 'r--', label='Numerical (RK45)', alpha=0.6)
plt.plot(t_eval, x_pinn, 'b-', label='PINN (Physics-Informed)')
plt.title(f"Mass-Spring-Damper (m={m}, k={k}, c={c})")
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.legend()
plt.grid(True)
plt.savefig("system3_pinn_vs_numerical.png")
plt.show()