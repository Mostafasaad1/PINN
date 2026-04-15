import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import os

# ==========================================
# 1. PHYSICAL PARAMETERS & ANALYTICAL BASIS
# ==========================================
m, k, c, g = 10.0, 200.0, 15.0, 9.81
x0, v0 = 1.2, 3.0
t_max = 5.0
MODEL_PATH = "system3_pinn_model.pth"

# ==========================================
# 2. PINN ARCHITECTURE
# ==========================================
class VibrationPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 3) # [x, y, lambda]
        )

    def forward(self, t):
        return self.net(t)

# Physics loss function (Same as before)
def get_physics_loss(model, t_collocation):
    t_collocation.requires_grad = True
    pred = model(t_collocation)
    x, y, lam = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    dx_dt = torch.autograd.grad(x, t_collocation, torch.ones_like(x), create_graph=True)[0]
    d2x_dt2 = torch.autograd.grad(dx_dt, t_collocation, torch.ones_like(dx_dt), create_graph=True)[0]
    dy_dt = torch.autograd.grad(y, t_collocation, torch.ones_like(y), create_graph=True)[0]
    d2y_dt2 = torch.autograd.grad(dy_dt, t_collocation, torch.ones_like(dy_dt), create_graph=True)[0]

    res_x = m * d2x_dt2 + c * dx_dt + k * x 
    res_y = m * d2y_dt2 + lam + m * g
    res_phi = y 
    return torch.mean(res_x**2) + torch.mean(res_y**2) + 10 * torch.mean(res_phi**2)

# ==========================================
# 3. TRAINING / LOADING LOGIC
# ==========================================
pinn = VibrationPINN()

if os.path.exists(MODEL_PATH):
    print(f"--- Found saved model: {MODEL_PATH} ---")
    print("Loading weights... Skipping training.")
    pinn.load_state_dict(torch.load(MODEL_PATH))
    pinn.eval() # Set to evaluation mode
else:
    print("--- No saved model found. Commencing Training... ---")
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    t_train = torch.linspace(0, t_max, 600).view(-1, 1)

    for epoch in range(12001):
        optimizer.zero_grad()
        loss_p = get_physics_loss(pinn, t_train)
        
        t0 = torch.tensor([[0.0]], requires_grad=True)
        p0 = pinn(t0)
        x0_p = p0[:, 0:1]
        v0_p = torch.autograd.grad(x0_p, t0, torch.ones_like(x0_p), create_graph=True)[0]
        loss_ic = (x0_p - x0)**2 + (v0_p - v0)**2 + (p0[:, 1:2])**2
        
        total_loss = loss_p + 50 * loss_ic
        total_loss.backward()
        optimizer.step()
        
        if epoch % 3000 == 0:
            print(f"Epoch {epoch:5d} | Loss: {total_loss.item():.8f}")

    # SAVE THE MODEL
    torch.save(pinn.state_dict(), MODEL_PATH)
    print(f"Training complete. Model saved to {MODEL_PATH}")

# ==========================================
# 4. INFERENCE & VISUALIZATION (Consistent with previous)
# ==========================================
# [Benchmark Code]
def mbd_ground_truth(t, state):
    x, v = state
    return [v, (-k*x - c*v) / m]

t_eval = np.linspace(0, t_max, 200)
sol = solve_ivp(mbd_ground_truth, [0, t_max], [x0, v0], t_eval=t_eval)

t_test = torch.tensor(t_eval, dtype=torch.float32).view(-1, 1)
with torch.no_grad():
    results = pinn(t_test)
    x_pinn = results[:, 0].numpy()

# Comparison Table
df_comp = pd.DataFrame({
    "Time (s)": t_eval[::20],
    "RK45 x(t)": sol.y[0][::20],
    "PINN x(t)": x_pinn[::20],
    "Error": np.abs(sol.y[0][::20] - x_pinn[::20])
})
print("\n--- PERFORMANCE VALIDATION ---")
print(df_comp.to_string(index=False))

# Plotting and Saving Figure
plt.figure(figsize=(10, 5))
plt.plot(t_eval, sol.y[0], 'k--', label='RK45 (Numerical)', alpha=0.5)
plt.plot(t_eval, x_pinn, 'b-', label='PINN (Stored Model)')
plt.title("System 3: Saved Model Validation")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.legend()
plt.grid(True)
plt.savefig("system3_reused_model.png")
plt.show()