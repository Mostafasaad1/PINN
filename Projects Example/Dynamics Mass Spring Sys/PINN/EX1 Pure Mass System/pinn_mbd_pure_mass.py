import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==============================================================================
# PINN ARCHITECTURE FOR CONSTRAINED MULTIBODY DYNAMICS (MBD)
# ==============================================================================
# System: Pure Mass on a 2D plane with a horizontal force and a Y-constraint.
# Methodology: Replaces the DAE matrix solver with a neural residual minimizer.
# ==============================================================================

class PINNSolver(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: time (t) -> Output: [x(t), y(t), lambda(t)]
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3) 
        )
        
    def forward(self, t):
        return self.net(t)

# 1. PHYSICAL CONSTANTS (From MBD Analysis Document)
m = 10.0      # Mass (kg)
F_ext = 20.0  # External Force in X (N)
g = 9.81      # Gravity (m/s^2)

# 2. TRAINING SETUP
model = PINNSolver()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Spatiotemporal Domain: 0 to 5 seconds
t_domain = torch.linspace(0, 5, 200).view(-1, 1).requires_grad_(True)
t_0 = torch.tensor([[0.0]], requires_grad=True)

print("🚀 Starting PINN Training: Minimizing DAE Residuals...")

for epoch in range(6001):
    optimizer.zero_grad()
    
    # --- PREDICTIONS ---
    q = model(t_domain)
    x, y, lam = q[:, 0:1], q[:, 1:2], q[:, 2:3]
    
    # --- AUTOMATIC DIFFERENTIATION (The Physics Engine) ---
    # First derivatives (Velocity)
    dx_dt = torch.autograd.grad(x, t_domain, torch.ones_like(x), create_graph=True)[0]
    dy_dt = torch.autograd.grad(y, t_domain, torch.ones_like(y), create_graph=True)[0]
    
    # Second derivatives (Acceleration)
    d2x_dt2 = torch.autograd.grad(dx_dt, t_domain, torch.ones_like(dx_dt), create_graph=True)[0]
    d2y_dt2 = torch.autograd.grad(dy_dt, t_domain, torch.ones_like(dy_dt), create_graph=True)[0]
    
    # --- RESIDUAL CALCULATIONS (Physics Constraints) ---
    # Eq 1: m*x'' = F_ext
    res_x = m * d2x_dt2 - F_ext
    
    # Eq 2: m*y'' + Phi_y^T * lambda = -m*g
    # For constraint y=0, the Jacobian Phi_q is [0, 1]. Thus Phi_q^T * lambda is [0; lambda]
    res_y = m * d2y_dt2 - (-m * g + lam)
    
    # Eq 3: Geometric Constraint Phi(q) = 0
    res_constraint = y 
    
    # --- BOUNDARY CONDITIONS (Initial Conditions) ---
    q_0 = model(t_0)
    x_0 = q_0[:, 0:1]
    # Velocity at t=0
    v_0 = torch.autograd.grad(x_0, t_0, torch.ones_like(x_0), create_graph=True)[0]
    
    loss_ic = torch.mean(x_0**2) + torch.mean(v_0**2) # x(0)=0, v(0)=0
    loss_physics = torch.mean(res_x**2) + torch.mean(res_y**2) + torch.mean(res_constraint**2)
    
    total_loss = loss_ic + loss_physics
    total_loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} | Residual Loss: {total_loss.item():.2e}")

# 3. VERIFICATION & PLOTTING
with torch.no_grad():
    t_test = torch.linspace(0, 5, 100).view(-1, 1)
    results = model(t_test).numpy()
    t_p = t_test.numpy()

# Analytical validation
x_analyt = 0.5 * (F_ext/m) * t_p**2

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t_p, x_analyt, 'k--', alpha=0.6, label='Analytical (1/2at^2)')
plt.plot(t_p, results[:, 0], 'r', label='PINN Prediction (x)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.title('System 1: Pure Mass Dynamics (Numerical Method Replacement)')

plt.subplot(2, 1, 2)
plt.axhline(y=m*g, color='k', linestyle='--', label='Theoretical λ (mg)')
plt.plot(t_p, results[:, 2], 'g', label='Inferred λ (Lagrange Multiplier)')
plt.ylabel('Reaction Force (N)')
plt.xlabel('Time (s)')
plt.legend()
plt.tight_layout()
plt.show()